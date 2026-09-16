#!/usr/bin/env python
"""Transductive LR LDA 在 ViT-B/16 FG10 上（对齐 ADAPT_transductive 的评估协议）。

ADAPT_transductive 的 setting（FG10 分支）：
- 单视图特征（无 63-view 增强；仅 imagenet 4 域走增强分支）
- clip_logits = 100·f@W（W=GPT_w_Custom_class_emb proto），soft label = softmax(clip_logits)
- seed=0 / DataLoader shuffle=True / batch=64（特征序=该协议）
- transductive：整域一次性估计参数 → 统一打分，不是在线流式

本 runner 把"判别器"换成 LR LDA（LowRankLDA）：
- 全量 update（streaming mu = soft label 加权均值，与 ADAPT 的 soft 权重一致）
- update 完毕后再统一 predict 全体（offline，非"每样本 predict 后 update"）
- 只报 LR LDA 判别器 acc 与 CLIP zero-shot 基线（不含 ADAPT 的 bank/相似度/cache 融合，
  那些是 ADAPT_transductive 独有模块，LR LDA 判别器不适用）
"""
import os
import sys
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from method import LowRankLDA

FE = '/home/liangyiwen/wufan/lowrank_selftrain/feats'
PTH = '/home/liangyiwen/wufan/ADAPT/pre_extracted_class_feat/ViT-B16/GPT_w_Custom_class_emb'
FG10 = ['fgvc_aircraft', 'caltech101', 'stanford_cars', 'dtd', 'eurosat',
        'oxford_flowers', 'food101', 'oxford_pets', 'sun397', 'ucf101']


@torch.no_grad()
def zero_shot(feats, tgt, Wt, dev, chunk=50000):
    ok = torch.zeros(feats.shape[0], dtype=torch.bool)
    for i in range(0, feats.shape[0], chunk):
        X = feats[i:i + chunk].to(dev)
        logits = 100.0 * X @ Wt.to(dev)
        ok[i:i + chunk] = (logits.argmax(1) == tgt[i:i + chunk].to(dev)).cpu()
    return 100.0 * ok.float().mean().item()


@torch.no_grad()
def run_transductive(feats, tgt, Wt, gpu, rank, sketch, shrink, eps, warmup=64, batch=4096):
    dev = f'cuda:{gpu}'
    Wt = Wt.to(dev)
    C, D = Wt.shape[1], feats.shape[1]
    N = feats.shape[0]
    m = LowRankLDA(C, D, dev, warmup, rank=rank, sketch_mult=sketch,
                            sigma_shrink=shrink, sigma_eps=eps)
    for t in range(N):                       # 全量 offline update（与 ADAPT 的整域估计对齐）
        x = feats[t].to(dev)
        ll = 100.0 * (x @ Wt)
        m.update(x, ll.softmax(0))
    ok = 0
    for i in range(0, N, batch):
        X = feats[i:i + batch].to(dev)
        scores = m.predict(X).t()            # (batch, C)  —— 统一打分
        ok += int((scores.argmax(1) == tgt[i:i + batch].to(dev)).sum().item())
    return 100.0 * ok / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=6)
    ap.add_argument('--rank', type=int, default=64)
    ap.add_argument('--sketch', type=int, default=2)
    ap.add_argument('--shrink', default='iso', choices=['iso', 'add', 'trace', 'tfull', 'trade'])
    ap.add_argument('--eps', type=float, default=0.0)
    ap.add_argument('--datasets', nargs='*', default=None)
    a = ap.parse_args()
    dsets = a.datasets or FG10
    rows = []
    for ds in dsets:
        t0 = time.time()
        d = np.load(os.path.join(FE, f'{ds}.npz'))
        feats = torch.from_numpy(d['feats']).float()
        tgt = torch.from_numpy(d['targets']).long()
        Wt = torch.load(os.path.join(PTH, f'{ds}.pth'), map_location='cpu').float()
        if Wt.shape[0] != feats.shape[1]:
            Wt = Wt.t()
        zs = zero_shot(feats, tgt, Wt, a.gpu)
        lr = run_transductive(feats, tgt, Wt, a.gpu, a.rank, a.sketch, a.shrink, a.eps)
        rows.append((ds, zs, lr))
        print(f'[lr-trans] {ds:15s} clip={zs:6.2f}  lr={lr:6.2f}  '
              f'(r={a.rank}/s={a.sketch}/{a.shrink}/{a.eps}) ({time.time()-t0:.0f}s)', flush=True)
    if len(rows) > 1:
        z = sum(r[1] for r in rows) / len(rows)
        l = sum(r[2] for r in rows) / len(rows)
        print(f'[lr-trans] AVG({len(rows)}域) clip={z:.2f}  lr={l:.2f}', flush=True)


if __name__ == '__main__':
    main()