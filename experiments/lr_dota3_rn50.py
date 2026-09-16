#!/usr/bin/env python
"""RN50 上对每个数据集一次性输出三路准确率：纯LDA / 原始CLIP(zero-shot) / DOTA加权融合。

- LDA = LowRankLDASelfTrain (rank=64, streaming mu) —— 项目主推 low-rank 判别器。
- CLIP = zero-shot (100*x@W logits argmax)。
- DOTA 融合 = clip_ll + clamp(rho*Nc.mean(), 0, eta) * lr_score，rho/eta 取官方 DOTA 超参
  (FG10 从 DOTA/configs/vit 各域 yaml；imagenet 系 = DOTA_RHO_ETA 即 lr_inet.py)。

三种预测来自同一次在线状态；预测非在线的 CLIP 准确率用 chunk 一次性计算。
"""
import os
import sys
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from method import LowRankLDASelfTrain

W = '/home/liangyiwen/wufan'
FE = f'{W}/ADAPT/feats/rn50'
PTH = f'{W}/ADAPT/pre_extracted_class_feat/RN50/GPT_w_Custom_class_emb'

FG10 = ['fgvc_aircraft', 'caltech101', 'stanford_cars', 'dtd', 'eurosat',
        'oxford_flowers', 'food101', 'oxford_pets', 'sun397', 'ucf101']
INET = ['imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r', 'imagenet']

# (rho, eta) —— FG10 官方 DOTA configs/vit yaml；(ds->yaml 映射: fgvc_aircraft->fgvc)
FG10_RHO_ETA = {
    'fgvc_aircraft': (0.01, 0.2), 'caltech101': (0.02, 0.3), 'stanford_cars': (0.01, 0.3),
    'dtd': (0.02, 0.3), 'eurosat': (0.05, 3.0), 'oxford_flowers': (0.02, 0.2),
    'food101': (0.01, 0.1), 'oxford_pets': (0.02, 0.3), 'sun397': (0.01, 0.2), 'ucf101': (0.02, 0.4),
}
# imagenet 系 —— 与官方 yaml 一致（imagenet/_a/_r/_s/_v）
INET_RHO_ETA = {
    'imagenet': (0.005, 0.2), 'imagenetv2': (0.01, 0.1), 'imagenet_sketch': (0.03, 0.8),
    'imagenet_a': (0.02, 0.5), 'imagenet_r': (0.0025, 0.3),
}


def load(ds):
    d = np.load(os.path.join(FE, f'singlev_{ds}.npz'))
    feats = torch.from_numpy(d['feats']).float()
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tgt = torch.from_numpy(d['targets']).long()
    Wt = torch.load(os.path.join(PTH, f'{ds}.pth'), map_location='cpu').float()
    if Wt.shape[0] != feats.shape[1]:          # expect (D, C)
        Wt = Wt.t()
    return feats, tgt, Wt


@torch.no_grad()
def clip_acc(feats, tgt, Wt, gpu, chunk=50000):
    dev = f'cuda:{gpu}'
    ok = torch.zeros(feats.shape[0], dtype=torch.bool)
    for i in range(0, feats.shape[0], chunk):
        X = feats[i:i + chunk].to(dev)
        logits = 100.0 * X @ Wt.to(dev)
        ok[i:i + chunk] = (logits.argmax(1) == tgt[i:i + chunk].to(dev)).cpu()
    return 100.0 * ok.float().mean().item()


@torch.no_grad()
def run_online(feats, tgt, Wt, gpu, rho, eta, warmup, rank, mu):
    dev = f'cuda:{gpu}'
    feats = feats.to(dev); tgt = tgt.to(dev); Wt = Wt.to(dev)
    N, D = feats.shape
    C = Wt.shape[1]
    m = LowRankLDASelfTrain(C, D, dev, warmup, rank=rank, sketch_mult=2, mu_mode=mu)
    lda_ok = dota_ok = 0
    for t in range(N):
        x = feats[t]
        clip_ll = 100.0 * x @ Wt
        m.update(x, clip_ll.softmax(0))
        if m.ready():
            lr = m.predict(x)
            lda_ok += int(lr.argmax().item() == tgt[t].item())
            w = min(max(rho * float(m.Nc.mean()), 0.0), eta)
            final = clip_ll + w * lr
            dota_ok += int(final.argmax().item() == tgt[t].item())
        else:
            lda_ok += int(clip_ll.argmax().item() == tgt[t].item())
            dota_ok += int(clip_ll.argmax().item() == tgt[t].item())
    return 100.0 * lda_ok / N, 100.0 * dota_ok / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', default='fg10', choices=['fg10', 'inet', 'all'])
    ap.add_argument('--gpu', type=int, default=5)
    ap.add_argument('--mu', default='streaming')
    ap.add_argument('--rank', type=int, default=64)
    ap.add_argument('--warmup', type=int, default=64)
    ap.add_argument('--datasets', nargs='*', default=None)
    a = ap.parse_args()
    if a.datasets:
        dsets = a.datasets
    elif a.group == 'fg10':
        dsets = FG10
    elif a.group == 'inet':
        dsets = INET
    else:
        dsets = FG10 + INET
    roe = {**FG10_RHO_ETA, **INET_RHO_ETA}
    torch.cuda.set_device(a.gpu)
    rows = []
    for ds in dsets:
        t0 = time.time()
        feats, tgt, Wt = load(ds)
        zs = clip_acc(feats, tgt, Wt, a.gpu)
        rho, eta = roe[ds]
        lda, dta = run_online(feats, tgt, Wt, a.gpu, rho, eta, a.warmup, a.rank, a.mu)
        rows.append((ds, zs, lda, dta))
        print(f'[dota3] {ds:16s} clip={zs:6.2f}  lda={lda:6.2f}  dota(ρ={rho},η={eta:g})={dta:6.2f}  '
              f'({time.time()-t0:.0f}s)', flush=True)
    if len(rows) > 1:
        z = sum(r[1] for r in rows) / len(rows); l = sum(r[2] for r in rows) / len(rows)
        d = sum(r[3] for r in rows) / len(rows)
        print(f'[dota3] AVG({len(rows)}域) clip={z:.2f}  lda={l:.2f}  dota={d:.2f}', flush=True)


if __name__ == '__main__':
    main()