#!/usr/bin/env python
"""imagenet 变体 transductive runner：ImageNet-C 15 域（全量 update → 统一打分）。

注意：ImageNet 系 5 域没有 transductive 协议结果（仅 FG10 与 C 有）。
Core = LowRankLDA（AAᵀ + σ²I FD streaming，见 method.py）。
"""
import os
import sys
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, '/home/liangyiwen/wufan/ADAPT')
from clip import clip as clip_pkg  # noqa: F401  （proto 加载无需模型，保留一致性）
sys.path.insert(0, '/home/liangyiwen/wufan/lowrank_selftrain')
from method import LowRankLDA

C_PTH = {
    'vit': '/home/liangyiwen/wufan/ADAPT/pre_extracted_class_feat/ViT-B16/GPT_w_Custom_class_emb/imagenet.pth',
    'rn50': '/home/liangyiwen/wufan/ADAPT/pre_extracted_class_feat/RN50/GPT_w_Custom_class_emb/imagenet.pth',
}
C_OUT = {
    'vit': '/mnt/newdata/liangyiwen/imagenet_c_feats',
    'rn50': '/mnt/newdata/liangyiwen/imagenet_c_feats_rn50',
}
C_OUT_AUG63 = '/mnt/newdata/liangyiwen/imagenet_c_feats_aug63'

CORDER = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
          'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
          'gaussian_noise', 'shot_noise', 'impulse_noise']


@torch.no_grad()
def c_zero_shot(feats, tgt, Wt, dev, chunk=50000):
    ok = torch.zeros(feats.shape[0], dtype=torch.bool)
    for i in range(0, feats.shape[0], chunk):
        X = feats[i:i + chunk].to(dev)
        logits = 100.0 * X @ Wt.to(dev)
        ok[i:i + chunk] = (logits.argmax(1) == tgt[i:i + chunk].to(dev)).cpu()
    return 100.0 * ok.float().mean().item()


@torch.no_grad()
def c_run_trans(corr, gpu, rank, sketch, shrink, eps, warmup=64, batch=4096,
                backbone='vit', cache='singlev'):
    fdir = C_OUT_AUG63 if cache == 'aug63' else C_OUT[backbone]
    d = np.load(os.path.join(fdir, f'{corr}.npz'))
    feats = torch.from_numpy(d['feats']).float()
    tgt = torch.from_numpy(d['targets']).long()
    if cache == 'aug63':
        clip_ll = torch.from_numpy(d['clip_ll']).float()
        C, D = clip_ll.shape[1], feats.shape[1]
        Wt, zs = None, 100.0 * (clip_ll.argmax(1) == tgt).float().mean().item()
    else:
        Wt = torch.load(C_PTH[backbone], map_location='cpu').float()
        if Wt.shape[0] != feats.shape[1]:
            Wt = Wt.t()
        C, D = Wt.shape[1], feats.shape[1]
        clip_ll, zs = None, None
    dev = f'cuda:{gpu}'
    if Wt is not None:
        Wt = Wt.to(dev)
    N = feats.shape[0]
    if zs is None:
        zs = c_zero_shot(feats, tgt, Wt, gpu)

    m = LowRankLDA(C, D, dev, warmup, rank=rank, sketch_mult=sketch,
                            sigma_shrink=shrink, sigma_eps=eps)
    for t in range(N):
        x = feats[t].to(dev)
        if clip_ll is not None:
            m.update(x, clip_ll[t].softmax(0).to(dev))
        else:
            m.update(x, (100.0 * (x @ Wt)).softmax(0))
    ok = 0
    for i in range(0, N, batch):
        X = feats[i:i + batch].to(dev)
        scores = m.predict(X).t()
        ok += int((scores.argmax(1) == tgt[i:i + batch].to(dev)).sum().item())
    return zs, 100.0 * ok / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=5)
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--rank', type=int, default=64)
    ap.add_argument('--sketch', type=int, default=1)
    ap.add_argument('--shrink', default='iso', choices=['iso', 'add', 'trace', 'tfull', 'trade'])
    ap.add_argument('--eps', type=float, default=0.0)
    ap.add_argument('--backbone', default='vit', choices=['vit', 'rn50'])
    ap.add_argument('--cache', default='singlev', choices=['singlev', 'aug63'])
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    rows = []
    for corr in (a.datasets or CORDER):
        t0 = time.time()
        zs, lr = c_run_trans(corr, a.gpu, a.rank, a.sketch, a.shrink, a.eps,
                             backbone=a.backbone, cache=a.cache)
        rows.append((corr, zs, lr))
        print(f'[ic-tr] {corr:18s} clip={zs:6.2f}  lr(trans)={lr:6.2f}  '
              f'(r={a.rank}/s={a.sketch}/{a.backbone}/{a.cache}) '
              f'({time.time()-t0:.0f}s)', flush=True)
    if len(rows) > 1:
        z = sum(r[1] for r in rows) / len(rows)
        l = sum(r[2] for r in rows) / len(rows)
        print(f'[ic-tr] AVG({len(rows)}域) clip={z:.2f}  lr(trans)={l:.2f}', flush=True)


if __name__ == '__main__':
    main()
