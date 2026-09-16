#!/usr/bin/env python
"""multi-view update 实验：每样本 6 个选中视图连发 update（统计量收 6 份），聚合特征 predict 一次。

纯 LR LDA（fuse=none），其余配置对齐 65.71 配方（r64/l=k/streaming/add ε=0.05）。
用法: python run_multi_update.py --gpu 2 [--fuse none]
"""
import os
import sys
import argparse
import time

import numpy as np
import torch

sys.path.insert(0, '/home/liangyiwen/wufan/lowrank_selftrain')
from method import LowRankLDASelfTrain

PTH_TEMPLATE = '/home/liangyiwen/wufan/ADAPT/pre_extracted_class_feat/{ARCH}/GPT_w_Custom_class_emb'
FEATS_DIR = '/home/liangyiwen/wufan/ADAPT/feats'
DATASETS = ['imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r', 'imagenet']


@torch.no_grad()
def run(ds, gpu, rank, sketch, shrink, sigma_eps, warmup=64):
    d = np.load(os.path.join(FEATS_DIR, f'feats_multi_{ds}.npz'))
    Fv = torch.from_numpy(d['feats']).float()             # [N, k, D] 未聚合视图
    fmean = torch.from_numpy(d['feats_mean']).float()     # [N, D] 聚合
    soft = torch.from_numpy(d['soft']).float()            # [N, C] 聚合软标签
    tgt = torch.from_numpy(d['targets']).long()
    N, K, D = Fv.shape
    C = soft.shape[1]
    dev = f'cuda:{gpu}'
    Fv, fmean, soft, tgt = Fv.to(dev), fmean.to(dev), soft.to(dev), tgt.to(dev)

    m = LowRankLDASelfTrain(C, D, dev, warmup, rank=rank, sketch_mult=sketch,
                            mu_mode='streaming', sigma_shrink=shrink, sigma_eps=sigma_eps)
    ok = 0
    for t in range(N):
        s = soft[t]
        for j in range(K):                                 # 同一样本 6 视图连发 update
            m.update(Fv[t, j], s)
        if m.ready():
            pred = int(m.predict(fmean[t]).argmax().item())
        else:
            pred = int(s.argmax().item())
        ok += int(pred == int(tgt[t]))
    return 100.0 * ok / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--datasets', nargs='*', default=DATASETS)
    ap.add_argument('--rank', type=int, default=64)
    ap.add_argument('--sketch', type=int, default=1)
    ap.add_argument('--shrink', default='add')
    ap.add_argument('--sigma_eps', type=float, default=0.05)
    a = ap.parse_args()
    per = {}
    for ds in a.datasets:
        t0 = time.time()
        acc = run(ds, a.gpu, a.rank, a.sketch, a.shrink, a.sigma_eps)
        per[ds] = acc
        print(f'[multi] {ds:16s} acc={acc:6.2f}%  r={a.rank}/s={a.sketch}/{a.shrink}{a.sigma_eps} '
              f'({time.time()-t0:.0f}s)', flush=True)
    print(f'[multi] AVG = {sum(per.values())/len(per):.2f}%', flush=True)


if __name__ == '__main__':
    main()
