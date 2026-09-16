#!/usr/bin/env python
"""Minimal FG10 runner for RN50 features (low-rank LDA, stream/selftrain, fuse none).

Reads RN50 singlev feats (ADAPT/feats/rn50) + RN50 text protos
(ADAPT/pre_extracted_class_feat/RN50). Mirrors lr_inet.py logic but for FG10
datasets (no DOTA rho/eta schedule -> fuse none).
"""
import os, torch
import numpy as np
import sys
sys.path.insert(0, '/home/liangyiwen/wufan/lowrank_selftrain')
from method import LowRankLDASelfTrain

W = '/home/liangyiwen/wufan'
FE = f'{W}/ADAPT/feats/rn50'
PTH = f'{W}/ADAPT/pre_extracted_class_feat/RN50/GPT_w_Custom_class_emb'
FG10 = ['fgvc_aircraft', 'caltech101', 'stanford_cars', 'dtd', 'eurosat',
        'oxford_flowers', 'food101', 'oxford_pets', 'sun397', 'ucf101']


def run_ds(ds, gpu, mu, rank=64, warmup=64):
    dev = f'cuda:{gpu}'
    d = np.load(os.path.join(FE, f'singlev_{ds}.npz'))
    feats = torch.from_numpy(d['feats']).float().to(dev)
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tgt = torch.from_numpy(d['targets']).long().to(dev)
    Wt = torch.load(os.path.join(PTH, f'{ds}.pth'), map_location='cpu').float().to(dev)
    if Wt.shape[0] != feats.shape[1]:
        Wt = Wt.t()
    C, D = Wt.shape[1], feats.shape[1]
    N = feats.shape[0]
    m = LowRankLDASelfTrain(C, D, dev, warmup, rank=rank, sketch_mult=2, mu_mode=mu)
    correct = 0
    for t in range(N):
        x = feats[t]
        clip = torch.softmax(100.0 * x @ Wt, 0)
        m.update(x, clip)
        pred = int(m.predict(x).argmax()) if m.ready() else int(clip.argmax())
        correct += int(pred == tgt[t].item())
    return 100.0 * correct / N


def zero_shot(ds, gpu):
    dev = f'cuda:{gpu}'
    d = np.load(os.path.join(FE, f'singlev_{ds}.npz'))
    feats = torch.from_numpy(d['feats']).float().to(dev)
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tgt = torch.from_numpy(d['targets']).long().to(dev)
    Wt = torch.load(os.path.join(PTH, f'{ds}.pth'), map_location='cpu').float().to(dev)
    if Wt.shape[0] != feats.shape[1]:
        Wt = Wt.t()
    logits = 100.0 * feats @ Wt
    return (logits.argmax(1) == tgt).float().mean().item() * 100


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=5)
    ap.add_argument('--mu', default='streaming')
    ap.add_argument('--datasets', nargs='*', default=None)
    a = ap.parse_args()
    dsets = a.datasets or FG10
    torch.cuda.set_device(a.gpu)
    zs, accs = {}, {}
    for ds in dsets:
        zs[ds] = zero_shot(ds, a.gpu)
        accs[ds] = run_ds(ds, a.gpu, a.mu)
        print(f'[fg10-rn50] {ds:16s} zs={zs[ds]:6.2f}  lr({a.mu})={accs[ds]:6.2f}', flush=True)
    print(f'[fg10-rn50] AVG zs={sum(zs.values())/len(dsets):.2f}  lr({a.mu})={sum(accs.values())/len(dsets):.2f}')