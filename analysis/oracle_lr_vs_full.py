#!/usr/bin/env python
"""Oracle-label comparison: LR LDA vs full-rank LDA, pure discriminator (no fusion).

Question: does the low-rank approximation lose discriminative power vs full-rank
when label noise is removed? 2x2 design: {LR r=64, Full D=512} x {soft CLIP labels,
oracle one-hot true labels}, streaming update-then-predict, aug63 features, same
Welford centering and add-shrinkage for both. Plus a batch (non-streaming) oracle
full-LDA ceiling per domain for reference.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根
from method import LowRankLDA

FEATS_DIR = os.environ.get('AUG63_FEATS_DIR', '/home/liangyiwen/wufan/ADAPT/feats')
DOMAINS = ['imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r', 'imagenet']


def load(ds):
    d = np.load(os.path.join(FEATS_DIR, f'aug63_{ds}.npz'))
    return (torch.from_numpy(d['feats']).float(),
            torch.from_numpy(d['targets']).long())


class FullLDA:
    """Full-rank streaming LDA: identical update semantics to LowRankLDA but
    keeps the complete scatter matrix T [D, D] instead of an FD sketch."""

    def __init__(self, C, D, device, warmup, eps=0.05):
        self.C, self.D, self.device, self.warmup = C, D, device, warmup
        self.eps = eps
        self.Sc = torch.zeros(C, D, device=device)
        self.Nc = torch.zeros(C, device=device)
        self.T = torch.zeros(D, D, device=device)
        self.n_cc = 0
        self.n_eff = torch.zeros((), device=device)
        self._epsI = eps * torch.eye(D, device=device)

    def update(self, x, s):
        c = int(s.argmax())
        w_c = s[c].clamp_min(0.0)
        mu_old = self.Sc[c] / self.Nc[c].clamp_min(1e-8)
        Nc_old = self.Nc[c]
        self.Sc += s.unsqueeze(1) * x.unsqueeze(0)
        self.Nc += s
        wf = torch.sqrt((Nc_old * w_c / (Nc_old + w_c)).clamp_min(1e-12))
        xp = wf * (x - mu_old)
        self.T += xp.unsqueeze(1) * xp.unsqueeze(0)
        self.n_cc += 1
        self.n_eff += w_c

    def ready(self):
        return self.n_cc > self.warmup

    def predict(self, x):
        mu = self.Sc / self.Nc.clamp_min(1e-8).unsqueeze(1)
        Sig = self.T / self.n_eff.clamp_min(1e-12) + self._epsI
        # solve instead of explicit inverse: Y = Sig^{-1} mu  [D, C]
        Y = torch.linalg.solve(Sig, mu.t())
        muPx = Y.t() @ x                                  # [C] = mu^T Sig^{-1} x
        muPmu = (mu * Y.t()).sum(1)                       # [C] = mu^T Sig^{-1} mu
        return muPx - 0.5 * muPmu


def run_streaming(ds, mode_cls, label_mode, gpu, warmup=64, rank=64, eps=0.05):
    F, tgt = load(ds)
    device = f'cuda:{gpu}'
    F, tgt = F.to(device), tgt.to(device)
    N, D = F.shape
    C = int(tgt.max()) + 1
    if label_mode == 'soft':
        W = torch.load(os.path.join(
            os.environ.get('PTH_ROOT',
                           '/home/liangyiwen/wufan/ADAPT/pre_extracted_class_feat/ViT-B16') + '/'
            'GPT_w_Custom_class_emb', f'{ds}.pth'),
            map_location='cpu', weights_only=True).float()
        if W.shape[0] != D:
            W = W.t()
        W = W.to(device)
    if mode_cls == 'lr':
        m = LowRankLDA(C, D, device, warmup, rank=rank, sketch_mult=1,
                       sigma_shrink='add', sigma_eps=eps)
    else:
        m = FullLDA(C, D, device, warmup, eps=eps)
    eye = torch.eye(C, device=device)
    ok, tot = 0, 0
    for t in range(N):
        x = F[t]
        s = eye[tgt[t]] if label_mode == 'oracle' else (100.0 * x @ W).softmax(0)
        m.update(x, s)
        if m.ready():
            pred = int(m.predict(x).argmax())
            ok += int(pred == int(tgt[t]))
            tot += 1                                      # acc 只算 ready 之后的样本
    return 100.0 * ok / tot


def batch_oracle(ds, gpu, eps=0.05):
    """Non-streaming ceiling: exact within-class stats from all true labels."""
    F, tgt = load(ds)
    device = f'cuda:{gpu}'
    F, tgt = F.to(device), tgt.to(device)
    N, D = F.shape
    C = int(tgt.max()) + 1
    Sc = torch.zeros(C, D, device=device).index_add_(0, tgt, F)
    Nc = torch.bincount(tgt, minlength=C).float().to(device)
    mu = Sc / Nc.unsqueeze(1)
    T = F.t() @ F - Sc.t() @ mu                           # within-class scatter
    Sig = T / (N - C) + eps * torch.eye(D, device=device)
    Y = torch.linalg.solve(Sig, mu.t())                   # [D, C]
    score = F @ Y - 0.5 * (mu * Y.t()).sum(1).unsqueeze(0)
    return 100.0 * (score.argmax(1) == tgt).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=64)
    ap.add_argument('--eps', type=float, default=0.05)
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)

    print(f'{"domain":16s} {"LR-soft":>8s} {"LR-oracle":>10s} '
          f'{"Full-soft":>10s} {"Full-oracle":>12s} {"batch-ceiling":>14s}')
    for ds in DOMAINS:
        t0 = time.time()
        lr_s = run_streaming(ds, 'lr', 'soft', a.gpu, a.warmup, eps=a.eps)
        lr_o = run_streaming(ds, 'lr', 'oracle', a.gpu, a.warmup, eps=a.eps)
        full_s = run_streaming(ds, 'full', 'soft', a.gpu, a.warmup, eps=a.eps)
        full_o = run_streaming(ds, 'full', 'oracle', a.gpu, a.warmup, eps=a.eps)
        ceil = batch_oracle(ds, a.gpu, eps=a.eps)
        print(f'{ds:16s} {lr_s:8.2f} {lr_o:10.2f} {full_s:10.2f} '
              f'{full_o:12.2f} {ceil:14.2f}   ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
