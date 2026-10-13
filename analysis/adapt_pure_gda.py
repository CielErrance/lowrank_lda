#!/usr/bin/env python
"""纯判别器对照：ADAPT KS-ridge GDA（bank 估计器）vs LR LDA（FD sketch 估计器）。

同协议：aug63 缓存特征流 + update-then-predict + 同一 CLIP logits 来源，
打分都取融合前的纯二次判别 argmax。隔离"分数估计器"单一变量。

bank/KS-ridge 语义逐行移植自 ADAPT/ADAPT_online.py（不动原文件）：
- 每类 ≤16 样本的 bank，新样本按 entropy 淘汰最差（低熵筛选 + 淘汰制）
- μ_c = EMA(bank 软加权均值, 文本原型)，α=0.9，未见类=文本原型
- Σ = KS-ridge: cov_inv = d·pinv((n−1)Σ + trace(Σ)·I)
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根
from method import LowRankLDA

FEATS_DIR = os.environ.get('AUG63_FEATS_DIR', '/home/liangyiwen/wufan/ADAPT/feats')
PTH = os.path.join(os.environ.get('PTH_ROOT',
    '/home/liangyiwen/wufan/ADAPT/pre_extracted_class_feat/ViT-B16'),
    'GPT_w_Custom_class_emb')
DOMAINS = ['imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r', 'imagenet']


def load(ds):
    d = np.load(os.path.join(FEATS_DIR, f'aug63_{ds}.npz'))
    W = torch.load(os.path.join(PTH, f'{ds}.pth'), map_location='cpu',
                   weights_only=True).float()
    F = torch.from_numpy(d['feats']).float()
    if W.shape[0] != F.shape[1]:
        W = W.t()
    return F, torch.from_numpy(d['targets']).long(), W


class ADAPTBankGDA:
    """ADAPT 估计器：bank + KS-ridge Σ + EMA μ 锚定（逐行移植语义）。"""

    def __init__(self, C, D, dev, W, bank_size=16, alpha=0.9):
        self.C, self.D, self.dev = C, D, dev
        self.bs, self.alpha = bank_size, alpha
        self.initial_mean = W.t().to(dev)                     # [C, D] 文本原型
        self.mus = self.initial_mean.clone()
        self.vecs = torch.zeros(C * bank_size, D, device=dev)
        self.labels = torch.full((C * bank_size,), -1, dtype=torch.long, device=dev)
        self.pro = torch.zeros(C * bank_size, C, device=dev)
        self.loss = torch.full((C * bank_size,), float('inf'), device=dev)
        self.cov_inv = None                                   # lazy until bank 有 ≥2 类

    def update(self, feat, prob_map, pred, loss):
        s, e = pred * self.bs, (pred + 1) * self.bs
        cnt = (self.labels == pred).sum().item()
        if cnt < self.bs:
            self.vecs[s + cnt], self.labels[s + cnt] = feat, pred
            self.pro[s + cnt], self.loss[s + cnt] = prob_map, loss
        else:
            mx, mi = self.loss[s:e].max(dim=0)
            if loss < mx:
                i = s + mi.item()
                self.vecs[i], self.labels[i] = feat, pred
                self.pro[i], self.loss[i] = prob_map, loss
        # μ 更新（pred 类，EMA 锚定文本原型）
        mask = self.labels == pred
        sel_v = self.vecs[mask]
        sel_p = self.pro[mask, pred].unsqueeze(1)
        new_mu = (sel_p * sel_v).sum(0) / sel_p.sum()
        self.mus[pred] = self.alpha * new_mu + (1 - self.alpha) * self.initial_mean[pred]
        # KS-ridge Σ（bank 全体样本，以当前 mus 居中）
        keys = torch.unique(self.labels[self.labels != -1])
        if len(keys) >= 2:
            centered = torch.cat([self.vecs[self.labels == c] - self.mus[c].unsqueeze(0)
                                  for c in keys])
            n, d = centered.shape
            if n > 1:
                Sig = centered.t().cov()
                tr = Sig.trace()
                self.cov_inv = d * torch.linalg.pinv(
                    (n - 1) * Sig + tr * torch.eye(d, device=self.dev))

    def ready(self):
        return self.cov_inv is not None

    def predict(self, x):
        # 与 ADAPT_online.py param_estimation 逐行同式
        W = torch.einsum('nd, dc -> cn', self.mus, self.cov_inv)          # [C, D]
        b = -torch.einsum('nd, dc, nc -> n', self.mus, self.cov_inv,
                          self.mus) / 2                                    # [C]
        return x @ W + b                                                   # W 为 [D, C]（ADAPT 原始布局）


class KSNoBank:
    """KS-ridge GDA，无 bank：全体样本流式（软标签加权 Welford 散度 + 逐类均值）。
    与 ADAPT 的差别仅在 Σ/μ 的样本来源：bank（低熵筛选+16淘汰+全量重算）→ 全流等权。
    use_ema=True 时 μ 保留 ADAPT 的 α-EMA 文本原型锚定。"""

    def __init__(self, C, D, dev, W, alpha=0.9, use_ema=True):
        self.C, self.D, self.dev = C, D, dev
        self.alpha, self.use_ema = alpha, use_ema
        self.initial_mean = W.t().to(dev)
        self.Sc = torch.zeros(C, D, device=dev)
        self.Nc = torch.zeros(C, device=dev)
        self.T = torch.zeros(D, D, device=dev)
        self.n_cc = 0
        self.n_eff = torch.zeros((), device=dev)
        self.cov_inv = None

    def update(self, feat, prob_map, pred, loss):
        w_c = prob_map[pred].clamp_min(0.0)
        Sc_old = self.Sc[pred].clone()
        Nc_old = self.Nc[pred]
        self.Sc += prob_map.unsqueeze(1) * feat.unsqueeze(0)
        self.Nc += prob_map
        mu_old = Sc_old / Nc_old.clamp_min(1e-8)
        wf = torch.sqrt((Nc_old * w_c / (Nc_old + w_c)).clamp_min(1e-12))
        xp = wf * (feat - mu_old)
        self.T += xp.unsqueeze(1) * xp.unsqueeze(0)
        self.n_cc += 1
        self.n_eff += w_c
        if self.n_cc > 2:
            Sig = self.T / (self.n_cc - 1)                       # 无偏样本协方差
            tr = Sig.trace()
            self.cov_inv = self.D * torch.linalg.pinv(
                (self.n_cc - 1) * Sig + tr * torch.eye(self.D, device=self.dev))

    def ready(self):
        return self.cov_inv is not None

    def predict(self, x):
        mu = self.Sc / self.Nc.clamp_min(1e-8).unsqueeze(1)
        if self.use_ema:
            seen = (self.Nc > 0).unsqueeze(1)
            mu = torch.where(seen, self.alpha * mu + (1 - self.alpha) * self.initial_mean,
                             self.initial_mean)
        W = torch.einsum('nd, dc -> cn', mu, self.cov_inv)       # [D, C]
        b = -torch.einsum('nd, dc, nc -> n', mu, self.cov_inv, mu) / 2
        return x @ W + b


@torch.no_grad()
def run_domain(ds, gpu, mode):
    F_, tgt, W = load(ds)
    dev = f'cuda:{gpu}'
    F_, tgt, W = F_.to(dev), tgt.to(dev), W.to(dev)
    N, D = F_.shape
    C = W.shape[1]
    if mode == 'ks':
        m = ADAPTBankGDA(C, D, dev, W)
    elif mode == 'ks_nobank':
        m = KSNoBank(C, D, dev, W, use_ema=True)
    elif mode == 'ks_nobank_noema':
        m = KSNoBank(C, D, dev, W, use_ema=False)
    else:
        m = LowRankLDA(C, D, dev, warmup=64, rank=64, sketch_mult=1,
                       sigma_shrink='add', sigma_eps=0.05)
    ok, tot = 0, 0
    for t in range(N):
        x = F_[t]
        clip_ll = 100.0 * x @ W
        pmap = clip_ll.softmax(0)
        pred = int(clip_ll.argmax())
        loss = -(pmap * pmap.log()).sum()                   # entropy（bank 筛选键）
        if mode in ('ks', 'ks_nobank', 'ks_nobank_noema'):
            m.update(x, pmap, pred, loss)
            ready = m.ready()
        else:
            m.update(x, pmap)
            ready = m.ready()
        if ready:
            sc = m.predict(x)
            ok += int(int(sc.argmax()) == int(tgt[t]))
            tot += 1
    return 100.0 * ok / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=5)
    ap.add_argument('--domains', nargs='*', default=DOMAINS)
    ap.add_argument('--mode', default='ks',
                    choices=['ks', 'ks_nobank', 'ks_nobank_noema', 'lr'])
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    for ds in a.domains:
        t0 = time.time()
        acc = run_domain(ds, a.gpu, a.mode)
        print(f'[pure-gda] {a.mode:16s} {ds:16s} {acc:6.2f}  ({time.time()-t0:.0f}s)',
              flush=True)


if __name__ == '__main__':
    main()
