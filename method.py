#!/usr/bin/env python
"""Independent implementation: explicit low-rank LDA (streaming).

Model: shared within-class covariance Sigma = A A^T + sigma^2 I, estimated
online with a Frequent-Directions (FD) sketch S [l, D] (block FD: one SVD per
l samples). Extracted as a standalone unit from
``online_softlabel/online_methods.py`` (LDALowRankMethod).

Mean update (streaming only): mu_c = Sc_c / Nc_c, CLIP soft-label weighted sums.
Optional text-prototype anchoring (mu_prior + mu_alpha, ADAPT initial_mean
semantics) is applied at predict time.
"""
from __future__ import annotations

import torch


class BaseMethod:
    """Minimal online-method interface (update-then-predict protocol)."""

    def __init__(self, C, D, device, warmup):
        self.C, self.D, self.device = C, D, device
        self.warmup = warmup
        self.n = 0

    def ready(self):
        return self.n >= self.warmup

    def update(self, x, s):
        self.n += 1

    def predict(self, x):
        raise NotImplementedError


class LowRankLDA(BaseMethod):
    """Explicit low-rank LDA (streaming soft-label weighted).

    Streaming state is only the FD sketch S [l, D] (l = sketch_mult * rank).
    Centered samples x' = x - mu_c (hard-class argmax centering) are accumulated
    into a block buffer; every l samples a single SVD truncates the 2l-row stack
    back to l rows (FD forgetting: subtract the smallest singular value squared).
    A [D, rank] and sigma^2 (residual isotropic variance) are DERIVED from S at
    predict time (not stored); Sigma^{-1} is applied via the Woodbury identity,
    so no D x D matrix is ever materialized.
    """

    def __init__(self, C, D, device, warmup, rank=64, jitter=1e-4, sketch_mult=2,
                 sigma_shrink='iso', sigma_eps=0.0,
                 mu_prior=None, mu_alpha=0.0):
        super().__init__(C, D, device, warmup)
        self.rank = rank
        self.l = sketch_mult * rank          # FD sketch height (number of kept directions)
        self.jitter = jitter                 # lower clamp for sigma^2
        self.mu_prior = mu_prior             # [C, D] 文本原型锚定（ADAPT initial_mean 对齐）；None=不用
        self.mu_alpha = mu_alpha             # EMA 权重: mu = α·(Sc/Nc) + (1-α)·prior（Nc=0 类用纯 prior）
        self.Sc = torch.zeros(C, D, device=device)       # per-class soft-label weighted sums
        self.Nc = torch.zeros(C, device=device)          # per-class soft-label weights (approx counts)
        self.S = torch.zeros(self.l, D, device=device)   # FD sketch
        self.buffer = torch.zeros(self.l, D, device=device)  # block FD accumulator
        self.buf_count = 0
        self.n_cc = 0                                    # centered rows seen (one per sample; ready() gate)
        self.n_eff = torch.zeros((), device=device)      # effective weight seen (covariance normalization)
        self.total_var = torch.zeros(D, device=device)   # per-dim sum x'^2
        self._eye_k = torch.eye(rank, device=device)
        self.sigma_shrink = sigma_shrink
        self.sigma_eps = sigma_eps

    def update(self, x, s):
        self.n += 1
        # streaming: CLIP soft-label weighted mean accumulation.
        c = int(s.argmax())
        w_c = s[c].clamp_min(0.0)
        mu_c_old = self.Sc[c] / self.Nc[c].clamp_min(1e-8)
        Nc_old = self.Nc[c]
        self.Sc += s.unsqueeze(1) * x.unsqueeze(0)
        self.Nc += s

        # Welford unbiased scatter increment: center at the OLD (pre-update) mean and
        # scale by sqrt(N_old*w/(N_old+w)) so the accumulated scatter equals the exact
        # within-class scatter at the FINAL mean - no drift, no need to re-center the
        # FD-compressed history (fixes the eurosat streaming-approx centering drift).
        wf = torch.sqrt((Nc_old * w_c / (Nc_old + w_c)).clamp_min(1e-12))
        xp = wf * (x - mu_c_old)
        self.total_var += xp * xp
        self.n_cc += 1
        self.n_eff += w_c
        self.buffer[self.buf_count] = xp
        self.buf_count += 1
        if self.buf_count == self.l:
            # block FD: stack sketch + l buffered samples, SVD, forget the smallest
            # singular direction, keep the top-l rows.
            S_aug = torch.cat([self.S, self.buffer], dim=0)      # [2l, D]
            _, sig, Vh = torch.linalg.svd(S_aug, full_matrices=False)
            sig2 = (sig * sig - sig[-1] * sig[-1]).clamp_min(0.0)
            self.S = (sig2[:self.l].sqrt().unsqueeze(1)) * Vh[:self.l]   # [l, D]
            self.buf_count = 0

    def ready(self):
        return self.n >= self.warmup and self.n_cc > self.l

    def predict(self, x):
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)                                 # (1, D)
        mu = self.Sc / self.Nc.clamp_min(1e-8).unsqueeze(1)      # [C, D]
        if self.mu_prior is not None and self.mu_alpha > 0:
            # ADAPT 式文本先验锚定：已见类 EMA 混合，未见类（Nc=0）直接用文本原型
            seen = (self.Nc > 0).unsqueeze(1)
            mu = torch.where(seen,
                             self.mu_alpha * mu + (1 - self.mu_alpha) * self.mu_prior,
                             self.mu_prior)
        nu = float(max(self.n_cc, 1))                            # row count (scale reference)
        ne = float(self.n_eff.clamp_min(1e-12))                  # effective weight (covariance normalization)
        jit = self.jitter * nu / ne                              # jitter floor scales with covariance scale
        k = self.rank
        # S's rows are orthogonal (SVD-guaranteed in update) and sig2 is descending, so
        # S S^T = diag(sig2[:l]) and the top-k subspace = the first k rows of S. This makes
        # eigh(SS) redundant (opt A) and A^T A diagonal (opt B). l=k and l=2k unify: both
        # take Sk = S[:k] (l=k: S[:k]=S; l=2k: top-k = first k rows).
        Sk = self.S[:k]                                          # [k, D]
        A = Sk.t() / (ne ** 0.5)                                 # [D, k]
        lam = (Sk * Sk).sum(1) / ne                              # [k] = diag(A^T A)
        sigma2 = ((self.total_var.sum() / ne) - lam.sum()) / (self.D - k)
        # sigma^2 I shrinkage overrides (keep AA^T+sigma^2 I form, tune the I-term)
        if self.sigma_shrink == 'scale':
            sigma2 = sigma2 * (1.0 + self.sigma_eps)
        elif self.sigma_shrink == 'add':
            sigma2 = sigma2 + self.sigma_eps
        elif self.sigma_shrink == 'trace':
            sigma2 = sigma2 + self.sigma_eps * (self.total_var.sum() / ne) / self.D
        elif self.sigma_shrink == 'tfull':
            # KS-ridge-style: replace sigma^2 with full trace mean (add trace(Σ)·I)
            sigma2 = (self.total_var.sum() / ne) / self.D
        elif self.sigma_shrink == 'trade':
            # blend residual sigma^2 with trace mean, eps in [0,1]
            tfull = (self.total_var.sum() / ne) / self.D
            sigma2 = (1.0 - self.sigma_eps) * sigma2 + self.sigma_eps * tfull
        sigma2 = float(sigma2.clamp_min(jit))

        # Woodbury: Sigma^{-1} = sigma^{-2} I - sigma^{-4} A (I + sigma^{-2} A^T A)^{-1} A^T.
        # M = I + (1/sigma2) A^T A is diagonal (= I + (1/sigma2) diag(lam)) -> elementwise
        # inverse; muA @ Minv becomes muA * Minv (O(Ck) vs O(Ck^2)).
        Minv = 1.0 / (1.0 + lam / sigma2)                        # [k] elementwise (opt B)
        muA = mu @ A                                             # [C, k]
        xA = x @ A                                               # [n, k]
        muAx = (muA * Minv) @ xA.t()                             # [C, n]
        muPx = (1.0 / sigma2) * (mu @ x.t() - (1.0 / sigma2) * muAx)
        muPmu = (1.0 / sigma2) * ((mu * mu).sum(1) - (1.0 / sigma2) * ((muA * Minv) * muA).sum(1))
        out = muPx - 0.5 * muPmu.unsqueeze(1)    # score_c = mu_c^T Sigma^{-1} x - 0.5 mu_c^T Sigma^{-1} mu_c
        return out.squeeze(-1) if single else out   # single: (C,1)->(C,)
