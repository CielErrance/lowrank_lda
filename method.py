#!/usr/bin/env python
"""Independent implementation: explicit low-rank LDA with self-training.

Model: shared within-class covariance Sigma = A A^T + sigma^2 I, estimated
online with a Frequent-Directions (FD) sketch S [l, D] (block FD: one SVD per
l samples). This is the "lowrank LDA + self-train" method extracted as a
standalone unit, matching the production configuration
(rank=64, sketch_mult=2 => l=2k, iso sigma^2, mu_mode='selftrain',
n_ref=200, temp=2.0) in ``online_softlabel/online_methods.py`` (LDALowRankMethod).

Two mean-update modes:
  * streaming : mu_c = Sc_c / Nc_c  (CLIP soft-label weighted sums)
  * selftrain : online EM. E-step computes p_LDA = softmax(score(x)/T) and a
                per-class reliability gate w_c = clamp(Nc_c / n_ref, 0, 1);
                the M-step updates Sc/Nc with the mixed label
                s_mix = w_c * p_LDA + (1 - w_c) * s_clip.

This is the selftrain (non-auto / non-entropy-gate) version only: the auto
entropy gate, EMA and adaptive margin machinery are deliberately not included.
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


class LowRankLDASelfTrain(BaseMethod):
    """Explicit low-rank LDA + optional self-training (online EM).

    Streaming state is only the FD sketch S [l, D] (l = sketch_mult * rank).
    Centered samples x' = x - mu_c (hard-class argmax centering) are accumulated
    into a block buffer; every l samples a single SVD truncates the 2l-row stack
    back to l rows (FD forgetting: subtract the smallest singular value squared).
    A [D, rank] and sigma^2 (residual isotropic variance) are DERIVED from S at
    predict time (not stored); Sigma^{-1} is applied via the Woodbury identity,
    so no D x D matrix is ever materialized.
    """

    def __init__(self, C, D, device, warmup, rank=64, jitter=1e-4, sketch_mult=2,
                 mu_mode='selftrain', temp=2.0, n_ref=200):
        super().__init__(C, D, device, warmup)
        self.rank = rank
        self.l = sketch_mult * rank          # FD sketch height (number of kept directions)
        self.jitter = jitter                 # lower clamp for sigma^2
        self.mu_mode = mu_mode               # 'streaming' or 'selftrain'
        self.temp = temp                     # LDA softmax temperature (selftrain E-step)
        self.n_ref = n_ref                   # per-class sample threshold for w_c
        self.Sc = torch.zeros(C, D, device=device)       # per-class soft-label weighted sums
        self.Nc = torch.zeros(C, device=device)          # per-class soft-label weights (approx counts)
        self.S = torch.zeros(self.l, D, device=device)   # FD sketch
        self.buffer = torch.zeros(self.l, D, device=device)  # block FD accumulator
        self.buf_count = 0
        self.n_cc = 0                                    # centered samples seen
        self.total_var = torch.zeros(D, device=device)   # per-dim sum x'^2
        self._eye_k = torch.eye(rank, device=device)

    def update(self, x, s):
        self.n += 1
        if self.mu_mode == 'selftrain' and self.n > self.warmup:
            # E-step: LDA discriminant from t-1 parameters; per-class reliability gate.
            p_lda = torch.softmax(self.predict(x) / self.temp, dim=0)
            w = (self.Nc / self.n_ref).clamp(0.0, 1.0)     # [C] 1 = full LDA trust
            s_mix = w * p_lda + (1 - w) * s                # mixed soft label
            c = int(s_mix.argmax())
            w_c = s_mix[c].clamp_min(0.0)                  # weight mass on the centering class
            mu_c_old = self.Sc[c] / self.Nc[c].clamp_min(1e-8)
            Nc_old = self.Nc[c]
            self.Sc += s_mix.unsqueeze(1) * x.unsqueeze(0)
            self.Nc += s_mix
        else:
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
        mu = self.Sc / self.Nc.clamp_min(1e-8).unsqueeze(1)      # [C, D]
        n = max(self.n_cc, 1)
        k = self.rank
        # S's rows are orthogonal (SVD-guaranteed in update) and sig2 is descending, so
        # S S^T = diag(sig2[:l]) and the top-k subspace = the first k rows of S. This makes
        # eigh(SS) redundant (opt A) and A^T A diagonal (opt B). l=k and l=2k unify: both
        # take Sk = S[:k] (l=k: S[:k]=S; l=2k: top-k = first k rows).
        Sk = self.S[:k]                                          # [k, D]
        A = Sk.t() / (n ** 0.5)                                  # [D, k]
        lam = (Sk * Sk).sum(1) / n                               # [k] = diag(A^T A)
        sigma2 = ((self.total_var.sum() / n) - lam.sum()) / (self.D - k)
        sigma2 = float(sigma2.clamp_min(self.jitter))

        # Woodbury: Sigma^{-1} = sigma^{-2} I - sigma^{-4} A (I + sigma^{-2} A^T A)^{-1} A^T.
        # M = I + (1/sigma2) A^T A is diagonal (= I + (1/sigma2) diag(lam)) -> elementwise
        # inverse; muA @ Minv becomes muA * Minv (O(Ck) vs O(Ck^2)).
        Minv = 1.0 / (1.0 + lam / sigma2)                        # [k] elementwise (opt B)
        muA = mu @ A                                             # [C, k]
        xA = x @ A                                               # [k]
        muAx = (muA * Minv) @ xA                                 # [C]
        muPx = (1.0 / sigma2) * (mu @ x - (1.0 / sigma2) * muAx)
        muPmu = (1.0 / sigma2) * ((mu * mu).sum(1) - (1.0 / sigma2) * ((muA * Minv) * muA).sum(1))
        return muPx - 0.5 * muPmu    # score_c = mu_c^T Sigma^{-1} x - 0.5 mu_c^T Sigma^{-1} mu_c
