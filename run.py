#!/usr/bin/env python
"""Run the standalone low-rank LDA + self-train method over the FG10 datasets.

bs=1 stream, update-then-predict, no ground-truth labels, no training data.
Writes a per-run CSV (results_lowrank_selftrain[_tag].csv). To parallelize
across GPUs, launch several invocations with --tag and disjoint --datasets,
then merge them with aggregate.py (or just use run_all.sh).

Default configuration is the production selftrain setting
(rank=64, sketch=2 => l=2k, mu=selftrain, nref=200, temp=2.0).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from data import DATASETS, FEATS_DIR, load_stream, write_result_row
from method import LowRankLDASelfTrain
from extract_features import extract_and_cache

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_NAME = 'lowrank_selftrain'


@torch.no_grad()
def run_dataset(dataset, device, feats_dir, warmup_frac, rank, sketch,
                mu_mode, temp, n_ref):
    feats, targets, W, soft = load_stream(dataset, device, feats_dir)
    N, D = feats.shape
    C = W.shape[0]
    warmup = max(8, int(warmup_frac * C))
    method = LowRankLDASelfTrain(C, D, device, warmup, rank=rank,
                                 sketch_mult=sketch, mu_mode=mu_mode,
                                 temp=temp, n_ref=n_ref)

    correct = 0
    lnC = float(np.log(C))
    nent_sum = 0.0
    for t in tqdm(range(N), desc=f'{dataset}', leave=False):
        x, s = feats[t], soft[t]
        nent_sum += float(-(s * s.clamp_min(1e-12).log()).sum().item())  # H(s_clip)
        method.update(x, s)
        if method.ready():
            pred = int(method.predict(x).argmax())
        else:
            pred = int(soft[t].argmax())          # bootstrap fallback -> CLIP
        correct += int(pred == int(targets[t]))
    acc = 100.0 * correct / N
    nent = (nent_sum / N) / lnC if lnC > 0 else 0.0  # normalized entropy in [0,1]
    return acc, N, C, nent


def main():
    ap = argparse.ArgumentParser(description='low-rank LDA + self-train (standalone)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--feats_dir', default=FEATS_DIR,
                    help='directory holding <dataset>.npz feature caches')
    ap.add_argument('--data_root', default=None,
                    help='root dir containing the 10 dataset folders; if given, missing '
                         'feature caches are extracted on the fly (see extract_features.py)')
    ap.add_argument('--datasets', nargs='*', default=DATASETS)
    ap.add_argument('--warmup_frac', type=float, default=1.0,
                    help='warmup samples = frac * n_classes')
    ap.add_argument('--rank', type=int, default=64, help='low-rank factor A rank k')
    ap.add_argument('--sketch', type=int, default=2,
                    help='FD sketch multiplier: l=sketch*rank (2=l=2k, 1=l=k no eigh)')
    ap.add_argument('--mu', default='selftrain', choices=['streaming', 'selftrain'],
                    help="mean update: streaming (CLIP) or selftrain (online EM, nref)")
    ap.add_argument('--temp', type=float, default=2.0,
                    help='selftrain: LDA softmax temperature')
    ap.add_argument('--nref', type=float, default=200,
                    help='selftrain: per-class sample threshold for w_c')
    ap.add_argument('--tag', default=None,
                    help='suffix for results csv (results_<method>_<tag>.csv) to allow parallel shards')
    args = ap.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    print(f'[run] method={METHOD_NAME} gpu={args.gpu} mu={args.mu} nref={args.nref} '
          f'rank={args.rank} sketch={args.sketch} temp={args.temp} '
          f'warmup_frac={args.warmup_frac} datasets={args.datasets}')

    per_ds, nent_ds = {}, {}
    for ds in args.datasets:
        npz = os.path.join(args.feats_dir, f'{ds}.npz')
        if not os.path.exists(npz):
            if args.data_root is not None:
                print(f'[run] {ds}: no cache, extracting from {args.data_root} ...')
                extract_and_cache(ds, args.gpu, args.data_root, force=False)
            else:
                print(f'[run] SKIP {ds}: no cache {npz} (pass --data_root to auto-extract)')
                continue
        t0 = time.time()
        acc, N, C, nent = run_dataset(ds, device, args.feats_dir, args.warmup_frac,
                                      args.rank, args.sketch, args.mu,
                                      args.temp, args.nref)
        per_ds[ds] = acc
        nent_ds[ds] = nent
        print(f'[run] {METHOD_NAME:16s} {ds:16s} acc={acc:6.2f}%  '
              f'(N={N}, C={C}, {time.time()-t0:.1f}s)  nent={nent:.4f}')

    if per_ds:
        print('[nent] CLIP normalized entropy H(s)/ln(C) per dataset (1=uniform, 0=confident):')
        for ds, v in nent_ds.items():
            print(f'[nent]   {ds:16s} nent={v:.4f}')
        avg = sum(per_ds.values()) / len(per_ds)
        print(f'[run] {METHOD_NAME:16s} AVG over {len(per_ds)} = {avg:.2f}%')
        csv_name = f'results_{METHOD_NAME}_{args.tag}.csv' if args.tag \
            else f'results_{METHOD_NAME}.csv'
        write_result_row(os.path.join(HERE, csv_name), METHOD_NAME, per_ds, avg)


if __name__ == '__main__':
    main()
