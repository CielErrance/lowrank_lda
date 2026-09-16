#!/usr/bin/env python
"""Merge per-shard result CSVs (results_lowrank_lda_shard*.csv) into one
full FG10 row and write results_lowrank_lda_all.csv."""
from __future__ import annotations

import csv
import glob
import os

from data import DATASETS

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD = 'lowrank_lda'


def main():
    per_ds = {}
    paths = sorted(glob.glob(os.path.join(HERE, f'results_{METHOD}_shard*.csv')))
    for p in paths:
        with open(p) as f:
            for r in csv.reader(f):
                if not r or r[0] != METHOD:
                    continue
                for d, v in zip(DATASETS, r[1:len(DATASETS) + 1]):
                    if d in per_ds or v in ('', 'nan'):
                        continue
                    per_ds[d] = float(v)
    if not per_ds:
        print('[agg] no shard results found; run run.py shards first')
        return

    colw = max(8, max((len(d) for d in DATASETS), default=8))
    avg = sum(per_ds.values()) / len(per_ds)
    row = [METHOD] + [f'{per_ds.get(d, float("nan")):.2f}' for d in DATASETS] + [f'{avg:.2f}']
    out = os.path.join(HERE, f'results_{METHOD}_all.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method'] + DATASETS + ['avg'])
        w.writerow(row)
    print(' '.join([f'{"method":<12}'] + [f'{d[:colw]:>{colw}}' for d in DATASETS] + [f'{"avg":>7}']))
    print(' '.join([f'{METHOD:<12}'] + [f'{v:>{colw}}' for v in row[1:-1]] + [f'{row[-1]:>7}']))
    print(f'\n[agg] wrote {out}')


if __name__ == '__main__':
    main()
