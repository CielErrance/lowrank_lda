#!/usr/bin/env python
"""Self-contained data loading for the lowrank_lda method.

Features are read from this repo's ``feats/`` cache (produced by
``extract_features.py``) and CLIP text prototypes from the bundled
``class_feat/GPT_w_Custom_class_emb/`` — the exact pre-extracted soft-label
source used for the published numbers. No dependencies on other wufan trees.
"""
from __future__ import annotations

import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# Feature caches (written by extract_features.py) and bundled text prototypes.
FEATS_DIR = os.path.join(HERE, 'feats')
CLASS_FEAT_DIR = os.path.join(HERE, 'class_feat', 'GPT_w_Custom_class_emb')

# FG10 evaluation suite (10 datasets, in the canonical order).
DATASETS = [
    'fgvc_aircraft', 'caltech101', 'stanford_cars', 'dtd',
    'eurosat', 'oxford_flowers', 'food101', 'oxford_pets', 'sun397', 'ucf101',
]


def load_text_protos(dataset, device, arch='ViT-B/16'):
    """Return normalized text prototypes W [C, D] and raw clip_weights [D, C]."""
    clip_weights = torch.load(
        os.path.join(CLASS_FEAT_DIR, f'{dataset}.pth'), map_location=device)
    W = clip_weights.T.float()
    W = W / W.norm(dim=-1, keepdim=True)
    return W, clip_weights.float()


def load_stream(dataset, device, feats_dir=None):
    """Load the cached stream; return feats [N, D] (unit-norm), targets [N],
    W [C, D], soft [N, C]."""
    feats_dir = feats_dir or FEATS_DIR
    npz = os.path.join(feats_dir, f'{dataset}.npz')
    d = np.load(npz)
    feats = torch.from_numpy(d['feats']).float().to(device)
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    targets = torch.from_numpy(d['targets']).long().to(device)
    W, _ = load_text_protos(dataset, device)
    logits = 100.0 * feats @ W.t()
    soft = torch.softmax(logits, dim=1)
    return feats, targets, W, soft


def write_result_row(csv_path, method, per_ds, avg):
    """Write/overwrite one method row in a CSV (keeps other methods' rows)."""
    import csv
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    header = ['method'] + DATASETS + ['avg']
    row = [method] + [f'{per_ds.get(d, float("nan")):.2f}' for d in DATASETS] + [f'{avg:.2f}']
    exists = os.path.exists(csv_path)
    rows = []
    if exists:
        with open(csv_path) as f:
            rows = list(csv.reader(f))
    rows = [r for r in rows if r and r[0] != method and r[0] != 'method']
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
        w.writerow(row)
    print(f'[result] wrote {method} -> {csv_path}')
