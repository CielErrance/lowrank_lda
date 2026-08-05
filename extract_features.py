#!/usr/bin/env python
"""Extract & cache CLIP ViT-B/16 image features (bs=1) for the FG10 datasets.

Self-contained replacement for ``online_softlabel``'s extraction: loads raw
images from ``--data_root`` (Zhou/CoOp layout, see datasets.py), encodes each
with CLIP ViT-B/16, and writes ``feats/<dataset>.npz``. Text prototypes are read
from the bundled ``class_feat/`` (reproducing the exact soft labels used in the
published numbers). ``set_random_seed(seed)`` + ``shuffle=True`` loader keeps
the stream order identical to the cached feature streams.

CLIP weights are auto-downloaded by ``clip.load('ViT-B/16')`` to
``~/.cache/clip`` on first use (network required once).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from clip import clip as clip_mod
from data import DATASETS, FEATS_DIR, load_text_protos
from datasets import build_test_loader, set_random_seed


@torch.no_grad()
def encode_batch(images, encoder, device):
    """Encode a batch of preprocessed images to unit-norm CLIP features."""
    images = images.to(device)
    features = encoder(images)
    return (features / features.norm(dim=-1, keepdim=True)).float()


def extract_and_cache(dataset, gpu, data_root, arch='ViT-B/16', seed=0, force=False):
    """Extract one dataset to ``feats/<dataset>.npz`` (skips if already cached)."""
    os.makedirs(FEATS_DIR, exist_ok=True)
    out = os.path.join(FEATS_DIR, f'{dataset}.npz')
    if os.path.exists(out) and not force:
        print(f'[extract] {dataset}: cache exists, skip ({out})')
        return out

    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    set_random_seed(seed)                       # fixes the shuffled stream order

    model, preprocess = clip_mod.load(arch, device=device)
    model.eval()
    _, clip_weights = load_text_protos(dataset, device, arch)

    loader = build_test_loader(dataset, preprocess, data_root, batch_size=1)
    feats, targets = [], []
    for images, tgt in tqdm(loader, desc=f'extract {dataset} (bs=1)'):
        features = encode_batch(images, model.encode_image, device)
        feats.append(features[0].detach().cpu())
        targets.append(int(tgt[0].item()))
    feats = torch.stack(feats).numpy().astype(np.float32)
    targets = np.array(targets, dtype=np.int64)
    np.savez(out, feats=feats, targets=targets)
    print(f'[extract] {dataset}: saved {feats.shape} -> {out}')
    return out


def main():
    ap = argparse.ArgumentParser(description='extract CLIP features for FG10')
    ap.add_argument('--datasets', nargs='*', default=DATASETS)
    ap.add_argument('--data_root', required=True,
                    help='root dir containing the 10 dataset folders '
                         '(each with its Zhou split_zhou_*.json inside)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    for ds in args.datasets:
        extract_and_cache(ds, args.gpu, args.data_root, force=args.force)


if __name__ == '__main__':
    main()
