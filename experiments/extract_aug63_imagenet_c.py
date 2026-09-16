#!/usr/bin/env python
"""ImageNet-C 63-view 增强聚合特征提取（对齐 imagenet 家族的 aug63 流水线）。

每样本: base + 63 RandomResizedCrop+Flip 视图 -> CLIP 特征 [64, D] ->
低熵 top-10% 选择 -> 聚合 mean 特征 + mean logits + soft。
缓存 /mnt/newdata/liangyiwen/imagenet_c_feats_aug63/<corr>.npz
（键: feats, clip_ll, soft, targets），供 imagenet_c_lr.py --cache aug63 消费。
"""
import os
import sys
import argparse
import time

import numpy as np
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '/home/liangyiwen/wufan/ADAPT')
from clip import clip as clip_pkg
from data.datautils import get_ood_preprocess

DATA_ROOT = '/mnt/newdata/liangyiwen/datasets/corruption/imagenet-c'
OUT_DIR = '/mnt/newdata/liangyiwen/imagenet_c_feats_aug63'
PTH = ('/home/liangyiwen/wufan/ADAPT/pre_extracted_class_feat/'
       'ViT-B16/GPT_w_Custom_class_emb/imagenet.pth')

CORDER = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost',
          'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
          'jpeg_compression', 'gaussian_noise', 'shot_noise', 'impulse_noise']


def collate_views(batch):
    views = torch.stack(batch[0][0])      # (V, 3, 224, 224)
    tgt = batch[0][1]
    return views, tgt


@torch.no_grad()
def extract(corr, model, weight, gpu, level='5', n_views=63, force=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'{corr}.npz')
    if os.path.exists(out) and not force:
        print(f'[aug63-c] {corr}: cache exists, skip', flush=True)
        return
    tf = get_ood_preprocess(n_views)
    testdir = os.path.join(DATA_ROOT, corr, level)
    ds = datasets.ImageFolder(testdir, transform=tf)
    loader = DataLoader(ds, batch_size=1, num_workers=8, collate_fn=collate_views,
                        shuffle=False, pin_memory=True)
    w = weight.float().cuda()
    feats, clip_ll, soft, targets = [], [], [], []
    t0 = time.time()
    for vi, (views, tgt) in enumerate(tqdm(loader, desc=corr, leave=False)):
        views = views.cuda(non_blocking=True)
        f = model.encode_image(views)
        f = f / f.norm(dim=-1, keepdim=True)
        ll = 100.0 * f.float() @ w                            # (V, C)
        ent = -(ll.softmax(1) * ll.log_softmax(1)).sum(1)
        k = max(int(ll.shape[0] * 0.1), 1)                    # top-10% 最低熵视图
        sel = torch.argsort(ent)[:k]
        feats.append(f[sel].mean(0).cpu())
        al = ll[sel].mean(0)                                  # (C,) logit ensemble
        clip_ll.append(al.cpu())
        soft.append(al.softmax(0).cpu())
        targets.append(int(tgt))
        if vi == 200:
            dt = time.time() - t0
            print(f'[aug63-c] {corr}: {dt / (vi + 1):.3f}s/样本, 预计单域 '
                  f'{dt / (vi + 1) * len(ds) / 3600:.1f}h', flush=True)
    np.savez(out, feats=torch.stack(feats).numpy(),
             clip_ll=torch.stack(clip_ll).numpy(),
             soft=torch.stack(soft).numpy(),
             targets=np.array(targets))
    print(f'[aug63-c] saved {corr}: {len(feats)} 样本 '
          f'({(time.time() - t0) / 3600:.2f}h) {time.strftime("%H:%M:%S")}', flush=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--corruptions', nargs='*', default=CORDER)
    ap.add_argument('--level', default='5')
    ap.add_argument('--n_views', type=int, default=63)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    dev = f'cuda:{a.gpu}'
    model, _ = clip_pkg.load('ViT-B/16', device=dev)
    model.eval()
    weight = torch.load(PTH, map_location='cpu').float()
    if weight.shape[0] != 512:
        weight = weight.t()
    for corr in a.corruptions:
        extract(corr, model, weight, a.gpu, a.level, a.n_views, a.force)


if __name__ == '__main__':
    main()
