#!/usr/bin/env python
"""imagenet 家族 aug63 变体提取：保留 top-10% 选中视图（不聚合），供"每视图独立 update"LR LDA 实验。

每样本: base + 63 视图 -> CLIP -> 熵 top-10% 选 6 视图 -> 保留 6 条特征（不 mean）+
聚合特征（6 视图 mean，作为 predict 用）+ 聚合 logit-ensemble 软标签（与 aug63 同约定）。
缓存 feats_multi_<ds>.npz（键: feats [N,k,D] 未聚合, feats_mean [N,D], soft [N,C], targets [N]）。

online 语义（正确版）：6 视图是样本到达后本地生成、同属 t 时刻的同一观测 —— 连发 6 次
update（统计量收 6 份），predict 用聚合特征打一次分。
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

ID_TO_DIRNAME = {
    'imagenet': 'domain_shift_datastes/imagenet/images/val',
    'imagenetv2': 'domain_shift_datastes/imagenetv2-matched-frequency-format-val/imagenetv2-matched-frequency-format-val',
    'imagenet_sketch': 'domain_shift_datastes/ImageNet-Sketch/images',
    'imagenet_a': 'domain_shift_datastes/imagenet-a/imagenet-a',
    'imagenet_r': 'domain_shift_datastes/imagenet-r/imagenet-r',
}
PTH_NAME = {'imagenet': 'imagenet', 'imagenetv2': 'imagenetv2', 'imagenet_sketch': 'imagenet_sketch',
            'imagenet_a': 'imagenet_a', 'imagenet_r': 'imagenet_r'}
OUT_DIR = '/home/liangyiwen/wufan/ADAPT/feats'


def collate_views(batch):
    views = torch.stack(batch[0][0])
    tgt = batch[0][1]
    return views, tgt


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_set', required=True, choices=list(ID_TO_DIRNAME))
    ap.add_argument('--data', default='/home/liangyiwen/datasets')
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--n_views', type=int, default=63)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    dev = f'cuda:{a.gpu}'

    out = os.path.join(OUT_DIR, f'feats_multi_{a.test_set}.npz')
    if os.path.exists(out) and not a.force:
        print(f'[multi] {a.test_set}: cache exists, skip', flush=True)
        return

    model, _ = clip_pkg.load('ViT-B/16', device=dev)
    model.eval()
    w = torch.load(os.path.join(
        '/home/liangyiwen/wufan/ADAPT/pre_extracted_class_feat/ViT-B16/GPT_w_Custom_class_emb',
        f'{PTH_NAME[a.test_set]}.pth'), map_location='cpu').float().cuda()

    tf = get_ood_preprocess(a.n_views)
    testdir = os.path.join(a.data, ID_TO_DIRNAME[a.test_set])
    ds = datasets.ImageFolder(testdir, transform=tf)
    loader = DataLoader(ds, batch_size=1, num_workers=8, collate_fn=collate_views,
                        shuffle=False, pin_memory=True)
    print(f'[multi] {a.test_set}: {len(ds)} imgs (n_views={a.n_views})', flush=True)

    feats, feats_mean, targets, soft = [], [], [], []
    t0 = time.time()
    for vi, (views, tgt) in enumerate(tqdm(loader, desc=a.test_set, leave=False)):
        views = views.cuda(non_blocking=True)
        f = model.encode_image(views)
        f = f / f.norm(dim=-1, keepdim=True)
        ll = 100.0 * f.float() @ w
        ent = -(ll.softmax(1) * ll.log_softmax(1)).sum(1)
        k = max(int(ll.shape[0] * 0.1), 1)
        sel = torch.argsort(ent)[:k]
        feats.append(f[sel].cpu())                          # (k, D) 未聚合
        feats_mean.append(f[sel].mean(0).cpu())             # (D,) 聚合（predict 用）
        targets.append(int(tgt))
        soft.append(ll[sel].mean(0).softmax(0).cpu())       # 聚合软标签（aug63 同约定）
        if vi == 200:
            dt = time.time() - t0
            print(f'[multi] {a.test_set}: {dt/(vi+1):.3f}s/样本, 预计 {dt/(vi+1)*len(ds)/3600:.1f}h',
                  flush=True)
    F = torch.stack(feats).numpy()                          # [N, k, D]
    np.savez(out, feats=F,
             feats_mean=torch.stack(feats_mean).numpy(),
             targets=np.array(targets),
             soft=torch.stack(soft).numpy())
    print(f'[multi] saved {a.test_set}: feats{F.shape}（每样本 {F.shape[1]} 视图未聚合） '
          f'({(time.time()-t0)/3600:.2f}h) {time.strftime("%H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
