#!/usr/bin/env python
"""imagenet 变体 online runner（--family inet5 = ImageNet 系 5 域；--family inc = ImageNet-C）。

- inet5：aug63/singlev 缓存特征，流式 update-then-predict，融合全家桶
  （dotaclip 生产 / alphamix·adaptmul 实验形态），μ 锚定、置信门限、hard 分配开关。
- inc：ImageNet-C 15 域流式（--mode extract 先提特征缓存；--cache singlev/aug63）。
Core = LowRankLDA（AAᵀ + σ²I FD streaming，见 method.py）。
"""
import os
import sys
import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/liangyiwen/wufan/lowrank_selftrain')
from method import LowRankLDA
from clip import clip as clip_pkg

PTH_TEMPLATE = '/home/liangyiwen/wufan/ADAPT/pre_extracted_class_feat/{ARCH}/GPT_w_Custom_class_emb'
FEATS_DIR = '/home/liangyiwen/wufan/ADAPT/feats'
IFI = '/home/liangyiwen/wufan/.imagenet_feats'

# DOTA's per-domain fusion schedule (rho = scale of w, eta = cap)
DOTA_RHO_ETA = {'imagenet': (0.005, 0.2), 'imagenetv2': (0.01, 0.1),
                'imagenet_sketch': (0.03, 0.8), 'imagenet_a': (0.02, 0.5),
                'imagenet_r': (0.0025, 0.3)}

# ADAPT's per-domain fusion temperature (compute_final_prediction: clip*exp(1/scale*logP))
ADAPT_SCALE = {'imagenet': 20.0, 'imagenetv2': 30.0, 'imagenet_sketch': 20.0,
               'imagenet_a': 40.0, 'imagenet_r': 100.0}


def load_data(ds, cache='singlev', arch='ViT-B16'):
    pth_dir = PTH_TEMPLATE.format(ARCH=arch)
    feats_dir = os.path.join(FEATS_DIR, 'rn50') if arch == 'RN50' else FEATS_DIR
    if cache == 'aug63':
        d = np.load(os.path.join(FEATS_DIR, f'aug63_{ds}.npz'))
        return (torch.from_numpy(d['feats']).float(),
                torch.from_numpy(d['targets']).long(),
                torch.load(os.path.join(pth_dir, f'{ds}.pth'), map_location='cpu').float(),
                torch.from_numpy(d['clip_ll']).float())
    if arch == 'RN50':
        npz = os.path.join(feats_dir, f'singlev_{ds}.npz')
    elif ds == 'imagenet':
        npz = os.path.join(IFI, 'imagenet.npz')
    else:
        npz = os.path.join(feats_dir, f'singlev_{ds}.npz')
    d = np.load(npz)
    feats = torch.from_numpy(d['feats']).float()
    targets = torch.from_numpy(d['targets']).long()
    weight = torch.load(os.path.join(pth_dir, f'{ds}.pth'), map_location='cpu').float()
    if weight.shape[0] != feats.shape[1]:      # (D, C) expected; else transpose
        weight = weight.t()
    return feats, targets, weight, None


@torch.no_grad()
def run_dataset(ds, gpu, rank, sketch, warmup, fuse, rho, eta, cache,
                shrink, sigma_eps, arch='ViT-B16', alpha=0.5, norm='minmax', mu_alpha=0.0,
                gate=0.0, hard_assign=False):
    if rho is None:                       # per-domain DOTA schedule
        rho, eta = DOTA_RHO_ETA[ds]
    device = f'cuda:{gpu}'
    feats, targets, weight, clip_ll_c = load_data(ds, cache, arch)
    feats = feats.to(device)
    if clip_ll_c is not None:
        clip_ll_c = clip_ll_c.to(device)
    N, D = feats.shape
    C = weight.shape[1]
    W = weight.to(device)
    method = LowRankLDA(C, D, device, warmup, rank=rank, sketch_mult=sketch,
                                 sigma_shrink=shrink, sigma_eps=sigma_eps,
                                 mu_prior=weight.t().float().to(device), mu_alpha=mu_alpha)

    def mixnorm(v):
        # 逐样本归一化（对 [n] 的 1 维 logit/score 向量），两种可逆/保序方案
        if norm == 'minmax':
            return (v - v.min()) / (v.max() - v.min()).clamp_min(1e-12)
        return v.softmax(0)               # softmax

    correct, total = 0, 0
    for t in tqdm(range(N), desc=f'{ds[:8]}/{fuse}', leave=False):
        x, target = feats[t], targets[t].item()
        clip_ll = clip_ll_c[t] if clip_ll_c is not None else 100.0 * x @ W
        soft = clip_ll.softmax(0)
        # E4 hard 分配：软标签质量集中于 pred 类（对齐 ADAPT img_pro[pred]·onehot 语义）
        if hard_assign:
            c_hat = int(soft.argmax())
            soft = torch.zeros_like(soft)
            soft[c_hat] = clip_ll[c_hat] / 100.0     # 保留该类概率作权重
        # E3 置信门限：仅高置信样本进统计量（模拟 ADAPT bank 低熵淘汰制）；gate=0 全收
        if gate <= 0.0 or float(soft.max()) >= gate:
            method.update(x, soft)
        if method.ready():
            lr_score = method.predict(x)
            if fuse == 'none':
                pred = int(lr_score.argmax())
            elif fuse == 'alphamix':  # alpha*norm(clip) + (1-alpha)*norm(lda)，逐样本归一化后线性混合
                final = alpha * mixnorm(clip_ll) + (1.0 - alpha) * mixnorm(lr_score)
                pred = int(final.argmax())
            elif fuse == 'adaptmul':  # ADAPT 乘性: clip * exp(1/scale * log_softmax(lda))，无 cache 项
                final = clip_ll * torch.exp(1.0 / ADAPT_SCALE[ds] * lr_score.log_softmax(0))
                pred = int(final.argmax())
            else:  # dotaclip
                w = min(max(rho * float(method.Nc.mean()), 0.0), eta)
                final = clip_ll + w * lr_score
                pred = int(final.argmax())
        else:
            pred = int(clip_ll.argmax())
        correct += int(pred == target)
        total += 1
    return 100.0 * correct / total


# ==================== ImageNet-C（--family inc）====================
C_DATA = '/home/liangyiwen/datasets'
C_PTH = {
    'vit': '/home/liangyiwen/wufan/ADAPT/pre_extracted_class_feat/ViT-B16/GPT_w_Custom_class_emb/imagenet.pth',
    'rn50': '/home/liangyiwen/wufan/ADAPT/pre_extracted_class_feat/RN50/GPT_w_Custom_class_emb/imagenet.pth',
}
C_OUT = {
    'vit': '/mnt/newdata/liangyiwen/imagenet_c_feats',        # ViT-B16 512-d
    'rn50': '/mnt/newdata/liangyiwen/imagenet_c_feats_rn50',  # RN50 1024-d
}
C_OUT_AUG63 = '/mnt/newdata/liangyiwen/imagenet_c_feats_aug63'  # 63-view 增强聚合（ViT）

# 评估顺序：Defo→Glas→Moti→Zoom→Snow→Fros→Fog→Brig→Cont→Elas→Pix→JPEG→Gauss→Shot→Impu
CORDER = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
          'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
          'gaussian_noise', 'shot_noise', 'impulse_noise']


def c_load_model(gpu, backbone='vit'):
    dev = f'cuda:{gpu}'
    arch = 'ViT-B/16' if backbone == 'vit' else 'RN50'
    model, preprocess = clip_pkg.load(arch, device=dev)
    model.eval()
    return model, preprocess, dev


def c_extract(corr, model, preprocess, dev, force=False, backbone='vit'):
    import torchvision  # noqa
    from data.datautils import build_test_loader
    os.makedirs(C_OUT[backbone], exist_ok=True)
    out = os.path.join(C_OUT[backbone], f'{corr}.npz')
    if os.path.exists(out) and not force:
        print(f'[ic-extract] {corr}: cache exists, skip', flush=True)
        return
    loader = build_test_loader('imagenet_c', preprocess, C_DATA, batch_size=64,
                               corruption=corr, level='5')
    feats, targets = [], []
    with torch.no_grad():
        for im, lbl in loader:
            im = im.to(dev)
            f = model.encode_image(im)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.float().cpu())
            targets.append(lbl)
    feats = torch.cat(feats).numpy()
    targets = torch.cat(targets).numpy()
    np.savez(out, feats=feats, targets=targets)
    print(f'[ic-extract] saved {corr}: {feats.shape} ({time.strftime("%H:%M:%S")})', flush=True)


@torch.no_grad()
def c_zero_shot(feats, tgt, Wt, dev, chunk=50000):
    ok = torch.zeros(feats.shape[0], dtype=torch.bool)
    for i in range(0, feats.shape[0], chunk):
        X = feats[i:i + chunk].to(dev)
        logits = 100.0 * X @ Wt.to(dev)
        ok[i:i + chunk] = (logits.argmax(1) == tgt[i:i + chunk].to(dev)).cpu()
    return 100.0 * ok.float().mean().item()


@torch.no_grad()
def c_run_online(corr, gpu, rank, sketch, shrink, eps, warmup=64, backbone='vit', cache='singlev'):
    fdir = C_OUT_AUG63 if cache == 'aug63' else C_OUT[backbone]
    d = np.load(os.path.join(fdir, f'{corr}.npz'))
    feats = torch.from_numpy(d['feats']).float()
    tgt = torch.from_numpy(d['targets']).long()
    if cache == 'aug63':
        # aug63 约定：软标签 = 选中视图 logit-ensemble；zs 用 ensemble argmax
        clip_ll = torch.from_numpy(d['clip_ll']).float()
        C, D = clip_ll.shape[1], feats.shape[1]
        Wt, zs = None, 100.0 * (clip_ll.argmax(1) == tgt).float().mean().item()
    else:
        Wt = torch.load(C_PTH[backbone], map_location='cpu').float()
        if Wt.shape[0] != feats.shape[1]:
            Wt = Wt.t()
        C, D = Wt.shape[1], feats.shape[1]
        clip_ll, zs = None, None
    dev = f'cuda:{gpu}'
    if Wt is not None:
        Wt = Wt.to(dev)
    N = feats.shape[0]
    if zs is None:
        zs = c_zero_shot(feats, tgt, Wt, gpu)

    m = LowRankLDA(C, D, dev, warmup, rank=rank, sketch_mult=sketch,
                            sigma_shrink=shrink, sigma_eps=eps)
    ok = 0
    for t in range(N):
        x = feats[t].to(dev)
        if clip_ll is not None:
            ll = clip_ll[t].to(dev)
        else:
            ll = 100.0 * (x @ Wt)
        m.update(x, ll.softmax(0))
        if m.ready():
            pred = int(m.predict(x).argmax().item())
        else:
            pred = int(ll.argmax().item())
        ok += int(pred == tgt[t].item())
    return zs, 100.0 * ok / N


def main_inc(a):
    import torchvision  # noqa: F401
    sys.path.insert(0, '/home/liangyiwen/wufan/ADAPT')
    if a.mode == 'extract':
        model, preprocess, dev = c_load_model(a.gpu, a.backbone)
        for corr in (a.datasets or CORDER):
            c_extract(corr, model, preprocess, dev, a.force, a.backbone)
        return
    torch.cuda.set_device(a.gpu)
    rows = []
    for corr in (a.datasets or CORDER):
        t0 = time.time()
        zs, lr = c_run_online(corr, a.gpu, a.rank, a.sketch, a.shrink, a.sigma_eps,
                              backbone=a.backbone, cache=a.cache)
        rows.append((corr, zs, lr))
        print(f'[ic-lr] {corr:18s} clip={zs:6.2f}  lr(online)={lr:6.2f}  '
              f'(r={a.rank}/s={a.sketch}/{a.backbone}/{a.cache}) '
              f'({time.time()-t0:.0f}s)', flush=True)
    if len(rows) > 1:
        z = sum(r[1] for r in rows) / len(rows)
        l = sum(r[2] for r in rows) / len(rows)
        print(f'[ic-lr] AVG({len(rows)}域) clip={z:.2f}  lr(online)={l:.2f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', default='inet5', choices=['inet5', 'inc'],
                    help='inet5=imagenet 系 5 域（缓存特征）; inc=ImageNet-C（在线流式）')
    ap.add_argument('--mode', default='run', choices=['run', 'extract'],
                    help='仅 family=inc: extract=提取 C 特征缓存')
    ap.add_argument('--datasets', nargs='*')
    ap.add_argument('--gpu', type=int, default=6)
    ap.add_argument('--rank', type=int, default=64)
    ap.add_argument('--sketch', type=int, default=1)
    ap.add_argument('--warmup', type=int, default=64)
    ap.add_argument('--fuse', default='dotaclip', choices=['none', 'dotaclip', 'alphamix', 'adaptmul'])
    ap.add_argument('--rho', type=float, default=None)
    ap.add_argument('--eta', type=float, default=None)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--norm', default='minmax', choices=['minmax', 'softmax'])
    ap.add_argument('--mu_alpha', type=float, default=0.0)
    ap.add_argument('--gate', type=float, default=0.0)
    ap.add_argument('--hard_assign', action='store_true')
    ap.add_argument('--cache', default='aug63', choices=['singlev', 'aug63'])
    ap.add_argument('--shrink', default=None, choices=['iso', 'scale', 'add', 'trace', 'tfull', 'trade'],
                    help='默认按 family：inet5=add(ε=0.05)、inc=iso(ε=0)')
    ap.add_argument('--sigma_eps', type=float, default=None,
                    help='默认按 family：inet5=0.05、inc=0.0')
    ap.add_argument('--arch', default='ViT-B16', choices=['ViT-B16', 'RN50'])
    ap.add_argument('--backbone', default='vit', choices=['vit', 'rn50'])
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--level', default='5')
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)

    # 生产默认按 family 区分：inet5=add 0.05、inc=iso 0.0
    if args.shrink is None:
        args.shrink = 'iso' if args.family == 'inc' else 'add'
    if args.sigma_eps is None:
        args.sigma_eps = 0.0 if args.family == 'inc' else 0.05

    if args.family == 'inc':
        main_inc(args)
        return

    per = {}
    for ds in (args.datasets or ['imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r', 'imagenet']):
        t0 = time.time()
        acc = run_dataset(ds, args.gpu, args.rank, args.sketch,
                          args.warmup, args.fuse, args.rho, args.eta, args.cache,
                          args.shrink, args.sigma_eps, args.arch, args.alpha, args.norm,
                          args.mu_alpha, args.gate, args.hard_assign)
        per[ds] = acc
        print(f'[lr] {ds:16s} acc={acc:6.2f}%  rank={args.rank} fuse={args.fuse}'
              f'{f"/a={args.alpha}/{args.norm}" if args.fuse == "alphamix" else ""}'
              f'{f"/prior={args.mu_alpha}" if args.mu_alpha > 0 else ""}'
              f'{f"/gate={args.gate}" if args.gate > 0 else ""}'
              f'{"hard" if args.hard_assign else ""} '
              f'cache={args.cache} shrink={args.shrink}(eps={args.sigma_eps}) arch={args.arch} '
              f'({time.time()-t0:.1f}s)', flush=True)
    print(f'[lr] AVG = {sum(per.values())/len(per):.2f}%  (cfg rank={args.rank} sketch={args.sketch} '
          f'fuse={args.fuse}'
          f'{f" alpha={args.alpha} norm={args.norm}" if args.fuse == "alphamix" else ""} '
          f'cache={args.cache} shrink={args.shrink} eps={args.sigma_eps})',
          flush=True)


if __name__ == '__main__':
    main()
