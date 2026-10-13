#!/usr/bin/env python
"""Extract DPE-CLIP (CuPL) text prototypes for the 5 ImageNet-family domains.

Faithful to DPE-CLIP/utils.py:clip_classifier:
    texts = [t.format(classname) for t in template] + cupl[classname]
    embedding = mean(normalize(encode_text(texts))); normalize  -> one column
with DPE's own per-domain template lists (imagenet/v2/a: 7 templates;
r/sketch: 80 templates) and gpt3_prompts/CuPL_prompts_imagenet.json.

Output: [D, C] pth per domain on the big disk, mirroring the ADAPT tier layout.
"""
import argparse
import ast
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根
sys.path.insert(0, os.environ.get('ADAPT_ROOT', '/home/liangyiwen/wufan/ADAPT'))
from clip import clip as clip_pkg
from data.cls_to_names import get_classnames

DPE = os.environ.get('DPE_ROOT', '/home/liangyiwen/wufan/DPE-CLIP')
CUPL = os.path.join(DPE, 'gpt3_prompts/CuPL_prompts_imagenet.json')
OUT_DIR = os.environ.get('DPE_OUT_DIR',
    '/mnt/newdata/liangyiwen/clip_class_emb/ViT-B16/GPT_w_DPE_class_emb')
DOMAINS = ['imagenet', 'imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r']


def dpe_templates(ds):
    """Read the per-domain template list straight out of the DPE dataset module."""
    src = open(os.path.join(DPE, 'datasets', f'{ds}.py')).read()
    m = re.search(r'^(?:imagenet_templates|template)\s*=\s*(\[.*?\])', src, re.M | re.S)
    assert m, f'no template list in {ds}.py'
    return ast.literal_eval(m.group(1))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--domains', nargs='*', default=DOMAINS)
    a = ap.parse_args()
    dev = f'cuda:{a.gpu}'
    torch.cuda.set_device(a.gpu)
    model, _ = clip_pkg.load('ViT-B/16', device=dev)
    model.eval()
    cupl = json.load(open(CUPL))
    os.makedirs(OUT_DIR, exist_ok=True)
    for ds in a.domains:
        classnames = get_classnames(ds)              # pth 列序
        tmpl = dpe_templates(ds)
        W = []
        for cls in classnames:
            key = cls.replace('_', ' ')
            prompts = [t.format(key) for t in tmpl] + cupl[key]
            prompts = [p.replace('..', '.') for p in prompts]
            with torch.no_grad():
                emb = model.encode_text(clip_pkg.tokenize(prompts).to(dev))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            e = emb.mean(0)
            W.append(e / e.norm())
        W = torch.stack(W, dim=1).float().cpu()      # [D, C]
        torch.save(W, os.path.join(OUT_DIR, f'{ds}.pth'))
        print(f'[dpe] {ds}: {len(classnames)} classes x {len(tmpl)}+{len(cupl[classnames[0].replace("_"," ")])} '
              f'prompts -> {tuple(W.shape)}', flush=True)


if __name__ == '__main__':
    main()
