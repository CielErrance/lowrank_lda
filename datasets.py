#!/usr/bin/env python
"""Dataset loading for feature extraction (self-contained).

Ported from vMFcache's ``data/fewshot_datasets.py`` + ``data/datautils.py``:
each FG10 dataset lives under ``<data_root>/<dirname>`` with the Zhou et al.
(CoOp) split JSONs inside. Image files are enumerated in split order and the
DataLoader runs with ``shuffle=True`` after ``set_random_seed(seed)`` so the
bs=1 stream order reproduces the cached feature streams exactly.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Data root sub-directory per dataset (same layout as vMFcache).
ID_TO_DIRNAME = {
    'oxford_flowers': 'oxford_flowers',
    'dtd': 'dtd',
    'oxford_pets': 'oxford_pets',
    'stanford_cars': 'stanford_cars',
    'ucf101': 'ucf101',
    'caltech101': 'caltech-101',
    'food101': 'food-101',
    'sun397': 'sun397',
    'fgvc_aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat',
}

_SUPPORTED = set(ID_TO_DIRNAME)


class _BaseJsonDataset(Dataset):
    """Generic dataset whose train/test split is a Zhou-style JSON list of
    (relative_image_path, label)."""
    def __init__(self, image_path, json_path, mode='test', transform=None):
        self.transform = transform
        self.image_path = image_path
        with open(json_path) as fp:
            samples = json.load(fp)[mode]
        self.image_list = [s[0] for s in samples]
        self.label_list = [s[1] for s in samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_path, self.image_list[idx])).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.label_list[idx]).long()


class _Aircraft(Dataset):
    """FGVC-Aircraft has a bespoke layout (variants.txt + images_variant_*.txt)."""
    def __init__(self, root, mode='test', transform=None):
        self.transform = transform
        with open(os.path.join(root, 'variants.txt')) as fp:
            cname = [l.replace('\n', '') for l in fp.readlines()]
        self.image_list, self.label_list = [], []
        with open(os.path.join(root, f'images_variant_{mode}.txt')) as fp:
            for line in fp.readlines():
                parts = line.replace('\n', '').split(' ')
                self.image_list.append(f'{parts[0]}.jpg')
                self.label_list.append(cname.index(' '.join(parts[1:])))
        self.root = root

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, 'images', self.image_list[idx])).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.label_list[idx]).long()


def build_test_loader(set_id, transform, data_root, batch_size, shuffle=True, num_workers=8):
    """Return a DataLoader over the test split of ``set_id``.

    Images live at ``<data_root>/<ID_to_DIRNAME[set_id]>/``. ``shuffle=True``
    (the default) reproduces the original cached stream order when combined
    with ``set_random_seed(0)`` before loader construction.
    """
    if set_id not in _SUPPORTED:
        raise NotImplementedError(f'unsupported dataset: {set_id}')
    root = os.path.join(data_root, ID_TO_DIRNAME[set_id])
    if set_id == 'fgvc_aircraft':
        testset = _Aircraft(root, mode='test', transform=transform)
    else:
        subdir, json_path = _PATH_SPEC[set_id]
        testset = _BaseJsonDataset(
            os.path.join(root, subdir),
            os.path.join(root, json_path),
            mode='test', transform=transform)
    return DataLoader(testset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


_PATH_SPEC = {
    'oxford_flowers': ('jpg', 'split_zhou_OxfordFlowers.json'),
    'food101': ('images', 'split_zhou_Food101.json'),
    'dtd': ('images', 'split_zhou_DescribableTextures.json'),
    'oxford_pets': ('images', 'split_zhou_OxfordPets.json'),
    'sun397': ('SUN397', 'split_zhou_SUN397.json'),
    'caltech101': ('101_ObjectCategories', 'split_zhou_Caltech101.json'),
    'ucf101': ('UCF-101-midframes', 'split_zhou_UCF101.json'),
    'stanford_cars': ('', 'split_zhou_StanfordCars.json'),
    'eurosat': ('2750', 'split_zhou_EuroSAT.json'),
}
