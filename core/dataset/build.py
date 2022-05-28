'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import random

from .SPEEDPLUSDataset  import SPEEDPLUSDataset
from .transforms        import *
from .target_generators import HeatmapGenerator

def _seed_worker(worker_id):
    """ Set seeds for dataloader workers. For more information, see below

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataset(cfg, split='train', load_labels=True):

    transforms = build_transforms(cfg, is_train=(split=='train'), load_labels=load_labels)

    # Specify target generators only if loading labels
    if load_labels:
        target_generators = []
        for h in cfg.MODEL.HEAD.LOSS_HEADS:
            if h == 'heatmap': g = HeatmapGenerator(
                [int(in_size/cfg.DATASET.OUTPUT_SIZE[0]) for in_size in cfg.DATASET.INPUT_SIZE],
                cfg.DATASET.NUM_KEYPOINTS, cfg.DATASET.SIGMA
            )
            else:
                g = None

            target_generators.append(g)
    else:
        target_generators = None

    dataset = SPEEDPLUSDataset(cfg, split, transforms, target_generators)

    return dataset

def get_dataloader(cfg, split='train', distributed=False, load_labels=True):

    if split=='train':
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        shuffle = cfg.TRAIN.SHUFFLE
        num_workers = min(cfg.TRAIN.IMAGES_PER_GPU, cfg.TRAIN.WORKERS)
    else:
        images_per_gpu = cfg.TEST.IMAGES_PER_GPU
        shuffle = False
        num_workers = 0

    dataset = build_dataset(cfg,
                            split=split,
                            load_labels=load_labels)

    if split=='train' and distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True
        )
        shuffle = False # Shuffling done by sampler
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_gpu,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_seed_worker,
        sampler=train_sampler
    )

    return data_loader
