'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time

from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_

from utils.utils     import AverageMeter, ProgressMeter
from utils.visualize import *

logger = logging.getLogger("Training")

def do_train(epoch, cfg, model, data_loader, optimizer, log_dir=None,
                device=torch.device('cpu'), scaler=None, rank=0):

    batch_time = AverageMeter('', 'ms', ':6.1f')
    data_time  = AverageMeter('', 'ms', ':6.1f')

    loss_names = []
    if 'heatmap' in cfg.MODEL.HEAD.LOSS_HEADS:
        loss_names += ['hmap']
    if 'efficientpose' in cfg.MODEL.HEAD.LOSS_HEADS:
        loss_names += ['cls', 'box', 'pose']
    if 'segmentation' in cfg.MODEL.HEAD.LOSS_HEADS:
        loss_names += ['seg']

    loss_meters = {}
    for name in loss_names:
        loss_meters[name] = AverageMeter(name, '', ':.2e')

    progress = ProgressMeter(
        len(data_loader),
        batch_time,
        list(loss_meters.values()),
        prefix="Training {:03d} ".format(epoch+1))

    # switch to train mode
    model.train()

    # Loop through dataloader
    end = time.time()
    for idx, (images, targets) in enumerate(data_loader):
        start = time.time()
        data_time.update((start - end)*1000)

        # Debug (uncomment)
        # imshow(images[0])
        # imshowbbox(images[0], targets['boundingbox'][0])
        # imshowheatmap(images[0], targets['heatmap'][0], 8)

        # Zero gradient
        optimizer.zero_grad(set_to_none=True)

        # Enable mixed-precision learning is scaler is provided
        with autocast(enabled=scaler is not None):
            loss, loss_items = model(images,
                                     is_train=True,
                                     gpu=device,
                                     **targets)

        # Compute & update gradient
        if scaler is not None:
            # Use mixed-precision
            scaler.scale(loss).backward()

            # Unscale before clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)

            # Update the scale for next iteration
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Record losses
        for k, v in loss_items.items():
            loss_meters[k].update(float(v), images.shape[0])

        # Elapsed time
        batch_time.update((time.time() - start)*1000)
        end = time.time()

        # Report progress to console
        if rank == 0:
            if cfg.VERBOSE:
                progress.display(idx)
            if idx+1 == len(data_loader):
                progress.display_summary()

    # TODO: tensorboard logging

