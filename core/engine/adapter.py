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

def do_adapt(epoch, cfg, model, bn_features, data_loader, optimizer, log_dir=None,
                device=torch.device('cpu'), scaler=None):

    batch_time = AverageMeter('', 'ms', ':6.1f')
    data_time  = AverageMeter('', 'ms', ':6.1f')

    entropy_meter = AverageMeter('ent', '', ':.2e')

    num_batches = int(cfg.ODR.NUM_TRAIN_SAMPLES / cfg.TRAIN.IMAGES_PER_GPU)
    progress = ProgressMeter(
        num_batches,
        batch_time,
        [entropy_meter],
        prefix="ODR {:03d} ".format(epoch+1))

    # For bn_features accumulation
    bn_mean    = {}
    bn_mean_sq = {}

    # Initialize global indices
    batch_idx = 0
    num_samples_seen = 0

    # Loop through dataloader
    end = time.time()
    loader_iter = iter(data_loader)
    while num_samples_seen < cfg.ODR.NUM_TRAIN_SAMPLES:

        # Next sample
        try:
            images = next(loader_iter)
        except StopIteration:
            loader_iter = iter(data_loader)
            images = next(loader_iter)

        start = time.time()
        data_time.update((start - end)*1000)

        # Debug (uncomment)
        # imshow(images[0])
        # imshow(torch.cat([images[0][0], images[1][0]], dim=2))
        # imshowbbox(images[0], targets['boundingbox'][0])
        # imshowheatmap(images[0], targets['heatmap'][0], 8)

        # compute output
        with autocast(enabled=scaler is not None):
            loss, loss_items = model(images,
                                     is_train=True,
                                     gpu=device,
                                     **{})

            loss = loss / cfg.ODR.IMAGES_PER_BATCH

        # Compute gradients
        loss.backward()

        entropy_meter.update(float(loss_items['ent']), cfg.TRAIN.IMAGES_PER_GPU)

        if cfg.MODEL.USE_GROUPNORM_BACKBONE:
            # Accumulate gradients just like in BN
            if (batch_idx + 1) % cfg.ODR.IMAGES_PER_BATCH == 0:
                # GroupNorm used -- simply update affine parameters
                # clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Zero-out gradients
                optimizer.zero_grad(set_to_none=True)
        else:
            # BatchNorm used -- manually update stats
            for name, feature in bn_features.items():
                # Mean & Mean of squares
                mean    = torch.mean(feature, (0,2,3)) / cfg.ODR.IMAGES_PER_BATCH
                mean_sq = torch.mean(feature.square(), (0,2,3)) / cfg.ODR.IMAGES_PER_BATCH

                if cfg.ODR.IMAGES_PER_BATCH == 1 or \
                    (batch_idx + 1) % cfg.ODR.IMAGES_PER_BATCH == 1:
                    bn_mean[name]    = mean
                    bn_mean_sq[name] = mean_sq
                else:
                    bn_mean[name]    = bn_mean[name] + mean
                    bn_mean_sq[name] = bn_mean_sq[name] + mean_sq

            # Compute & update gradient
            if (batch_idx + 1) % cfg.ODR.IMAGES_PER_BATCH == 0:
                # clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Zero-out gradients
                optimizer.zero_grad(set_to_none=True)

                # Update running stats
                for name, module in model.named_modules():
                    if name in bn_mean.keys():
                        # Current mean & var
                        mean = bn_mean[name]
                        var  = bn_mean_sq[name] - mean.square()

                        # Running mean & var
                        running_mean = module.running_mean
                        running_var  = module.running_var

                        # Update running mean & var
                        mean = (1 - module.momentum) * running_mean + \
                                module.momentum * mean
                        var  = (1 - module.momentum) * running_var + \
                                module.momentum * var

                        # Update
                        with torch.no_grad():
                            module.running_mean.copy_(mean)
                            module.running_var.copy_(var)

                # Zero-out mean/mean of squares
                bn_mean = {}
                bn_mean_sq = {}

        # Record
        batch_time.update((time.time() - start)*1000)
        end = time.time()

        if cfg.VERBOSE:
            progress.display(batch_idx+1)
        if batch_idx+1 == num_batches:
            progress.display_summary()

        # Update indices
        batch_idx += 1
        num_samples_seen += cfg.TRAIN.IMAGES_PER_GPU

    # TODO: tensorboard logging

