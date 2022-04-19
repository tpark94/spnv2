'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch

logger = logging.getLogger(__name__)

def _get_trainable_param(module):
    return filter(lambda p: p.requires_grad, module.parameters())

def get_optimizer(cfg, model):

    logger.info(f'Creating optimizer: {cfg.TRAIN.OPTIMIZER}')
    logger.info(f'   - Initial LR: {cfg.TRAIN.LR}')

    param = _get_trainable_param(model)

    # Keyword arguments
    kwargs = {'lr': cfg.TRAIN.LR, 'weight_decay': cfg.TRAIN.WD}
    if cfg.TRAIN.OPTIMIZER in ['Adam', 'AdamW']:
        kwargs['betas'] = (cfg.TRAIN.GAMMA1, cfg.TRAIN.GAMMA2)
        if cfg.FP16:
            kwargs['eps'] = 1e-4
    else:
        kwargs['momentum'] = cfg.TRAIN.GAMMA1

    # Create optimizer
    optimizer = getattr(torch.optim, cfg.TRAIN.OPTIMIZER)(param, **kwargs)

    return optimizer

def get_scaler(cfg):
    scaler = None
    if cfg.FP16 and cfg.CUDA:
        scaler = torch.cuda.amp.GradScaler()
        logger.info('Mixed-precision training: ENABLED')

    return scaler

def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate based on schedule"""

    lr_step = cfg.TRAIN.LR_STEP
    if not isinstance (lr_step, (list, tuple)):
        lr_step = [lr_step]

    if cfg.TRAIN.SCHEDULER == 'step':
        lr = cfg.TRAIN.LR * (cfg.TRAIN.LR_FACTOR ** sum([epoch >= s for s in lr_step]))
    else:
        NotImplementedError('Only step-wise scheduler is implemented')

    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = cfg.TRAIN.LR
        else:
            param_group['lr'] = lr

    logger.info(f'Current epoch learning rate: {lr:.2e}')