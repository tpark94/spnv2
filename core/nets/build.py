'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import *
from .heads     import *
from utils.utils import num_total_parameters

logger = logging.getLogger(__name__)

def get_scaled_parameters(phi):
    """ Get all relevant scaled parameters to build EfficientPose.
        From the official EfficientPose repository:
            https://github.com/ybkscht/EfficientPose/blob/main/model.py

        Args:
            phi: EfficientPose scaling hyperparameter phi

        Returns:
        Dictionary containing the scaled parameters
    """
    # info tuples with scalable parameters
    # image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    bifpn_widths = (64, 88, 112, 160, 224, 288, 384)
    bifpn_depths = (3, 4, 5, 6, 7, 7, 8)
    subnet_depths = (3, 3, 3, 4, 4, 4, 5)
    subnet_iteration_steps = (1, 1, 1, 2, 2, 2, 3)
    num_groups_gn = (4, 4, 7, 10, 14, 18, 24) # try to get 16 channels per group

    parameters = {"bifpn_width": bifpn_widths[phi],
                  "bifpn_depth": bifpn_depths[phi],
                  "subnet_depth": subnet_depths[phi],
                  "subnet_num_iteration_steps": subnet_iteration_steps[phi],
                  "num_groups_gn": num_groups_gn[phi]}

    return parameters

def _build_backbone(cfg):
    """ Create EfficientDet backbone

        Args:
            cfg: Object containing configuration parameters

        Returns:
            backbone: EfficientDet (EfficientNet + BiFPN) backbone network
    """
    assert 'efficientdet' in cfg.MODEL.BACKBONE.NAME, \
        'Only efficientdet backbone is supported at the moment'

    scaled_parameters = get_scaled_parameters(cfg.MODEL.EFFICIENTDET_PHI)
    backbone = EfficientDet(cfg, scaled_parameters)

    logger.info(f'   - Backbone: {cfg.MODEL.BACKBONE.NAME} (# param: {num_total_parameters(backbone):,d})')

    if cfg.MODEL.USE_GROUPNORM_BACKBONE:
        logger.info(f'   - GroupNorm built for backbone')
        logger.info(f'   - Pretrained model loaded from {cfg.MODEL.BACKBONE.PRETRAINED}')

    return backbone

def _build_heads(cfg):
    """ Build head networks based on the configuration. Individual head networks
        take input from the output of the backbone network.

    Args:
        cfg ([type]): Object containing configuration parameters

    Returns:
        heads: nn.ModuleList of head networks
    """
    heads   = []

    scaled_parameters = get_scaled_parameters(cfg.MODEL.EFFICIENTDET_PHI)

    # Create heads
    for i, name in enumerate(cfg.MODEL.HEAD.NAMES):
        if name == 'heatmap':
            head = HeatmapHead(scaled_parameters['bifpn_width'],
                               scaled_parameters['subnet_depth'],
                               cfg.DATASET.NUM_KEYPOINTS,
                               cfg.MODEL.USE_GROUPNORM_HEADS,
                               cfg.MODEL.GROUPNORM_SIZE)

        elif name == 'efficientpose':
            head = EfficientPoseHead(cfg,
                                     scaled_parameters['bifpn_width'],
                                     scaled_parameters['subnet_depth'],
                                     scaled_parameters['subnet_num_iteration_steps'],
                                     cfg.MODEL.USE_GROUPNORM_HEADS,
                                     scaled_parameters['num_groups_gn'])

        elif name == 'segmentation':
            head = SegmentationHead(scaled_parameters['bifpn_width'],
                                    scaled_parameters['subnet_depth'],
                                    cfg.MODEL.USE_GROUPNORM_HEADS,
                                    cfg.MODEL.GROUPNORM_SIZE)

        else:
            logger.error(f'{name}-type head is not defined or imported')

        logger.info(f'   - Head #{i+1}: {name} (# param: {num_total_parameters(head):,d})')

        heads.append(head)

    if cfg.MODEL.USE_GROUPNORM_HEADS:
        logger.info(f'   - GroupNorm built for prediction heads')

    return nn.ModuleList(heads)

def _shannon_entropy(x):
    """ Shannon entropy of pixel-wise logits """
    b = torch.sigmoid(x) * F.logsigmoid(x)
    b = -1.0 * b.mean()
    return b

class SPNv2(nn.Module):
    ''' Generic ConvNet consisting of a backbone and (possibly multiple) heads
        for different tasks
    '''
    def __init__(self, cfg):
        super().__init__()
        logger.info('Creating SPNv2 ...')

        # Build backbone
        self.backbone = _build_backbone(cfg)

        # Build task-specific heads
        self.heads      = _build_heads(cfg)
        self.head_names = cfg.MODEL.HEAD.NAMES

        # Which heads to compute loss?
        self.loss_h_idx = [self.head_names.index(h) for h in cfg.MODEL.HEAD.LOSS_HEADS]

        # Loss factors
        self.loss_factors = cfg.MODEL.HEAD.LOSS_FACTORS

        # Which head for inference?
        self.test_h_idx = [self.head_names.index(h) for h in cfg.TEST.HEAD]

        # Entropy minimization for segmentation?
        self.min_entropy = cfg.ODR.MIN_ENTROPY

    def forward(self, x, is_train=False, gpu=torch.device('cpu'), **targets):

        # Backbone forward pass
        x = self.backbone(x.to(gpu, non_blocking=True))

        if is_train:
            # Training - prediction heads
            loss = 0
            losses = {}
            for i, head in enumerate(self.heads):
                # --- Supervised loss if specified --- #
                if i in self.loss_h_idx:
                    if self.head_names[i] == 'efficientpose':
                        head_targets = {
                            k: v.to(gpu, non_blocking=True) for k, v in targets.items() \
                                if k in ['boundingbox', 'rotationmatrix', 'translation']
                        }
                    elif self.head_names[i] == 'heatmap':
                        head_targets = {
                            k: v.to(gpu, non_blocking=True) for k, v in targets.items() \
                                if k in ['heatmap']
                        }
                    elif self.head_names[i] == 'segmentation':
                        head_targets = {
                            k: v.to(gpu, non_blocking=True) for k, v in targets.items() \
                                if k in ['mask']
                        }
                    else:
                        raise NotImplementedError(f'{self.head_names[i]} is not implemented')

                    # Through i-th head
                    loss_i, loss_items = head(x, **head_targets)

                    # Append individual loss
                    loss   = loss + self.loss_factors[i] * loss_i
                    losses = {**losses, **loss_items}

            # --- Unsupervised loss --- #
            # Min entropy via segmentation
            if self.min_entropy and 'segmentation' in self.head_names:
                i = self.head_names.index('segmentation')
                logit  = self.heads[i](x) # [B, 1, H, W]
                loss_i = _shannon_entropy(logit)
                loss_items = {'ent': loss_i.detach()}

                loss   = loss + 1.0 * loss_i
                losses = {**losses, **loss_items}

            return loss, losses
        else:
            out = []
            for i in self.test_h_idx:
                out.append(self.heads[i](x))
            return out

def _check_bn_exists(module, module_name):
    """ Check if BN layers exist in a module """
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d):
            warnings.warn(f'GroupNorm is activated for {module_name} but found a BatchNorm layer at {name}!')

def build_spnv2(cfg):
    net = SPNv2(cfg)

    # if using group_norm, make sure there's no BN layers
    if cfg.MODEL.USE_GROUPNORM_BACKBONE:
        _check_bn_exists(net.backbone, 'backbone')
    if cfg.MODEL.USE_GROUPNORM_HEADS:
        _check_bn_exists(net.heads, 'heads')

    if cfg.MODEL.PRETRAIN_FILE:
        load_dict = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location='cpu')
        net.load_state_dict(load_dict, strict=True)

        logger.info(f'   - Pretrained model loaded from {cfg.MODEL.PRETRAIN_FILE}')

    return net