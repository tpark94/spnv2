'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch.nn as nn

from ..layers import DepthwiseSeparableConv
from ..loss   import LossHeatmapMSE

logger = logging.getLogger(__name__)

# Basic keyword arguments for DepthwiseSeparableConv
# - No normalization, no activation
conv_kwargs = {
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'norm_layer': None,
    'act_layer': None
}

# BatchNorm2d keyward arguments
bn_norm_kwargs = {'momentum': 0.003, 'eps': 1e-4}
gn_norm_kwargs = {'eps': 1e-4, 'affine': True}

class HeatmapHead(nn.Module):
    def __init__(self, in_channels, depth, num_keypoints, use_group_norm, group_norm_size):
        super(HeatmapHead, self).__init__()
        self.depth = depth

        self.mid_channels = min(2 * in_channels, 256)

        # Conv layers are shared across all feature levels
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv(
                in_channels if _ == 0 else self.mid_channels,
                self.mid_channels,
                bias=False,
                **conv_kwargs
            ) for _ in range(depth)
        ])

        # Normalization layers must be specific to each path
        num_groups = int(self.mid_channels / group_norm_size)
        self.bns = nn.ModuleList([
            nn.GroupNorm(num_groups, self.mid_channels, **gn_norm_kwargs) if use_group_norm else \
                nn.BatchNorm2d(self.mid_channels, **bn_norm_kwargs) for _ in range(depth)
        ])

        # SiLU (Swish) activation
        self.act = nn.SiLU(inplace=True)

        # Final head layer
        self.head = DepthwiseSeparableConv(
            self.mid_channels,
            num_keypoints,
            bias=True,
            **conv_kwargs
        )

        self.loss = LossHeatmapMSE()

    def forward(self, features, **targets):
        # x: list of feature maps from P2 ~ P7 levels of BiFPN
        # Following the semantic segmentation studies of EfficientDet,
        # take input from P2 only
        feature = features[0]

        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i](feature)
            feature = self.act(feature)
        heatmap = self.head(feature)  # [B, K, H, W]

        if targets:
            loss = self.loss(heatmap, targets['heatmap'])
            loss_item = {'hmap': loss.detach()}
            return loss, loss_item
        else:
            return heatmap