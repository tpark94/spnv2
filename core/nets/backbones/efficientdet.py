'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from .efficientnet import *
from .bifpn        import BiFPNLayer

import logging

logger = logging.getLogger(__name__)

bn_norm_kwargs = {'momentum': 0.01, 'eps': 1e-4, 'affine': True}
gn_norm_kwargs = {'eps': 1e-4, 'affine': True}

'''
Based on

EfficientPose (official, tensorflow): https://github.com/ybkscht/EfficientPose
EfficientDet  (official, tensorflow): https://github.com/google/automl
EfficientDet  (unofficial, pytorch):  https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
'''

class MyGroupNorm16(nn.GroupNorm):
    def __init__(self, num_channels):
        super().__init__(int(num_channels/16), num_channels)

class MyGroupNorm8(nn.GroupNorm):
    def __init__(self, num_channels):
        super().__init__(int(num_channels/8), num_channels)

class EfficientDet(nn.Module):
    def __init__(self, cfg, scaled_parameters):
        super().__init__()

        # Build EfficientNet backbone
        if not cfg.MODEL.USE_GROUPNORM_BACKBONE:
            # Regular with BN (pretrained on ImageNet)
            self.efficientnet = eval(f'efficientnet_b{cfg.MODEL.EFFICIENTDET_PHI}')(
                pretrained=True
            )
        else:
            # BN layers replaced with GN
            # NOTE: group size for GN in backbone is fixed
            if cfg.MODEL.EFFICIENTDET_PHI == 0:
                norm = MyGroupNorm16
            else:
                norm = MyGroupNorm8

            # Build model with GN
            self.efficientnet = eval(f'efficientnet_b{cfg.MODEL.EFFICIENTDET_PHI}')(
                norm_layer=norm,
                num_classes=100 if cfg.MODEL.BACKBONE.PRETRAINED else 1000
            )

            if cfg.MODEL.BACKBONE.PRETRAINED:
                # Model pre-trained on ImageNet-100?
                self.efficientnet.load_state_dict(
                    torch.load(cfg.MODEL.BACKBONE.PRETRAINED, map_location='cpu')['state_dict']
                )
            else:
                # If not pre-trained, set initial weights of residual connection layers to zero
                # This makes the residual layer act as an identity function, which stabilizes training
                for m in self.efficientnet.modules():
                    if isinstance(m, MBConv):
                        for n, m2 in m.named_modules():
                            if isinstance(m2, (nn.BatchNorm2d, nn.GroupNorm)) \
                                and n=='block.3.1' and m.use_res_connect:
                                nn.init.zeros_(m2.weight)

        # Remove last Convolutional head (1x1), AvgPool and FC layers
        self.efficientnet = nn.ModuleList(self.efficientnet.children())[:-2][0][:-1]

        # Use P2 features (x4)?
        # - Always compute from P2, but only heatmap/segmentation heads use them
        # - Bounding box/rotation/translation use P3 ~ P7 for computational efficiency
        self.use_p2 = True

        # BiFPN model
        self.bifpn = nn.Sequential(
            *[
                BiFPNLayer(cfg.MODEL.BACKBONE.EXTRA.EFFICIENTNET_CHANNELS,
                           scaled_parameters['bifpn_width'],
                           first_time=True if _ == 0 else False,
                           attention=True,
                           use_p2=self.use_p2,
                           use_group_norm=cfg.MODEL.USE_GROUPNORM_BACKBONE,
                           group_norm_size=cfg.MODEL.GROUPNORM_SIZE)
                for _ in range(scaled_parameters['bifpn_depth'])
            ]
        )

    def forward(self, x):
        features = [None] # P1 feature placeholder
        for idx in range(len(self.efficientnet)):
            x = self.efficientnet[idx](x)

            # According to official implementation, extract features if
            # - next MBConv stage has stride 2, or
            # - at the last stage
            # The returned features have strides x2, x4, x8, x16, x32, and we only take the
            # last three levels for BiFPN layers, where initially, x64, x128 layers are simply downsampled
            # from the last feature map.
            # Additionally, include P2 feature if performing heatmap prediction or ssegmentation.
            if idx == 2:
                features.append(x) if self.use_p2 else features.append(None)

            if idx in [3,5,7]:
                features.append(x)

        # Return BiFPN features
        return self.bifpn(features)

if __name__=='__main__':
    net = EfficientDet()

    x = torch.rand(1, 3, 512, 768)
    print(EfficientDet(x).shape)