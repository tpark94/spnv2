'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

# See
# https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142
# for channel sizes

EFFICIENTDET_D0 = CN()
EFFICIENTDET_D0.EFFICIENTNET_CHANNELS = [16, 24, 40, 112, 320] # P1, P2, P3, P4, P5 channel sizes

EFFICIENTDET_D3 = CN()
EFFICIENTDET_D3.EFFICIENTNET_CHANNELS = [24, 32, 48, 136, 384]

EFFICIENTDET_D5 = CN()
EFFICIENTDET_D5.EFFICIENTNET_CHANNELS = [24, 40, 64, 176, 512]

EFFICIENTDET_D6 = CN()
EFFICIENTDET_D6.EFFICIENTNET_CHANNELS = [32, 40, 72, 200, 576]

EFFICIENTDET_EXTRAS = {
    'efficientdet_d0': EFFICIENTDET_D0,
    'efficientdet_d3': EFFICIENTDET_D3,
    'efficientdet_d5': EFFICIENTDET_D5,
    'efficientdet_d6': EFFICIENTDET_D6
}