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

class LossHeatmapMSE(nn.Module):
    def __init__(self):
        super(LossHeatmapMSE, self).__init__()

    def forward(self, x, y):
        assert x.size() == y.size()

        # MSE loss for heatmaps
        loss = torch.square(x - y)
        loss = loss.sum(dim=(2,3)).mean(dim=1).mean()

        return loss
