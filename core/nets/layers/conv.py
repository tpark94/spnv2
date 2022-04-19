'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class ConvBnAct2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True,
                 norm_layer=None, num_groups=None, norm_kwargs=None, act_layer=None):
        super(ConvBnAct2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)

        # Batch/Group norm norm
        self.bn = None
        if norm_layer is not None:
            norm_args = [num_groups, out_channels] if num_groups is not None else [out_channels]
            if norm_kwargs is not None and isinstance(norm_kwargs, dict):
                self.bn = norm_layer(*norm_args, **norm_kwargs)
            else:
                self.bn = norm_layer(*norm_args)

        # Activation
        self.act = act_layer() if act_layer is not None else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)

        return x

class DepthwiseSeparableConv(nn.Module):
    ''' Depthwise Separable Convolution layer for EfficientDet, implementation from
    https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/efficientdet/model.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True,
                 norm_layer=None, num_groups=None, norm_kwargs=None, act_layer=None):
        super(DepthwiseSeparableConv, self).__init__()
        # Depth-wise (disable bias by default)
        self.conv_dw = nn.Conv2d(in_channels, in_channels,
                                 kernel_size=kernel_size, stride=stride, groups=in_channels,
                                 padding=padding, bias=False)

        # Point-wise
        self.conv_pw = nn.Conv2d(in_channels, out_channels,
                                 kernel_size=1, stride=1, padding=0,
                                 bias=bias)

        # Batch norm
        self.bn = None
        if norm_layer is not None:
            norm_args = [num_groups, out_channels] if num_groups is not None else [out_channels]
            if norm_kwargs is not None and isinstance(norm_kwargs, dict):
                self.bn = norm_layer(*norm_args, **norm_kwargs)
            else:
                self.bn = norm_layer(*norm_args)

        # Activation
        self.act = act_layer(inplace=True) if act_layer is not None else None

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)

        return x