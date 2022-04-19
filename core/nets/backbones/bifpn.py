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
import torch.nn as nn

from ..layers import DepthwiseSeparableConv, ConvBnAct2D

logger = logging.getLogger(__name__)

# Normalization layer keyword arguments
bn_norm_kwargs = {'momentum': 0.01, 'eps': 1e-4, 'affine': True}
gn_norm_kwargs = {'eps': 1e-4, 'affine': True}

'''
Adopted from the unofficial pytorch implementation at

https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
'''

class Conv1x1Block(nn.Module):
    """ 1x1 Conv + Norm """
    def __init__(self, in_channels, out_channels, use_group_norm=False, group_norm_size=16):
        super(Conv1x1Block, self).__init__()

        if not use_group_norm:
            norm_layer  = nn.BatchNorm2d
            norm_kwargs = bn_norm_kwargs
            num_groups  = None
        else:
            norm_layer  = nn.GroupNorm
            norm_kwargs = gn_norm_kwargs
            num_groups  = int(out_channels / group_norm_size)

        self.conv = ConvBnAct2D(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                norm_layer=norm_layer, num_groups=num_groups, norm_kwargs=norm_kwargs, act_layer=None)

    def forward(self, x):
        x = self.conv(x)
        return x

class SeparableConvBlock(nn.Module):
    """ Depthwise Separable Conv + Norm """
    def __init__(self, in_channels, out_channels=None, use_group_norm=False, group_norm_size=16):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        if not use_group_norm:
            norm_layer  = nn.BatchNorm2d
            norm_kwargs = bn_norm_kwargs
            num_groups  = None
        else:
            norm_layer  = nn.GroupNorm
            norm_kwargs = gn_norm_kwargs
            num_groups  = int(out_channels / group_norm_size)

        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                           norm_layer=norm_layer, num_groups=num_groups, norm_kwargs=norm_kwargs, act_layer=None)

    def forward(self, x):
        x = self.conv(x)
        return x

class BiFPNLayer(nn.Module):
    """ Always fuse P2 ~ P7, as opposed to P3 ~ P7 for original implementation
    """
    def __init__(
        self,
        features,
        num_channels,
        first_time=False,
        epsilon=1e-4,
        attention=True,
        use_p2=False,
        use_group_norm=False,
        group_norm_size=16):
        """
        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
        """
        super(BiFPNLayer, self).__init__()
        self.epsilon = epsilon

        # Use P2?
        self.use_p2 = use_p2

        # Conv layers
        self.conv6_up   = SeparableConvBlock(num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)
        self.conv5_up   = SeparableConvBlock(num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)
        self.conv4_up   = SeparableConvBlock(num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)
        self.conv3_up   = SeparableConvBlock(num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)
        self.conv4_down = SeparableConvBlock(num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)
        self.conv5_down = SeparableConvBlock(num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)
        self.conv6_down = SeparableConvBlock(num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)
        self.conv7_down = SeparableConvBlock(num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)

        if use_p2:
            self.conv3_down = SeparableConvBlock(num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)
            self.conv2_up   = SeparableConvBlock(num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p5_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p6_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p7_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if use_p2:
            self.p3_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.p2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')

        self.swish = nn.SiLU(inplace=True)

        # Initial layer
        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = Conv1x1Block(features[4], num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)
            self.p4_down_channel = Conv1x1Block(features[3], num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)
            self.p3_down_channel = Conv1x1Block(features[2], num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)

            self.p5_to_p6 = nn.Sequential(
                Conv1x1Block(features[4], num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.p6_to_p7 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.p4_down_channel_2 = Conv1x1Block(features[3], num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)
            self.p5_down_channel_2 = Conv1x1Block(features[4], num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)

            if use_p2:
                self.p2_down_channel = Conv1x1Block(features[1], num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)
                self.p3_down_channel_2 = Conv1x1Block(features[2], num_channels, use_group_norm=use_group_norm, group_norm_size=group_norm_size)

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        if use_p2:
            self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p2_w1_relu = nn.ReLU()
            self.p3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
            self.p3_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            outs = self._forward_fast_attention(inputs)
        else:
            outs = self._forward(inputs)

        return outs

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            _, p2, p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

            if self.use_p2:
                p2_in = self.p2_down_channel(p2)

        else:
            # (P2_0), P3_0, P4_0, P5_0, P6_0 and P7_0
            if self.use_p2:
                p2_in, p3_in, p4_in, p5_in, p6_in, p7_in = inputs
            else:
                p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

        if self.use_p2:
            p3_up = p3_out
            # Weights for P2_0 and P3_1 to P2_2
            p2_w1 = self.p2_w1_relu(self.p2_w1)
            weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
            # Connections for P2_0 and P3_1 to P2_2 respectively
            p2_out = self.conv2_up(self.swish(weight[0] * p2_in + weight[1] * self.p2_upsample(p3_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)
            if self.use_p2: p3_in = self.p3_down_channel_2(p3)

        if self.use_p2:
            # Weights for P3_0, P3_1 and P2_2 to P3_2
            p3_w2 = self.p3_w2_relu(self.p3_w2)
            weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
            # Connections for P3_0, P3_1 and P2_2 to P3_2 respectively
            p3_out = self.conv3_down(
                self.swish(weight[0] * p3_in + weight[1] * p3_up + weight[2] * self.p3_downsample(p2_out)))

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        if self.use_p2:
            return [p2_out, p3_out, p4_out, p5_out, p6_out, p7_out]
        else:
            return [p3_out, p4_out, p5_out, p6_out, p7_out]

    def _forward(self, inputs):
        if self.first_time:
            _, p2, p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        return [p3_out, p4_out, p5_out, p6_out, p7_out]