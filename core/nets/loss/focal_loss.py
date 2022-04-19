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
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.alpha   = 0.25
        self.gamma   = 2.0

    def forward(self, classification, anchor_states):
        """ Args:
                classification (torch.Tensor): [B, sum(AHW)] logits
                anchor_states  (torch.Tensor): [B, sum(AHW)]
        """
        p = torch.sigmoid(classification) # Head returns logits

        positive_indices = torch.eq(anchor_states,  1)
        ignore_indices   = torch.eq(anchor_states, -1)

        num_positive_indices = positive_indices.sum()

        # Focal loss
        # - Apply alpha to anchors with IoU > 0.5, 1-alpha to IoU < 0.5
        alpha = torch.where(positive_indices, self.alpha, 1 - self.alpha)
        focal_weight = torch.where(positive_indices, 1 - p, p)
        focal_weight = alpha * focal_weight.pow(self.gamma)

        cls_loss = focal_weight * F.binary_cross_entropy_with_logits(
                    classification, anchor_states, reduction='none')

        # Ignore (zero loss) 0.4 < iou < 0.5
        zeros    = torch.zeros_like(cls_loss)
        cls_loss = torch.where(ignore_indices, zeros, cls_loss)
        cls_loss = cls_loss.sum().div(torch.clamp(num_positive_indices, min=1.0))

        return cls_loss

