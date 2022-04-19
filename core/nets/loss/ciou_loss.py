'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn

def ciou_loss(bbox1, bbox2, return_iou=False):
    ''' Complete IoU loss, adapted from the official repo

    https://github.com/Zzh-tju/CIoU/blob/master/layers/modules/multibox_loss.py

    Bounding boxes [..., 4] are organized as [xmin, ymin, xmax, ymax] (pix)
    '''
    w1  = bbox1[...,2] - bbox1[...,0]
    h1  = bbox1[...,3] - bbox1[...,1]
    xc1 = bbox1[...,0] + w1/2
    yc1 = bbox1[...,1] + h1/2

    w2  = bbox2[...,2] - bbox2[...,0]
    h2  = bbox2[...,3] - bbox2[...,1]
    xc2 = bbox2[...,0] + w2/2
    yc2 = bbox2[...,1] + h2/2

    # Individual bbox area
    area1 = w1 * h1
    area2 = w2 * h2

    # Intersection
    inter_l = torch.max(xc1 - w1 / 2, xc2 - w2 / 2)
    inter_r = torch.min(xc1 + w1 / 2, xc2 + w2 / 2)
    inter_t = torch.max(yc1 - h1 / 2, yc2 - h2 / 2)
    inter_b = torch.min(yc1 + h1 / 2, yc2 + h2 / 2)
    inter_area = torch.clamp((inter_r - inter_l), min=0) * \
                        torch.clamp((inter_b - inter_t), min=0)

    # Union
    union = area1 + area2 - inter_area + 1e-16

    # IoU
    iou = inter_area / union
    if return_iou: return iou

    # Central point distance
    c_l = torch.min(xc1 - w1 / 2, xc2 - w2 / 2)
    c_r = torch.max(xc1 + w1 / 2, xc2 + w2 / 2)
    c_t = torch.min(yc1 - h1 / 2, yc2 - h2 / 2)
    c_b = torch.max(yc1 + h1 / 2, yc2 + h2 / 2)
    inter_diag = (xc2 - xc1).pow(2) + (yc2 - yc1).pow(2)
    c_diag = torch.clamp((c_r - c_l), min=0).pow(2) + \
                    torch.clamp((c_b - c_t), min=0).pow(2)

    # Normalized central point distance
    u = (inter_diag) / c_diag

    # Aspect ratio
    # v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    v = (4 / (math.pi ** 2)) * (torch.atan2(w2, h2) - torch.atan2(w1, h1)).pow(2) # torch.atan causes NaN loss
    with torch.no_grad():
        S = (iou>0.5).half()
        alpha = S*v/(1-iou+v)

    # Complete IoU
    cious = iou - u - alpha * v
    cious = torch.clamp(cious, min=-1.0, max=1.0)

    return torch.sum(1-cious)

class CIoULoss(nn.Module):
    """ Class wrapper for ciou_loss """
    def __init__(self):
        super(CIoULoss, self).__init__()

    def forward(self, bbox_pred, bbox_gt, anchor_states):
        """[summary]

        Args:
            bbox_pred ([type]): [description]
            bbox_gt ([type]): [description]
            anchor_states ([type]): [description]
        """
        bbox_gt = bbox_gt.unsqueeze(1).repeat(1,anchor_states.shape[1],1)

        positive_indices = torch.eq(anchor_states, 1)
        num_positive_indices = positive_indices.sum()

        # Apply bbox loss to anchors with IoU > 0.5
        bbox_pred = bbox_pred[positive_indices, :]
        bbox_gt   = bbox_gt[positive_indices, :]

        bbox_loss = ciou_loss(bbox_pred, bbox_gt, return_iou=False)

        return bbox_loss.div(num_positive_indices)
