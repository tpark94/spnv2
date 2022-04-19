'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch

from ..loss import ciou_loss

def create_anchors(image_size=(768, 512), scales=None, ratios=None):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117

    Returns:
        anchor_boxes: [N x 4] anchor boxes [xmin, ymin, xmax, ymax] (pix) in input image dimension
        strides: record of strides associated with eacn anchor boxes
    """
    # Basic scaling factor
    anchor_scale = 4

    # Pyramid levels from EfficientDet to use
    # Even if segmentation head exists that uses P2, ignore it
    # (otherwise too many anchor boxes)
    pyramid_levels = [3, 4, 5, 6, 7]

    strides = [2 ** x for x in pyramid_levels]
    if scales is None: scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], dtype=np.float32)
    if ratios is None: ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

    # Generates multiscale anchor boxes.
    H, W = image_size

    boxes_all   = []
    strides_all = []
    for stride in strides:
        boxes_level   = []
        for scale in scales:
            for ratio in ratios:
                if W % stride != 0:
                    raise ValueError('input size must be divisible by the stride.')
                base_anchor_size = anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2

                x = np.arange(int(stride / 2), W, stride)
                y = np.arange(int(stride / 2), H, stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # [xmin, ymin, xmax, ymax]
                boxes = np.vstack((xv - anchor_size_x_2, yv - anchor_size_y_2,
                                   xv + anchor_size_x_2, yv + anchor_size_y_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))

                # strides
                [strides_all.append(stride) for _ in range(len(xv))]

        # concat anchors on the same level to the reshape NxAx4
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))

    anchor_boxes = np.vstack(boxes_all).astype(np.float32)
    anchor_boxes = torch.from_numpy(anchor_boxes)
    strides_all  = torch.tensor(strides_all, dtype=torch.float32)

    assert anchor_boxes.shape[0] == len(strides_all)

    return anchor_boxes, strides_all

def compute_anchor_state(anchors, bbox_gt):
    """ Assign anchor states based on IoU w.r.t. ground-truth bounding box.
        +1: Positive, 0: Negative, -1: Ignore.

        Bounding boxes must be in [xmin, ymin, xmax, ymax] format.

    Args:
        anchors (torch.Tensor): [sum(AHW), 4]
        bbox_gt (torch.Tensor): [B, 4] ground-truth bounding box
    """
    # if bbox_gt.ndim == 1:
    #     assert len(bbox_gt) == 4
    #     bbox_gt = bbox_gt[None, :]

    B = bbox_gt.shape[0]
    N = anchors.shape[0]

    # Anchors and bboxes into same shape
    anchors = anchors.unsqueeze(0).repeat(B,1,1)
    bbox_gt = bbox_gt.unsqueeze(1).repeat(1,N,1)

    anchor_state = torch.ones((B, N), dtype=bbox_gt.dtype,
                                 device=bbox_gt.device).mul(-1)

    iou = ciou_loss(anchors, bbox_gt, return_iou=True)

    # IoU > 0.5 -> positive
    anchor_state[iou > 0.5] = 1

    # IoU < 0.4 -> negative
    anchor_state[iou < 0.4] = 0

    return anchor_state