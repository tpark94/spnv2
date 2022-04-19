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

from utils.postprocess import rot_6d_to_matrix

class TransformationLoss(nn.Module):
    def __init__(self, model_3d_points):
        super(TransformationLoss, self).__init__()
        self.model_3d_points = model_3d_points # [3, M]

    def forward(self, r_raw_pr, t_pr, R_gt, t_gt, anchor_states):
        B = r_raw_pr.shape[0]

        loss_total = 0
        for b in range(B):
            # Apply transformation loss to anchors with IoU > 0.5
            positive_indices = torch.eq(anchor_states[b], 1)

            if positive_indices.sum() > 0:
                # 6D rotation -> DCM [N, 3, 3]
                r6d  = r_raw_pr[b, positive_indices, :]
                R_pr = rot_6d_to_matrix(r6d)

                # Translation [N, 3]
                t_pr_i = t_pr[b, positive_indices, :]

                # Ground-truth [N, 3, 3], [N, 3]
                R_gt_i = R_gt[b].view(1,3,3).repeat((positive_indices.sum(), 1, 1))
                t_gt_i = t_gt[b].view(1,3).repeat((positive_indices.sum(), 1))

                # Model points [N, 3, M]
                mx = self.model_3d_points.view(1,3,-1).repeat((positive_indices.sum(), 1, 1))

                # Transformation loss
                proj_pr = torch.bmm(R_pr, mx) + t_pr_i.view(-1,3,1)
                proj_gt = torch.bmm(R_gt_i, mx) + t_gt_i.view(-1,3,1)

                # loss = torch.sum((proj_pr - proj_gt) ** 2, dim=1).mean()
                loss = torch.linalg.vector_norm(proj_pr - proj_gt, ord=2, dim=1)
                loss_total += loss.mean()

        return loss_total.div(B)

