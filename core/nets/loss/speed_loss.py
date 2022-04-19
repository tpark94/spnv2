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

class SPEEDLoss(nn.Module):
    def __init__(self):
        super(SPEEDLoss, self).__init__()

    def forward(self, r_raw_pr, t_pr, R_gt, t_gt, anchor_states):
        B, N = anchor_states.shape

        # loss_total = []
        # for b in range(B):
        # Apply transformation loss to anchors with IoU > 0.5
        positive_indices = torch.eq(anchor_states, 1)

        # if num_positive_indices > 0:
        # 6D rotation -> DCM [sum(Ni), 3, 3]
        r6d  = r_raw_pr[positive_indices, :]
        R_pr = rot_6d_to_matrix(r6d)

        # Translation [sum(Ni), 3]
        t_pr = t_pr[positive_indices, :]

        # Ground-truth pose into same shape, then take positive indices
        R_gt = R_gt.view(B,1,3,3).repeat(1,N,1,1)
        R_gt = R_gt[positive_indices, :, :]
        t_gt = t_gt.view(B,1,3).repeat(1,N,1)
        t_gt = t_gt[positive_indices, :]

        # Rotation error
        Rdot  = torch.bmm(R_pr, R_gt.transpose(1,2)) # [sum(Ni), 3, 3]
        trace = Rdot.diagonal(offset=0, dim1=1, dim2=2).sum(-1) # Batch trace
        trace = (trace - 1.0)/2.0

        # Gradient diverges as argument to torch.acos approaches +/- 1 ...
        # Solution: clamp the value to never reach exactly +/- 1
        rot_err = torch.acos(trace.clamp(-1+1e-6, 1-1e-6)) # [rad]

        # Translation error (normalized)
        pos_err = torch.linalg.vector_norm(t_pr - t_gt, ord=2, dim=1)
        pos_err = pos_err.div(torch.linalg.vector_norm(t_gt, ord=2, dim=1))

        # Total loss
        loss = rot_err + pos_err

        return loss.mean()

