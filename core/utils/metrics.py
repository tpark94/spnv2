'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

''' Utility functions for evaluating various performance metrics '''

def bbox_iou(box1, box2, x1y1x2y2=False):
    # Dimension check
    if box1.ndim == 1 and box2.ndim == 1:
        # Both are (4,), convert to [1, 4]
        assert len(box1) == 4 and len(box2) == 4, 'Bounding boxes should have 4 parameters'
        box1, box2 = box1[None, :], box2[None, :]
    elif box1.ndim == 2 and box2.ndim == 2:
        assert box1.shape[1] == box2.shape[1] and box1.shape[1] == 4, \
            'Both bounding boxes should have [N, 4] shape'
    else:
        if box1.ndim == 2:
            assert box1.shape[1] == 4
            box2 = box2[None, :]
        elif box2.ndim == 2:
            assert box2.shape[1] == 4
            box1 = box1[None, :]
        else:
            raise ValueError('Double-check bounding box shapes')

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    else:
        # x, y, w, h = box
        b1_x1, b1_x2 = box1[:,0] - box1[:,2] / 2, box1[:,0] + box1[:,2] / 2
        b1_y1, b1_y2 = box1[:,1] - box1[:,3] / 2, box1[:,1] + box1[:,3] / 2
        b2_x1, b2_x2 = box2[:,0] - box2[:,2] / 2, box2[:,0] + box2[:,2] / 2
        b2_y1, b2_y2 = box2[:,1] - box2[:,3] / 2, box2[:,1] + box2[:,3] / 2

    # Intersection area
    inter_area = np.clip(np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1), 0, None) * \
                 np.clip(np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1), 0, None)

    # Union Area
    union_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area + 1e-16

    iou = inter_area / union_area  # iou

    return iou

def segment_iou(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection + 1e-16
    iou   = intersection / union

    return iou

def error_translation(t_pr, t_gt):
    t_pr = np.reshape(t_pr, (3,))
    t_gt = np.reshape(t_gt, (3,))

    return np.sqrt(np.sum(np.square(t_gt - t_pr)))

def _error_orientation_quaternion(q_pr, q_gt):
    # q must be [qvec, qcos]
    q_pr = np.reshape(q_pr, (4,))
    q_gt = np.reshape(q_gt, (4,))

    qdot = np.abs(np.dot(q_pr, q_gt))
    qdot = np.minimum(qdot, 1.0)
    return np.rad2deg(2*np.arccos(qdot)) # [deg]

def _error_orientation_rotationmatrix(R_pr, R_gt):
    assert R_pr.shape == R_gt.shape and R_pr.ndim == 2
    assert np.abs(np.linalg.det(R_pr) - 1) < 1e-6, \
        f'Determinant of R_pr is {np.linalg.det(R_pr)}'

    Rdot = np.dot(R_pr, np.transpose(R_gt))
    trace = (np.trace(Rdot) - 1.0)/2.0
    trace = np.clip(trace, -1.0, 1.0)
    return np.rad2deg(np.arccos(trace))

def error_orientation(ori_pr, ori_gt, representation='quaternion'):
    assert representation in ['quaternion', 'rotationmatrix'], \
        'Orientation representation must be either quaternion or rotationmatrix'

    if representation == 'quaternion':
        return _error_orientation_quaternion(ori_pr, ori_gt)
    else:
        return _error_orientation_rotationmatrix(ori_pr, ori_gt)

def speed_score(t_pr, ori_pr, t_gt, ori_gt, representation='quaternion',
                    applyThreshold=True, theta_q=0.5, theta_t=0.005):
    # theta_q: rotation threshold [deg]
    # theta_t: normalized translation threshold [m/m]
    err_t = error_translation(t_pr, t_gt)
    err_q = error_orientation(ori_pr, ori_gt, representation) # [deg]

    t_gt = np.reshape(t_gt, (3,))
    speed_t = err_t / np.sqrt(np.sum(np.square(t_gt)))
    speed_q = np.deg2rad(err_q)

    # Check if within threshold
    if applyThreshold and err_q < theta_q:
        speed_q = 0.0

    if applyThreshold and speed_t < theta_t:
        speed_t = 0.0

    speed = speed_t + speed_q

    return speed_t, speed_q, speed


