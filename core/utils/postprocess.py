'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

import logging
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Heatmap post-processing
# -----------------------------------------------------------------------
def solve_pose_from_heatmaps(heatmaps, image_size, threshold, camera, keypts_true_3D):
    # heatmaps: [Nk x H x W]
    nK, nH, nW = heatmaps.shape

    # Take max
    maxVal, maxInd = heatmaps.view(nK, -1).max(-1)

    # Apply heatmaps until we have enough keypoints detected
    visibleIdx = []
    while len(visibleIdx) < 5 and threshold >= 0:
        visibleIdx = torch.nonzero(maxVal > threshold)
        threshold -= 0.1

    if len(visibleIdx) < 5:
        logger.info('Detected sample with less than 5 keypoints detected')
        return None, np.array([1,0,0,0]), np.array([0,0,5])
    else:
        # Extract keypoint locations in [pix]
        keypts = torch.zeros(nK, 2)
        for k in range(nK):
            if k in visibleIdx:
                y = torch.floor(maxInd[k] / nW)
                x = maxInd[k] - nW * y
                keypts[k,0] = x / nW * image_size[0]
                keypts[k,1] = y / nH * image_size[1]

        # PnP
        q, t = pnp(keypts_true_3D[visibleIdx], keypts[visibleIdx].numpy(),
                        camera['cameraMatrix'], camera['distCoeffs'])

        # Correction based on known SPEED+ pose distribution
        t = np.clip(t, 0, 10)

        return keypts, q, t

# -----------------------------------------------------------------------
# Bounding box post-processing
# -----------------------------------------------------------------------
@torch.jit.script
def raw_output_to_bbox(bbox_raw, image_size, anchors):
    """[summary]

    Args:
        bbox_raw (torch.Tensor): [B x M x 4] predicted RAW bounding boxes
        anchors (torch.Tensor): [B x N x 4] anchor boxes

    Returns:
        [type]: [description]
    """
    anchor_width    = anchors[:,2] - anchors[:,0]
    anchor_height   = anchors[:,3] - anchors[:,1]
    anchor_center_x = anchors[:,0] + anchor_width/2
    anchor_center_y = anchors[:,1] + anchor_height/2

    # Current box prediction (xc, yc, w, h)
    x = anchor_center_x + anchor_width * bbox_raw[:,:,0]
    y = anchor_center_y + anchor_height * bbox_raw[:,:,1]
    w = anchor_width * torch.exp(bbox_raw[:,:,2])
    h = anchor_height * torch.exp(bbox_raw[:,:,3])

    # Into [xmin, ymin, xmax, ymax]
    bbox = torch.zeros_like(bbox_raw)
    bbox[:,:,0] = torch.clamp(x - w/2, 0, image_size[0]-1)
    bbox[:,:,1] = torch.clamp(y - h/2, 0, image_size[1]-1)
    bbox[:,:,2] = torch.clamp(x + w/2, 0, image_size[0]-1)
    bbox[:,:,3] = torch.clamp(y + h/2, 0, image_size[1]-1)

    return bbox

# -----------------------------------------------------------------------
# Rotation/Translation regression post-processing (EfficientPose)
# -----------------------------------------------------------------------
@torch.jit.script
def delta_xy_tz_to_translation(translation_raw,
                               image_size, input_size,
                               anchors, strides, cameraMatrix):
    dx, dy, tz = translation_raw[:,:,0], translation_raw[:,:,1], translation_raw[:,:,2]

    cx = dx * strides + (anchors[:,0] + anchors[:,2])/2
    cy = dy * strides + (anchors[:,1] + anchors[:,3])/2

    # (cx, cy) are in terms of input image size, so convert to original size
    cx = cx / input_size[0] * image_size[0]
    cy = cy / input_size[1] * image_size[1]

    tx = (cx - cameraMatrix[0,2]) * tz / cameraMatrix[0,0]
    ty = (cy - cameraMatrix[1,2]) * tz / cameraMatrix[1,1]

    return torch.stack((tx, ty, tz), dim=-1) # [B, N, 3]

@torch.jit.script
def rot_6d_to_matrix(r):
    # r : (.., 6)
    r1_raw = r[..., :3]
    r2_raw = r[..., 3:]

    # First column
    r1 = F.normalize(r1_raw, p=2.0, dim=-1) # [..., 3]

    # Second column
    dot = torch.sum(r1 * r2_raw, dim=-1, keepdim=True)
    r2 = r2_raw - dot * r1
    r2 = F.normalize(r2, p=2.0, dim=-1)

    # Third column
    r3 = torch.cross(r1, r2, dim=-1)

    # Into matrix
    dcm = torch.stack((r1, r2, r3), dim=-1)

    return dcm

# -----------------------------------------------------------------------
# Functions regarding 3D keypoints projection
# -----------------------------------------------------------------------
def quat2dcm(q):
    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """
    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3), dtype=np.float32)

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

def project_keypoints(q, r, K, dist, keypoints):
    """ Projecting 3D keypoints to 2D
        q: quaternion (np.array)
        r: position   (np.array)
        K: camera intrinsic (3,3) (np.array)
        dist: distortion coefficients (5,) (np.array)
        keypoints: N x 3 or 3 x N (np.array)
    """
    # Make sure keypoints are 3 x N
    if keypoints.shape[0] != 3:
        keypoints = np.transpose(keypoints)

    # Keypoints into 4 x N homogenous coordinates
    keypoints = np.vstack((keypoints, np.ones((1, keypoints.shape[1]))))

    # transformation to image frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    xyz      = np.dot(pose_mat, keypoints) # [3 x N]
    x0, y0   = xyz[0,:] / xyz[2,:], xyz[1,:] / xyz[2,:] # [1 x N] each

    # apply distortion
    r2 = x0*x0 + y0*y0
    cdist = 1 + dist[0]*r2 + dist[1]*r2*r2 + dist[4]*r2*r2*r2
    x  = x0*cdist + dist[2]*2*x0*y0 + dist[3]*(r2 + 2*x0*x0)
    y  = y0*cdist + dist[2]*(r2 + 2*y0*y0) + dist[3]*2*x0*y0

    # apply camera matrix
    points2D = np.vstack((K[0,0]*x + K[0,2], K[1,1]*y + K[1,2]))

    return points2D

def pnp(points_3D, points_2D, cameraMatrix, distCoeffs=None, rvec=None, tvec=None, useExtrinsicGuess=False):
    if distCoeffs is None:
        distCoeffs = np.zeros((5, 1), dtype=np.float32)

    assert points_3D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    points_3D = np.ascontiguousarray(points_3D).reshape((-1,1,3))
    points_2D = np.ascontiguousarray(points_2D).reshape((-1,1,2))

    _, R_exp, t = cv2.solvePnP(points_3D,
                               points_2D,
                               cameraMatrix,
                               distCoeffs, rvec, tvec, useExtrinsicGuess,
                               flags=cv2.SOLVEPNP_EPNP)

    R_pr, _ = cv2.Rodrigues(R_exp)

    # To (q, t)
    q = R.from_matrix(R_pr).as_quat() # [qvec, qcos]
    t = np.squeeze(t)

    return q[[3,0,1,2]], t
