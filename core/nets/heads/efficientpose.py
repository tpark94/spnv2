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

from ..layers import DepthwiseSeparableConv
from ..loss   import FocalLoss, CIoULoss, TransformationLoss, SPEEDLoss
from .anchors import create_anchors, compute_anchor_state
from utils.utils import load_camera_intrinsics, load_cad_model
from utils.postprocess import raw_output_to_bbox, delta_xy_tz_to_translation

logger = logging.getLogger(__name__)

# Basic keyword arguments for DepthwiseSeparableConv
# - No normalization, no activation
conv_kwargs = {
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'norm_layer': None,
    'act_layer': None
}

# BatchNorm2d keyward arguments
bn_norm_kwargs = {'momentum': 0.003, 'eps': 1e-5, 'affine': True}
gn_norm_kwargs = {'eps': 1e-5, 'affine': True}

class ClassNet(nn.Module):
    def __init__(self, in_channels, depth, num_anchors, use_group_norm, num_groups):
        super(ClassNet, self).__init__()
        self.depth = depth

        # Conv layers are shared across all feature levels
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv(
                in_channels,
                in_channels,
                bias=False,
                **conv_kwargs
            ) for _ in range(depth)
        ])

        # Normalization layers must be specific to each path
        self.bns = nn.ModuleList([
                nn.ModuleList([
                    nn.GroupNorm(num_groups, in_channels, **gn_norm_kwargs) if use_group_norm else \
                        nn.BatchNorm2d(in_channels, **bn_norm_kwargs) for _ in range(depth)
            ]) for _ in range(3, 8)
        ])

        # SiLU (Swish) activation
        self.act = nn.SiLU(inplace=True)

        # Final head layer
        self.head = DepthwiseSeparableConv(
            in_channels,
            num_anchors,
            bias=True,
            **conv_kwargs
        )

    def forward(self, x):
        # x: list of feature maps from P2 ~ P7 levels of BiFPN
        # Follow original EfficientPose/EfficientDet and take P3 ~ P7 only
        outputs = []
        for level, feature in enumerate(x[1:]):
            for i in range(self.depth):
                feature = self.convs[i](feature)
                feature = self.bns[level][i](feature)
                feature = self.act(feature)
            feature = self.head(feature)  # [B, A, Hi, Wi]

            # Reshape: [B, A, Hi, Wi] -> [B, AHiWi]
            feature = feature.permute(0, 2, 3, 1)
            feature = feature.contiguous().view(feature.shape[0], -1)

            outputs.append(feature)

        # Concatenate into [B, sum(AHiWi)]
        outputs = torch.cat(outputs, dim=1)
        return outputs

class BoxNet(nn.Module):
    def __init__(self, in_channels, depth, num_anchors, use_group_norm, num_groups):
        super(BoxNet, self).__init__()
        self.depth = depth

        # Conv layers are shared across all feature levels
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv(
                in_channels,
                in_channels,
                bias=False,
                **conv_kwargs
            ) for _ in range(depth)
        ])

        # Normalization layers must be specific to each path
        self.bns = nn.ModuleList([
                nn.ModuleList([
                    nn.GroupNorm(num_groups, in_channels, **gn_norm_kwargs) if use_group_norm else \
                        nn.BatchNorm2d(in_channels, **bn_norm_kwargs) for _ in range(depth)
            ]) for _ in range(3, 8)
        ])

        # SiLU (Swish) activation
        self.act = nn.SiLU(inplace=True)

        self.head = DepthwiseSeparableConv(
            in_channels,
            num_anchors * 4,
            bias=True,
            **conv_kwargs
        )

    def forward(self, x):
        # x: list of feature maps from P2 ~ P7 levels of BiFPN
        # Follow original EfficientPose/EfficientDet and take P3 ~ P7 only
        outputs = []
        for level, feature in enumerate(x[1:]):
            for i in range(self.depth):
                feature = self.convs[i](feature)
                feature = self.bns[level][i](feature)
                feature = self.act(feature)
            feature = self.head(feature)  # [B, A, Hi, Wi]

            # Reshape: [B, A, Hi, Wi] -> [B, AHiWi]
            feature = feature.permute(0, 2, 3, 1)
            feature = feature.contiguous().view(feature.shape[0], -1, 4)

            outputs.append(feature)

        # Concatenate into [B, sum(AHiWi), 4]
        outputs = torch.cat(outputs, dim=1)
        return outputs

class IterativeRotationSubnet(nn.Module):
    def __init__(self, in_channels, out_channels, depth, num_iterations, num_anchors, use_group_norm, num_groups):
        super(IterativeRotationSubnet, self).__init__()
        self.depth = depth
        self.num_iterations = num_iterations

        # Conv layers are shared across all feature levels
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv(
                in_channels if _ == 0 else out_channels,
                out_channels,
                bias=False,
                **conv_kwargs
            ) for _ in range(depth)
        ])

        # Normalization layers must be specific to each path and module
        self.bns = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.GroupNorm(num_groups, out_channels, **gn_norm_kwargs) if use_group_norm else \
                        nn.BatchNorm2d(out_channels, **bn_norm_kwargs) for _ in range(depth)
                ]) for _ in range(3, 8)
            ]) for _ in range(num_iterations)
        ])


        # SiLU (Swish) activation
        self.act = nn.SiLU(inplace=True)

        self.head = DepthwiseSeparableConv(
            out_channels,
            num_anchors * 6,
            bias=True,
            **conv_kwargs
        )

    def forward(self, feature, level, iter):
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[iter][level][i](feature)
            feature = self.act(feature)
        feature = self.head(feature)  # [B, 6A, Hi, Wi]

        return feature

class RotationNet(nn.Module):
    def __init__(self, in_channels, depth, num_iterations, num_anchors, use_group_norm, num_groups):
        super(RotationNet, self).__init__()
        self.depth = depth
        self.num_iterations = num_iterations

        # Conv layers are shared across all feature levels
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv(
                in_channels,
                in_channels,
                bias=False,
                **conv_kwargs
            ) for _ in range(depth)
        ])

        # Normalization layers must be specific to each path and module
        self.bns = nn.ModuleList([
            nn.ModuleList([
                nn.GroupNorm(num_groups, in_channels, **gn_norm_kwargs) if use_group_norm else \
                    nn.BatchNorm2d(in_channels, **bn_norm_kwargs) for _ in range(depth)
            ]) for _ in range(3, 8)
        ])

        # SiLU (Swish) activation
        self.act = nn.SiLU(inplace=True)

        # Final head
        self.head = DepthwiseSeparableConv(
            in_channels,
            num_anchors * 6,
            bias=True,
            **conv_kwargs
        )

        # Iterative refinement module
        self.iterative_submodel = IterativeRotationSubnet(in_channels + num_anchors * 6,
                                                          in_channels,
                                                          depth-1,
                                                          num_iterations,
                                                          num_anchors,
                                                          use_group_norm,
                                                          num_groups)

    def forward(self, x):
        # x: list of feature maps from P2 ~ P7 levels of BiFPN
        # Follow original EfficientPose/EfficientDet and take P3 ~ P7 only
        outputs = []
        for level, feature in enumerate(x[1:]):
            # Head for initial rotation
            for i in range(self.depth):
                feature = self.convs[i](feature)
                feature = self.bns[level][i](feature)
                feature = self.act(feature)
            rotation = self.head(feature)  # [B, 6A, Hi, Wi]

            # Rotation refinement
            for i in range(self.num_iterations):
                delta_rotation = self.iterative_submodel(
                    torch.cat([feature, rotation], dim=1),
                    level, i
                )
                rotation += delta_rotation

            # Reshape: [B, 6A, Hi, Wi] -> [B, AHiWi, 6]
            rotation = rotation.permute(0, 2, 3, 1)
            rotation = rotation.contiguous().view(rotation.shape[0], -1, 6)

            outputs.append(rotation)

        # Concatenate into [B, sum(AHiWi), 4]
        outputs = torch.cat(outputs, dim=1)
        return outputs

class IterativeTranslationSubnet(nn.Module):
    def __init__(self, in_channels, out_channels, depth, num_iterations, num_anchors, use_group_norm, num_groups):
        super(IterativeTranslationSubnet, self).__init__()
        self.depth = depth
        self.num_iterations = num_iterations

        # Conv layers are shared across all feature levels
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv(
                in_channels if _ == 0 else out_channels,
                out_channels,
                bias=False,
                **conv_kwargs
            ) for _ in range(depth)
        ])

        # Normalization layers must be specific to each path and module
        self.bns = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.GroupNorm(num_groups, out_channels, **gn_norm_kwargs) if use_group_norm else \
                        nn.BatchNorm2d(out_channels, **bn_norm_kwargs) for _ in range(depth)
                ]) for _ in range(3, 8)
            ]) for _ in range(num_iterations)
        ])

        # SiLU (Swish) activation
        self.act = nn.SiLU(inplace=True)

        # Two heads for (x, y) and z predictions
        self.head_xy = DepthwiseSeparableConv(
            out_channels,
            num_anchors * 2,
            bias=True,
            **conv_kwargs
        )
        self.head_z = DepthwiseSeparableConv(
            out_channels,
            num_anchors,
            bias=True,
            **conv_kwargs
        )

    def forward(self, feature, level, iter):
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[iter][level][i](feature)
            feature = self.act(feature)
        out_xy = self.head_xy(feature) # [B, 2A, Hi, Wi]
        out_z  = self.head_z(feature)  # [B,  A, Hi, Wi]

        return out_xy, out_z

class TranslationNet(nn.Module):
    def __init__(self, in_channels, depth, num_iterations, num_anchors, use_group_norm, num_groups):
        super(TranslationNet, self).__init__()
        self.depth = depth
        self.num_iterations = num_iterations

        # Conv layers are shared across all feature levels
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv(
                in_channels,
                in_channels,
                bias=False,
                **conv_kwargs
            ) for _ in range(depth)
        ])

        # Normalization layers must be specific to each path and module
        self.bns = nn.ModuleList([
            nn.ModuleList([
                nn.GroupNorm(num_groups, in_channels, **gn_norm_kwargs) if use_group_norm else \
                    nn.BatchNorm2d(in_channels, **bn_norm_kwargs) for _ in range(depth)
            ]) for _ in range(3, 8)
        ])

        # SiLU (Swish) activation
        self.act = nn.SiLU(inplace=True)

        # Final head - two for (x,y) and z predictions
        self.head_xy = DepthwiseSeparableConv(
            in_channels,
            num_anchors * 2,
            bias=True,
            **conv_kwargs
        )
        self.head_z = DepthwiseSeparableConv(
            in_channels,
            num_anchors,
            bias=True,
            **conv_kwargs
        )

        # Iterative refinement module
        self.iterative_submodel = IterativeTranslationSubnet(in_channels + num_anchors * 3,
                                                             in_channels,
                                                             depth-1,
                                                             num_iterations,
                                                             num_anchors,
                                                             use_group_norm,
                                                             num_groups)

    def forward(self, x):
        # x: list of feature maps from P2 ~ P7 levels of BiFPN
        # Follow original EfficientPose/EfficientDet and take P3 ~ P7 only
        outputs = []
        for level, feature in enumerate(x[1:]):
            # Head for initial rotation
            for i in range(self.depth):
                feature = self.convs[i](feature)
                feature = self.bns[level][i](feature)
                feature = self.act(feature)
            translation_xy = self.head_xy(feature) # [B, 2A, Hi, Wi]
            translation_z  = self.head_z(feature)  # [B,  A, Hi, Wi]

            # Rotation refinement
            for i in range(self.num_iterations):
                delta_xy, delta_z = self.iterative_submodel(
                    torch.cat([feature, translation_xy, translation_z], dim=1),
                    level, i
                )
                translation_xy += delta_xy
                translation_z  += delta_z

            # [B, 3A, Hi, Wi]
            translation = torch.cat([translation_xy, translation_z], dim=1)

            # Reshape: [B, 3A, Hi, Wi] -> [B, AHiWi, 3]
            translation = translation.permute(0, 2, 3, 1)
            translation = translation.contiguous().view(translation.shape[0], -1, 3)

            outputs.append(translation)

        # Concatenate into [B, sum(AHiWi), 3]
        outputs = torch.cat(outputs, dim=1)
        return outputs

class EfficientPoseHead(nn.Module):
    def __init__(self, cfg, in_channels, depth, num_iter, use_group_norm, group_norm_size):
        super(EfficientPoseHead, self).__init__()
        self.image_size     = torch.as_tensor(cfg.DATASET.IMAGE_SIZE)
        self.input_size     = torch.as_tensor(cfg.DATASET.INPUT_SIZE)
        self.bbox_threshold = cfg.TEST.BBOX_THRESHOLD
        self.pose_loss_type = cfg.MODEL.HEAD.POSE_REGRESSION_LOSS
        self.loss_factors   = cfg.MODEL.HEAD.EFFICIENTPOSE_LOSS_FACTOR

        # Anchor boxes (in image input size)
        anchors, strides = create_anchors(
                    image_size=self.input_size,
                    scales=eval(cfg.MODEL.HEAD.ANCHOR_SCALE),
                    ratios=cfg.MODEL.HEAD.ANCHOR_RATIO)
        self.register_buffer('anchors', anchors)
        self.register_buffer('strides', strides)

        # Camera
        self.camera = load_camera_intrinsics(cfg.DATASET.CAMERA)
        self.camera = torch.from_numpy(self.camera['cameraMatrix'])

        if self.pose_loss_type == 'transformation':
            # CAD model for transformation loss
            model_points = load_cad_model(cfg.DATASET.CADMODEL,
                                               num_points=500) # [Np, 3]
            model_points = torch.from_numpy(model_points).t() # [3, Np]
            self.register_buffer('model_points', model_points)

        ##### Classification
        # - Output: [B, sum(AHiWi)]
        # - Binary classification of object presence per anchor
        self.class_net = ClassNet(in_channels,
                                  depth,
                                  cfg.DATASET.NUM_ANCHORS,
                                  use_group_norm,
                                  group_norm_size)

        # Focal loss for object presence/absence classification
        self.focal_loss = FocalLoss()

        ##### Bounding box detection
        # - Output: list of [B, sum(AHiWi), 4]
        # - (xmin, ymin, xmax, ymax) detection per anchor
        self.box_net = BoxNet(in_channels,
                              depth,
                              cfg.DATASET.NUM_ANCHORS,
                              use_group_norm,
                              group_norm_size)

        # Complete IoU loss for bounding box detection
        self.ciou_loss = CIoULoss()

        ##### Rotation / Translation
        self.rotation_net = RotationNet(in_channels,
                                        depth,
                                        num_iter,
                                        cfg.DATASET.NUM_ANCHORS,
                                        use_group_norm,
                                        group_norm_size)

        self.translation_net = TranslationNet(in_channels,
                                              depth,
                                              num_iter,
                                              cfg.DATASET.NUM_ANCHORS,
                                              use_group_norm,
                                              group_norm_size)

        # Transformation loss or SPEED metric loss
        if self.pose_loss_type == 'transformation':
            self.pose_loss = TransformationLoss(self.model_points)
        elif self.pose_loss_type == 'speed':
            self.pose_loss = SPEEDLoss()

    def forward(self, features, **targets):
        # Apply each heads to feature maps
        classification = self.class_net(features) # [B, sum(AHiWi)]

        box_regression = self.box_net(features)   # [B, sum(AHiWi), 4]

        rotation_raw_6d = self.rotation_net(features) # [B, sum(AHiWi), 6]

        translation_raw = self.translation_net(features) # [B, sum(AHiWi), 3]

        # Post-processing and loss computation are done in float32
        # since using FP16 here causes worse loss convergence ...
        with torch.cuda.amp.autocast(enabled=False):

            # Cast the outputs to float from half
            classification  = classification.float()
            box_regression  = box_regression.float()
            rotation_raw_6d = rotation_raw_6d.float()
            translation_raw = translation_raw.float()

            # Post-processing
            bbox_prediction = raw_output_to_bbox(box_regression,
                                                self.input_size,
                                                self.anchors)

            translation = delta_xy_tz_to_translation(translation_raw,
                                                    self.image_size,
                                                    self.input_size,
                                                    self.anchors,
                                                    self.strides,
                                                    self.camera) # [B, sum(AHiWi), 3]

            if targets:
                # Prepare anchor state (positive/negative/ignore)
                anchor_states = compute_anchor_state(self.anchors,
                                                     targets['boundingbox'])

                # Losses
                cls_loss = self.focal_loss(classification,
                                           anchor_states)

                box_loss = self.ciou_loss(bbox_prediction,
                                          targets['boundingbox'],
                                          anchor_states)

                pose_loss = self.pose_loss(rotation_raw_6d,
                                           translation,
                                           targets['rotationmatrix'],
                                           targets['translation'],
                                           anchor_states)

                loss_total = self.loss_factors[0] * cls_loss \
                           + self.loss_factors[1] * box_loss \
                           + self.loss_factors[2] * pose_loss
                loss_items = {'cls':  cls_loss.detach(),
                              'box':  box_loss.detach(),
                              'pose': pose_loss.detach()}

                return loss_total, loss_items

            else:
                return classification, bbox_prediction, rotation_raw_6d, translation
