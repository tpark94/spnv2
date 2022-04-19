'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import patches as patches

import cv2
import numpy as np
import torch

def _tensor_to_numpy_image(image):
    std   = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1)
    mean  = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1)
    image = image * std + mean
    image = image.mul(255).clamp(0,255).permute(1,2,0).byte().cpu().numpy()

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def imshow(image, savefn=None):
    # image: torch.tensor (3 x H x W)

    # image to numpy.array
    image = _tensor_to_numpy_image(image)

    # plot
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    if savefn is not None:
        plt.savefig(savefn, bbox_inches='tight', pad_inches=0)

def imshowbbox(image, bbox, normalized=True):
    image = _tensor_to_numpy_image(image)
    xmin, ymin, xmax, ymax = bbox.numpy()
    if normalized:
        xmin *= image.shape[1]
        ymin *= image.shape[0]
        xmax *= image.shape[1]
        ymax *= image.shape[0]

    plt.imshow(image)
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin],
                color='g', linewidth=1.5)
    plt.show()

def showheatmap(heatmaps, keyptId=-1):
    # heatmaps: torch.tensor (N x H x W)

    # heatmap to numpy.array (N x H x W)
    heatmaps = heatmaps.mul(255).clamp(0,255).byte().cpu().numpy()

    if keyptId > -1:
        # Single keypoint
        heatmap = heatmaps[keyptId, :, :]
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

    plt.imshow(heatmap_color)
    plt.axis('off')
    plt.show()

def imshowheatmap(image, heatmaps, keyptId=-1):
    # image:    torch.tensor (3 x H x W)
    # heatmaps: torch.tensor (N x H x W)

    # image to numpy.array (H x W x 3)
    image = image.mul(255).clamp(0,255).permute(1,2,0).byte().cpu().numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Scale heatmap to [0, 1]
    # hmax = torch.max(heatmaps.view(heatmaps.shape[0], -1), dim=1)[0]
    # heatmaps /= hmax.view(-1,1,1)

    # heatmap to numpy.array (N x H x W)
    heatmaps = heatmaps.mul(255).clamp(0,255).byte().cpu().numpy()

    # Resize image
    num_keypts, height, width = heatmaps.shape
    image = cv2.resize(image, (int(width), int(height)))
    image = np.expand_dims(image, axis=-1)

    if keyptId > -1:
        # Single keypoint
        heatmap = heatmaps[keyptId, :, :]
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        image_fused   = (heatmap_color * 0.7 + image * 0.3) / 255 # [0,1] for float

    plt.imshow(image_fused)
    plt.axis('off')
    plt.show()
