'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

logger = logging.getLogger(__name__)

class HeatmapGenerator():
    """ Adopted from the repository below:
    https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/blob/master/lib/dataset/target_generators/target_generators.py
    """
    def __init__(self, output_res, num_keypts, sigma=-1):
        self.output_res = output_res # (W, H)
        self.num_keypts = num_keypts

        if sigma < 0:
            # sigma = self.output_res/64
            logger.warning('Warning! Set sigma > 0 for heatmaps')

        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1

        # Gaussian kernel
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypts):
        # keypts: [num_keypts, 2]
        assert keypts.ndim == 2 and keypts.shape[0] == self.num_keypts, \
                     'Keypoints must be [N x 2]'

        hms = np.zeros((self.num_keypts, self.output_res[1], self.output_res[0]),
                       dtype=np.float32)
        sigma = self.sigma

        for idx, pt in enumerate(keypts):
            x = int(pt[0] * self.output_res[0]) # scale keypoint location
            y = int(pt[1] * self.output_res[1])
            if x < 0 or y < 0 or \
                x >= self.output_res[0] or y >= self.output_res[1]:
                continue

            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

            c, d = max(0, -ul[0]), min(br[0], self.output_res[0]) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], self.output_res[1]) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], self.output_res[0])
            aa, bb = max(0, ul[1]), min(br[1], self.output_res[1])

            hms[idx, aa:bb, cc:dd] = np.maximum(
                hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])

        return hms

if __name__=='__main__':
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    h = HeatmapGenerator((25, 25), 1, sigma=2, normalize=True)
    keypts = np.reshape(np.array([0.5, 0.5, 1], dtype=np.float32), (1, 1, 3))
    heatmap = np.squeeze(h(keypts))

    print(heatmap)