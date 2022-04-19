'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Tuple, Iterable, Union

import random
import numpy as np
import albumentations as A

""" The function below is adopted from the Albumentation's official implementation
    of the class CoarseDropout. It is modified to accept the bounding box as an input
    to force the dropout regions within the prescribed bounding box.
"""

class CoarseDropout(A.DualTransform):
    """CoarseDropout of the rectangular regions in the image.

        Modified from official Albumentations implementation to restrict the
        dropout regions within the target bounding box

    Args:
        max_holes (int): Maximum number of regions to zero out.
        max_height (int, float): Maximum height of the hole.
        If float, it is calculated as a fraction of the image height.
        max_width (int, float): Maximum width of the hole.
        If float, it is calculated as a fraction of the image width.
        min_holes (int): Minimum number of regions to zero out. If `None`,
            `min_holes` is be set to `max_holes`. Default: `None`.
        min_height (int, float): Minimum height of the hole. Default: None. If `None`,
            `min_height` is set to `max_height`. Default: `None`.
            If float, it is calculated as a fraction of the image height.
        min_width (int, float): Minimum width of the hole. If `None`, `min_height` is
            set to `max_width`. Default: `None`.
            If float, it is calculated as a fraction of the image width.
        fill_value (int, float, list of int, list of float): value for dropped pixels.
        mask_fill_value (int, float, list of int, list of float): fill value for dropped pixels
            in mask. If `None` - mask is not affected. Default: `None`.
    Targets:
        image, mask, keypoints
    Image types:
        uint8, float32
    Reference:
    |  https://arxiv.org/abs/1708.04552
    |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
    """

    def __init__(
        self,
        max_holes: int = 8,
        min_holes: Optional[int] = None,
        max_ratio: float = 0.33,
        min_ratio: float = 0.1,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(CoarseDropout, self).__init__(always_apply, p)
        self.max_holes = max_holes
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        if not 0 < self.min_holes <= self.max_holes:
            raise ValueError("Invalid combination of min_holes and max_holes. Got: {}".format([min_holes, max_holes]))

    def apply(
        self,
        img: np.ndarray,
        fill_value: Union[int, float] = 0,
        holes: Iterable[Tuple[int, int, int, int]] = (),
        **params
    ) -> np.ndarray:
        fill_value = random.randint(0, 256)
        return A.functional.cutout(img, holes, fill_value)

    def apply_to_mask(
        self,
        img: np.ndarray,
        mask_fill_value: Union[int, float] = 0,
        holes: Iterable[Tuple[int, int, int, int]] = (),
        **params
    ) -> np.ndarray:
        return A.functional.cutout(img, holes, 0)

    def get_params_dependent_on_targets(self, params):
        img  = params["image"]
        height, width = img.shape[:2]
        bbox = params["bboxes"][0] # Single object [xmin, ymin, xmax, ymax] (pix)

        xmin = int(width * bbox[0])
        ymin = int(height * bbox[1])
        xmax = int(width * bbox[2])
        ymax = int(height * bbox[3])

        # Min/Max Height/Width
        min_height = int(self.min_ratio * (ymax - ymin))
        max_height = int(self.max_ratio * (ymax - ymin))
        min_width  = int(self.min_ratio * (xmax - xmin))
        max_width  = int(self.max_ratio * (xmax - xmin))

        holes = []
        for _n in range(random.randint(self.min_holes, self.max_holes)):

            hole_height = random.randint(min_height, max_height)
            hole_width = random.randint(min_width, max_width)

            y1 = random.randint(ymin, ymax - hole_height)
            x1 = random.randint(xmin, xmax - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def get_transform_init_args_names(self):
        return (
            "max_holes",
            "min_holes"
        )