'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import albumentations as A

""" The function below is adopted from the Albumentation's official implementation
    of the class RandomSunFlare. It is modified to accept the bounding box as an input
    to force the sun flares within the prescribed bounding box.
"""

class RandomSunFlare(A.DualTransform):
    """Simulates Sun Flare for the image
    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Modified from official Albumentations implementation to restrict the flare
    location within the target's bounding box

    Args:
        flare_roi (float, float, float, float): region of the image where flare will
            appear (x_min, y_min, x_max, y_max). All values should be in range [0, 1].
        angle_lower (float): should be in range [0, `angle_upper`].
        angle_upper (float): should be in range [`angle_lower`, 1].
        num_flare_circles_lower (int): lower limit for the number of flare circles.
            Should be in range [0, `num_flare_circles_upper`].
        num_flare_circles_upper (int): upper limit for the number of flare circles.
            Should be in range [`num_flare_circles_lower`, inf].
        src_radius (int):
        src_color ((int, int, int)): color of the flare
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        angle_lower=0,
        angle_upper=1,
        num_flare_circles_lower=6,
        num_flare_circles_upper=10,
        always_apply=False,
        p=0.5,
    ):
        super(RandomSunFlare, self).__init__(always_apply, p)

        if not 0 <= angle_lower < angle_upper <= 1:
            raise ValueError(
                "Invalid combination of angle_lower nad angle_upper. Got: {}".format((angle_lower, angle_upper))
            )
        if not 0 <= num_flare_circles_lower < num_flare_circles_upper:
            raise ValueError(
                "Invalid combination of num_flare_circles_lower nad num_flare_circles_upper. Got: {}".format(
                    (num_flare_circles_lower, num_flare_circles_upper)
                )
            )

        self.angle_lower = angle_lower
        self.angle_upper = angle_upper
        self.num_flare_circles_lower = num_flare_circles_lower
        self.num_flare_circles_upper = num_flare_circles_upper

    def apply(self, image, src_radius=400, flare_center_x=0.5, flare_center_y=0.5, circles=(), **params):
        clr = random.choice([0, 255])
        src_color = (clr, clr, clr)
        return A.functional.add_sun_flare(
            image,
            flare_center_x,
            flare_center_y,
            src_radius,
            src_color,
            circles,
        )

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        bbox = params["bboxes"][0] # Single object [xmin, ymin, xmax, ymax] (pix)
        height, width = img.shape[:2]

        # Flare angle
        angle = 2 * math.pi * random.uniform(self.angle_lower, self.angle_upper)

        # Flare location
        flare_center_x = random.uniform(bbox[0], bbox[2])
        flare_center_y = random.uniform(bbox[1], bbox[3])

        flare_center_x = int(width * flare_center_x)
        flare_center_y = int(height * flare_center_y)

        # Number of circles
        num_circles = random.randint(self.num_flare_circles_lower, self.num_flare_circles_upper)

        # Random color
        src_color = random.randint(0, 256)

        # Random radius smaller than target
        src_radius = int(random.uniform(0.2, 0.4) * max(width, height))

        # Create circles
        circles = []

        x = []
        y = []

        for rand_x in range(0, width, 10):
            rand_y = math.tan(angle) * (rand_x - flare_center_x) + flare_center_y
            x.append(rand_x)
            y.append(2 * flare_center_y - rand_y)

        for _i in range(num_circles):
            alpha = random.uniform(0.05, 0.2)
            r = random.randint(0, len(x) - 1)
            rad = random.randint(1, max(height // 100 - 2, 2))

            r_color = random.randint(max(src_color - 50, 0), src_color)
            g_color = random.randint(max(src_color - 50, 0), src_color)
            b_color = random.randint(max(src_color - 50, 0), src_color)

            circles += [
                (
                    alpha,
                    (int(x[r]), int(y[r])),
                    pow(rad, 3),
                    (r_color, g_color, b_color),
                )
            ]

        return {
            "circles": circles,
            "src_radius": src_radius,
            "flare_center_x": flare_center_x,
            "flare_center_y": flare_center_y,
        }

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def get_transform_init_args(self):
        return {
            "angle_lower": self.angle_lower,
            "angle_upper": self.angle_upper,
            "num_flare_circles_lower": self.num_flare_circles_lower,
            "num_flare_circles_upper": self.num_flare_circles_upper,
        }