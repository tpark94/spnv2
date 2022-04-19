'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .randomsunflare import RandomSunFlare
from .coarsedropout  import CoarseDropout

def build_transforms(cfg, is_train=True, load_labels=True):
    transforms = []

    # Resize the image
    # transforms = [A.Resize(cfg.DATASET.INPUT_SIZE[1],
    #                        cfg.DATASET.INPUT_SIZE[0])]

    # Add augmentation if training, skip if not
    if is_train:
        if cfg.AUGMENT.ADJUST_BRIGHTNESS_CONTRAST:
            transforms += [A.RandomBrightnessContrast(brightness_limit=0.2,
                                                      contrast_limit=0.2,
                                                      p=cfg.AUGMENT.P)]
        if cfg.AUGMENT.APPLY_RANDOM_ERASE:
            transforms += [CoarseDropout(max_holes=5,
                                         min_holes=1,
                                         max_ratio=0.5,
                                         min_ratio=0.2,
                                         p=cfg.AUGMENT.P)]
        if cfg.AUGMENT.APPLY_SOLAR_FLARE:
            transforms += [RandomSunFlare(num_flare_circles_lower=1,
                                          num_flare_circles_upper=10,
                                          p=cfg.AUGMENT.P)]
        if cfg.AUGMENT.APPLY_BLUR:
            transforms += [A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3,9)),
                    A.MedianBlur(blur_limit=(3,7)),
                    A.GlassBlur(sigma=0.5,
                                max_delta=2)
                ], p=cfg.AUGMENT.P
            )]
        if cfg.AUGMENT.APPLY_NOISE:
            transforms += [A.OneOf(
                [
                    A.GaussNoise(var_limit=40**2), # variance [pix]
                    A.ISONoise(color_shift=(0.1, 0.5),
                               intensity=(0.5, 1.0))
                ], p=cfg.AUGMENT.P
            )]

    # Normalize by ImageNet stats, then turn into tensor
    transforms += [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                   ToTensorV2()]

    # Compose and return
    if load_labels:
        transforms = A.Compose(
            transforms,
            A.BboxParams(format='albumentations',       # [xmin, ymin, xmax, ymax] (normalized)
                         label_fields=['class_labels']) # Placeholder
        )
    else:
        transforms = A.Compose(
            transforms
        )

    return transforms

if __name__=='__main__':

    from PIL import Image
    data = Image.open('/Users/taehapark/SLAB/Dataset/speedplus/synthetic/images/img000100.jpg').convert('RGB')

    transforms = build_transforms((480, 320))

    data = transforms(data)
    data = T.functional.to_pil_image(data)

    data.show()



