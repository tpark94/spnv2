'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm

import _init_paths

from config import cfg, update_config
from utils.utils import load_camera_intrinsics, load_tango_3d_keypoints
from utils.postprocess import project_keypoints

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing SPEED+.')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                help="Modify config options using the command-line",
                default=None,
                nargs=argparse.REMAINDER)

    # additional argument unrelated to cfg
    parser.add_argument('--jsonfile',
                        required=True,
                        type=str)
    parser.add_argument('--no_masks',
                        dest='load_masks',
                        action='store_false')
    parser.add_argument('--no_labels',
                        dest='load_labels',
                        action='store_false')

    args = parser.parse_args()

    return args

def main():

    args = parse_args()
    update_config(cfg, args)

    datadir = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.DATANAME)

    # Read labels from JSON file
    jsonfile = os.path.join(datadir, args.jsonfile)
    print(f'Reading JSON file from {jsonfile}...')
    with open(jsonfile, 'r') as f:
        labels = json.load(f) # list

    # Read camera
    camera = load_camera_intrinsics(cfg.DATASET.CAMERA)

    # Read Tango 3D keypoints
    keypts3d = load_tango_3d_keypoints(cfg.DATASET.KEYPOINTS) # (11, 3) [m]

    # Where to save CSV?
    if cfg.DATASET.DATANAME == 'speedplus':
        domain, split = args.jsonfile.split('/')
    elif cfg.DATASET.DATANAME == 'prisma25':
        domain, split = '', args.jsonfile
    else:
        raise NotImplementedError('Only accepting speedplus and prisma25')
    outdir = os.path.join(datadir, domain, 'labels')
    if not os.path.exists(outdir): os.makedirs(outdir)
    csvfile = os.path.join(outdir, split.replace('json', 'csv'))
    print(f'Label CSV file will be saved to {csvfile}')

    # Where to save resized image?
    imagedir = os.path.join(datadir, domain,
            f'images_{cfg.DATASET.INPUT_SIZE[0]}x{cfg.DATASET.INPUT_SIZE[1]}_RGB')
    if not os.path.exists(imagedir): os.makedirs(imagedir)
    print(f'Resized images will be saved to {imagedir}')

    if args.load_masks:
        maskdir = os.path.join(datadir, domain,
            f'masks_{int(cfg.DATASET.INPUT_SIZE[0]/cfg.DATASET.OUTPUT_SIZE[0])}x{int(cfg.DATASET.INPUT_SIZE[1]/cfg.DATASET.OUTPUT_SIZE[0])}')
        if not os.path.exists(maskdir): os.makedirs(maskdir)
        print(f'Resized masks will be saved to {maskdir}')

    # Open
    csv = open(csvfile, 'w')

    for idx in tqdm(range(len(labels))):

        # ---------- Read image & resize & save
        filename = labels[idx]['filename']
        image    = cv2.imread(os.path.join(datadir, domain, 'images', filename), cv2.IMREAD_COLOR)
        image    = cv2.resize(image, cfg.DATASET.INPUT_SIZE)
        cv2.imwrite(os.path.join(imagedir, filename), image)

        # ---------- Read mask & resize & save
        if args.load_masks:
            mask = cv2.imread(os.path.join(datadir, domain, 'masks', filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, [int(s / cfg.DATASET.OUTPUT_SIZE[0]) for s in cfg.DATASET.INPUT_SIZE])
            cv2.imwrite(os.path.join(maskdir, filename), mask)

        # ---------- Read labels
        if args.load_labels:
            q_vbs2tango = np.array(labels[idx]['q_vbs2tango_true'], dtype=np.float32)
            r_Vo2To_vbs = np.array(labels[idx]['r_Vo2To_vbs_true'], dtype=np.float32)

        # ---------- Project keypoints & origin
        if args.load_labels:
            # Attach origin
            keypts3d_origin = np.concatenate((np.zeros((3,1), dtype=np.float32),
                                            np.transpose(keypts3d)), axis=1) # [3, 12]

            keypts2d = project_keypoints(q_vbs2tango,
                                        r_Vo2To_vbs,
                                        camera['cameraMatrix'],
                                        camera['distCoeffs'],
                                        keypts3d_origin) # (2, 12)

            keypts2d[0] = keypts2d[0] / camera['Nu']
            keypts2d[1] = keypts2d[1] / camera['Nv']

            # Into vector (x0, y0, kx1, ky1, ..., kx11, ky11)
            keypts2d_vec = np.reshape(np.transpose(keypts2d), (24,))

        # ---------- Bounding box labels
        # If masks are available, get them from masks
        # If not, use keypoints instead
        if args.load_labels:
            if args.load_masks:
                seg  = np.where(mask > 0)
                xmin = np.min(seg[1]) / camera['Nu']
                ymin = np.min(seg[0]) / camera['Nv']
                xmax = np.max(seg[1]) / camera['Nu']
                ymax = np.max(seg[0]) / camera['Nv']
            else:
                xmin = np.min(keypts2d[0])
                ymin = np.min(keypts2d[1])
                xmax = np.max(keypts2d[0])
                ymax = np.max(keypts2d[1])

        # CSV row
        row = [filename]

        if args.load_labels:
            row = row + [xmin, ymin, xmax, ymax] \
                      + q_vbs2tango.tolist() \
                      + r_Vo2To_vbs.tolist() \
                      + keypts2d_vec.tolist()

        row = ', '.join([str(e) for e in row])

        # Write
        csv.write(row + '\n')

    csv.close()

    print('done\n\n')

if __name__=='__main__':
    main()