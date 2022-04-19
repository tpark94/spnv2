'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import argparse

import torch

import _init_paths

from config import cfg, update_config
from nets   import build_spnv2
from dataset import get_dataloader
from engine.inference  import do_valid
from utils.utils import set_seeds_cudnn, create_logger_directories, \
                        load_camera_intrinsics, load_tango_3d_keypoints

def parse_args():
    parser = argparse.ArgumentParser(description='Test on SPNv2')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args

def main(cfg):
    args = parse_args()
    update_config(cfg, args)

    # Load model to test
    test_model = osp.join(cfg.OUTPUT_DIR, cfg.TEST.MODEL_FILE)
    if not osp.exists(test_model) or osp.isdir(test_model):
        test_model = osp.join(cfg.OUTPUT_DIR, cfg.MODEL.BACKBONE.NAME,
                                    cfg.EXP_NAME, 'model_best.pth.tar')
    cfg.defrost()
    cfg.TEST.MODEL_FILE = test_model
    cfg.freeze()

    # Logger & directories
    logger, output_dir, _ = create_logger_directories(cfg, 'test')

    # Set all seeds & cudNN
    set_seeds_cudnn(cfg, seed=cfg.SEED)

    # GPU?
    device = torch.device('cuda:0') if cfg.CUDA and torch.cuda.is_available() else torch.device('cpu')

    # Complete network
    model = build_spnv2(cfg)

    # Load checkpoint
    if cfg.TEST.MODEL_FILE:
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location='cpu'), strict=True)
        logger.info('   - Model loaded from {}'.format(cfg.TEST.MODEL_FILE))
    model = model.to(device)

    # Dataloaders
    test_loader = get_dataloader(cfg, split='test', load_labels=True)

    # For validation
    camera = load_camera_intrinsics(cfg.DATASET.CAMERA)
    keypts_true_3D = load_tango_3d_keypoints(cfg.DATASET.KEYPOINTS)

    # ---------------------------------------
    # Main Test
    # ---------------------------------------
    score = do_valid(0,
                     cfg,
                     model,
                     test_loader,
                     camera,
                     keypts_true_3D,
                     valid_fraction=None,
                     log_dir=output_dir,
                     device=device)

if __name__=='__main__':
    main(cfg)
