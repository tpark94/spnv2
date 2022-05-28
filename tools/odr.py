'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import torch

import _init_paths

from config  import cfg, update_config
from nets    import build_spnv2
from dataset import get_dataloader
from solver  import get_optimizer, get_scaler
from engine.adapter    import do_adapt
from engine.inference  import do_valid
from utils.checkpoints import save_checkpoint
from utils.utils import set_seeds_cudnn, setup_logger, create_logger_directories, \
                        load_camera_intrinsics, load_tango_3d_keypoints, \
                        write_model_info, num_trainable_parameters

def parse_args():
    parser = argparse.ArgumentParser(description='TTDR')

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

    # Set global seed, if not provided
    seed = int(time.time()) if cfg.SEED is None else cfg.SEED

    cfg.defrost()
    cfg.SEED = seed
    cfg.freeze()

    _, output_dir, log_dir = \
        create_logger_directories(cfg, phase='train', write_cfg_to_file=True)

    # No distributed training for TTDR
    main_worker(
        output_dir,
        log_dir
    )

def main_worker(output_dir, log_dir):

    # Set all seeds & cudNN
    set_seeds_cudnn(cfg, seed=cfg.SEED)

    # setup logger
    logger = setup_logger(log_dir, 0, 'train', to_console=False)

    # build network
    model = build_spnv2(cfg)

    # GPU device
    device = torch.device('cuda')
    model  = model.to(device)

    # write model summary to file
    write_model_info(model, log_dir)

    # disable entire model grads
    model.eval()
    model.requires_grad_(False)

    # -------------------------------------------------------------
    # For gradient accumulation with BatchNorm layers
    # -------------------------------------------------------------
    # Here, we manually update BatchNorm's running stats
    def get_bn_features_from_name(name):
        def bn_feature_hook(module, input, output):
            bn_features[name] = input[0].detach()
        return bn_feature_hook

    bn_features = {}
    handles = []
    for n, m in model.backbone.named_modules():
        if not cfg.MODEL.USE_GROUPNORM_BACKBONE and isinstance(m, torch.nn.BatchNorm2d):
            # (1) Set BatchNorm layers to eval mode, so that running states are
            #     not updated by every forward() calls
            m.requires_grad_(True)
            m.eval()

            # (2) Keep the record of input features to BatchNorm layers
            h = m.register_forward_hook(get_bn_features_from_name(n))
            handles.append(h)

        elif cfg.MODEL.USE_GROUPNORM_BACKBONE and isinstance(m, torch.nn.GroupNorm):
            # GroupNorm layers -- simply allow requires_grad
            m.requires_grad_(True)

    logger.info(f'Total number of parameters with requires_grad=True')
    logger.info(f'   - {num_trainable_parameters(model):,d}')

    # Dataloaders
    train_loader = get_dataloader(cfg,
                                  split='train',
                                  distributed=False,
                                  load_labels=False) # No labels during TTDR
    val_loader   = get_dataloader(cfg,
                                  split='val',
                                  distributed=False,
                                  load_labels=True)

    # Optimizer & scaler for mixed-precision training
    optimizer = get_optimizer(cfg, model)
    scaler    = get_scaler(cfg)

    # For validation
    camera = load_camera_intrinsics(cfg.DATASET.CAMERA)
    keypts_true_3D = load_tango_3d_keypoints(cfg.DATASET.KEYPOINTS)

    # ---------------------------------------
    # Main ODR
    # ---------------------------------------
    # Single epoch training
    do_adapt(0,
             cfg,
             model,
             bn_features,
             train_loader,
             optimizer,
             log_dir=log_dir,
             device=device,
             scaler=scaler)

    # Remove hooks
    for h in handles:
        h.remove()

    # Validate on the fraction of dataset
    # score = 0
    score = do_valid(0,
                    cfg,
                    model,
                    val_loader,
                    camera,
                    keypts_true_3D,
                    valid_fraction=1.0,
                    log_dir=output_dir,
                    device=device)

    # Save
    save_checkpoint({
        'epoch': 1,
        'backbone': cfg.MODEL.BACKBONE.NAME,
        'heads': cfg.MODEL.HEAD.NAMES,
        'state_dict': model.state_dict(),
        'best_state_dict': model.state_dict(),
        'best_score': score,
        'optimizer': optimizer.state_dict(),
        'scaler': None
    }, False, True, output_dir)

    logger.info('\n\n')

if __name__=='__main__':
    main(cfg)
