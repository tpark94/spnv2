'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from yacs.config import CfgNode as CN

from .efficientdet import EFFICIENTDET_EXTRAS

_C = CN()

# ------------------------------------------------------------------------------ #
# Basic settings
# ------------------------------------------------------------------------------ #
_C.ROOT       = '/media/shared/Jeff/SLAB/spnv2'     # Root directory of this project (.../spnv2)
_C.OUTPUT_DIR = 'outputs'                           # Name of the folder to save training outputs
_C.LOG_DIR    = 'log'                               # Name of the folder to save trainings logs
_C.EXP_NAME   = 'exp1'                              # Current experiment name

# Basic settings
_C.CUDA        = False                              # Use GPU?
_C.FP16        = False                              # Use mixed precision?
_C.AUTO_RESUME = True                               # Pick up from the last available training session?
_C.PIN_MEMORY  = True
_C.SEED        = None                               # Random seed. If None, seed is determined based on computer time
_C.VERBOSE     = False

# cudNN related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Distributed training
_C.DIST = CN()
_C.DIST.RANK = 0
_C.DIST.BACKEND = 'nccl'
_C.DIST.MULTIPROCESSING_DISTRIBUTED = False

# ------------------------------------------------------------------------------ #
# Dataset-related parameters
# ------------------------------------------------------------------------------ #
_C.DATASET = CN()

# - Basic directory & files
_C.DATASET.ROOT      = '/Users/taehapark/SLAB/Dataset'      # Root directory of all datasets
_C.DATASET.DATANAME  = 'speedplus'                          # Dataset name
_C.DATASET.CAMERA    = 'camera.json'                        # .json file containing camera parameters
_C.DATASET.KEYPOINTS = 'models/tangoPoints.mat'             # .mat file containing [3 x N] keypoints (m)
_C.DATASET.CADMODEL  = 'models/tango.ply'                   # .ply file containing target 3D model

# - Dataset characteristics
_C.DATASET.NUM_KEYPOINTS   = 11                             # Number of keypoints to detect
_C.DATASET.NUM_ANCHORS     = 9                              # Number of anchors for bounding box
_C.DATASET.MAX_NUM_OBJECTS = 1                              # Duh

# - I/O
_C.DATASET.IMAGE_SIZE  = [1920, 1200]                       # Original image size [W, H]
_C.DATASET.INPUT_SIZE  = [768, 512]                         # Input size to CNN [W, H]
_C.DATASET.OUTPUT_SIZE = [4, 8, 16, 32]                     # Reduce factor w.r.t. input_size at each feature level

# - Head-specific
_C.DATASET.SIGMA = 2                                        # Heatmap Gaussian kernel parameter

# ------------------------------------------------------------------------------ #
# Training-related parameters
# ------------------------------------------------------------------------------ #
_C.TRAIN = CN()

# - Learning rate & scheduler
_C.TRAIN.LR        = 0.001
_C.TRAIN.SCHEDULER = 'step'
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP   = [90, 110]

# - Optimizer
_C.TRAIN.OPTIMIZER = 'SGD'                                  # Optimizer name. Must be same as PyTorch optimizer name
_C.TRAIN.WD        = 0.0001                                 # Weight decay factor
_C.TRAIN.GAMMA1    = 0.9                                    # Momentum factor
_C.TRAIN.GAMMA2    = 0.0                                    # Secondary momentum factor for Adam optimizers

# - Epochs
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH   = 100
_C.TRAIN.VALID_FREQ  = 20
_C.TRAIN.VALID_FRACTION = None                              # Fraction of validation set on which inference is conducted

# - Batches
_C.TRAIN.IMAGES_PER_GPU = 32                                # Batch size PER GPU, NOT across all GPUs!
_C.TRAIN.SHUFFLE        = True
_C.TRAIN.WORKERS        = 4

# - CSV files
_C.TRAIN.TRAIN_CSV  = 'synthetic/splits/train.csv'
_C.TRAIN.VAL_CSV    = 'synthetic/splits/validation.csv'

# ------------------------------------------------------------------------------ #
# Data augmentation
# ------------------------------------------------------------------------------ #
_C.AUGMENT = CN()

# - Basic augmentations (besides resizing)
_C.AUGMENT.P = 0.5
_C.AUGMENT.ADJUST_BRIGHTNESS_CONTRAST = False
_C.AUGMENT.APPLY_BLUR                 = False
_C.AUGMENT.APPLY_SOLAR_FLARE          = False
_C.AUGMENT.APPLY_NOISE                = False
_C.AUGMENT.APPLY_RANDOM_ERASE         = False

# - Texture randomization
_C.AUGMENT.APPLY_TEXTURE_RANDOMIZATION = False
_C.AUGMENT.RANDOM_TEXTURE = CN()
_C.AUGMENT.RANDOM_TEXTURE.ALPHA = 0.5                       # Degree of texture randomization
_C.AUGMENT.RANDOM_TEXTURE.PROB  = 0.5                       # Probability of applying texture randomization

# ------------------------------------------------------------------------------ #
# Testing-related parameters
# ------------------------------------------------------------------------------ #
_C.TEST = CN()
_C.TEST.MODEL_FILE = ''                                     # Location of model weights (.pth.tar) w.r.t. OUTPUT_DIR
                                                            # If not provided, the default path is constructed according to
                                                            #       MODEL.BACKBONE.NAME/EXP_NAME/model_best.pth.tar
_C.TEST.IMAGES_PER_GPU = 1                                  # Batch size (only 1 is supported)
_C.TEST.HEAD = ['heatmap']                                  # Which prediction heads to evaluate?
_C.TEST.HEATMAP_THRESHOLD = 0.5                             # Keypoints detection threshold
_C.TEST.BBOX_THRESHOLD    = 0.5                             # Bounding box detection threshold
_C.TEST.TEST_CSV = 'lightbox/splits/lightbox.csv'           # CSV file to containing images to test on

# - Thresholds for SPEED metric (SPEED+ lightbox, sunlamp)
_C.TEST.SPEED_THRESHOLD_Q = 0.169    # [deg]
_C.TEST.SPEED_THRESHOLD_T = 2.173e-3 # [m/m]

# ------------------------------------------------------------------------------ #
# ODR-related parameters
# ------------------------------------------------------------------------------ #
_C.ODR = CN()
_C.ODR.MIN_ENTROPY       = False                            # Perform ODR?
_C.ODR.NUM_TRAIN_SAMPLES = 1024                             # Total number of samples to observe
_C.ODR.IMAGES_PER_BATCH  = 4                                # Batch size for updating BN layer statistics

# ------------------------------------------------------------------------------ #
# Model-related parameters
# ------------------------------------------------------------------------------ #
_C.MODEL = CN()
_C.MODEL.PRETRAIN_FILE     = None                           # Path to pre-trained model to load
_C.MODEL.EFFICIENTDET_PHI  = 0                              # EfficientNet scaling parameter
_C.MODEL.FIND_UNUSED_PARAM = False                          # See find_unused_parameters argument of
                                                            # torch.nn.parallel.DistributedDataParallel

# - Normalization layers
_C.MODEL.USE_GROUPNORM_BACKBONE = False                     # Use GroupNorm in the backbone?
_C.MODEL.USE_GROUPNORM_HEADS    = False                     # Use GroupNorm in the prediction heads?
_C.MODEL.GROUPNORM_SIZE = 16                                # Group size in the prediction heads
                                                            # - For backbone, size is fixed depending on PHI

# - Backbone
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'efficientdet_d3'                  # Name of the backbone: 'efficientdet_d#'
_C.MODEL.BACKBONE.PRETRAINED = None                         # Pre-trained model for the backbone only
_C.MODEL.BACKBONE.EXTRA = CN(new_allowed=True)

# - Heads
_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.NAMES = ['heatmap']                           # Prediction heads to BUILD
_C.MODEL.HEAD.LOSS_HEADS = ['heatmap']                      # Prediction heads to TRAIN and compute loss
_C.MODEL.HEAD.LOSS_FACTORS = [1.0]                          # Scaling factors for losses from each prediction heads
_C.MODEL.HEAD.LOSS_NUMS = [2]                               # Number of loss items per each prediction head
_C.MODEL.HEAD.EFFICIENTPOSE_LOSS_FACTOR = None              # Loss scaling factors for EfficientPose head: [cls, bbox, pose]
_C.MODEL.HEAD.ANCHOR_SCALE = None
_C.MODEL.HEAD.ANCHOR_RATIO = None
_C.MODEL.HEAD.POSE_REGRESSION_LOSS = 'transformation'       # Pose regression loss -- 'transformation' or 'speed'


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # Load backbone model extras
    if 'efficientdet' in cfg.MODEL.BACKBONE.NAME:
        assert 'efficientdet_d' in cfg.MODEL.BACKBONE.NAME, \
            "Only efficientdet's are supported at the moment."
        cfg.MODEL.BACKBONE.EXTRA = EFFICIENTDET_EXTRAS[cfg.MODEL.BACKBONE.NAME]
    else:
        raise AssertionError("Only efficientdet backbones are supported at the moment.")

    # Full paths to auxiliary files
    cfg.DATASET.CAMERA    = join(cfg.DATASET.ROOT, cfg.DATASET.DATANAME, cfg.DATASET.CAMERA)
    cfg.DATASET.KEYPOINTS = join(cfg.DATASET.ROOT, cfg.DATASET.KEYPOINTS)
    cfg.DATASET.CADMODEL  = join(cfg.DATASET.ROOT, cfg.DATASET.CADMODEL)

    # Assert types
    if not isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)):
        cfg.DATASET.OUTPUT_SIZE = [cfg.DATASET.OUTPUT_SIZE]
    if not isinstance(cfg.MODEL.HEAD.NAMES, (list, tuple)):
        cfg.MODEL.HEAD.NAMES = [cfg.MODEL.HEAD.NAMES]
    if not isinstance(cfg.MODEL.HEAD.LOSS_FACTORS, (list, tuple)):
        cfg.MODEL.HEAD.LOSS_FACTORS = [cfg.MODEL.HEAD.LOSS_FACTORS]

    if cfg.MODEL.HEAD.POSE_REGRESSION_LOSS not in ['speed', 'transformation']:
        raise ValueError('Pose regression loss must be either transformation or speed')

    # TODO: There may be other useful checks to include

    cfg.freeze()

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)