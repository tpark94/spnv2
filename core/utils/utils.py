'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import random
import logging
import time
import json
from plyfile  import PlyData
from pathlib  import Path
from scipy.io import loadmat
from enum     import Enum

import torch
from torchinfo import summary

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# For reporting & logging
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """ Computes and stores the average and current value

        Modified from
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self, name, unit='-', fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.unit = unit
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __str__(self):
        fmtstr = '' if not self.name else '{name} '
        fmtstr += '{val' + self.fmt + '} ({avg' + self.fmt + '}) {unit}'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        fmtstr += ' {unit}'

        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    """ Prints training progress

        Modified from
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self, num_batches, timer, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.num_batches = num_batches
        self.timer  = timer
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = ['\r' + self.prefix + self.batch_fmtstr.format(batch) \
                      + ' [' + str(self.timer) + ']']
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)

        if batch < self.num_batches:
            sys.stdout.write(msg)
            sys.stdout.flush()
        else:
            sys.stdout.write('\r')
            sys.stdout.flush()
            sys.stdout.write(msg[1:]+'\n')

    def display_summary(self):
        entries = [" *"]
        entries += ['Time: ' + self.timer.summary()]
        entries += [meter.summary() for meter in self.meters]
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class NoOp:
    # https://discuss.pytorch.org/t/ddp-training-log-issue/125808
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            """Accept every signature by doing non-operation."""
            pass

        return no_op

def setup_logger(log_dir, rank, phase, to_console=True):
    if rank == 0:
        # File to save logger outputs
        log_file = os.path.join(log_dir, f'{phase}_rank{rank}.log')

        # Configure logger formats
        format  = '%(asctime)-15s %(message)s'
        datefmt = '%Y/%m/%d %H:%M:%S'
        logging.basicConfig(filename=str(log_file),
                            datefmt=datefmt,
                            format=format,
                            level=logging.INFO)

        # Root logger to the file
        logger = logging.getLogger()

        # Stream handler to the console
        if to_console:
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter(fmt=format, datefmt=datefmt))
            logger.addHandler(console)
    else:
        logger = NoOp()

    return logger

def create_logger_directories(cfg, phase='train', write_cfg_to_file=True):
    # Where to save outputs (e.g., checkpoints)
    output_dir = Path(cfg.OUTPUT_DIR) / cfg.MODEL.BACKBONE.NAME / cfg.EXP_NAME

    # Where to save logs
    time_str = time.strftime('%Y%m%d_%H_%M_%S')
    log_dir  = Path(cfg.LOG_DIR) / cfg.MODEL.BACKBONE.NAME / cfg.EXP_NAME / \
                    f'{phase}_{time_str}'

    # Master rank make output directories
    if cfg.DIST.RANK == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        while not output_dir.exists():
            time.sleep(5)

    # Create logger
    log_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logger(log_dir, cfg.DIST.RANK, phase, to_console=True)

    if write_cfg_to_file:
        with open(log_dir / 'config.txt', 'w') as f:
            f.write(str(cfg))

    logger.info(f'Outputs (e.g., checkpoints) are saved at:   {output_dir}')
    logger.info(f'Messages and tensorboard logs are saved at: {log_dir}')

    return logger, str(output_dir), str(log_dir)

# -----------------------------------------------------------------------
# Functions regarding 3D model loading
def load_tango_3d_keypoints(mat_dir):
    vertices  = loadmat(mat_dir)['tango3Dpoints'] # [3 x 11]
    corners3D = np.transpose(np.array(vertices, dtype=np.float32)) # [11 x 3]

    return corners3D

def load_cad_model(ply_dir, num_points=500):
    model  = PlyData.read(ply_dir)
    vertex = model['vertex']
    points_3d = np.stack([vertex[:]['x'],
                          vertex[:]['y'],
                          vertex[:]['z']], axis = -1) # [N x 3]
    points_3d /= 1000 # mm -> m

    num_model_points = points_3d.shape[0]
    if num_points:
        if num_points > num_model_points:
            raise AssertionError(f'num_points must be less than or equal to current model size ({num_model_points})')
        elif num_points == num_model_points:
            return points_3d
        else:
            # Sample num_points vertices
            step_size = (num_model_points // num_points) - 1
            if step_size < 1:
                step_size = 1
            points_3d = points_3d[::step_size, :]
            return points_3d[:num_points, :]

def load_camera_intrinsics(camera_json):
    with open(camera_json) as f:
        cam = json.load(f)

    # Into np.array
    cam = {
        k: np.array(v, dtype=np.float32) for k, v in cam.items()
    }

    return cam

# -----------------------------------------------------------------------
# Miscellaneous functions.
def set_seeds_cudnn(cfg, seed=None):
    if seed is None:
        seed = int(time.time())

    logger.info(f'Random seed: {seed}')

    # Set seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cfg.CUDA and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark     = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled       = cfg.CUDNN.ENABLED

        # empty any cached GPU memory
        torch.cuda.empty_cache()

def _write_model_info(module, log_dir, filename, depth=3):
    model_summary = summary(module,
                            depth=depth,
                            col_names=["kernel_size", "num_params"],
                            row_settings=["var_names"],
                            verbose=0)
    with open(os.path.join(log_dir, filename), 'w') as f:
        f.write(str(model_summary))

def write_model_info(model, log_dir):
    _write_model_info(model.backbone, log_dir, 'backbone_simple.txt', depth=3)
    _write_model_info(model.backbone, log_dir, 'backbone_all.txt', depth=10)
    _write_model_info(model.heads, log_dir, 'heads_simple.txt', depth=3)
    _write_model_info(model.heads, log_dir, 'heads_all.txt', depth=10)

def num_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

def num_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum(p.numel() for p in model_parameters)
