#!/bin/bash
#
#SBATCH --job-name=full_config_dist_train
#
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err
#
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#
#SBATCH --partition=gpu
#SBATCH --constraint=GPU_GEN:VLT
#SBATCH --gpus=4
#SBATCH --gpus-per-task=4
#
#SBATCH --mail-user=tpark94@stanford.edu
#SBATCH --mail-type=ALL

# Load Python & CUDA modules. Assumes PyTorch 1.10 is installed in the python
# module below in advance.
ml reset
ml load python/3.9.0
ml load cuda/10.2.89

# Given limited memory in the user's home directory,
# keep the dataset in group_home and all the logs and outputs in scratch.
# Note: make sure the directories below already exist!
# Note: contents in scratch are purged 90 days after last modification!
ROOT=$HOME/spnv2
DROOT=$GROUP_HOME/datasets
LOG=$SCRATCH/spnv2/logs
OUTPUT=$SCRATCH/spnv2/outputs

# Experiment config file
CFG=$ROOT/experiments/offline_train_full_config_phi3_BN.yaml
# CFG=$ROOT/experiments/offline_train_full_config_phi6_GN.yaml

# python3 tools/train.py --cfg $CFG \
#     DIST.MULTIPROCESSING_DISTRIBUTED True \
#     LOG_DIR $LOG OUTPUT_DIR $OUTPUT ROOT $ROOT DATASET.ROOT $DROOT VERBOSE False

python3 tools/test.py --cfg $CFG \
    TEST.TEST_CSV lightbox/labels/lightbox.csv \
    LOG_DIR $LOG OUTPUT_DIR $OUTPUT ROOT $ROOT DATASET.ROOT $DROOT VERBOSE False

python3 tools/test.py --cfg $CFG \
    TEST.TEST_CSV sunlamp/labels/sunlamp.csv \
    LOG_DIR $LOG OUTPUT_DIR $OUTPUT ROOT $ROOT DATASET.ROOT $DROOT VERBOSE False
