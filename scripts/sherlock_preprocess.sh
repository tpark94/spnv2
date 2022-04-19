#!/bin/bash
#
#SBATCH --job-name=preprocess
#
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00

ROOT=$HOME/spnv2
DROOT=$GROUP_HOME/datasets

for JSON in synthetic/train.json synthetic/validation.json
do
    echo $JSON
    python3 tools/preprocess.py --jsonfile $JSON \
            --cfg experiments/offline_train_full_config.yaml \
            ROOT $ROOT DATASET.ROOT $DROOT --no_mask
done

for JSON in lightbox/test.json sunlamp/test.json
do
    echo $JSON
    python3 tools/preprocess.py --jsonfile $JSON \
            --cfg experiments/offline_train_full_config.yaml \
            ROOT $ROOT DATASET.ROOT $DROOT --no_mask --no_labels
done