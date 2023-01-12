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

#ROOT=$HOME/spnv2
#DROOT=$GROUP_HOME/datasets

ROOT=/media/shared/Jeff/SLAB/spnv2
DROOT=/home/jeffpark/SLAB/Dataset

for JSON in synthetic/train.json synthetic/validation.json
do
    echo $JSON
    python3 tools/preprocess.py --jsonfile $JSON \
            --cfg experiments/offline_train_full_config_phi3_BN.yaml \
            ROOT $ROOT DATASET.ROOT $DROOT
done

# for JSON in lightbox/test.json sunlamp/test.json
# do
#     echo $JSON
#     python3 tools/preprocess.py --jsonfile $JSON \
#             --cfg experiments/offline_train_full_config_phi3_BN.yaml \
#             ROOT $ROOT DATASET.ROOT $DROOT --no_mask --no_labels
# done