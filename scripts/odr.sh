#! /usr/bin/env bash

################################################################################
# ODR (Single)
################################################################################

PRETRAIN=outputs/efficientdet_d3/full_config/model_best.pth.tar

EXP='experiments/odr_B4_N1024.yaml'
EXP_NAME='odr/BN_B4_N1024'

for DOMAIN in lightbox sunlamp; do

    CSV=$DOMAIN/labels/$DOMAIN.csv

    NUM_SAMPLES=1024
    BATCH_SIZE=4

    python3 tools/odr.py --cfg $EXP VERBOSE True \
        MODEL.PRETRAIN_FILE $PRETRAIN \
        TRAIN.TRAIN_CSV $CSV TRAIN.VAL_CSV $CSV \
        ODR.NUM_TRAIN_SAMPLES $NUM_SAMPLES \
        ODR.IMAGES_PER_BATCH $BATCH_SIZE

done