#!/bin/bash

DATASET_NAME=miniimagenet
TEST_RESOLUTION=84
DEVICE=cuda:0 # the
CKPT_PATH=~/Projects/brain-train-pefsl-checkpoints/
BACKBONE_NAME=resnet9
RESOLUTION=84
USE_STRIDES=
FMAPS=32

python  main.py \
--dataset ${DATASET_NAME} \
--steps "[['manifold mixup'],['rotations']]" \
--backbone ${BACKBONE_NAME} \
--feature-processing ME \
--batch-size 376 \
--epochs 500 \
--skip-epochs 450 \
--cosine \
--lr 0.1 \
--gamma 0.9 \
--few-shot \
--save-backbone ${CKPT_PATH}${DATASET_NAME}_${BACKBONE_NAME}_${RESOLUTION}x${RESOLUTION}.pt \
--save-features-prefix ${CKPT_PATH}${DATASET_NAME}_${BACKBONE_NAME}_${RESOLUTION}x${RESOLUTION}.pt \
--training-image-size ${RESOLUTION} \
--test-image-size ${TEST_RESOLUTION} \
--dataset-path ~/datasets/ \
--device ${DEVICE} \
--deterministic \
--leaky \
--feature-maps ${FMAPS} \
$STRIDES \
--wandb brain-imt --wandbProjectName PEFSL --wandb-dir ~/wandb --wandb-sweep ./config.yaml
