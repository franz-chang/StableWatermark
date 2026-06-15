#!/bin/bash
#===============================================================================
# DRIVE 视网膜数据集训练脚本
# Digital Retinal Images - 40张眼底图像
# 小样本数据集，需要数据增强
#===============================================================================

DATASET_NAME="drive"
DATASET_DISPLAY="DRIVE Retina (视网膜血管)"
BATCH_SIZE=4
EPOCHS=100
IMAGE_SIZE=512
MAX_SAMPLES=40  # DRIVE只有40张图像

source "$(dirname "${BASH_SOURCE[0]}")/_train_template.sh"
main "$@"