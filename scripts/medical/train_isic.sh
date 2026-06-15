#!/bin/bash
#===============================================================================
# ISIC 皮肤病变数据集训练脚本
# ISIC 2018 Skin Lesion Analysis - ~25,000张图像
#===============================================================================

DATASET_NAME="isic"
DATASET_DISPLAY="ISIC Skin Lesion (皮肤病变)"
BATCH_SIZE=8
EPOCHS=40
IMAGE_SIZE=512
MAX_SAMPLES=5000

source "$(dirname "${BASH_SOURCE[0]}")/_train_template.sh"
main "$@"