#!/bin/bash
#===============================================================================
# Brain MRI 脑肿瘤数据集训练脚本
# Brain Tumor MRI Classification - ~7,000张图像
#===============================================================================

DATASET_NAME="brainmri"
DATASET_DISPLAY="Brain MRI (脑肿瘤)"
BATCH_SIZE=8
EPOCHS=50
IMAGE_SIZE=512
MAX_SAMPLES=5000

source "$(dirname "${BASH_SOURCE[0]}")/_train_template.sh"
main "$@"