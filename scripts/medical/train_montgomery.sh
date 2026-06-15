#!/bin/bash
#===============================================================================
# Montgomery 肺结核 X 光数据集训练脚本
# Montgomery County TB Screening - 800张图像
# 小样本数据集
#===============================================================================

DATASET_NAME="montgomery"
DATASET_DISPLAY="Montgomery TB (肺结核筛查)"
BATCH_SIZE=4
EPOCHS=100
IMAGE_SIZE=512
MAX_SAMPLES=800

source "$(dirname "${BASH_SOURCE[0]}")/_train_template.sh"
main "$@"