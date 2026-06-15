#!/bin/bash
#===============================================================================
# LiTS 肝脏肿瘤 CT 数据集训练脚本
# Liver Tumor Segmentation Challenge - 131个3D CT扫描
#===============================================================================

DATASET_NAME="lits"
DATASET_DISPLAY="LiTS Liver CT (肝脏肿瘤)"
BATCH_SIZE=4
EPOCHS=60
IMAGE_SIZE=384  # CT切片常用较小尺寸
MAX_SAMPLES=2000

source "$(dirname "${BASH_SOURCE[0]}")/_train_template.sh"
main "$@"