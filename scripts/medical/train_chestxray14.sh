#!/bin/bash
#===============================================================================
# ChestX-ray14 胸部X光数据集训练脚本
# NIH 胸部X光数据库 - 112,120张图像
#===============================================================================

DATASET_NAME="chestxray14"
DATASET_DISPLAY="NIH ChestX-ray14 (胸部X光)"

source "$(dirname "${BASH_SOURCE[0]}")/_train_template.sh"