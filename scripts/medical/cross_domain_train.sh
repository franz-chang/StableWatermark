#!/bin/bash
#===============================================================================
# 跨领域联合训练脚本
# 在多个医疗数据集上联合训练水印模型
# 测试水印在不同模态间的泛化能力
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
DATA_DIR="$PROJECT_ROOT/data"
OUTPUT_DIR="$PROJECT_ROOT/outputs/medical/cross_domain"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }

echo ""
echo "=============================================================================="
echo " StableWatermark - 跨领域联合训练"
echo "=============================================================================="
echo ""

# 可用的数据集
DATASETS=("chestxray14" "brainmri" "isic")

# 检查每个数据集
AVAILABLE_DATA=""
for ds in "${DATASETS[@]}"; do
    DATA_PATH="$DATA_DIR/$ds"
    if [ -d "$DATA_PATH" ]; then
        IMAGE_COUNT=$(find "$DATA_PATH" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l)
        if [ "$IMAGE_COUNT" -gt 0 ]; then
            log_info "✓ $ds: $IMAGE_COUNT 张图像"
            AVAILABLE_DATA="$AVAILABLE_DATA --data_roots $ds $DATA_PATH"
        else
            log_info "✗ $ds: 目录为空"
        fi
    else
        log_info "✗ $ds: 目录不存在"
    fi
done

if [ -z "$AVAILABLE_DATA" ]; then
    echo ""
    echo "[错误] 没有可用的数据集"
    echo "请先下载数据集:"
    echo "  python $PROJECT_ROOT/data/download_medical_data.py --dataset all --save_dir $DATA_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/logs"

echo ""
echo "------------------------------------------------------------------------------"
echo " 训练配置"
echo "------------------------------------------------------------------------------"
echo " 模式:      跨领域联合训练"
echo " 数据集:    ${DATASETS[*]}"
echo " Batch:     8"
echo " Epochs:    40"
echo " Image:     512x512"
echo " 最大样本:  每数据集2000"
echo " 输出:      $OUTPUT_DIR"
echo "------------------------------------------------------------------------------"
echo ""

# 运行训练
log_info "启动跨领域训练..."

python3 "$PROJECT_ROOT/experiments/train_medical.py" \
    --experiment cross_domain \
    --data_root "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 40 \
    --batch_size 8 \
    $AVAILABLE_DATA

echo ""
echo "=============================================================================="
log_success "跨领域训练完成!"
echo "=============================================================================="
echo ""
echo "检查点: $OUTPUT_DIR/checkpoints/"
echo "日志:   $OUTPUT_DIR/logs/"
echo ""