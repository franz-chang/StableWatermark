#!/bin/bash
#===============================================================================
# 单独数据集训练脚本
# 快速训练指定的医疗数据集
#===============================================================================

#!/bin/bash

# 数据集名称
DATASET_NAME="chestxray14"
DISPLAY_NAME="NIH ChestX-ray14"

# 设置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DATA_DIR="$PROJECT_ROOT/data"
OUTPUT_DIR="$PROJECT_ROOT/outputs/medical"

# 训练参数
BATCH_SIZE=16
EPOCHS=30
IMAGE_SIZE=512
MAX_SAMPLES=5000
LEARNING_RATE=0.0001

# 创建输出目录
RUN_DIR="$OUTPUT_DIR/$DATASET_NAME"
mkdir -p "$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR/logs"
mkdir -p "$RUN_DIR/results"

echo ""
echo "=============================================================================="
echo " StableWatermark 医疗数据集训练"
echo "=============================================================================="
echo ""
echo " 数据集: $DISPLAY_NAME"
echo " 路径:   $DATA_DIR/$DATASET_NAME"
echo " 输出:   $RUN_DIR"
echo ""
echo " 参数:"
echo "   - 批次大小:  $BATCH_SIZE"
echo "   - 训练轮数:  $EPOCHS"
echo "   - 图像尺寸:  $IMAGE_SIZE"
echo "   - 最大样本:  $MAX_SAMPLES"
echo "   - 学习率:    $LEARNING_RATE"
echo ""
echo "=============================================================================="
echo ""

# 检查数据是否存在
if [ ! -d "$DATA_DIR/$DATASET_NAME" ]; then
    echo "[错误] 数据目录不存在: $DATA_DIR/$DATASET_NAME"
    echo ""
    echo "请先下载数据集:"
    echo "  python data/download_medical_data.py --dataset $DATASET_NAME --save_dir $DATA_DIR"
    echo ""
    exit 1
fi

# 统计图像数量
IMAGE_COUNT=$(find "$DATA_DIR/$DATASET_NAME" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
echo "[信息] 发现 $IMAGE_COUNT 张图像"
echo ""

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "[错误] 数据目录为空"
    echo "请确保图像文件在: $DATA_DIR/$DATASET_NAME"
    exit 1
fi

# 开始训练
echo "[开始] 启动训练..."
echo ""

python3 "$PROJECT_ROOT/main.py" \
    --train \
    --data_root "$DATA_DIR/$DATASET_NAME" \
    --dataset_type "$DATASET_NAME" \
    --output_dir "$RUN_DIR" \
    --num_epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --image_size "$IMAGE_SIZE" \
    --max_samples "$MAX_SAMPLES" \
    --learning_rate "$LEARNING_RATE"

echo ""
echo "[完成] 训练完成!"
echo " 检查点: $RUN_DIR/checkpoints/"
echo ""
echo "评估模型:"
echo "  python main.py --evaluate_only \\
      --checkpoint $RUN_DIR/checkpoints/latest.pt \\
      --data_root $DATA_DIR/$DATASET_NAME \\
      --output_dir $RUN_DIR/results"
echo ""
echo "=============================================================================="