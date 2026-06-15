#!/bin/bash
#===============================================================================
# 医疗数据集训练模板脚本
# 被各数据集专用脚本引用
#===============================================================================

# ============================================================================
# 数据集配置 - 每个脚本需要定义这些变量
# ============================================================================
# DATASET_NAME      - 数据集内部名称 (用于路径等)
# DATASET_DISPLAY   - 显示名称
# BATCH_SIZE        - 批次大小
# EPOCHS           - 训练轮数
# IMAGE_SIZE        - 图像尺寸
# MAX_SAMPLES       - 最大样本数

# 默认值 (可被覆盖)
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-30}
IMAGE_SIZE=${IMAGE_SIZE:-512}
MAX_SAMPLES=${MAX_SAMPLES:-5000}
LEARNING_RATE=${LEARNING_RATE:-0.0001}

# ============================================================================
# 路径设置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
DATA_DIR="$PROJECT_ROOT/data"
OUTPUT_DIR="$PROJECT_ROOT/outputs/medical"

# ============================================================================
# 颜色和日志
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ============================================================================
# 初始化
# ============================================================================
init_dataset() {
    # 检查是否已定义必要变量
    if [ -z "$DATASET_NAME" ]; then
        log_error "DATASET_NAME 未定义"
        exit 1
    fi

    DATASET_DISPLAY=${DATASET_DISPLAY:-"$DATASET_NAME"}

    # 创建输出目录
    RUN_DIR="$OUTPUT_DIR/$DATASET_NAME"
    mkdir -p "$RUN_DIR/checkpoints"
    mkdir -p "$RUN_DIR/logs"
    mkdir -p "$RUN_DIR/results"
}

# ============================================================================
# 检查数据
# ============================================================================
check_data() {
    if [ ! -d "$DATA_DIR/$DATASET_NAME" ]; then
        log_error "数据目录不存在: $DATA_DIR/$DATASET_NAME"
        echo ""
        echo "请先下载数据集:"
        echo "  python $PROJECT_ROOT/data/download_medical_data.py --dataset $DATASET_NAME --save_dir $DATA_DIR"
        echo ""
        return 1
    fi

    IMAGE_COUNT=$(find "$DATA_DIR/$DATASET_NAME" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l)

    if [ "$IMAGE_COUNT" -eq 0 ]; then
        log_error "数据目录为空: $DATA_DIR/$DATASET_NAME"
        echo "请确保图像文件存在于该目录"
        return 1
    fi

    log_success "发现 $IMAGE_COUNT 张图像"
    return 0
}

# ============================================================================
# 打印配置
# ============================================================================
print_config() {
    echo ""
    echo "=============================================================================="
    echo " StableWatermark - $DATASET_DISPLAY"
    echo "=============================================================================="
    echo ""
    echo " 数据集: $DATASET_DISPLAY"
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
}

# ============================================================================
# 运行训练
# ============================================================================
run_training() {
    log_info "启动训练..."

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

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_success "训练完成!"
    else
        log_error "训练失败 (退出码: $exit_code)"
    fi

    return $exit_code
}

# ============================================================================
# 运行评估
# ============================================================================
run_evaluation() {
    local checkpoint="${1:-$RUN_DIR/checkpoints/latest.pt}"

    if [ ! -f "$checkpoint" ]; then
        log_error "检查点不存在: $checkpoint"
        return 1
    fi

    log_info "运行评估..."

    python3 "$PROJECT_ROOT/main.py" \
        --evaluate_only \
        --checkpoint "$checkpoint" \
        --data_root "$DATA_DIR/$DATASET_NAME" \
        --dataset_type "$DATASET_NAME" \
        --output_dir "$RUN_DIR/results"

    return $?
}

# ============================================================================
# 主流程
# ============================================================================
main() {
    local mode="${1:-train}"  # train, eval, both

    # 初始化
    init_dataset

    # 检查数据
    check_data || exit 1

    # 打印配置
    print_config

    # 执行
    case "$mode" in
        train)
            run_training
            ;;
        eval)
            run_evaluation
            ;;
        both|all)
            run_training && run_evaluation
            ;;
        *)
            log_error "未知模式: $mode"
            echo "可用模式: train, eval, both"
            exit 1
            ;;
    esac

    echo ""
    echo "=============================================================================="
    echo " 完成!"
    echo "=============================================================================="
}

# 如果被直接运行 (而非source)
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi