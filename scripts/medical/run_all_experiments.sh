#!/bin/bash
# =============================================================================
# StableWatermark - 医疗数据集完整实验脚本
# =============================================================================
# 支持 ChestX-ray14, ISIC, DRIVE, BrainMRI, LiTS 等数据集
# =============================================================================

set -e  # 遇到错误立即退出

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/medical"
MODELS_DIR="${PROJECT_ROOT}/models"

# 默认参数
DEFAULT_EPOCHS=30
DEFAULT_BATCH_SIZE=8
DEFAULT_MAX_SAMPLES=5000
DEFAULT_IMAGE_SIZE=512

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 帮助信息
show_help() {
    cat << EOF
使用方法: $0 [选项] <命令>

命令:
  setup          - 设置环境并下载数据集
  list           - 列出所有支持的数据集
  train          - 运行训练
  evaluate       - 运行评估
  all            - 运行完整流程 (训练+评估)
  cross_domain   - 跨领域联合训练
  benchmark      - 基准对比实验
  ablation       - 消融实验

数据集:
  chestxray14    - NIH 胸部X光 (112K)
  isic           - 皮肤病变 (25K)
  drive          - 视网膜血管 (40)
  brainmri       - 脑肿瘤 MRI (7K)
  lits           - 肝脏 CT (131)
  montgomery     - 肺结核 X光 (800)
  all_medical    - 所有医疗数据集

选项:
  -d, --dataset DATASET   指定数据集 (默认: chestxray14)
  -e, --epochs NUM        训练轮数 (默认: $DEFAULT_EPOCHS)
  -b, --batch_size NUM    批次大小 (默认: $DEFAULT_BATCH_SIZE)
  -m, --max_samples NUM   最大样本数 (默认: $DEFAULT_MAX_SAMPLES)
  -s, --image_size NUM    图像尺寸 (默认: $DEFAULT_IMAGE_SIZE)
  -o, --output DIR        输出目录 (默认: $OUTPUT_DIR)
  -h, --help              显示帮助

示例:
  $0 list
  $0 setup chestxray14
  $0 train -d chestxray14 -e 50
  $0 all -d isic
  $0 cross_domain
  $0 benchmark
EOF
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."

    # Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi

    # PyTorch
    if ! python3 -c "import torch" 2>/dev/null; then
        log_error "PyTorch 未安装"
        log_info "运行: pip install torch torchvision"
        exit 1
    fi

    # diffusers
    if ! python3 -c "import diffusers" 2>/dev/null; then
        log_warn "diffusers 未安装，SD水印功能不可用"
        log_info "运行: pip install diffusers transformers"
    fi

    log_success "依赖检查通过"
}

# 下载医疗数据集
download_datasets() {
    local dataset="$1"

    log_info "下载数据集: $dataset"

    if [ "$dataset" = "all_medical" ]; then
        log_info "下载所有医疗数据集..."
        python3 "${PROJECT_ROOT}/data/download_medical_data.py" --dataset all --save_dir "$DATA_DIR"
    else
        python3 "${PROJECT_ROOT}/data/download_medical_data.py" --dataset "$dataset" --save_dir "$DATA_DIR"
    fi

    log_success "数据下载完成"
}

# 列出数据集
list_datasets() {
    echo ""
    echo "========================================================================"
    echo " StableWatermark 支持的医疗数据集"
    echo "========================================================================"
    echo ""

    python3 -c "
from data import MEDICAL_DATASET_REGISTRY
for name, info in MEDICAL_DATASET_REGISTRY.items():
    print(f'  [{name}]')
    print(f'    名称: {info.name}')
    print(f'    模态: {info.modality}')
    print(f'    部位: {info.body_part}')
    print(f'    样本: ~{info.avg_samples}')
    print(f'    下载: {info.download_url[:50]}...' if len(info.download_url) > 50 else f'    下载: {info.download_url}')
    print()
"

    echo "========================================================================"
    echo ""
    echo "使用方法:"
    echo "  $0 setup chestxray14    # 下载 chestxray14"
    echo "  $0 setup all_medical    # 下载所有数据集"
    echo ""
}

# 获取数据集参数
get_dataset_config() {
    local dataset="$1"

    case "$dataset" in
        chestxray14)
            echo "batch_size=16 epochs=30 image_size=512"
            ;;
        isic)
            echo "batch_size=8 epochs=40 image_size=512"
            ;;
        drive)
            echo "batch_size=4 epochs=100 image_size=512"
            ;;
        brainmri)
            echo "batch_size=8 epochs=50 image_size=512"
            ;;
        lits)
            echo "batch_size=4 epochs=60 image_size=384"
            ;;
        montgomery)
            echo "batch_size=4 epochs=100 image_size=512"
            ;;
        shenzhen)
            echo "batch_size=4 epochs=100 image_size=512"
            ;;
        rsnaBreast)
            echo "batch_size=16 epochs=30 image_size=512"
            ;;
        *)
            echo "batch_size=$DEFAULT_BATCH_SIZE epochs=$DEFAULT_EPOCHS image_size=$DEFAULT_IMAGE_SIZE"
            ;;
    esac
}

# 训练单数据集
train_single() {
    local dataset="$1"
    local epochs="${2:-$DEFAULT_EPOCHS}"
    local batch_size="${3:-$DEFAULT_BATCH_SIZE}"
    local max_samples="${4:-$DEFAULT_MAX_SAMPLES}"

    local data_path="$DATA_DIR/$dataset"
    local config=$(get_dataset_config "$dataset")

    # 解析配置
    eval "declare -A c=($config)"
    epochs=${c[epochs]:-$epochs}
    batch_size=${c[batch_size]:-$batch_size}
    image_size=${c[image_size]:-$DEFAULT_IMAGE_SIZE}

    log_info "训练数据集: $dataset"
    log_info "  数据路径: $data_path"
    log_info "  Epochs: $epochs"
    log_info "  Batch Size: $batch_size"
    log_info "  Image Size: $image_size"
    log_info "  Max Samples: $max_samples"

    # 检查数据是否存在
    if [ ! -d "$data_path" ] || [ -z "$(ls -A "$data_path" 2>/dev/null | grep -E '\.(jpg|png|jpeg)$')" ]; then
        log_error "数据集不存在: $data_path"
        log_info "请先运行: $0 setup $dataset"
        exit 1
    fi

    # 创建输出目录
    local run_output="$OUTPUT_DIR/$dataset"
    mkdir -p "$run_output/checkpoints"
    mkdir -p "$run_output/logs"

    log_info "输出目录: $run_output"

    # 运行训练 (使用 main.py)
    python3 "${PROJECT_ROOT}/main.py" \
        --train \
        --data_root "$data_path" \
        --dataset_type "$dataset" \
        --output_dir "$run_output" \
        --num_epochs "$epochs" \
        --batch_size "$batch_size" \
        --image_size "$image_size" \
        --max_samples "$max_samples"

    log_success "训练完成: $run_output"
}

# 训练跨领域
train_cross_domain() {
    log_info "跨领域联合训练"

    local data_roots=""
    for dataset in chestxray14 brainmri isic; do
        if [ -d "$DATA_DIR/$dataset" ]; then
            data_roots="$data_roots $dataset:$DATA_DIR/$dataset"
        fi
    done

    if [ -z "$data_roots" ]; then
        log_error "没有找到有效的数据集"
        exit 1
    fi

    local run_output="$OUTPUT_DIR/cross_domain"
    mkdir -p "$run_output"

    log_info "跨领域数据集: ${data_roots//:/ }"
    log_info "输出目录: $run_output"

    python3 "${PROJECT_ROOT}/experiments/train_medical.py" \
        --experiment cross_domain \
        --data_root "$DATA_DIR" \
        --output_dir "$run_output" \
        --num_epochs 40 \
        --batch_size 8

    log_success "跨领域训练完成"
}

# 评估
evaluate() {
    local dataset="$1"
    local checkpoint="$2"

    if [ -z "$checkpoint" ]; then
        checkpoint="$OUTPUT_DIR/$dataset/checkpoints/latest.pt"
    fi

    log_info "评估数据集: $dataset"
    log_info "检查点: $checkpoint"

    if [ ! -f "$checkpoint" ]; then
        log_error "检查点不存在: $checkpoint"
        exit 1
    fi

    local data_path="$DATA_DIR/$dataset"
    local results_dir="$OUTPUT_DIR/$dataset/results"
    mkdir -p "$results_dir"

    python3 "${PROJECT_ROOT}/main.py" \
        --evaluate_only \
        --checkpoint "$checkpoint" \
        --data_root "$data_path" \
        --dataset_type "$dataset" \
        --output_dir "$results_dir"

    log_success "评估完成: $results_dir"
}

# 基准对比实验
run_benchmark() {
    log_info "运行基准对比实验"

    local output_dir="$OUTPUT_DIR/benchmark"
    mkdir -p "$output_dir"

    echo ""
    echo "========================================================================"
    echo " 基准对比实验配置"
    echo "========================================================================"
    echo ""

    # 测试数据集列表
    local datasets=("chestxray14" "brainmri" "isic" "drive")

    for dataset in "${datasets[@]}"; do
        local data_path="$DATA_DIR/$dataset"
        if [ -d "$data_path" ] && [ -n "$(ls -A "$data_path" 2>/dev/null | grep -E '\.(jpg|png|jpeg)$')" ]; then
            echo "  ✓ $dataset - 可用"
            echo "    数据: $data_path"
        else
            echo "  ✗ $dataset - 不可用 (运行: ./run_medical.sh setup $dataset)"
        fi
    done

    echo ""
    echo "运行以下基准测试:"
    echo "  1. Bit Accuracy (水印提取准确率)"
    echo "  2. PSNR (峰值信噪比)"
    echo "  3. SSIM (结构相似性)"
    echo "  4. 抗攻击能力"
    echo ""
    echo "查看基准测试结果: $output_dir"

    # 创建基准测试报告脚本
    cat > "$output_dir/benchmark_report.py" << 'BENCH_EOF'
#!/usr/bin/env python3
"""
医疗数据集基准测试报告生成
"""

import os
import json
from datetime import datetime

def generate_benchmark_report(output_dir):
    """生成基准测试报告"""
    report = {
        "title": "StableWatermark 医疗数据集基准测试报告",
        "timestamp": datetime.now().isoformat(),
        "datasets": [],
        "metrics": {
            "bit_accuracy": {"clean": [], "noisy": [], "compressed": []},
            "psnr": [],
            "ssim": []
        }
    }

    # 收集结果
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.json') and 'result' in f:
                with open(os.path.join(root, f)) as fp:
                    data = json.load(fp)
                    report["datasets"].append(data)

    # 保存报告
    report_path = os.path.join(output_dir, "benchmark_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {report_path}")
    return report

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    generate_benchmark_report(output_dir)
BENCH_EOF

    log_success "基准测试脚本已生成: $output_dir/benchmark_report.py"
}

# 消融实验
run_ablation() {
    local dataset="${1:-chestxray14}"

    log_info "运行消融实验: $dataset"

    local output_dir="$OUTPUT_DIR/ablation/$dataset"
    mkdir -p "$output_dir"

    echo ""
    echo "========================================================================"
    echo " 消融实验配置"
    echo "========================================================================"
    echo ""
    echo "  #1 Baseline:    λ_rec=1.0, λ_msg=10.0, λ_adv=0.5"
    echo "  #2 No-Adversarial: λ_rec=1.0, λ_msg=10.0, λ_adv=0.0"
    echo "  #3 High-Message: λ_rec=1.0, λ_msg=20.0, λ_adv=0.5"
    echo "  #4 Low-Rec:     λ_rec=0.5, λ_msg=10.0, λ_adv=0.5"
    echo ""

    # 生成消融实验配置
    python3 << ABLATION_EOF
import json
import os

ablation_configs = [
    {
        "name": "baseline",
        "lambda_rec": 1.0,
        "lambda_msg": 10.0,
        "lambda_adv": 0.5,
        "description": "默认配置"
    },
    {
        "name": "no_adversarial",
        "lambda_rec": 1.0,
        "lambda_msg": 10.0,
        "lambda_adv": 0.0,
        "description": "无对抗损失"
    },
    {
        "name": "high_message",
        "lambda_rec": 1.0,
        "lambda_msg": 20.0,
        "lambda_adv": 0.5,
        "description": "高消息权重"
    },
    {
        "name": "low_reconstruction",
        "lambda_rec": 0.5,
        "lambda_msg": 10.0,
        "lambda_adv": 0.5,
        "description": "低重建权重"
    }
]

output_dir = "$output_dir"
os.makedirs(output_dir, exist_ok=True)

for i, conf in enumerate(ablation_configs):
    config_path = os.path.join(output_dir, f"config_{conf['name']}.json")
    with open(config_path, 'w') as f:
        json.dump(conf, f, indent=2)
    print(f"  Config {i+1}: {config_path}")

print(f"\n配置已保存到: {output_dir}")
ABLATION_EOF

    log_success "消融实验配置已生成: $output_dir"
    log_info "手动运行每个配置进行训练和评估"
}

# ============================================================================
# 主程序
# ============================================================================

# 解析参数
DATASET="chestxray14"
EPOCHS=$DEFAULT_EPOCHS
BATCH_SIZE=$DEFAULT_BATCH_SIZE
MAX_SAMPLES=$DEFAULT_MAX_SAMPLES
IMAGE_SIZE=$DEFAULT_IMAGE_SIZE

# 处理参数
CMD=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -m|--max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        -s|--image_size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        setup|list|train|evaluate|all|cross_domain|benchmark|ablation)
            CMD="$1"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# 执行命令
case "$CMD" in
    setup)
        check_dependencies
        download_datasets "$DATASET"
        ;;
    list)
        list_datasets
        ;;
    train)
        check_dependencies
        train_single "$DATASET" "$EPOCHS" "$BATCH_SIZE" "$MAX_SAMPLES"
        ;;
    evaluate)
        check_dependencies
        evaluate "$DATASET"
        ;;
    all)
        check_dependencies
        train_single "$DATASET" "$EPOCHS" "$BATCH_SIZE" "$MAX_SAMPLES"
        evaluate "$DATASET"
        ;;
    cross_domain)
        check_dependencies
        train_cross_domain
        ;;
    benchmark)
        check_dependencies
        run_benchmark
        ;;
    ablation)
        check_dependencies
        run_ablation "$DATASET"
        ;;
    *)
        show_help
        exit 1
        ;;
esac

log_success "完成!"