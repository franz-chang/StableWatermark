"""
医疗数据集实验配置

预设配置用于不同医疗数据集的水印实验
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .base_config import (
    ModelConfig, TrainingConfig, DatasetConfig, AttackConfig, ExperimentConfig
)


# 医疗数据集特定配置
@dataclass
class MedicalDatasetPreset:
    """医疗数据集预设配置"""
    name: str
    dataset_type: str
    gray_to_rgb: bool  # 是否将灰度图转换为RGB
    target_size: int  # 目标图像尺寸
    description: str
    recommended_epochs: int
    recommended_batch_size: int


# 数据集预设注册表
MEDICAL_PRESETS: Dict[str, MedicalDatasetPreset] = {
    "chestxray14": MedicalDatasetPreset(
        name="NIH ChestX-ray14",
        dataset_type="chestxray14",
        gray_to_rgb=True,
        target_size=512,
        description="胸部X光 - 112K样本，适合大规模训练",
        recommended_epochs=30,
        recommended_batch_size=16
    ),
    "isic": MedicalDatasetPreset(
        name="ISIC Skin Lesion",
        dataset_type="isic",
        gray_to_rgb=False,  # 已是RGB
        target_size=512,
        description="皮肤镜图像 - 彩色图像，适合彩色水印",
        recommended_epochs=40,
        recommended_batch_size=8
    ),
    "drive": MedicalDatasetPreset(
        name="DRIVE Retina",
        dataset_type="drive",
        gray_to_rgb=True,
        target_size=512,
        description="视网膜图像 - 40张图像，建议数据增强",
        recommended_epochs=100,
        recommended_batch_size=4
    ),
    "brainmri": MedicalDatasetPreset(
        name="Brain Tumor MRI",
        dataset_type="brainmri",
        gray_to_rgb=True,
        target_size=512,
        description="脑肿瘤MRI - MRI图像，对比度调整",
        recommended_epochs=50,
        recommended_batch_size=8
    ),
    "lits": MedicalDatasetPreset(
        name="LiTS Liver CT",
        dataset_type="lits",
        gray_to_rgb=True,
        target_size=384,
        description="肝脏CT - 切片图像，需要窗口化",
        recommended_epochs=60,
        recommended_batch_size=4
    ),
    "montgomery": MedicalDatasetPreset(
        name="Montgomery TB X-ray",
        dataset_type="montgomery",
        gray_to_rgb=True,
        target_size=512,
        description="肺结核X光 - 胸部X光，少量样本",
        recommended_epochs=100,
        recommended_batch_size=4
    ),
}


def get_medical_experiment_config(
    dataset_type: str,
    data_root: str = "./data",
    output_dir: str = "./outputs/medical"
) -> ExperimentConfig:
    """
    获取医疗数据集实验配置

    Args:
        dataset_type: 数据集类型 (chestxray14, isic, drive, etc.)
        data_root: 数据根目录
        output_dir: 输出目录

    Returns:
        ExperimentConfig: 预设的实验配置
    """
    preset = MEDICAL_PRESETS.get(dataset_type)

    if preset is None:
        print(f"Warning: Unknown dataset type '{dataset_type}', using default config")
        preset = MEDICAL_PRESETS["chestxray14"]

    # 构建数据路径
    dataset_path = f"{data_root}/{dataset_type}"

    return ExperimentConfig(
        model=ModelConfig(
            watermark_bits=48,
            hidden_dim=320,
            gumbel_temperature=1.0,
        ),
        training=TrainingConfig(
            batch_size=preset.recommended_batch_size,
            num_epochs=preset.recommended_epochs,
            learning_rate=1e-4,
            lambda_rec=1.0,
            lambda_msg=10.0,
            lambda_adv=0.5,
        ),
        dataset=DatasetConfig(
            name=dataset_type,
            data_root=dataset_path,
            image_size=preset.target_size,
            max_samples=5000,  # 限制样本数用于快速实验
            num_workers=4,
            gray_to_rgb=preset.gray_to_rgb,
        ),
        attack=AttackConfig(
            attacks=["gaussian_noise", "jpeg_compression", "center_crop"],
        ),
        output_dir=f"{output_dir}/{dataset_type}",
        seed=42,
    )


def get_cross_domain_config(
    data_roots: Dict[str, str],
    output_dir: str = "./outputs/cross_domain"
) -> ExperimentConfig:
    """
    获取跨领域实验配置

    Args:
        data_roots: 数据集名称到路径的映射
        output_dir: 输出目录

    Returns:
        ExperimentConfig: 跨领域实验配置
    """
    return ExperimentConfig(
        model=ModelConfig(
            watermark_bits=48,
            hidden_dim=320,
            gumbel_temperature=1.0,
        ),
        training=TrainingConfig(
            batch_size=8,
            num_epochs=40,
            learning_rate=1e-4,
            lambda_rec=1.0,
            lambda_msg=10.0,
            lambda_adv=0.5,
        ),
        dataset=DatasetConfig(
            name="multi_domain",
            data_root=",".join(data_roots.values()),
            image_size=512,
            max_samples=2000,
            num_workers=4,
        ),
        attack=AttackConfig(
            attacks=["gaussian_noise", "jpeg_compression", "center_crop", "combined"],
        ),
        output_dir=output_dir,
        seed=42,
    )


def create_medical_experiment_scripts():
    """
    生成医疗数据集实验脚本

    Returns:
        Dict[str, str]: 脚本名称到内容的映射
    """
    scripts = {}

    for dataset_type, preset in MEDICAL_PRESETS.items():
        script_content = f'''#!/bin/bash
# {preset.name} 水印实验脚本
# {preset.description}

DATASET_TYPE="{dataset_type}"
DATA_DIR="./data"
OUTPUT_DIR="./outputs/{dataset_type}"

# 训练
python main.py \\
    --train \\
    --dataset_type {dataset_type} \\
    --data_root "${{DATA_DIR}}/{dataset_type}" \\
    --output_dir "$OUTPUT_DIR" \\
    --batch_size {preset.recommended_batch_size} \\
    --num_epochs {preset.recommended_epochs} \\
    --image_size {preset.target_size} \\
    --max_samples 5000

# 评估
python main.py \\
    --evaluate_only \\
    --checkpoint "$OUTPUT_DIR/checkpoints/latest.pt" \\
    --data_root "${{DATA_DIR}}/{dataset_type}" \\
    --output_dir "$OUTPUT_DIR/results"
'''
        scripts[f"train_{dataset_type}.sh"] = script_content

    # 跨领域实验脚本
    cross_domain_script = '''#!/bin/bash
# 跨领域医疗数据集实验

DATASET_TYPES=("chestxray14" "brainmri" "isic")
DATA_DIR="./data"
OUTPUT_DIR="./outputs/cross_domain"

# 创建数据根字典
DATA_ROOTS=""
for dt in "${DATASET_TYPES[@]}"; do
    if [ -d "$DATA_DIR/$dt" ]; then
        DATA_ROOTS="$DATA_ROOTS --data_roots.$dt $DATA_DIR/$dt"
    fi
done

# 运行跨领域实验
python experiments/train_medical.py \\
    --experiment cross_domain \\
    --output_dir "$OUTPUT_DIR" \\
    --batch_size 8 \\
    --num_epochs 40 \\
    --image_size 512 \\
    $DATA_ROOTS
'''
    scripts["train_cross_domain.sh"] = cross_domain_script

    return scripts


# 自动生成并保存脚本
if __name__ == "__main__":
    print("Generating medical experiment scripts...")
    scripts = create_medical_experiment_scripts()

    scripts_dir = "./scripts/medical"
    import os
    os.makedirs(scripts_dir, exist_ok=True)

    for name, content in scripts.items():
        path = os.path.join(scripts_dir, name)
        with open(path, 'w') as f:
            f.write(content)
        os.chmod(path, 0o755)
        print(f"  Created: {path}")

    print("Done!")