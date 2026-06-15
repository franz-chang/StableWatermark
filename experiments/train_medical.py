#!/usr/bin/env python3
"""
医疗数据集水印训练脚本

支持单数据集训练和跨领域联合训练

Usage:
    # 单数据集训练
    python experiments/train_medical.py --dataset chestxray14 --data_root ./data

    # 跨领域训练
    python experiments/train_medical.py --experiment cross_domain \
        --data_roots.chestxray14 ./data/chestxray14 \
        --data_roots.brainmri ./data/brainmri \
        --data_roots.isic ./data/isic

    # 消融实验
    python experiments/train_medical.py --experiment ablation --dataset chestxray14
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.medical_config import (
    get_medical_experiment_config,
    get_cross_domain_config,
    MEDICAL_PRESETS,
)
from data import (
    get_medical_dataloader,
    get_multi_domain_dataloader,
    MedicalImageDataset,
    MultiDomainMedicalDataset,
)


class MedicalWatermarkTrainer:
    """医疗数据集水印训练器"""

    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dataloader = None
        self.model = None
        self.optimizer = None

    def setup_dataloader(self, **kwargs) -> DataLoader:
        """设置数据加载器"""
        if self.config.dataset.name == "multi_domain":
            # 跨领域数据加载
            data_roots = kwargs.get('data_roots', {})
            self.dataloader = get_multi_domain_dataloader(
                data_roots=data_roots,
                batch_size=self.config.training.batch_size,
                image_size=self.config.dataset.image_size,
            )
        else:
            # 单数据集
            self.dataloader = get_medical_dataloader(
                data_root=self.config.dataset.data_root,
                dataset_type=self.config.dataset.name,
                batch_size=self.config.training.batch_size,
                image_size=self.config.dataset.image_size,
                max_samples=self.config.dataset.max_samples,
            )
        return self.dataloader

    def print_config(self):
        """打印配置信息"""
        print("\n" + "=" * 70)
        print("Medical Watermark Training Configuration")
        print("=" * 70)
        print(f"Dataset: {self.config.dataset.name}")
        print(f"Image Size: {self.config.dataset.image_size}")
        print(f"Batch Size: {self.config.training.batch_size}")
        print(f"Epochs: {self.config.training.num_epochs}")
        print(f"Max Samples: {self.config.dataset.max_samples}")
        print(f"Device: {self.device}")
        print(f"Output Dir: {self.config.output_dir}")
        print("=" * 70 + "\n")


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Medical Dataset Watermark Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 实验类型
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default='single',
        choices=['single', 'cross_domain', 'ablation', 'benchmark'],
        help='实验类型'
    )

    # 数据集参数
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='chestxray14',
        choices=list(MEDICAL_PRESETS.keys()),
        help='数据集类型'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='./data',
        help='数据根目录'
    )
    parser.add_argument(
        '--data_roots',
        type=str,
        nargs='+',
        action='append',
        help='多数据集路径 (格式: dataset_name path)'
    )

    # 训练参数
    parser.add_argument('--batch_size', '-b', type=int, default=None)
    parser.add_argument('--num_epochs', '-n', type=int, default=None)
    parser.add_argument('--image_size', '-s', type=int, default=None)
    parser.add_argument('--max_samples', '-m', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-4)

    # 输出
    parser.add_argument('--output_dir', '-o', type=str, default='./outputs/medical')
    parser.add_argument('--device', type=str, default='cuda')

    # 选项
    parser.add_argument('--dry_run', action='store_true', help='仅打印配置不训练')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练路径')

    return parser


def parse_data_roots(data_roots_args) -> Dict[str, str]:
    """解析数据根目录参数"""
    data_roots = {}
    if data_roots_args:
        for item in data_roots_args:
            if len(item) >= 2:
                dataset_name = item[0]
                path = item[1]
                data_roots[dataset_name] = path
            elif len(item) == 1:
                # 尝试解析 key=value 格式
                parts = item[0].split('=')
                if len(parts) == 2:
                    data_roots[parts[0]] = parts[1]
    return data_roots


def run_single_experiment(args):
    """运行单数据集实验"""
    print(f"\n{'='*70}")
    print(f"Running Single Dataset Experiment: {args.dataset}")
    print(f"{'='*70}")

    # 构建数据路径
    data_root = os.path.join(args.data_root, args.dataset)

    # 获取配置
    config = get_medical_experiment_config(
        dataset_type=args.dataset,
        data_root=args.data_root,
        output_dir=args.output_dir,
    )

    # 覆盖命令行参数
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.image_size:
        config.dataset.image_size = args.image_size
    config.dataset.max_samples = args.max_samples
    config.training.learning_rate = args.lr

    # 创建训练器
    trainer = MedicalWatermarkTrainer(config, device=args.device)
    trainer.print_config()

    if args.dry_run:
        print("Dry run - exiting")
        return

    # 检查数据是否存在
    if not os.path.exists(data_root):
        print(f"Warning: Data directory not found: {data_root}")
        print("Please download the dataset first:")
        print(f"  python data/download_medical_data.py --dataset {args.dataset} --save_dir {args.data_root}")
        return

    # 统计数据集大小
    import glob
    images = glob.glob(os.path.join(data_root, '**/*.[jp][pn][g]'), recursive=True)
    print(f"Found {len(images)} images in {data_root}")

    # 设置数据加载器
    trainer.setup_dataloader()

    # 训练提示
    print(f"\nDataset ready at: {data_root}")
    print(f"To train, integrate with main.py training loop")
    print(f"Use config from configs.medical_config.get_medical_experiment_config('{args.dataset}')")


def run_cross_domain_experiment(args):
    """运行跨领域实验"""
    print(f"\n{'='*70}")
    print(f"Running Cross-Domain Medical Experiment")
    print(f"{'='*70}")

    # 解析数据根目录
    data_roots = parse_data_roots(args.data_roots)

    if not data_roots:
        # 默认使用项目目录结构
        for preset_name in MEDICAL_PRESETS.keys():
            default_path = os.path.join(args.data_root, preset_name)
            if os.path.exists(default_path):
                data_roots[preset_name] = default_path

    if not data_roots:
        print("Error: No valid data roots found. Please specify with --data_roots")
        print("Examples:")
        print("  --data_roots chestxray14 ./data/chestxray14")
        print("  --data_roots brainmri ./data/brainmri isic ./data/isic")
        return

    # 获取配置
    config = get_cross_domain_config(
        data_roots=data_roots,
        output_dir=args.output_dir,
    )

    # 覆盖参数
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size

    # 创建训练器
    trainer = MedicalWatermarkTrainer(config, device=args.device)
    trainer.print_config()

    print("Datasets loaded:")
    for name, path in data_roots.items():
        status = "✓" if os.path.exists(path) else "✗"
        print(f"  {status} {name}: {path}")

    if args.dry_run:
        print("\nDry run - exiting")
        return

    print("\nCross-domain training ready.")
    print("Integrate with main.py training loop using MultiDomainMedicalDataset")


def run_ablation_experiment(args):
    """运行消融实验"""
    print(f"\n{'='*70}")
    print(f"Running Ablation Study: {args.dataset}")
    print(f"{'='*70}")

    # 消融实验配置
    ablation_configs = [
        {"name": "baseline", "lambda_rec": 1.0, "lambda_msg": 10.0, "lambda_adv": 0.5},
        {"name": "no_adversarial", "lambda_rec": 1.0, "lambda_msg": 10.0, "lambda_adv": 0.0},
        {"name": "high_msg", "lambda_rec": 1.0, "lambda_msg": 20.0, "lambda_adv": 0.5},
        {"name": "low_rec", "lambda_rec": 0.5, "lambda_msg": 10.0, "lambda_adv": 0.5},
    ]

    print("\nAblation configurations:")
    for conf in ablation_configs:
        print(f"  {conf['name']}: rec={conf['lambda_rec']}, msg={conf['lambda_msg']}, adv={conf['lambda_adv']}")

    if args.dry_run:
        print("\nDry run - exiting")
        return

    print("\nTo run ablation study, iterate through configs and call train_lora.py or main.py")
    print("Results will be saved to separate subdirectories in output_dir")


def run_benchmark_experiment(args):
    """运行基准对比实验"""
    print(f"\n{'='*70}")
    print(f"Running Benchmark Experiment")
    print(f"{'='*70}")

    benchmark_datasets = ['chestxray14', 'brainmri', 'isic', 'drive']

    print("\nBenchmark datasets:")
    for ds in benchmark_datasets:
        preset = MEDICAL_PRESETS.get(ds)
        if preset:
            print(f"  - {ds}: {preset.name}")
            print(f"    Epochs: {preset.recommended_epochs}, Batch: {preset.recommended_batch_size}")

    if args.dry_run:
        print("\nDry run - exiting")
        return

    print("\nBenchmark will evaluate watermark quality (Bit Accuracy, PSNR, SSIM)")
    print("across multiple medical imaging modalities.")


def main():
    parser = create_parser()
    args = parser.parse_args()

    # 根据实验类型调用对应函数
    if args.experiment == 'single':
        run_single_experiment(args)
    elif args.experiment == 'cross_domain':
        run_cross_domain_experiment(args)
    elif args.experiment == 'ablation':
        run_ablation_experiment(args)
    elif args.experiment == 'benchmark':
        run_benchmark_experiment(args)


if __name__ == "__main__":
    main()