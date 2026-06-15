#!/usr/bin/env python3
"""
医疗数据集基准测试脚本

对多个医疗数据集进行水印质量基准测试

测试指标:
- Bit Accuracy: 水印提取准确率
- PSNR: 峰值信噪比 (图像质量)
- SSIM: 结构相似性指数
- 抗攻击能力: 高斯噪声、JPEG压缩、裁剪等

Usage:
    python experiments/benchmark_medical.py --dataset chestxray14
    python experiments/benchmark_medical.py --all
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from configs.medical_config import MEDICAL_PRESETS
from data import MEDICAL_DATASET_REGISTRY


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    dataset: str
    modality: str
    num_samples: int
    metrics: Dict[str, float]
    attack_results: Dict[str, Dict[str, float]]
    timestamp: str


class MedicalBenchmark:
    """医疗数据集基准测试器"""

    def __init__(self, data_dir: str, output_dir: str, device: str = "cuda"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results: List[BenchmarkResult] = []

        os.makedirs(output_dir, exist_ok=True)

    def count_images(self, dataset_path: str) -> int:
        """统计图像数量"""
        count = 0
        for root, dirs, files in os.walk(dataset_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    count += 1
        return count

    def benchmark_dataset(self, dataset_name: str) -> Optional[BenchmarkResult]:
        """
        对单个数据集进行基准测试

        Returns:
            BenchmarkResult 或 None (如果数据集不存在)
        """
        dataset_path = os.path.join(self.data_dir, dataset_name)

        if not os.path.exists(dataset_path):
            print(f"  数据集不存在: {dataset_path}")
            return None

        num_samples = self.count_images(dataset_path)
        if num_samples == 0:
            print(f"  数据集为空: {dataset_path}")
            return None

        print(f"  图像数量: {num_samples}")

        # 获取数据集信息
        info = MEDICAL_DATASET_REGISTRY.get(dataset_name)

        # 创建模拟结果 (实际应用中需要真实模型推理)
        # TODO: 集成真实的水印模型进行测试
        result = BenchmarkResult(
            dataset=dataset_name,
            modality=info.modality if info else "Unknown",
            num_samples=num_samples,
            metrics={
                "bit_accuracy_clean": 0.0,
                "psnr_clean": 0.0,
                "ssim_clean": 0.0,
            },
            attack_results={
                "gaussian_noise": {"bit_accuracy": 0.0, "psnr": 0.0},
                "jpeg_compression": {"bit_accuracy": 0.0, "psnr": 0.0},
                "center_crop": {"bit_accuracy": 0.0, "psnr": 0.0},
                "combined": {"bit_accuracy": 0.0, "psnr": 0.0},
            },
            timestamp=datetime.now().isoformat(),
        )

        print(f"  模态: {result.modality}")

        return result

    def run_benchmark(
        self,
        dataset_names: List[str],
        use_pretrained: bool = False
    ) -> Dict[str, BenchmarkResult]:
        """
        运行基准测试

        Args:
            dataset_names: 数据集名称列表
            use_pretrained: 是否使用预训练模型

        Returns:
            数据集名称到结果的映射
        """
        print("\n" + "=" * 70)
        print("开始医疗数据集基准测试")
        print("=" * 70)

        results = {}
        for name in dataset_names:
            print(f"\n[{name}]")
            result = self.benchmark_dataset(name)
            if result:
                results[name] = result
                self.results.append(result)

        return results

    def generate_report(self) -> str:
        """生成基准测试报告"""
        report_path = os.path.join(self.output_dir, f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        report_data = {
            "title": "StableWatermark Medical Dataset Benchmark",
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "results": [asdict(r) for r in self.results],
            "summary": self._generate_summary()
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return report_path

    def _generate_summary(self) -> Dict:
        """生成摘要统计"""
        if not self.results:
            return {}

        summary = {
            "total_datasets": len(self.results),
            "total_samples": sum(r.num_samples for r in self.results),
            "modalities": list(set(r.modality for r in self.results)),
        }

        # 按模态分组统计
        by_modality = {}
        for r in self.results:
            if r.modality not in by_modality:
                by_modality[r.modality] = []
            by_modality[r.modality].append(r.dataset)

        summary["datasets_by_modality"] = by_modality

        return summary

    def print_results_table(self):
        """打印结果表格"""
        print("\n" + "=" * 100)
        print("基准测试结果汇总")
        print("=" * 100)
        print()
        print(f"{'数据集':<15} {'模态':<12} {'样本数':<10} {'Clean BA':<12} {'Clean PSNR':<12} {'Combined BA':<12}")
        print("-" * 100)

        for r in self.results:
            clean_ba = f"{r.metrics.get('bit_accuracy_clean', 0):.2%}"
            clean_psnr = f"{r.metrics.get('psnr_clean', 0):.2f}"
            combined_ba = f"{r.attack_results.get('combined', {}).get('bit_accuracy', 0):.2%}"

            print(f"{r.dataset:<15} {r.modality:<12} {r.num_samples:<10} {clean_ba:<12} {clean_psnr:<12} {combined_ba:<12}")

        print("-" * 100)
        print()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Medical Dataset Watermark Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        nargs='+',
        default=None,
        help='数据集名称 (支持多个)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='测试所有数据集'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='数据目录'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='./outputs/benchmark',
        help='输出目录'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备 (cuda/cpu)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有支持的数据集'
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # 列出数据集
    if args.list:
        print("\n支持的医疗数据集:")
        print("-" * 60)
        for name, info in MEDICAL_DATASET_REGISTRY.items():
            preset = MEDICAL_PRESETS.get(name)
            epochs = preset.recommended_epochs if preset else "N/A"
            batch = preset.recommended_batch_size if preset else "N/A"
            print(f"  {name:<15} {info.modality:<12} 样本:~{info.avg_samples:<8} epochs:{epochs} batch:{batch}")
        return

    # 确定要测试的数据集
    if args.all:
        dataset_names = list(MEDICAL_PRESETS.keys())
    elif args.dataset:
        dataset_names = args.dataset
    else:
        # 默认测试4个主要数据集
        dataset_names = ['chestxray14', 'brainmri', 'isic', 'drive']

    print(f"\n将测试以下数据集: {dataset_names}")

    # 创建基准测试器
    benchmark = MedicalBenchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )

    # 运行测试
    results = benchmark.run_benchmark(dataset_names)

    if results:
        # 打印结果
        benchmark.print_results_table()

        # 生成报告
        report_path = benchmark.generate_report()
        print(f"\n报告已保存: {report_path}")
    else:
        print("\n没有找到可用的数据集")
        print("请先下载数据集:")
        print(f"  python data/download_medical_data.py --dataset <name> --save_dir {args.data_dir}")


if __name__ == "__main__":
    main()