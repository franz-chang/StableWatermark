#!/usr/bin/env python3
"""
StableWatermark - 主程序入口

基于 Gumbel-Softmax 的 Stable Diffusion 水印算法实现
"""

import torch
import argparse
import os
import sys
from datetime import datetime

# 添加当前目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'scripts'))

try:
    from scripts.run_experiment import main as run_experiment_main
except ImportError:
    try:
        from run_experiment import main as run_experiment_main
    except ImportError:
        run_experiment_main = None


def quick_demo():
    """快速演示"""
    print("="*60)
    print("StableWatermark - Quick Demo")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 导入模型
    from models import WatermarkEncoder, WatermarkDecoder
    from modules.gumbel_softmax import gumbel_softmax
    from utils.attack import GaussianNoise, JPEGCompression
    from utils.metrics import calculate_bit_accuracy

    # 创建模型
    print("\n1. Creating models...")
    encoder = WatermarkEncoder(feature_dim=1280, message_bits=48, hidden_dim=256)
    decoder = WatermarkDecoder(input_channels=3, message_bits=48, hidden_dim=512)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # 生成测试数据
    print("\n2. Generating test data...")
    batch_size = 4
    images = torch.rand(batch_size, 3, 256, 256, device=device)
    messages = torch.randint(0, 2, (batch_size, 48), device=device).float()

    print(f"  Images shape: {images.shape}")
    print(f"  Messages shape: {messages.shape}")

    # 嵌入水印
    print("\n3. Embedding watermark...")
    watermarked = encoder.embed_message(images, messages)
    print(f"  Watermarked shape: {watermarked.shape}")

    # 提取水印
    print("\n4. Extracting watermark...")
    decoder.eval()
    with torch.no_grad():
        predicted_logits = decoder(watermarked)
        predicted_probs = torch.sigmoid(predicted_logits)

    bit_acc = calculate_bit_accuracy(predicted_probs, messages)
    print(f"  Predicted probabilities: {predicted_probs[0, :10].cpu().numpy()}...")
    print(f"  Bit Accuracy: {bit_acc:.4f}")

    # 应用攻击测试
    print("\n5. Testing robustness...")
    attacks = [
        ("Gaussian Noise (σ=0.03)", GaussianNoise(sigma=0.03)),
        ("JPEG Compression (Q=75)", JPEGCompression(quality=75)),
    ]

    for name, attack in attacks:
        attacked = attack(watermarked)
        with torch.no_grad():
            attacked_probs = torch.sigmoid(decoder(attacked))
        acc = calculate_bit_accuracy(attacked_probs, messages)
        print(f"  {name}: Bit Accuracy = {acc:.4f}")

    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)

    return watermarked, messages


def main():
    parser = argparse.ArgumentParser(
        description="StableWatermark - Gumbel-Softmax based Image Watermarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick demo
  python main.py --demo

  # Train model
  python main.py --train --data_root ./data/coco --num_epochs 50

  # Evaluate model
  python main.py --evaluate_only --output_dir ./outputs

  # Full experiment
  python main.py --data_root ./data --num_epochs 100 --batch_size 16
        """
    )

    # 模式选择
    parser.add_argument("--demo", action="store_true",
                        help="Run quick demo")
    parser.add_argument("--train", action="store_true",
                        help="Train model")
    parser.add_argument("--evaluate_only", action="store_true",
                        help="Only run evaluation")

    # 数据参数
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Path to data directory")
    parser.add_argument("--dataset_type", type=str, default="synthetic",
                        choices=["coco", "generated", "synthetic"],
                        help="Dataset type")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum number of samples")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # 设备
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, default: auto)")

    # 输出
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")

    args = parser.parse_args()

    # 设置设备
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.demo:
        quick_demo()
    else:
        # 运行完整实验
        run_experiment_main(args)


if __name__ == "__main__":
    main()