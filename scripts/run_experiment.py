#!/usr/bin/env python3
"""
StableWatermark 实验脚本

运行水印模型的训练和评估
"""

import argparse
import os
import sys
import torch
import random
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import get_config, get_config_str
from training.trainer import create_trainer
from data.dataset import get_dataloader
from utils.attack import AttackConfig, get_all_attacks
from utils.metrics import MetricsTracker
from utils.visualization import plot_training_curves, plot_attack_results


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """主函数"""
    print("="*60)
    print("StableWatermark - Gumbel-Softmax based Image Watermarking")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print()

    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取配置
    config = get_config()
    config.device = args.device

    # 打印配置
    print(get_config_str(config))

    # 准备数据
    print("\n" + "="*40)
    print("Preparing Data")
    print("="*40)

    train_dataloader = get_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_samples=args.max_samples,
        dataset_type=args.dataset_type,
        shuffle=True,
        message_bits=config.model.watermark_bits
    )

    val_dataloader = get_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_samples=args.max_samples // 5 if args.max_samples else 100,
        dataset_type=args.dataset_type,
        shuffle=False,
        message_bits=config.model.watermark_bits
    )

    print(f"  Training samples: {len(train_dataloader.dataset)}")
    print(f"  Validation samples: {len(val_dataloader.dataset)}")

    # 创建训练器
    print("\n" + "="*40)
    print("Creating Trainer")
    print("="*40)

    trainer_config = {
        'lambda_rec': config.training.lambda_rec,
        'lambda_msg': config.training.lambda_msg,
        'lambda_adv': config.training.lambda_adv,
        'learning_rate': config.training.learning_rate,
        'weight_decay': config.training.weight_decay,
        'num_epochs': args.num_epochs,
        'log_every': config.training.log_every
    }

    trainer = create_trainer(
        feature_dim=config.model.hidden_dim,
        message_bits=config.model.watermark_bits,
        hidden_dim=config.model.message_dim,
        use_discriminator=True,
        device=args.device,
        config=trainer_config
    )

    print(f"  Model parameters:")
    encoder_params = sum(p.numel() for p in trainer.encoder.parameters())
    decoder_params = sum(p.numel() for p in trainer.decoder.parameters())
    disc_params = sum(p.numel() for p in trainer.discriminator.parameters())
    print(f"    Encoder: {encoder_params:,}")
    print(f"    Decoder: {decoder_params:,}")
    print(f"    Discriminator: {disc_params:,}")
    print(f"    Total: {encoder_params + decoder_params + disc_params:,}")

    # 训练
    if not args.evaluate_only:
        print("\n" + "="*40)
        print("Training")
        print("="*40)

        # 攻击配置
        attack_cfg = AttackConfig(
            gaussian_noise_sigma=0.03,
            salt_pepper_prob=0.05,
            blur_kernel=5,
            crop_scale=0.8,
            jpeg_quality=75
        )

        trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=args.num_epochs,
            output_dir=args.output_dir,
            attack_config=attack_cfg
        )

        # 绘制训练曲线
        print("\n" + "="*40)
        print("Generating Training Curves")
        print("="*40)

        plot_training_curves(
            trainer.history,
            save_path=os.path.join(args.output_dir, "training_curves.png")
        )

    # 最终评估
    print("\n" + "="*40)
    print("Final Evaluation")
    print("="*40)

    attack_cfg = AttackConfig(
        gaussian_noise_sigma=0.05,
        salt_pepper_prob=0.1,
        blur_kernel=7,
        crop_scale=0.7,
        jpeg_quality=70
    )

    results = trainer.evaluate(val_dataloader, attack_cfg)

    # 打印干净样本的结果
    print("\nClean Samples:")
    for metric, value in results['clean'].items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")

    # 打印攻击测试结果
    print("\nAttack Tests:")
    for attack_name, metrics in results['attacked'].items():
        print(f"\n  {attack_name}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"    {metric}: {value:.4f}")

    # 保存结果
    results_path = os.path.join(args.output_dir, "final_evaluation.json")
    import json

    def convert_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(x) for x in obj]
        else:
            return obj

    results = convert_to_serializable(results)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # 绘制攻击结果表格
    attack_results_for_plot = results['attacked']
    plot_attack_results(
        attack_results_for_plot,
        save_path=os.path.join(args.output_dir, "attack_results.png")
    )

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


def evaluate_experiment(args):
    """评估已训练的模型"""
    print("Loading trained model for evaluation...")

    device = args.device
    output_dir = args.output_dir

    # 加载模型
    from models import WatermarkEncoder, WatermarkDecoder

    encoder = WatermarkEncoder(feature_dim=1280, message_bits=48, hidden_dim=256)
    decoder = WatermarkDecoder(input_channels=3, message_bits=48, hidden_dim=512)

    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    # 准备数据
    val_dataloader = get_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_samples=args.max_samples // 5 if args.max_samples else 100,
        dataset_type=args.dataset_type,
        shuffle=False,
        message_bits=48
    )

    # 攻击配置
    attack_cfg = AttackConfig(
        gaussian_noise_sigma=0.03,
        salt_pepper_prob=0.05,
        blur_kernel=5,
        crop_scale=0.8,
        jpeg_quality=75
    )

    # 评估
    from training.trainer import Trainer
    trainer = Trainer(
        encoder=encoder,
        decoder=decoder,
        discriminator=None,
        device=device
    )

    results = trainer.evaluate(val_dataloader, attack_cfg)

    # 打印结果
    print("\nClean Samples:")
    for metric, value in results['clean'].items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")

    print("\nAttack Tests:")
    for attack_name, metrics in results['attacked'].items():
        print(f"\n  {attack_name}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"    {metric}: {value:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StableWatermark Training and Evaluation")

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
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")

    # 输出
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")

    # 模式
    parser.add_argument("--evaluate_only", action="store_true",
                        help="Only run evaluation, skip training")

    args = parser.parse_args()

    if args.evaluate_only:
        evaluate_experiment(args)
    else:
        main(args)