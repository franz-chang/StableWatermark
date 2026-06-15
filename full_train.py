#!/usr/bin/env python3
"""
StableWatermark - 完整训练脚本

使用合成数据和完整配置进行训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import json
from datetime import datetime
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import WatermarkEncoder, WatermarkDecoder, Discriminator
from data.dataset import SyntheticDataset
from training.losses import WatermarkLoss
from utils.attack import get_all_attacks, AttackConfig
from utils.metrics import calculate_bit_accuracy, calculate_psnr, calculate_ssim


def full_train(
    epochs: int = 50,
    batch_size: int = 16,
    image_size: int = 128,
    num_samples: int = 1000,
    device: str = None,
    output_dir: str = "./outputs/full_train"
):
    """完整训练"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*70)
    print("StableWatermark - Full Training")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Config: {epochs} epochs, batch_size={batch_size}, samples={num_samples}")
    print("="*70)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存配置
    config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'image_size': image_size,
        'num_samples': num_samples,
        'device': device,
        'feature_dim': 256,
        'hidden_dim': 128,
        'message_bits': 48
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # 数据集
    print("\n[1/4] Creating dataset...")
    dataset = SyntheticDataset(num_samples=num_samples, image_size=image_size, message_bits=48)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batches per epoch: {len(dataloader)}")

    # 模型
    print("\n[2/4] Creating models...")
    feature_dim = 256
    hidden_dim = 128

    encoder = WatermarkEncoder(
        feature_dim=feature_dim,
        message_bits=48,
        hidden_dim=hidden_dim
    )
    decoder = WatermarkDecoder(
        input_channels=3,
        message_bits=48,
        hidden_dim=256
    )
    discriminator = Discriminator(input_channels=3, ndf=32, n_layers=3)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    discriminator = discriminator.to(device)

    # 通道投影
    channel_proj = nn.Sequential(
        nn.Conv2d(3, feature_dim, kernel_size=1),
        nn.ReLU()
    ).to(device)
    channel_back = nn.Conv2d(feature_dim, 3, kernel_size=1).to(device)

    # 统计参数
    total_params = sum(p.numel() for p in encoder.parameters()) + \
                   sum(p.numel() for p in decoder.parameters()) + \
                   sum(p.numel() for p in discriminator.parameters())
    print(f"  Total parameters: {total_params:,}")

    # 损失和优化器
    loss_fn = WatermarkLoss(lambda_rec=1.0, lambda_msg=10.0, lambda_adv=0.5)

    lr = 1e-4
    opt = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr, weight_decay=1e-5
    )
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr * 0.5)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    # 训练历史
    history = {
        'epoch': [],
        'loss': [],
        'bit_acc': [],
        'psnr': [],
        'ssim': [],
        'time': []
    }

    best_bit_acc = 0.0
    best_epoch = 0

    # 训练
    print("\n[3/4] Training...")
    print("-" * 70)
    print(f"{'Epoch':^6} | {'Loss':^10} | {'Bit Acc':^10} | {'PSNR':^8} | {'SSIM':^8} | {'Time':^8}")
    print("-" * 70)

    start_time = time.time()

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        discriminator.train()

        epoch_loss = 0
        epoch_bit_acc = 0
        epoch_psnr = 0
        epoch_ssim = 0
        num_batches = 0

        epoch_start = time.time()

        for batch_idx, (images, messages) in enumerate(dataloader):
            images = images.to(device)
            messages = messages.to(device)

            # 特征提取
            features = channel_proj(
                torch.nn.functional.interpolate(
                    images, size=(32, 32), mode='bilinear', align_corners=False
                )
            )

            # 编码
            watermarked_features = encoder.embed_message(features, messages)
            watermarked_images = torch.sigmoid(channel_back(watermarked_features))

            # 调整尺寸
            watermarked_for_loss = torch.nn.functional.interpolate(
                watermarked_images, size=(image_size, image_size),
                mode='bilinear', align_corners=False
            )

            # 解码
            pred_logits = decoder(watermarked_images)
            pred_probs = torch.sigmoid(pred_logits)

            # 计算损失
            loss_dict = loss_fn(
                watermarked_for_loss, images,
                pred_logits, messages,
                discriminator_output=discriminator(watermarked_images),
                disc_real_output=discriminator(images),
                disc_fake_output=discriminator(watermarked_images.detach())
            )

            # 更新编码器和解码器
            opt.zero_grad()
            loss_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                max_norm=1.0
            )
            opt.step()

            # 更新判别器 (每2步更新一次)
            if batch_idx % 2 == 0:
                opt_disc.zero_grad()
                loss_dict.get('loss_adv_discriminator', torch.tensor(0.0, device=device)).backward()
                opt_disc.step()

            # 指标
            bit_acc = calculate_bit_accuracy(pred_probs, messages)
            psnr_val = calculate_psnr(watermarked_for_loss, images)
            ssim_val = calculate_ssim(watermarked_for_loss, images)

            epoch_loss += loss_dict['loss'].item()
            epoch_bit_acc += bit_acc
            epoch_psnr += psnr_val
            epoch_ssim += ssim_val
            num_batches += 1

        # 更新学习率
        scheduler.step()

        # 计算平均值
        epoch_loss /= num_batches
        epoch_bit_acc /= num_batches
        epoch_psnr /= num_batches
        epoch_ssim /= num_batches
        epoch_time = time.time() - epoch_start

        # 记录历史
        history['epoch'].append(epoch + 1)
        history['loss'].append(epoch_loss)
        history['bit_acc'].append(epoch_bit_acc)
        history['psnr'].append(epoch_psnr)
        history['ssim'].append(epoch_ssim)
        history['time'].append(epoch_time)

        # 打印进度
        print(f"{epoch+1:^6} | {epoch_loss:^10.4f} | {epoch_bit_acc:^10.4f} | {epoch_psnr:^8.2f} | {epoch_ssim:^8.4f} | {epoch_time:^8.1f}s")

        # 保存最佳模型
        if epoch_bit_acc > best_bit_acc:
            best_bit_acc = epoch_bit_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'opt_encoder_state_dict': opt.state_dict(),
                'opt_disc_state_dict': opt_disc.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_bit_acc': best_bit_acc
            }, os.path.join(output_dir, 'best_model.pt'))

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'history': history,
                'config': config
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt'))

    total_time = time.time() - start_time

    print("-" * 70)
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Best Bit Accuracy: {best_bit_acc:.4f} at epoch {best_epoch}")

    # 保存训练历史
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # 攻击测试
    print("\n[4/4] Attack Robustness Test...")
    print("-" * 70)

    encoder.eval()
    decoder.eval()

    # 加载最佳模型
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    attacks = get_all_attacks(AttackConfig(
        gaussian_noise_sigma=0.03,
        salt_pepper_prob=0.05,
        blur_kernel=5,
        crop_scale=0.8,
        jpeg_quality=75
    ))

    attack_results = {}

    with torch.no_grad():
        # 使用测试数据
        test_images, test_messages = next(iter(dataloader))
        test_images = test_images.to(device)
        test_messages = test_messages.to(device)

        # 嵌入水印
        test_features = channel_proj(
            torch.nn.functional.interpolate(
                test_images, size=(32, 32), mode='bilinear', align_corners=False
            )
        )
        wm_features = encoder.embed_message(test_features, test_messages)
        wm_images = torch.sigmoid(channel_back(wm_features))

        # 原始准确率
        original_probs = torch.sigmoid(decoder(wm_images))
        original_acc = calculate_bit_accuracy(original_probs, test_messages)
        print(f"  Original (no attack):     {original_acc:.4f}")
        attack_results['original'] = {'bit_accuracy': original_acc}

        # 测试各种攻击
        for name, attack_fn in sorted(attacks.items()):
            attacked_images = attack_fn(wm_images)
            attacked_probs = torch.sigmoid(decoder(attacked_images))
            attacked_acc = calculate_bit_accuracy(attacked_probs, test_messages)
            print(f"  {name:28s}: {attacked_acc:.4f}")
            attack_results[name] = {'bit_accuracy': attacked_acc}

    # 保存攻击结果
    with open(os.path.join(output_dir, 'attack_results.json'), 'w') as f:
        json.dump(attack_results, f, indent=2)

    print("\n" + "="*70)
    print("✓ Full Training Completed!")
    print(f"  Output: {output_dir}")
    print("="*70)

    return history, attack_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./outputs/full_train")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("\n⚠️  Warning: Running on CPU. Training will be slow.")
        print("    Consider using GPU for faster training.\n")

    full_train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.samples,
        device=device,
        output_dir=args.output_dir
    )