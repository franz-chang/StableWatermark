#!/usr/bin/env python3
"""
快速训练实验 - 验证完整水印训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import WatermarkEncoder, WatermarkDecoder, Discriminator
from data.dataset import SyntheticDataset
from training.losses import WatermarkLoss
from utils.attack import get_all_attacks, AttackConfig
from utils.metrics import calculate_bit_accuracy, calculate_psnr


def quick_train(
    epochs: int = 5,
    batch_size: int = 8,
    image_size: int = 128,
    num_samples: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """快速训练"""
    print("\n" + "="*60)
    print("StableWatermark - Quick Training Experiment")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Config: {epochs} epochs, batch_size={batch_size}, samples={num_samples}")
    print("="*60)

    # 数据集
    print("\n1. Creating dataset...")
    dataset = SyntheticDataset(num_samples=num_samples, image_size=image_size, message_bits=48)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 模型
    print("\n2. Creating models...")
    feature_dim = 256
    encoder = WatermarkEncoder(feature_dim=feature_dim, message_bits=48, hidden_dim=128)
    decoder = WatermarkDecoder(input_channels=3, message_bits=48, hidden_dim=256)
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

    # 损失和优化器
    loss_fn = WatermarkLoss(lambda_rec=1.0, lambda_msg=10.0, lambda_adv=0.5)
    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    opt_disc = optim.Adam(discriminator.parameters(), lr=1e-4)

    # 训练
    print("\n3. Training...")
    print("-" * 60)

    history = {'loss': [], 'bit_acc': [], 'psnr': []}
    global_step = 0

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        discriminator.train()

        epoch_loss = 0
        epoch_bit_acc = 0
        epoch_psnr = 0
        num_batches = 0

        for batch_idx, (images, messages) in enumerate(dataloader):
            images = images.to(device)
            messages = messages.to(device)

            # 特征提取
            features = channel_proj(
                torch.nn.functional.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)
            )

            # 编码
            watermarked_features = encoder.embed_message(features, messages)
            watermarked_images = torch.sigmoid(channel_back(watermarked_features))

            # 解码
            pred_logits = decoder(watermarked_images)
            pred_probs = torch.sigmoid(pred_logits)

            # 调整尺寸以匹配原始图像用于损失计算
            watermarked_for_loss = torch.nn.functional.interpolate(
                watermarked_images, size=(image_size, image_size), mode='bilinear', align_corners=False
            )

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
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            opt.step()

            # 更新判别器
            opt_disc.zero_grad()
            loss_dict.get('loss_adv_discriminator', torch.tensor(0.0)).backward()
            opt_disc.step()

            # 指标
            bit_acc = calculate_bit_accuracy(pred_probs, messages)
            psnr_val = calculate_psnr(watermarked_for_loss, images)

            epoch_loss += loss_dict['loss'].item()
            epoch_bit_acc += bit_acc
            epoch_psnr += psnr_val
            num_batches += 1
            global_step += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Step {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {loss_dict['loss'].item():.4f} | Bit Acc: {bit_acc:.4f} | PSNR: {psnr_val:.2f}")

        # 记录历史
        history['loss'].append(epoch_loss / num_batches)
        history['bit_acc'].append(epoch_bit_acc / num_batches)
        history['psnr'].append(epoch_psnr / num_batches)

        print(f"\n  Epoch {epoch+1} Summary: Avg Loss={history['loss'][-1]:.4f}, "
              f"Avg Bit Acc={history['bit_acc'][-1]:.4f}, Avg PSNR={history['psnr'][-1]:.2f}")

    # 攻击测试
    print("\n4. Attack Robustness Test...")
    print("-" * 60)

    encoder.eval()
    decoder.eval()

    attacks = get_all_attacks(AttackConfig())

    with torch.no_grad():
        for name, attack_fn in attacks.items():
            # 测试一批数据
            test_images, test_messages = next(iter(dataloader))
            test_images = test_images.to(device)
            test_messages = test_messages.to(device)

            # 嵌入
            test_features = channel_proj(
                torch.nn.functional.interpolate(test_images, size=(32, 32), mode='bilinear', align_corners=False)
            )
            wm_features = encoder.embed_message(test_features, test_messages)
            wm_images = torch.sigmoid(channel_back(wm_features))

            # 攻击
            attacked_images = attack_fn(wm_images)

            # 提取
            attacked_probs = torch.sigmoid(decoder(attacked_images))
            attacked_acc = calculate_bit_accuracy(attacked_probs, test_messages)

            print(f"  {name:20s}: Bit Accuracy = {attacked_acc:.4f}")

    print("\n" + "="*60)
    print("✓ Quick Training Experiment Completed!")
    print("="*60)

    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quick_train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.samples,
        device=device
    )