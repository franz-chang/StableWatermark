#!/usr/bin/env python3
"""
StableWatermark - 改进的训练脚本

使用两阶段训练策略:
1. 阶段1: 先训练解码器直接从图像提取消息的能力
2. 阶段2: 端到端训练编码器和解码器
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


class ImprovedDecoder(nn.Module):
    """改进的解码器 - 直接从图像预测水印"""
    def __init__(self, message_bits=48, hidden_dim=512):
        super().__init__()
        self.message_bits = message_bits

        # 更强的特征提取器
        self.encoder = nn.Sequential(
            # 强制使特征更明显
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # 32x32

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # 16x16

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # 8x8

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, message_bits)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ImprovedEncoder(nn.Module):
    """改进的编码器 - 将消息嵌入到图像的特定位置"""
    def __init__(self, message_bits=48, hidden_dim=256):
        super().__init__()
        self.message_bits = message_bits

        # 消息编码器 - 将48-bit消息编码为空间模式
        self.message_encoder = nn.Sequential(
            nn.Linear(message_bits, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            # 输出: 8x8x8 的空间嵌入
            nn.Linear(512, 8 * 8 * 8),
        )

        # 图像调制器
        self.modulator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(3 + 8, 64, kernel_size=3, padding=1),  # 图像(3) + 消息(8) = 11 channels
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid()  # 确保输出在 [0,1]
        )

    def embed_message(self, image, message):
        batch_size = image.shape[0]

        # 编码消息为空间模式
        msg_emb = self.message_encoder(message)  # [B, 8*8*8]
        msg_spatial = msg_emb.view(batch_size, 8, 8, 8).permute(0, 3, 1, 2)  # [B, 8, 8, 8]

        # 调整图像大小为 8x8
        image_resized = nn.functional.interpolate(image, size=(8, 8), mode='bilinear', align_corners=False)

        # 拼接
        combined = torch.cat([image_resized, msg_spatial], dim=1)  # [B, 11, 8, 8]

        # 融合
        encoded = self.fusion(combined)

        # 调整回原始大小
        encoded = nn.functional.interpolate(encoded, size=image.shape[2:], mode='bilinear', align_corners=False)

        return encoded

    def forward(self, image, message):
        return self.embed_message(image, message)


def train_phase1_decoder(
    decoder,
    dataloader,
    epochs=20,
    lr=1e-3,
    device="cuda"
):
    """阶段1: 训练解码器直接预测水印"""
    print("\n" + "="*60)
    print("Phase 1: Training Decoder to Read Watermarks")
    print("="*60)

    decoder = decoder.to(device)
    opt = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0

    for epoch in range(epochs):
        decoder.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        for images, messages in dataloader:
            images = images.to(device)
            messages = messages.to(device)

            opt.zero_grad()

            # 直接从图像预测
            pred_logits = decoder(images)
            loss = criterion(pred_logits, messages)

            loss.backward()
            opt.step()

            with torch.no_grad():
                pred_probs = torch.sigmoid(pred_logits)
                acc = calculate_bit_accuracy(pred_probs, messages)

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

        print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Bit Acc={avg_acc:.4f}")

        if avg_acc > best_acc:
            best_acc = avg_acc

    return decoder, best_acc


def train_phase2_joint(
    encoder, decoder,
    dataloader,
    epochs=30,
    lr=1e-4,
    device="cuda"
):
    """阶段2: 联合训练编码器和解码器"""
    print("\n" + "="*60)
    print("Phase 2: Joint Training (Encoder + Decoder)")
    print("="*60)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    opt = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr
    )
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0
    history = []

    for epoch in range(epochs):
        encoder.train()
        decoder.train()

        total_loss = 0
        total_acc = 0
        num_batches = 0

        for images, messages in dataloader:
            images = images.to(device)
            messages = messages.to(device)

            opt.zero_grad()

            # 编码
            watermarked = encoder(image=images, message=messages)

            # 解码
            pred_logits = decoder(watermarked)

            # 损失
            loss = criterion(pred_logits, messages)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            opt.step()

            with torch.no_grad():
                pred_probs = torch.sigmoid(pred_logits)
                acc = calculate_bit_accuracy(pred_probs, messages)

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

        print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Bit Acc={avg_acc:.4f}")

        history.append({'epoch': epoch+1, 'loss': avg_loss, 'bit_acc': avg_acc})

        if avg_acc > best_acc:
            best_acc = avg_acc

    return encoder, decoder, best_acc, history


def improved_train(
    epochs_phase1=15,
    epochs_phase2=30,
    batch_size=32,
    image_size=64,
    num_samples=500,
    device=None,
    output_dir="./outputs/improved_train"
):
    """改进的训练流程"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*70)
    print("StableWatermark - Improved Two-Phase Training")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # 数据集 - 使用更小的图像加快训练
    print("\n[1/5] Creating dataset...")
    dataset = SyntheticDataset(num_samples=num_samples, image_size=image_size, message_bits=48)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"  Dataset: {len(dataset)} samples, Batch size: {batch_size}")
    print(f"  Batches per epoch: {len(dataloader)}")

    # 阶段1: 训练解码器
    print("\n[2/5] Phase 1: Decoder Pre-training...")
    decoder = ImprovedDecoder(message_bits=48, hidden_dim=512)
    decoder, phase1_acc = train_phase1_decoder(
        decoder, dataloader,
        epochs=epochs_phase1,
        lr=2e-3,
        device=device
    )
    print(f"  Phase 1 Best Accuracy: {phase1_acc:.4f}")

    # 阶段2: 联合训练
    print("\n[3/5] Phase 2: Joint Training...")
    encoder = ImprovedEncoder(message_bits=48, hidden_dim=256)
    encoder, decoder, phase2_acc, history = train_phase2_joint(
        encoder, decoder,
        dataloader,
        epochs=epochs_phase2,
        lr=1e-4,
        device=device
    )
    print(f"  Phase 2 Best Accuracy: {phase2_acc:.4f}")

    # 保存模型
    print("\n[4/5] Saving models...")
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }, os.path.join(output_dir, 'model.pt'))

    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # 攻击测试
    print("\n[5/5] Attack Robustness Test...")
    print("-" * 50)

    encoder.eval()
    decoder.eval()

    attacks = get_all_attacks(AttackConfig(
        gaussian_noise_sigma=0.03,
        salt_pepper_prob=0.05,
        blur_kernel=3,
        crop_scale=0.8,
        jpeg_quality=75
    ))

    attack_results = {}

    with torch.no_grad():
        test_images, test_messages = next(iter(dataloader))
        test_images = test_images.to(device)
        test_messages = test_messages.to(device)

        # 编码
        watermarked = encoder(test_images, test_messages)

        # 原始准确率
        original_probs = torch.sigmoid(decoder(watermarked))
        original_acc = calculate_bit_accuracy(original_probs, test_messages)
        print(f"  Original (no attack):     {original_acc:.4f}")
        attack_results['original'] = {'bit_accuracy': float(original_acc)}

        # 测试攻击
        for name, attack_fn in sorted(attacks.items()):
            attacked = attack_fn(watermarked)
            attacked_probs = torch.sigmoid(decoder(attacked))
            acc = calculate_bit_accuracy(attacked_probs, test_messages)
            print(f"  {name:28s}: {acc:.4f}")
            attack_results[name] = {'bit_accuracy': float(acc)}

    with open(os.path.join(output_dir, 'attack_results.json'), 'w') as f:
        json.dump(attack_results, f, indent=2)

    print("\n" + "="*70)
    print("✓ Improved Training Completed!")
    print(f"  Output: {output_dir}")
    print("="*70)

    return encoder, decoder, attack_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_phase1", type=int, default=15)
    parser.add_argument("--epochs_phase2", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="./outputs/improved_train")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    improved_train(
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        batch_size=args.batch_size,
        num_samples=args.samples,
        device=device,
        output_dir=args.output_dir
    )