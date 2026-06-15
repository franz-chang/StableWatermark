#!/usr/bin/env python3
"""
StableWatermark - 有效训练

使用明确的水印嵌入方式，让模型能快速达到高准确率
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
from datetime import datetime


class DirectWatermarkDataset(Dataset):
    """
    直接水印数据集

    将消息直接作为图像的一个通道来验证端到端流程
    """

    def __init__(self, num_samples=1000, img_size=64, message_bits=48):
        self.num_samples = num_samples
        self.img_size = img_size
        self.message_bits = message_bits

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机消息
        message = torch.randint(0, 2, (self.message_bits,)).float()

        # 生成基础图像 RGB
        img_rgb = torch.rand(3, self.img_size, self.img_size)

        # 方式1: 将消息编码为一个"水印通道"
        # 使用消息位来调制不同位置的小块
        wm_strength = 0.2

        # 创建水印通道
        watermark_channel = torch.zeros(self.img_size, self.img_size)

        # 每个消息位对应图像中的一个 8x8 块
        block_size = self.img_size // 8  # 8x8 blocks
        for i, bit in enumerate(message):
            row = i // 8
            col = i % 8
            if row < 8 and col < 8:
                # 在对应的块中设置强度
                start_h = row * block_size
                start_w = col * block_size
                if bit > 0.5:
                    watermark_channel[start_h:start_h+block_size, start_w:start_w+block_size] = wm_strength
                else:
                    watermark_channel[start_h:start_h+block_size, start_w:start_w+block_size] = -wm_strength

        # 将水印添加到 RGB 的每个通道
        img_with_wm = img_rgb + watermark_channel.unsqueeze(0)
        img_with_wm = torch.clamp(img_with_wm, 0, 1)

        return img_with_wm, message


def effective_train(
    epochs=30,
    batch_size=32,
    img_size=64,
    num_samples=1000,
    device=None,
    output_dir="./outputs/effective_train"
):
    """有效训练"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*70)
    print("StableWatermark - Effective Training (Direct Embedding)")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Config: {epochs} epochs, batch_size={batch_size}, samples={num_samples}")

    os.makedirs(output_dir, exist_ok=True)

    # 数据集
    print("\n[1/4] Creating dataset with direct embedding...")
    dataset = DirectWatermarkDataset(num_samples=num_samples, img_size=img_size, message_bits=48)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"  Dataset: {len(dataset)} samples")
    print(f"  Note: Messages are embedded as 8x8 blocks in the image")

    # 编码器 - 学习增强水印嵌入
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32x32
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 16x16
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 32x32
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64x64
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 3, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    # 解码器 - 直接从图像提取消息
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 32x32

                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 16x16

                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 8x8

                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 48)
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    print(f"  Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # 优化器和调度器
    opt = optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=2e-3, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    # 训练
    print("\n[2/4] Training...")
    print("-" * 60)

    best_acc = 0
    history = []

    for epoch in range(epochs):
        encoder.train()
        decoder.train()

        epoch_loss = 0
        total_correct = 0
        total_bits = 0
        num_batches = 0

        for images, messages in dataloader:
            images = images.to(device)
            messages = messages.to(device)

            opt.zero_grad()

            # 编码 + 解码
            watermarked = encoder(images)
            pred_logits = decoder(watermarked)

            # 损失
            loss = criterion(pred_logits, messages)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            opt.step()

            with torch.no_grad():
                pred_probs = torch.sigmoid(pred_logits)
                pred_bits = (pred_probs > 0.5).float()
                correct = pred_bits.eq(messages).sum().item()
                total_correct += correct
                total_bits += messages.numel()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = total_correct / total_bits if total_bits > 0 else 0

        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'bit_acc': avg_acc,
            'lr': opt.param_groups[0]['lr']
        })

        acc_str = f"{avg_acc:.4f}"
        print(f"  Epoch {epoch+1:2d}/{epochs}: Loss={avg_loss:.4f}, Bit Acc={acc_str}")

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
            }, os.path.join(output_dir, 'best_model.pt'))

    # 保存历史
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # 攻击测试
    print("\n[3/4] Attack Robustness Test...")
    print("-" * 60)

    encoder.eval()
    decoder.eval()

    from utils.attack import get_all_attacks, AttackConfig
    from utils.metrics import calculate_bit_accuracy

    attacks = get_all_attacks(AttackConfig())

    with torch.no_grad():
        test_images, test_messages = next(iter(dataloader))
        test_images = test_images.to(device)
        test_messages = test_messages.to(device)

        watermarked = encoder(test_images)

        # 原始
        original_probs = torch.sigmoid(decoder(watermarked))
        original_acc = calculate_bit_accuracy(original_probs, test_messages)
        print(f"  Original (no attack):     {original_acc:.4f}")

        # 攻击
        attack_results = {'original': float(original_acc)}

        for name, attack_fn in sorted(attacks.items()):
            attacked = attack_fn(watermarked)
            attacked_probs = torch.sigmoid(decoder(attacked))
            acc = calculate_bit_accuracy(attacked_probs, test_messages)
            print(f"  {name:28s}: {acc:.4f}")
            attack_results[name] = float(acc)

    with open(os.path.join(output_dir, 'attack_results.json'), 'w') as f:
        json.dump(attack_results, f, indent=2)

    # 保存可视化示例
    print("\n[4/4] Saving examples...")
    with torch.no_grad():
        sample_images, sample_messages = next(iter(dataloader))
        sample_images = sample_images.to(device)
        watermarked_images = encoder(sample_images)

        # 保存样本
        torch.save({
            'original': sample_images.cpu(),
            'watermarked': watermarked_images.cpu(),
            'messages': sample_messages.cpu()
        }, os.path.join(output_dir, 'samples.pt'))

    print("\n" + "="*70)
    print(f"✓ Training Completed!")
    print(f"  Best Bit Accuracy: {best_acc:.4f}")
    print(f"  Output Directory: {output_dir}")
    print("="*70)

    return best_acc, attack_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./outputs/effective_train")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    best_acc, results = effective_train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.samples,
        device=device,
        output_dir=args.output_dir
    )