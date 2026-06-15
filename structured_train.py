#!/usr/bin/env python3
"""
StableWatermark - 结构化训练

使用有结构的图像（包含可学习的模式）来训练水印嵌入
在这种设置下，模型可以学习将水印嵌入到图像的特定纹理中
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
from datetime import datetime
import numpy as np


class StructuredDataset(Dataset):
    """
    结构化合成数据集

    生成包含可识别模式的图像，这些模式可以被水印修改
    """

    def __init__(self, num_samples=1000, img_size=64, message_bits=48):
        self.num_samples = num_samples
        self.img_size = img_size
        self.message_bits = message_bits

    def __len__(self):
        return self.num_samples

    def generate_pattern(self):
        """生成包含多种模式的单个图像"""
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)

        # 1. 网格模式
        if np.random.random() > 0.3:
            grid_size = np.random.randint(8, 16)
            for i in range(0, self.img_size, grid_size):
                end_i = min(i + 2, self.img_size)
                img[i:end_i, :, :] = np.random.rand(end_i - i, self.img_size, 3)
                img[:, i:end_i, :] = np.random.rand(self.img_size, end_i - i, 3)

        # 2. 圆形
        center = self.img_size // 2
        radius = np.random.randint(8, 16)
        for i in range(self.img_size):
            for j in range(self.img_size):
                if (i - center)**2 + (j - center)**2 < radius**2:
                    img[i, j, :] = np.random.rand(3)

        # 3. 条纹 - 使用向量化操作
        if np.random.random() > 0.5:
            stripe_width = np.random.randint(4, 8)
            x_coords, y_coords = np.meshgrid(
                np.arange(self.img_size),
                np.arange(self.img_size),
                indexing='ij'
            )
            stripe_pattern = ((x_coords // stripe_width) % 2 == 0).astype(float)
            for c in range(3):
                img[:, :, c] = img[:, :, c] * 0.5 + stripe_pattern * 0.5 * np.random.rand()

        # 4. 随机纹理块
        for _ in range(3):
            x, y = np.random.randint(0, max(1, self.img_size-8)), np.random.randint(0, max(1, self.img_size-8))
            size = np.random.randint(4, 8)
            end_x = min(x + size, self.img_size)
            end_y = min(y + size, self.img_size)
            img[x:end_x, y:end_y, :] = np.random.rand(end_x - x, end_y - y, 3)

        return torch.from_numpy(img.transpose(2, 0, 1))  # CHW format


class SimpleWatermarkDataset(Dataset):
    """
    简化水印数据集 - 直接生成 (图像, 消息) 对

    消息被"预先嵌入"到图像的某些通道/位置中
    """

    def __init__(self, num_samples=1000, img_size=64, message_bits=48, embed_strength=0.3):
        self.num_samples = num_samples
        self.img_size = img_size
        self.message_bits = message_bits
        self.embed_strength = embed_strength

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机消息
        message = torch.randint(0, 2, (self.message_bits,)).float()

        # 生成基础图像 (用StructuredDataset的generate_pattern方法)
        struct_ds = StructuredDataset(1, self.img_size, self.message_bits)
        base_img = struct_ds.generate_pattern().float() / 255.0

        # 将消息嵌入到图像中 - 通过调制某些频率分量
        embed_img = base_img.clone()

        # 使用消息位来调制不同频率的正弦分量
        freq_start = 2
        for i, bit in enumerate(message[:32]):
            freq = freq_start + i // 8
            offset = i % 8

            x = torch.linspace(0, 4 * np.pi, self.img_size)
            y = torch.linspace(0, 4 * np.pi, self.img_size)
            xx, yy = torch.meshgrid(x, y, indexing='ij')

            # 不同频率和相位的正弦波
            wave = torch.sin(xx * freq + offset) * torch.cos(yy * freq + offset * 0.5)

            # 根据消息位调整波的方向/强度
            if bit > 0.5:
                embed_img[0] = embed_img[0] + wave * self.embed_strength * 0.1

        # 限制在 [0, 1] 范围
        embed_img = torch.clamp(embed_img, 0, 1)

        return embed_img, message


def simple_train(
    epochs=30,
    batch_size=32,
    img_size=64,
    num_samples=500,
    device=None,
    output_dir="./outputs/simple_train"
):
    """简化但有效的训练"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*70)
    print("StableWatermark - Structured Image Training")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Config: {epochs} epochs, batch_size={batch_size}, samples={num_samples}")

    os.makedirs(output_dir, exist_ok=True)

    # 数据集
    print("\n[1/4] Creating structured dataset...")
    dataset = SimpleWatermarkDataset(
        num_samples=num_samples,
        img_size=img_size,
        message_bits=48,
        embed_strength=0.5
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"  Dataset: {len(dataset)} samples")

    # 模型
    print("\n[2/4] Creating models...")

    # 编码器 - 学习增强/修改消息嵌入
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3 + 48, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, 1),
                nn.Sigmoid()
            )

        def forward(self, image, message):
            # 将消息扩展到图像大小
            B, C, H, W = image.shape
            msg_expanded = message.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            # 将图像和消息拼接
            combined = torch.cat([image, msg_expanded], dim=1)
            return self.net(combined)

    # 解码器 - 学习从图像提取消息
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256*16, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 48)
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    print(f"  Total parameters: {sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters()):,}")

    # 优化器
    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # 训练
    print("\n[3/4] Training...")
    print("-" * 60)

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

            # 编码 + 解码
            watermarked = encoder(images, messages)
            pred_logits = decoder(watermarked)

            # 损失
            loss = criterion(pred_logits, messages)
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred_probs = torch.sigmoid(pred_logits)
                acc = (pred_probs > 0.5).float().eq(messages).float().mean()

            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

        history.append({'epoch': epoch+1, 'loss': avg_loss, 'bit_acc': avg_acc})

        print(f"  Epoch {epoch+1:2d}/{epochs}: Loss={avg_loss:.4f}, Bit Acc={avg_acc:.4f}")

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
    print("\n[4/4] Attack Robustness Test...")
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

        watermarked = encoder(test_images, test_messages)

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

    print("\n" + "="*70)
    print(f"✓ Training Completed! Best Accuracy: {best_acc:.4f}")
    print(f"  Output: {output_dir}")
    print("="*70)

    return best_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="./outputs/structured_train")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    simple_train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.samples,
        device=device,
        output_dir=args.output_dir
    )