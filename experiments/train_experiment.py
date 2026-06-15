#!/usr/bin/env python3
"""
StableWatermark 实验训练脚本

完整训练 StableWatermark 和基线方法
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import random

from models.tri_domain_encoder import SimpleTriDomainEncoder
from models.blind_decoder import SimpleBlindDecoder
from baselines.stegastamp import StegaStampBaseline
from baselines.dwt_dct import SimpleDCTWatermark
from baselines.mbeb import MBEBWatermark
from utils.attack import get_all_attacks, AttackConfig
from utils.metrics import calculate_bit_accuracy, calculate_psnr, calculate_ssim
from modules.message_construction import RandomMessageGenerator


class COCOImageDataset(Dataset):
    """COCO 图像数据集"""

    def __init__(self, image_dir: str, image_size: int = 256, max_samples: int = None):
        self.image_dir = image_dir
        self.image_size = image_size
        self.max_samples = max_samples

        # 获取所有图像文件
        self.image_paths = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))

        if max_samples and len(self.image_paths) > max_samples:
            random.seed(42)
            self.image_paths = random.sample(self.image_paths, max_samples)

        self.transform = lambda x: x

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')

        # 调整大小
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.float32) / 255.0

        # 转换为 CHW 格式
        image = torch.from_numpy(image.transpose(2, 0, 1))

        return image


class ImageWatermarkDataset(Dataset):
    """带水印的图像数据集"""

    def __init__(
        self,
        base_dataset: Dataset,
        message_bits: int = 48,
        encoder_type: str = "stablewatermark",
        encoder_path: str = None
    ):
        self.base_dataset = base_dataset
        self.message_bits = message_bits
        self.encoder_type = encoder_type

        # 创建消息生成器
        self.msg_generator = RandomMessageGenerator(message_bits=message_bits)

        # 延迟加载编码器
        self.encoder = None
        self.baseline_wm = None

        # 对于传统方法，直接使用
        if encoder_type == "dwt_dct":
            self.baseline_wm = SimpleDCTWatermark(message_bits=message_bits)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image = self.base_dataset[idx]
        message = self.msg_generator.generate(1)[0]

        return image, message


def train_stablewatermark(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cuda",
    output_dir: str = "./outputs/stablewatermark"
):
    """训练 StableWatermark"""

    print("\n" + "="*60)
    print("Training StableWatermark")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # 初始化模型
    encoder = SimpleTriDomainEncoder(in_channels=3, message_bits=48, hidden_dim=128)
    decoder = SimpleBlindDecoder(in_channels=3, message_bits=48, hidden_dim=256)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # 优化器
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 消息生成器
    msg_generator = RandomMessageGenerator(message_bits=48)

    best_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        encoder.train()
        decoder.train()

        train_loss = 0
        train_acc = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images in pbar:
            images = images.to(device)
            B = images.shape[0]

            # 生成消息
            messages = msg_generator.generate(B, device)

            optimizer.zero_grad()

            # 嵌入 + 提取
            watermarked = encoder(images, messages)
            pred_logits, _ = decoder(watermarked)

            # 损失
            loss = nn.BCEWithLogitsLoss()(pred_logits, messages)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            # 指标
            with torch.no_grad():
                pred_probs = torch.sigmoid(pred_logits)
                acc = calculate_bit_accuracy(pred_probs, messages)

            train_loss += loss.item()
            train_acc += acc
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc:.4f}"})

        scheduler.step()

        avg_loss = train_loss / num_batches
        avg_acc = train_acc / num_batches

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_acc)

        # 验证
        if val_loader:
            val_loss, val_acc = evaluate_stablewatermark(encoder, decoder, val_loader, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict()
                }, os.path.join(output_dir, 'best_model.pt'))

            print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Train Acc={avg_acc:.4f}, Val Acc={val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Train Acc={avg_acc:.4f}")

    # 保存训练历史
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return encoder, decoder, best_acc


def evaluate_stablewatermark(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda"
):
    """评估 StableWatermark"""
    encoder.eval()
    decoder.eval()

    msg_generator = RandomMessageGenerator(message_bits=48)

    total_loss = 0
    total_acc = 0
    num_batches = 0

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            B = images.shape[0]

            messages = msg_generator.generate(B, device)
            watermarked = encoder(images, messages)
            pred_logits, _ = decoder(watermarked)

            loss = nn.BCEWithLogitsLoss()(pred_logits, messages).item()
            acc = calculate_bit_accuracy(torch.sigmoid(pred_logits), messages)

            total_loss += loss
            total_acc += acc
            num_batches += 1

    return total_loss / num_batches, total_acc / num_batches


def run_robustness_test(
    method_name: str,
    encoder,
    decoder,
    loader: DataLoader,
    device: str = "cuda"
):
    """运行鲁棒性测试"""
    print(f"\n--- {method_name} Robustness Test ---")

    msg_generator = RandomMessageGenerator(message_bits=48)
    attacks = get_all_attacks(AttackConfig(
        gaussian_noise_sigma=0.03,
        salt_pepper_prob=0.05,
        blur_kernel=5,
        crop_scale=0.8,
        jpeg_quality=75
    ))

    results = {}

    with torch.no_grad():
        # 获取一批测试数据
        images = next(iter(loader))[0].to(device)[:8]
        messages = msg_generator.generate(8, device)

        # 嵌入水印
        watermarked = encoder(images, messages)

        # 原始准确率
        pred_logits, _ = decoder(watermarked)
        clean_acc = calculate_bit_accuracy(torch.sigmoid(pred_logits), messages)
        results['Clean'] = clean_acc

        # 对各种攻击的鲁棒性
        for name, attack_fn in attacks.items():
            attacked = attack_fn(watermarked)
            pred_logits, _ = decoder(attacked)
            acc = calculate_bit_accuracy(torch.sigmoid(pred_logits), messages)
            results[name] = acc

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/coco")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="./outputs/experiments")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Samples: {args.num_samples}")

    # 检查数据集是否可用
    if os.path.exists(args.data_dir):
        dataset = COCOImageDataset(
            image_dir=args.data_dir,
            image_size=args.image_size,
            max_samples=args.num_samples
        )
        print(f"Loaded {len(dataset)} images from {args.data_dir}")
    else:
        print(f"Warning: {args.data_dir} not found. Using synthetic data.")
        from data.dataset import SyntheticDataset
        dataset = SyntheticDataset(num_samples=args.num_samples, image_size=args.image_size)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 训练 StableWatermark
    os.makedirs(args.output_dir, exist_ok=True)
    encoder, decoder, best_acc = train_stablewatermark(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir
    )

    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    # 鲁棒性测试
    results = run_robustness_test("StableWatermark", encoder, decoder, val_loader, device)

    # 保存结果
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    for name, acc in results.items():
        print(f"  {name}: {acc:.4f}")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()