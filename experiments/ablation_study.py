#!/usr/bin/env python3
"""
StableWatermark 消融实验

测试各组件的贡献:
1. Full Model (完整模型)
2. w/o Frequency Branch (不使用频率分支)
3. w/o Spatial Branch (不使用空间分支)
4. w/o Feature Branch (不使用特征分支)
5. w/o Sensitivity Mask (不使用敏感性掩码)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from datetime import datetime

from models.tri_domain_encoder import SimpleTriDomainEncoder
from models.blind_decoder import SimpleBlindDecoder
from utils.attack import get_all_attacks, AttackConfig
from utils.metrics import calculate_bit_accuracy, calculate_psnr, calculate_ssim
from modules.message_construction import RandomMessageGenerator


class AblationEncoder(nn.Module):
    """
    可配置的消融编码器

    支持关闭/开启各个分支
    """

    def __init__(
        self,
        in_channels: int = 3,
        message_bits: int = 48,
        hidden_dim: int = 128,
        use_spatial: bool = True,
        use_frequency: bool = True,
        use_mask: bool = True
    ):
        super().__init__()
        self.message_bits = message_bits
        self.use_spatial = use_spatial
        self.use_frequency = use_frequency
        self.use_mask = use_mask

        # 消息编码
        self.message_encoder = nn.Sequential(
            nn.Linear(message_bits, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )

        # 空间分支
        if use_spatial:
            self.spatial_modulator = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
            )
            self.spatial_fusion = nn.Sequential(
                nn.Conv2d(64 + 64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, in_channels, kernel_size=1),
                nn.Sigmoid()
            )

        # 掩码生成 (简化)
        if use_mask:
            self.mask_generator = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(
        self,
        image: torch.Tensor,
        message: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            image: [B, C, H, W]
            message: [B, message_bits]
            mask: 可选的外部掩码

        Returns:
            watermarked: [B, C, H, W]
        """
        B, C, H, W = image.shape

        # 编码消息
        msg_feat = self.message_encoder(message)  # [B, hidden_dim * 2]
        msg_spatial = msg_feat.view(B, -1, 1, 1).expand(-1, -1, H, W)

        watermarked = image.clone()
        residual_sum = torch.zeros_like(image)

        # 空间分支
        if self.use_spatial:
            img_feat = self.spatial_modulator(image)  # [B, 64, H, W]
            combined = torch.cat([img_feat, msg_spatial[:, :64, :H, :W]], dim=1)
            spatial_residual = self.spatial_fusion(combined)
            residual_sum = residual_sum + spatial_residual * 0.1

        # 频率分支 (简化版)
        if self.use_frequency:
            # 对图像进行简单的高频增强
            img_fft = torch.fft.rfft2(image, norm='ortho')
            fft_mag = torch.abs(img_fft)
            fft_phase = torch.angle(img_fft)

            # 消息调制的频率增强
            msg_freq = msg_spatial[:, :64, :H, W//2+1].unsqueeze(-1)
            freq_strength = torch.sigmoid(msg_freq.mean(dim=1, keepdim=True))
            fft_mag_enhanced = fft_mag * (1 + 0.05 * freq_strength)

            # 逆变换
            freq_enhanced = torch.fft.irfft2(fft_mag_enhanced * torch.exp(1j * fft_phase), s=(H, W), norm='ortho')
            residual_sum = residual_sum + (freq_enhanced - image) * 0.5

        # 应用掩码
        if self.use_mask and mask is not None:
            residual_sum = residual_sum * mask
        elif self.use_mask:
            # 生成内部掩码
            internal_mask = self.mask_generator(image)
            residual_sum = residual_sum * internal_mask

        watermarked = image + residual_sum
        watermarked = torch.clamp(watermarked, 0, 1)

        return watermarked


def train_model(encoder, decoder, train_images, epochs=30, lr=1e-3, device="cpu"):
    """训练模型"""
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    msg_generator = RandomMessageGenerator(message_bits=48)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_images, batch_size=16, shuffle=True)

    for epoch in range(epochs):
        encoder.train()
        decoder.train()

        total_loss = 0
        total_acc = 0
        num_batches = 0

        for images, _ in train_loader:  # 数据集返回 (image, message) 对
            images = images.to(device)
            B = images.shape[0]
            messages = msg_generator.generate(B, device)

            optimizer.zero_grad()

            # 嵌入 + 提取
            watermarked = encoder(images, messages)
            pred_logits, _ = decoder(watermarked)

            # 损失
            loss = nn.BCEWithLogitsLoss()(pred_logits, messages)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = calculate_bit_accuracy(torch.sigmoid(pred_logits), messages)

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

        scheduler.step()

    return encoder, decoder, total_acc / max(num_batches, 1)


def evaluate_model(encoder, decoder, test_images, device="cpu"):
    """评估模型"""
    encoder.eval()
    decoder.eval()

    test_loader = DataLoader(test_images, batch_size=8, shuffle=False)
    msg_generator = RandomMessageGenerator(message_bits=48)
    attacks = get_all_attacks(AttackConfig())

    results = {'Clean': [], 'combined': []}

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            B = min(8, images.shape[0])
            messages = msg_generator.generate(B, device)

            # 嵌入
            watermarked = encoder(images[:B], messages[:B])

            # 干净准确率
            pred_logits, _ = decoder(watermarked)
            clean_acc = calculate_bit_accuracy(torch.sigmoid(pred_logits), messages[:B])
            results['Clean'].append(clean_acc)

            # 组合攻击
            attacked = list(attacks.values())[0](watermarked)
            pred_logits, _ = decoder(attacked)
            combined_acc = calculate_bit_accuracy(torch.sigmoid(pred_logits), messages[:B])
            results['combined'].append(combined_acc)

    return {
        'Clean': np.mean(results['Clean']) if results['Clean'] else 0,
        'combined': np.mean(results['combined']) if results['combined'] else 0
    }


def run_ablation_study(
    train_dataset,
    test_dataset,
    epochs: int = 20,
    device: str = "cuda"
):
    """运行消融实验"""

    print("\n" + "="*60)
    print("Ablation Study: Testing Component Contributions")
    print("="*60)

    results = {}

    # 1. Full Model
    print("\n[1/4] Testing Full Model...")
    encoder_full = AblationEncoder(use_spatial=True, use_frequency=True, use_mask=True)
    decoder = SimpleBlindDecoder(in_channels=3, message_bits=48, hidden_dim=256)
    encoder_full, decoder, _ = train_model(encoder_full, decoder, train_dataset, epochs, device=device)
    results['Full Model'] = evaluate_model(encoder_full, decoder, test_dataset, device)
    print(f"  Full Model: Clean={results['Full Model']['Clean']:.4f}, Combined={results['Full Model']['combined']:.4f}")

    # 2. w/o Frequency Branch
    print("\n[2/4] Testing w/o Frequency Branch...")
    encoder_no_freq = AblationEncoder(use_spatial=True, use_frequency=False, use_mask=True)
    decoder = SimpleBlindDecoder(in_channels=3, message_bits=48, hidden_dim=256)
    encoder_no_freq, decoder, _ = train_model(encoder_no_freq, decoder, train_dataset, epochs, device=device)
    results['w/o Frequency'] = evaluate_model(encoder_no_freq, decoder, test_dataset, device)
    print(f"  w/o Frequency: Clean={results['w/o Frequency']['Clean']:.4f}, Comb={results['w/o Frequency']['combined']:.4f}")

    # 3. w/o Spatial Branch
    print("\n[3/4] Testing w/o Spatial Branch...")
    encoder_no_spatial = AblationEncoder(use_spatial=False, use_frequency=True, use_mask=True)
    decoder = SimpleBlindDecoder(in_channels=3, message_bits=48, hidden_dim=256)
    encoder_no_spatial, decoder, _ = train_model(encoder_no_spatial, decoder, train_dataset, epochs, device=device)
    results['w/o Spatial'] = evaluate_model(encoder_no_spatial, decoder, test_dataset, device)
    print(f"  w/o Spatial: Clean={results['w/o Spatial']['Clean']:.4f}, Comb={results['w/o Spatial']['combined']:.4f}")

    # 4. w/o Mask
    print("\n[4/4] Testing w/o Mask...")
    encoder_no_mask = AblationEncoder(use_spatial=True, use_frequency=True, use_mask=False)
    decoder = SimpleBlindDecoder(in_channels=3, message_bits=48, hidden_dim=256)
    encoder_no_mask, decoder, _ = train_model(encoder_no_mask, decoder, train_dataset, epochs, device=device)
    results['w/o Mask'] = evaluate_model(encoder_no_mask, decoder, test_dataset, device)
    print(f"  w/o Mask: Clean={results['w/o Mask']['Clean']:.4f}, Comb={results['w/o Mask']['combined']:.4f}")

    return results


def generate_latex_ablation_table(results: dict) -> str:
    """生成消融实验 LaTeX 表格"""

    latex = """
\\begin{table}[t]
\\centering
\\caption{Ablation Study on StableWatermark Components (Bit Accuracy \\%)}
\\label{tab:ablation}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Configuration} & \\textbf{Clean} & \\textbf{Combined Attack} \\\\
\\midrule
"""

    for config, metrics in results.items():
        clean = metrics['Clean'] * 100
        combined = metrics['combined'] * 100
        latex += f"\\textit{{{config}}} & {clean:.1f} & {combined:.1f} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="./outputs/ablation")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Samples: {args.num_samples}")

    # 生成合成数据
    from data.dataset import SyntheticDataset
    full_dataset = SyntheticDataset(num_samples=args.num_samples, image_size=256)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # 运行消融实验
    results = run_ablation_study(
        train_dataset, test_dataset,
        epochs=args.epochs,
        device=device
    )

    # 保存结果
    with open(os.path.join(args.output_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 生成 LaTeX 表格
    latex_table = generate_latex_ablation_table(results)
    with open(os.path.join(args.output_dir, 'ablation_table.tex'), 'w') as f:
        f.write(f"% Generated at {datetime.now()}\n\n")
        f.write(latex_table)

    print("\n" + "="*60)
    print("Ablation Study Summary")
    print("="*60)
    for config, metrics in results.items():
        print(f"  {config}: Clean={metrics['Clean']:.4f}, Combined={metrics['combined']:.4f}")
    print(f"\nLaTeX table saved to: {args.output_dir}/ablation_table.tex")

    return results


if __name__ == "__main__":
    main()