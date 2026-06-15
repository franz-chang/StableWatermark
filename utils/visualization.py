"""
可视化工具

用于可视化水印嵌入效果和攻击结果
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import List, Union, Optional, Dict
import os


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """将 tensor 转换为 numpy 图像"""
    img = tensor.cpu().detach()
    if img.dim() == 4:
        img = img[0]
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img


def visualize_watermark(
    original: torch.Tensor,
    watermarked: torch.Tensor,
    predicted_bits: Optional[torch.Tensor] = None,
    target_bits: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    title: str = "Watermark Visualization"
) -> plt.Figure:
    """
    可视化水印嵌入效果

    Args:
        original: 原始图像
        watermarked: 含水印图像
        predicted_bits: 预测的水印比特
        target_bits: 目标水印比特
        save_path: 保存路径
        title: 标题

    Returns:
        fig: matplotlib 图形
    """
    fig, axes = plt.subplots(1, 4 if predicted_bits is not None else 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)

    # 原始图像
    axes[0].imshow(tensor_to_image(original))
    axes[0].set_title("Original")
    axes[0].axis('off')

    # 含水印图像
    axes[1].imshow(tensor_to_image(watermarked))
    axes[1].set_title("Watermarked")
    axes[1].axis('off')

    # 差异图
    diff = (watermarked - original).abs().mean(dim=1, keepdim=True)
    diff_img = tensor_to_image(diff.expand(-1, 3, -1, -1))
    axes[2].imshow(diff_img)
    axes[2].set_title("Difference (amplified)")
    axes[2].axis('off')

    # 水印比特
    if predicted_bits is not None and target_bits is not None:
        axes[3].clear()
        bits_to_plot = 48
        x = np.arange(bits_to_plot)

        pred = predicted_bits[:bits_to_plot].cpu().numpy()
        targets = target_bits[:bits_to_plot].cpu().numpy()

        width = 0.35
        axes[3].bar(x - width/2, targets, width, label='Target', alpha=0.7)
        axes[3].bar(x + width/2, pred, width, label='Predicted', alpha=0.7)
        axes[3].set_xlabel('Bit Position')
        axes[3].set_ylabel('Value')
        axes[3].set_title('Watermark Bits')
        axes[3].legend()
        axes[3].set_ylim(-0.1, 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    return fig


def visualize_attacks(
    original: torch.Tensor,
    attacked_images: Dict[str, torch.Tensor],
    bit_accuracies: Dict[str, float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    可视化各种攻击的效果

    Args:
        original: 原始图像
        attacked_images: 攻击后的图像字典 {attack_name: image}
        bit_accuracies: 各个攻击下的比特准确率
        save_path: 保存路径

    Returns:
        fig: matplotlib 图形
    """
    num_attacks = len(attacked_images)
    fig, axes = plt.subplots(2, num_attacks + 1, figsize=(4 * (num_attacks + 1), 8))

    # 第一行: 原始图像 + 各种攻击后的图像
    axes[0, 0].imshow(tensor_to_image(original))
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')

    for i, (attack_name, attacked) in enumerate(attacked_images.items()):
        axes[0, i + 1].imshow(tensor_to_image(attacked))
        axes[0, i + 1].set_title(f"{attack_name}")
        axes[0, i + 1].axis('off')

    # 第二行: 差异图
    axes[1, 0].imshow(np.zeros_like(tensor_to_image(original)))
    axes[1, 0].set_title("Original")
    axes[1, 0].axis('off')

    for i, (attack_name, attacked) in enumerate(attacked_images.items()):
        diff = (attacked - original).abs().mean(dim=1, keepdim=True)
        diff_img = tensor_to_image(diff.expand(-1, 3, -1, -1))
        axes[1, i + 1].imshow(diff_img)
        acc = bit_accuracies.get(attack_name, 0)
        axes[1, i + 1].set_title(f"Bit Acc: {acc:.2%}")
        axes[1, i + 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attack visualization to {save_path}")

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制训练曲线

    Args:
        history: 训练历史 {metric_name: [values]}
        save_path: 保存路径

    Returns:
        fig: matplotlib 图形
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Training Curves", fontsize=14)

    # 损失曲线
    if 'loss' in history:
        ax = axes[0, 0]
        ax.plot(history['loss'], label='Total Loss', linewidth=2)
        if 'loss_rec' in history:
            ax.plot(history['loss_rec'], label='Reconstruction Loss', alpha=0.7)
        if 'loss_msg' in history:
            ax.plot(history['loss_msg'], label='Message Loss', alpha=0.7)
        if 'loss_adv' in history:
            ax.plot(history['loss_adv'], label='Adversarial Loss', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 比特准确率
    if 'bit_accuracy' in history:
        ax = axes[0, 1]
        ax.plot(history['bit_accuracy'], linewidth=2, color='green')
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.set_title('Bit Accuracy')
        ax.grid(True, alpha=0.3)

    # 消息准确率
    if 'message_accuracy' in history:
        ax = axes[1, 0]
        ax.plot(history['message_accuracy'], linewidth=2, color='purple')
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.set_title('Message Accuracy')
        ax.grid(True, alpha=0.3)

    # 图像质量指标
    if 'psnr' in history or 'ssim' in history:
        ax = axes[1, 1]
        if 'psnr' in history:
            ax.plot(history['psnr'], label='PSNR (dB)', linewidth=2)
        if 'ssim' in history:
            ax2 = ax.twinx()
            ax2.plot(history['ssim'], label='SSIM', linewidth=2, color='orange')
            ax2.set_ylabel('SSIM')
        ax.set_xlabel('Step')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Image Quality')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    return fig


def plot_attack_results(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制攻击测试结果表格

    Args:
        results: 结果字典 {attack_name: {metric: value}}
        save_path: 保存路径

    Returns:
        fig: matplotlib 图形
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # 创建表格数据
    attack_names = list(results.keys())
    metrics = list(results[attack_names[0]].keys())

    table_data = []
    for attack_name in attack_names:
        row = [attack_name]
        for metric in metrics:
            value = results[attack_name].get(metric, 0)
            if metric in ['bit_accuracy', 'message_accuracy']:
                row.append(f"{value:.2%}")
            elif metric in ['psnr']:
                row.append(f"{value:.2f} dB")
            else:
                row.append(f"{value:.4f}")
        table_data.append(row)

    # 创建表格
    col_labels = ['Attack'] + [m.replace('_', ' ').title() for m in metrics]
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # 设置表格样式
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', weight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#D9E2F3')

    plt.title('Robustness Evaluation Results', fontsize=14, pad=20)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attack results to {save_path}")

    return fig


def save_image_grid(
    images: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True
):
    """
    保存图像网格

    Args:
        images: 图像tensor [batch_size, channels, height, width]
        save_path: 保存路径
        nrow: 每行图像数量
        padding: 间距
        normalize: 是否归一化
    """
    from torchvision.utils import save_image as tv_save_image

    tv_save_image(
        images,
        save_path,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=(0, 1)
    )


def test_visualization():
    """测试可视化功能"""
    print("Testing visualization...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建测试数据
    original = torch.rand(1, 3, 256, 256, device=device)
    watermarked = original + torch.randn_like(original) * 0.02
    watermarked = torch.clamp(watermarked, 0, 1)

    predicted_bits = torch.rand(48, device=device)
    target_bits = torch.randint(0, 2, (48,), device=device).float()

    # 测试基本可视化
    fig = visualize_watermark(
        original, watermarked,
        predicted_bits, target_bits,
        title="Test Watermark Visualization"
    )
    print(f"  Created visualization figure")

    # 测试训练曲线
    history = {
        'loss': np.random.randn(100).cumsum(),
        'bit_accuracy': np.random.rand(100) * 0.1 + 0.9,
        'psnr': np.random.rand(100) * 5 + 25,
        'ssim': np.random.rand(100) * 0.1 + 0.95
    }
    fig = plot_training_curves(history)
    print(f"  Created training curves")

    # 测试攻击结果表格
    results = {
        'Gaussian Noise': {'bit_accuracy': 0.95, 'psnr': 28.5, 'ssim': 0.92},
        'JPEG Compression': {'bit_accuracy': 0.87, 'psnr': 30.2, 'ssim': 0.95},
        'Center Crop': {'bit_accuracy': 0.72, 'psnr': 22.1, 'ssim': 0.78},
    }
    fig = plot_attack_results(results)
    print(f"  Created attack results table")

    plt.close('all')
    print("✓ Visualization tests passed")


if __name__ == "__main__":
    test_visualization()