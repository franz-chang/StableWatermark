"""
评估指标模块

用于评估水印的质量和鲁棒性
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Tuple, Optional


def calculate_bit_accuracy(
    predicted: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    计算比特准确率

    Args:
        predicted: 预测概率 [batch_size, num_bits]
        target: 目标比特 [batch_size, num_bits] (0 或 1)
        threshold: 二值化阈值

    Returns:
        accuracy: 比特准确率 (0-1)
    """
    predicted_bits = (predicted > threshold).float()
    correct = (predicted_bits == target).float().mean()
    return correct.item()


def calculate_message_accuracy(
    predicted: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    计算完整消息准确率

    一个消息只有在所有比特都正确时才认为正确

    Args:
        predicted: 预测概率
        target: 目标比特

    Returns:
        accuracy: 消息级别的准确率
    """
    predicted_bits = (predicted > threshold).float()
    correct_per_message = (predicted_bits == target).all(dim=1).float()
    return correct_per_message.mean().item()


def calculate_bit_error_rate(
    predicted: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    计算比特错误率

    Args:
        predicted: 预测概率
        target: 目标比特

    Returns:
        ber: 比特错误率 (0-1)
    """
    predicted_bits = (predicted > threshold).float()
    errors = (predicted_bits != target).float().mean()
    return errors.item()


def calculate_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0
) -> float:
    """
    计算峰值信噪比

    Args:
        img1: 图像1 [batch_size, channels, height, width]
        img2: 图像2
        data_range: 数据范围

    Returns:
        psnr: 平均 PSNR 值 (dB)
    """
    batch_size = img1.shape[0]
    psnr_values = []

    for i in range(batch_size):
        img1_np = img1[i].cpu().detach().numpy().transpose(1, 2, 0)
        img2_np = img2[i].cpu().detach().numpy().transpose(1, 2, 0)

        p = psnr(img1_np, img2_np, data_range=data_range)
        psnr_values.append(p)

    return np.mean(psnr_values)


def calculate_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0
) -> float:
    """
    计算结构相似性指数

    Args:
        img1: 图像1 [batch_size, channels, height, width]
        img2: 图像2
        data_range: 数据范围

    Returns:
        ssim: 平均 SSIM 值
    """
    batch_size = img1.shape[0]
    ssim_values = []

    for i in range(batch_size):
        img1_np = img1[i].cpu().detach().numpy().transpose(1, 2, 0)
        img2_np = img2[i].cpu().detach().numpy().transpose(1, 2, 0)

        # 处理灰度图像
        if img1_np.shape[2] == 1:
            img1_np = img1_np.squeeze(2)
            img2_np = img2_np.squeeze(2)
            s = ssim(img1_np, img2_np, data_range=data_range)
        else:
            s = ssim(img1_np, img2_np, channel_axis=2, data_range=data_range)

        ssim_values.append(s)

    return np.mean(ssim_values)


def calculate_lpips(
    img1: torch.Tensor,
    img2: torch.Tensor
) -> float:
    """
    计算感知距离 (简化版 LPIPS)

    使用 VGG 特征距离作为近似

    Note: 这里只是一个简化实现，实际应该使用预训练的 VGG 网络
    """
    # 简化：使用 L2 距离作为替代
    diff = (img1 - img2) ** 2
    return diff.mean().item()


def calculate_mse(
    img1: torch.Tensor,
    img2: torch.Tensor
) -> float:
    """计算均方误差"""
    return F.mse_loss(img1, img2).item()


def calculate_mae(
    img1: torch.Tensor,
    img2: torch.Tensor
) -> float:
    """计算平均绝对误差"""
    return F.l1_loss(img1, img2).item()


class MetricsTracker:
    """指标跟踪器"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有指标"""
        self.bit_accuracy = []
        self.message_accuracy = []
        self.bit_error_rate = []
        self.psnr = []
        self.ssim = []
        self.mse = []

    def update(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        watermarked: Optional[torch.Tensor] = None,
        original: Optional[torch.Tensor] = None
    ):
        """更新指标"""
        self.bit_accuracy.append(calculate_bit_accuracy(predicted, target))
        self.message_accuracy.append(calculate_message_accuracy(predicted, target))
        self.bit_error_rate.append(calculate_bit_error_rate(predicted, target))

        if watermarked is not None and original is not None:
            self.psnr.append(calculate_psnr(watermarked, original))
            self.ssim.append(calculate_ssim(watermarked, original))
            self.mse.append(calculate_mse(watermarked, original))

    def get_summary(self) -> dict:
        """获取指标摘要"""
        result = {}

        if self.bit_accuracy:
            result['bit_accuracy'] = np.mean(self.bit_accuracy)
            result['bit_accuracy_std'] = np.std(self.bit_accuracy)

        if self.message_accuracy:
            result['message_accuracy'] = np.mean(self.message_accuracy)
            result['message_accuracy_std'] = np.std(self.message_accuracy)

        if self.bit_error_rate:
            result['bit_error_rate'] = np.mean(self.bit_error_rate)
            result['bit_error_rate_std'] = np.std(self.bit_error_rate)

        if self.psnr:
            result['psnr'] = np.mean(self.psnr)
            result['psnr_std'] = np.std(self.psnr)

        if self.ssim:
            result['ssim'] = np.mean(self.ssim)
            result['ssim_std'] = np.std(self.ssim)

        if self.mse:
            result['mse'] = np.mean(self.mse)
            result['mse_std'] = np.std(self.mse)

        return result


def evaluate_watermark_performance(
    encoder,
    decoder,
    images: torch.Tensor,
    messages: torch.Tensor,
    device: str = "cuda"
) -> dict:
    """
    评估端到端水印性能

    Args:
        encoder: 水印编码器
        decoder: 水印解码器
        images: 原始图像
        messages: 水印消息
        device: 设备

    Returns:
        metrics: 性能指标字典
    """
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # 编码
        watermarked = encoder(images, messages)

        # 解码
        predicted_logits = decoder(watermarked)
        predicted_probs = torch.sigmoid(predicted_logits)

        # 计算指标
        bit_acc = calculate_bit_accuracy(predicted_probs, messages)
        msg_acc = calculate_message_accuracy(predicted_probs, messages)
        psnr_val = calculate_psnr(watermarked, images)
        ssim_val = calculate_ssim(watermarked, images)

    return {
        'bit_accuracy': bit_acc,
        'message_accuracy': msg_acc,
        'psnr': psnr_val,
        'ssim': ssim_val
    }


def test_metrics():
    """测试评估指标"""
    print("Testing metrics...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 4
    num_bits = 48

    # 测试数据
    predicted = torch.rand(batch_size, num_bits, device=device)
    target = (torch.rand(batch_size, num_bits, device=device) > 0.5).float()

    # 计算指标
    bit_acc = calculate_bit_accuracy(predicted, target)
    msg_acc = calculate_message_accuracy(predicted, target)
    ber = calculate_bit_error_rate(predicted, target)

    print(f"  Bit Accuracy: {bit_acc:.4f}")
    print(f"  Message Accuracy: {msg_acc:.4f}")
    print(f"  Bit Error Rate: {ber:.4f}")

    # 测试图像质量指标
    img1 = torch.rand(batch_size, 3, 256, 256, device=device)
    img2 = img1 + torch.randn_like(img1) * 0.05

    psnr_val = calculate_psnr(img1, img2)
    ssim_val = calculate_ssim(img1, img2)
    mse_val = calculate_mse(img1, img2)

    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    print(f"  MSE: {mse_val:.6f}")

    # 测试指标跟踪器
    tracker = MetricsTracker()
    tracker.update(predicted, target, img1 + 0.01, img1)
    tracker.update(predicted, target, img1 + 0.02, img1)

    summary = tracker.get_summary()
    print(f"  Tracker Summary: {summary}")

    print("✓ Metrics tests passed")


if __name__ == "__main__":
    test_metrics()