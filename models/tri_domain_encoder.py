"""
三域水印编码器 (Tri-Domain Watermark Encoder)

参考论文 Section III-C 的设计，实现：
1. 空间分支 (Spatial Branch)
2. 频率分支 (Frequency Branch)
3. 特征分支 (Feature Branch)
4. 分支融合 (Branch Fusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class FrequencyTransform(nn.Module):
    """
    可微分的频率变换模块

    支持 DCT (离散余弦变换) 的近似实现
    """

    def __init__(self, mode: str = "dct"):
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [B, C, H, W]

        Returns:
            freq: 频率域表示
        """
        # 简化的 DCT 实现
        # 在实际应用中可以使用更复杂的实现
        if self.mode == "dct":
            # 沿 H 和 W 维度分别进行 DCT
            x_dct = self._dct_2d(x)
            return x_dct
        elif self.mode == "fft":
            return torch.fft.fft2(x)
        else:
            return x

    def inverse(self, freq: torch.Tensor) -> torch.Tensor:
        """逆变换"""
        if self.mode == "dct":
            return self._idct_2d(freq)
        elif self.mode == "fft":
            return torch.fft.ifft2(freq).real
        else:
            return freq

    def _dct_2d(self, x: torch.Tensor) -> torch.Tensor:
        """简化的 2D DCT"""
        # 使用 FFT 的实部作为 DCT 的近似
        freq = torch.fft.rfft2(x, norm='ortho')
        return freq

    def _idct_2d(self, freq: torch.Tensor) -> torch.Tensor:
        """简化的 2D 逆 DCT"""
        return torch.fft.irfft2(freq, norm='ortho', s=x.shape[-2:] if hasattr(x, 'shape') else None)


class MessageProjection(nn.Module):
    """
    消息投影模块 (Key-conditioned Message Code)

    将水印消息投影到高维空间
    """

    def __init__(self, message_bits: int = 48, hidden_dim: int = 256, key_dim: int = 64):
        super().__init__()
        self.message_bits = message_bits
        self.hidden_dim = hidden_dim

        # 消息嵌入 + 键嵌入的融合
        self.message_net = nn.Sequential(
            nn.Linear(message_bits, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.key_net = nn.Sequential(
            nn.Linear(key_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, message: torch.Tensor, key: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            message: 消息 [B, message_bits]
            key: 密钥 [B, key_dim] (可选)

        Returns:
            z: 消息代码 [B, hidden_dim]
        """
        if key is None:
            # 使用默认密钥
            key = torch.zeros(message.shape[0], 64, device=message.device)

        msg_feat = self.message_net(message)
        key_feat = self.key_net(key)

        # 融合消息和密钥
        z = msg_feat * torch.sigmoid(key_feat)
        return z


class SpatialBranch(nn.Module):
    """
    空间分支 (Spatial Branch)

    在像素域嵌入水印
    """

    def __init__(
        self,
        in_channels: int = 3,
        message_bits: int = 48,
        hidden_dim: int = 256
    ):
        super().__init__()

        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # H/2, W/2
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # H/4, W/4
            nn.ReLU(inplace=True),
        )

        # 消息到残差的映射
        self.message_proj = nn.Sequential(
            nn.Linear(message_bits, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 残差解码器
        self.residual_decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim + 256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
            nn.Tanh()  # 输出 [-1, 1] 范围的残差
        )

    def forward(
        self,
        image: torch.Tensor,
        message: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            image: 输入图像 [B, C, H, W]
            message: 水印消息 [B, message_bits]
            mask: 嵌入掩码 [B, 1, H, W] (可选)

        Returns:
            residual: 空间域残差 [B, C, H, W]
        """
        # 编码图像
        img_feat = self.image_encoder(image)  # [B, 256, H/4, W/4]

        # 投影消息
        msg_feat = self.message_proj(message)  # [B, hidden_dim]
        msg_feat = msg_feat.unsqueeze(-1).unsqueeze(-1)  # [B, hidden_dim, 1, 1]
        msg_feat = msg_feat.expand(-1, -1, img_feat.shape[2], img_feat.shape[3])  # [B, hidden_dim, H/4, W/4]

        # 拼接并解码
        combined = torch.cat([img_feat, msg_feat], dim=1)
        residual = self.residual_decoder(combined)  # [B, 3, H, W]

        # 应用掩码
        if mask is not None:
            residual = residual * mask

        return residual


class FrequencyBranch(nn.Module):
    """
    频率分支 (Frequency Branch)

    在频率域嵌入水印 (DCT/FFT)
    """

    def __init__(
        self,
        in_channels: int = 3,
        message_bits: int = 48,
        hidden_dim: int = 256,
        embed_strength: float = 0.1
    ):
        super().__init__()
        self.embed_strength = embed_strength

        # 消息到频率系数的映射
        self.message_proj = nn.Sequential(
            nn.Linear(message_bits, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 频率选择掩码生成器 (选择中频)
        self.mask_generator = nn.Conv2d(3, 1, kernel_size=1)

    def forward(
        self,
        image: torch.Tensor,
        message: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            image: 输入图像 [B, C, H, W]
            message: 水印消息 [B, message_bits]

        Returns:
            residual: 频率域残差 [B, C, H, W]
        """
        # 投影消息
        msg_feat = self.message_proj(message)  # [B, hidden_dim]

        # 简化的频率嵌入
        # 对每个通道和频率调制
        B, C, H, W = image.shape

        # 创建频率掩码
        freq_mask = self._create_frequency_mask(B, H, W, device=image.device)

        # 消息调制的频率系数
        msg_scale = msg_feat.mean(dim=1, keepdim=True)  # [B, 1]
        freq_coef = msg_scale.unsqueeze(-1).unsqueeze(-1) * freq_mask

        # 在频率域添加水印
        freq = torch.fft.rfft2(image, norm='ortho')
        freq_watermarked = freq + self.embed_strength * freq_coef * torch.exp(1j * freq_coef)
        watermarked = torch.fft.irfft2(freq_watermarked, s=(H, W), norm='ortho')

        residual = watermarked - image

        if mask is not None:
            residual = residual * mask

        return residual

    def _create_frequency_mask(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: str
    ) -> torch.Tensor:
        """创建中频选择掩码"""
        # 创建频率坐标
        u = torch.arange(height, device=device).float()
        v = torch.arange(width // 2 + 1, device=device).float()
        U, V = torch.meshgrid(u, v, indexing='ij')

        # 计算频率距离 (避免中心 DC 分量)
        freq_dist = torch.sqrt(U ** 2 + V ** 2)

        # 中频范围 (避免低频和高频)
        low_thresh = height * 0.1
        high_thresh = height * 0.4

        mask = ((freq_dist > low_thresh) & (freq_dist < high_thresh)).float()
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        return mask


class FeatureBranch(nn.Module):
    """
    特征分支 (Feature Branch)

    在医学特征空间嵌入水印 (可选)
    """

    def __init__(
        self,
        message_bits: int = 48,
        feature_dim: int = 512,
        hidden_dim: int = 256
    ):
        super().__init__()

        # 消息到特征方向的映射
        self.message_proj = nn.Sequential(
            nn.Linear(message_bits, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(
        self,
        features: torch.Tensor,
        message: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: 预训练特征 [B, feature_dim]
            message: 水印消息 [B, message_bits]

        Returns:
            target_direction: 目标特征方向
        """
        return self.message_proj(message)


class TriDomainEncoder(nn.Module):
    """
    三域水印编码器

    结合空间、频率和特征三个域的水印嵌入
    """

    def __init__(
        self,
        in_channels: int = 3,
        message_bits: int = 48,
        hidden_dim: int = 256,
        use_frequency: bool = True,
        use_feature: bool = False,
        alpha: float = 0.6,  # 空间分支权重
        beta: float = 0.3,   # 频率分支权重
        gamma: float = 0.1   # 特征分支权重
    ):
        super().__init__()

        self.message_bits = message_bits
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_frequency = use_frequency
        self.use_feature = use_feature

        # 消息投影
        self.message_proj = MessageProjection(message_bits, hidden_dim)

        # 空间分支
        self.spatial_branch = SpatialBranch(in_channels, message_bits, hidden_dim)

        # 频率分支
        if use_frequency:
            self.frequency_branch = FrequencyBranch(in_channels, message_bits, hidden_dim)

        # 特征分支 (需要预训练特征)
        if use_feature:
            self.feature_branch = FeatureBranch(message_bits, feature_dim=512)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2 if use_frequency else in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=1),
            nn.Sigmoid()  # 确保权重在 [0, 1]
        )

    def forward(
        self,
        image: torch.Tensor,
        message: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            image: 输入图像 [B, C, H, W]
            message: 水印消息 [B, message_bits]
            mask: 敏感性掩码 [B, 1, H, W] (可选)
            features: 预训练特征 (可选)

        Returns:
            watermarked: 含水印图像 [B, C, H, W]
        """
        # 空间域残差
        res_spatial = self.spatial_branch(image, message, mask)

        # 频率域残差
        if self.use_frequency and self.beta > 0:
            res_freq = self.frequency_branch(image, message, mask)
        else:
            res_freq = torch.zeros_like(image)

        # 特征域残差 (简化处理)
        if self.use_feature and features is not None and self.gamma > 0:
            res_feat = self.feature_branch(features, message)
        else:
            res_feat = torch.zeros(1, device=image.device)

        # 融合残差
        # 简化的融合：直接加权求和
        if self.use_frequency:
            combined_res = self.alpha * res_spatial + self.beta * res_freq
            # 使用融合层进一步处理
            fusion_input = torch.cat([res_spatial, res_freq], dim=1)
            fusion_weight = self.fusion(fusion_input)
            combined_res = fusion_weight * res_spatial + (1 - fusion_weight) * res_freq
        else:
            combined_res = res_spatial

        # 添加到原始图像
        watermarked = image + combined_res
        watermarked = torch.clamp(watermarked, 0, 1)

        return watermarked


class SimpleTriDomainEncoder(nn.Module):
    """
    简化版三域编码器

    用于快速实验验证
    """

    def __init__(
        self,
        in_channels: int = 3,
        message_bits: int = 48,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.message_bits = message_bits

        # 消息编码
        self.message_encoder = nn.Sequential(
            nn.Linear(message_bits, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )

        # 图像调制器
        self.modulator = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )

        # 融合网络 (调整输入通道)
        self.fusion = nn.Sequential(
            nn.Conv2d(64 + 64, 128, kernel_size=3, padding=1),  # img_feat (64) + msg (64)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 频率掩码
        self.register_buffer('freq_mask', self._create_freq_mask(64))

    def _create_freq_mask(self, size: int) -> torch.Tensor:
        """创建中频掩码"""
        mask = torch.zeros(1, 1, size, size)
        center = size // 2
        for i in range(size):
            for j in range(size):
                dist = abs(i - center) + abs(j - center)
                if 0.2 * size < dist < 0.4 * size:
                    mask[0, 0, i, j] = 1.0
        return mask

    def forward(
        self,
        image: torch.Tensor,
        message: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            image: [B, C, H, W], 范围 [0, 1]
            message: [B, message_bits], 二进制消息
            mask: [B, 1, H, W] 或 None

        Returns:
            watermarked: [B, C, H, W], 范围 [0, 1]
        """
        B, C, H, W = image.shape

        # 编码消息为空间调制
        msg_feat = self.message_encoder(message)  # [B, hidden_dim * 2]
        msg_spatial = msg_feat.view(B, -1, 1, 1).expand(-1, -1, H, W)

        # 图像特征
        img_feat = self.modulator(image)  # [B, 64, H, W]

        # 添加频率域信息 (简化的FFT)
        img_fft = torch.fft.rfft2(img_feat, norm='ortho')
        fft_magnitude = torch.abs(img_fft)
        fft_phase = torch.angle(img_fft)

        # 消息调制的频率增强
        msg_freq_strength = msg_feat[:, :fft_magnitude.shape[1]].unsqueeze(-1).unsqueeze(-1)
        fft_magnitude_warped = fft_magnitude * (1 + 0.1 * torch.sigmoid(msg_freq_strength))

        # 逆变换
        img_freq_enhanced = torch.fft.irfft2(fft_magnitude_warped * torch.exp(1j * fft_phase), s=(H, W), norm='ortho')

        # 融合
        combined = torch.cat([img_feat, msg_spatial[:, :64, :H, :W]], dim=1)
        residual = self.fusion(combined)

        # 应用掩码
        if mask is not None:
            residual = residual * mask

        # 添加到原始图像
        watermarked = image + 0.1 * residual
        watermarked = torch.clamp(watermarked, 0, 1)

        return watermarked


def test_tri_domain_encoder():
    """测试三域编码器"""
    print("\nTesting Tri-Domain Encoder...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试简化版
    encoder = SimpleTriDomainEncoder(in_channels=3, message_bits=48, hidden_dim=128)
    encoder = encoder.to(device)

    # 输入
    image = torch.rand(2, 3, 256, 256, device=device)
    message = torch.randint(0, 2, (2, 48), device=device).float()
    mask = torch.ones(2, 1, 256, 256, device=device) * 0.8

    # 前向传播
    watermarked = encoder(image, message, mask)

    print(f"  Input shape: {image.shape}")
    print(f"  Output shape: {watermarked.shape}")
    print(f"  Max pixel diff: {(watermarked - image).abs().max().item():.4f}")

    assert watermarked.shape == image.shape
    assert watermarked.min() >= 0 and watermarked.max() <= 1

    # 梯度测试
    watermarked.sum().backward()
    assert any(p.grad is not None for p in encoder.parameters())
    print("  ✓ Gradient flow works")

    print("✓ Tri-Domain Encoder tests passed")


if __name__ == "__main__":
    test_tri_domain_encoder()