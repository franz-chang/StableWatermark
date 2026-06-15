"""
水印解码器

从含水印图像/特征中提取水印消息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from modules.conv_blocks import ResBlock, DownBlock, ConvBlock


class WatermarkDecoder(nn.Module):
    """
    水印解码器

    从图像或特征中提取 48-bit 水印消息
    """

    def __init__(
        self,
        input_channels: int = 3,  # 3 for images, feature_dim for features
        message_bits: int = 48,
        hidden_dim: int = 256,
        output_dim: Optional[int] = None
    ):
        """
        Args:
            input_channels: 输入通道数
            message_bits: 水印比特数
            hidden_dim: 隐藏层维度
            output_dim: 输出维度，默认与 message_bits 相同
        """
        super().__init__()
        self.input_channels = input_channels
        self.message_bits = message_bits
        self.output_dim = output_dim or message_bits

        # 特征编码器 - 将输入转换为统一特征
        self.encoder = nn.Sequential(
            ConvBlock(input_channels, 64, activation="relu"),
            nn.MaxPool2d(2),
            ConvBlock(64, 128, activation="relu"),
            nn.MaxPool2d(2),
            ConvBlock(128, 256, activation="relu"),
            nn.MaxPool2d(2),
            ConvBlock(256, hidden_dim, activation="relu"),
        )

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 消息解码器
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, self.output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        解码水印

        Args:
            x: 输入 [batch_size, channels, height, width]

        Returns:
            logits: 水印消息的 logits [batch_size, message_bits]
        """
        # 编码
        feat = self.encoder(x)  # [B, hidden_dim, H, W]

        # 全局池化
        feat = self.global_pool(feat)  # [B, hidden_dim, 1, 1]

        # 解码
        logits = self.decoder(feat)  # [B, message_bits]

        return logits


class DeepWatermarkDecoder(nn.Module):
    """
    深度水印解码器

    适合从多种尺度的含水印特征中提取水印
    """

    def __init__(
        self,
        feature_dims: Dict[str, int] = None,
        message_bits: int = 48,
        hidden_dim: int = 512
    ):
        super().__init__()

        if feature_dims is None:
            feature_dims = {
                "down_block_2": 640,
                "mid_block": 1280,
                "up_block_2": 640
            }

        self.feature_dims = feature_dims
        self.message_bits = message_bits
        self.hidden_dim = hidden_dim

        # 每层特征的处理
        self.layer_processors = nn.ModuleDict()
        for name, dim in feature_dims.items():
            self.layer_processors[name] = nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Conv2d(dim, hidden_dim // 4, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_dim // 4),
                nn.ReLU(inplace=True),
                ResBlock(hidden_dim // 4, hidden_dim // 2),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_dim // 2),
                nn.ReLU(inplace=True),
            )

        # 特征融合
        total_dim = len(feature_dims) * (hidden_dim // 2)

        # 消息解码器
        self.message_decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, message_bits)
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        从多尺度特征解码水印

        Args:
            features: 特征字典 {layer_name: feature}

        Returns:
            logits: 水印消息的 logits [batch_size, message_bits]
        """
        processed = []

        for name in self.feature_dims:
            if name in features:
                feat = self.layer_processors[name](features[name])
                processed.append(feat)

        # 拼接并解码
        if len(processed) == 0:
            raise ValueError("No valid features found")

        fused = torch.cat(processed, dim=1)  # [B, total_dim, 4, 4]
        logits = self.message_decoder(fused)  # [B, message_bits]

        return logits

    def decode_probability(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        返回解码概率（伯努利分布参数）

        Args:
            features: 特征字典

        Returns:
            probs: [batch_size, message_bits] 水印位为1的概率
        """
        logits = self.forward(features)
        return torch.sigmoid(logits)


class PatchBasedDecoder(nn.Module):
    """
    基于 Patch 的水印解码器

    为每个水印比特使用独立的解码头
    """

    def __init__(
        self,
        feature_dim: int = 1280,
        message_bits: int = 48,
        patch_size: int = 16
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.message_bits = message_bits
        self.patch_size = patch_size

        # 特征处理
        self.feature_conv = nn.Sequential(
            nn.Conv2d(feature_dim, 512, kernel_size=3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
        )

        # 每个比特独立的解码头
        self.bit_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            for _ in range(message_bits)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 特征图 [batch_size, feature_dim, H, W]

        Returns:
            logits: [batch_size, message_bits]
        """
        feat = self.feature_conv(x)

        outputs = []
        for head in self.bit_heads:
            out = head(feat)  # [batch_size, 1]
            outputs.append(out)

        return torch.cat(outputs, dim=1)


def test_watermark_decoder():
    """测试水印解码器"""
    print("Testing WatermarkDecoder...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试基本解码器
    decoder = WatermarkDecoder(
        input_channels=3,
        message_bits=48
    ).to(device)

    # 模拟图像输入
    batch_size = 4
    images = torch.randn(batch_size, 3, 256, 256, device=device)

    logits = decoder(images)
    print(f"  Input shape: {images.shape}")
    print(f"  Output logits shape: {logits.shape}")

    assert logits.shape == (batch_size, 48)

    # 测试解码概率
    probs = torch.sigmoid(logits)
    assert (probs >= 0).all() and (probs <= 1).all()

    print("✓ WatermarkDecoder 测试通过")

    # 测试从特征解码
    print("Testing DeepWatermarkDecoder from features...")

    feature_decoder = DeepWatermarkDecoder(
        feature_dims={
            "down_block_2": 640,
            "mid_block": 1280,
            "up_block_2": 640
        },
        message_bits=48
    ).to(device)

    features = {
        "down_block_2": torch.randn(batch_size, 640, 32, 32, device=device),
        "mid_block": torch.randn(batch_size, 1280, 16, 16, device=device),
        "up_block_2": torch.randn(batch_size, 640, 32, 32, device=device)
    }

    logits = feature_decoder(features)
    probs = feature_decoder.decode_probability(features)

    print(f"  Output logits shape: {logits.shape}")
    print(f"  Output probs shape: {probs.shape}")
    print(f"  Probs range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")

    assert logits.shape == (batch_size, 48)
    assert probs.shape == (batch_size, 48)

    print("✓ DeepWatermarkDecoder 测试通过")

    # 测试 PatchBasedDecoder
    print("Testing PatchBasedDecoder...")

    patch_decoder = PatchBasedDecoder(feature_dim=1280, message_bits=48).to(device)
    feat = torch.randn(batch_size, 1280, 16, 16, device=device)
    logits = patch_decoder(feat)

    print(f"  Output shape: {logits.shape}")
    assert logits.shape == (batch_size, 48)

    print("✓ PatchBasedDecoder 测试通过")


if __name__ == "__main__":
    test_watermark_decoder()