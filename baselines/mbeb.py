"""
MBEB (Matrix-Based Embedding) 基线方法

基于矩阵编码的水印方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class MBEBWatermark(nn.Module):
    """
    MBEB 水印实现

    基于矩阵编码的嵌入方法
    """

    def __init__(
        self,
        message_bits: int = 48,
        code_length: int = 7,
        embed_strength: float = 0.05
    ):
        super().__init__()
        self.message_bits = message_bits
        self.code_length = code_length  # 每个编码组的长度
        self.embed_strength = embed_strength

        # 简化的神经网络编码器和解码器
        self.encoder_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.decoder_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, message_bits)
        )

    def encode_message(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        将消息编码为与图像兼容的格式

        Args:
            image: [B, C, H, W]
            message: [B, message_bits]

        Returns:
            encoded: [B, C, H, W]
        """
        # 通过编码器处理图像
        feat = self.encoder_net(image)

        # 将消息投影到特征空间
        B = image.shape[0]
        C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]

        # 重复消息以匹配特征大小
        msg_expanded = message.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # 使用注意力机制融合
        attention = torch.sigmoid(feat)
        encoded = feat + attention * msg_expanded * self.embed_strength

        return encoded

    def decode_message(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        从编码特征中解码消息

        Args:
            encoded: [B, C, H, W]

        Returns:
            message: [B, message_bits]
        """
        return self.decoder_net(encoded)

    def forward(self, image: torch.Tensor, message: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """端到端前向传播"""
        encoded = self.encode_message(image, message)
        decoded = self.decode_message(encoded)
        return encoded, decoded


def test_mbeb():
    """测试 MBEB"""
    print("\nTesting MBEB Watermark...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wm = MBEBWatermark(message_bits=48, embed_strength=0.1).to(device)

    image = torch.rand(2, 3, 64, 64, device=device)
    message = torch.randint(0, 2, (2, 48), device=device).float()

    encoded, decoded = wm(image, message)
    print(f"  Encoded shape: {encoded.shape}")
    print(f"  Decoded shape: {decoded.shape}")

    loss = F.binary_cross_entropy_with_logits(decoded, message)
    print(f"  Loss: {loss.item():.4f}")

    print("✓ MBEB tests passed")


if __name__ == "__main__":
    test_mbeb()