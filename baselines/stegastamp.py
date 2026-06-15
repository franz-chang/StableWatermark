"""
StegaStamp 类似方法

基于深度学习的水印方法，作为基线对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class StegaStampEncoder(nn.Module):
    """
    类 StegaStamp 编码器

    使用深度学习在图像中嵌入水印
    """

    def __init__(
        self,
        in_channels: int = 3,
        message_bits: int = 48,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.message_bits = message_bits

        # 消息编码器
        self.message_encoder = nn.Sequential(
            nn.Linear(message_bits, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )

        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),  # H/2, W/2
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),  # H/4, W/4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),  # H/8, W/8
        )

        # 解码器 (生成残差)
        self.residual_decoder = nn.Sequential(
            nn.ConvTranspose2d(256 + hidden_dim * 2, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: 图像 [B, 3, H, W]
            message: 消息 [B, message_bits]

        Returns:
            watermarked: 含水印图像 [B, 3, H, W]
        """
        # 编码消息
        msg_feat = self.message_encoder(message)  # [B, hidden_dim * 2]
        B = msg_feat.shape[0]

        # 编码图像
        img_feat = self.image_encoder(image)  # [B, 256, H/8, W/8]

        # 将消息扩展到空间维度
        H, W = img_feat.shape[2], img_feat.shape[3]
        msg_spatial = msg_feat.view(B, -1, 1, 1).expand(-1, -1, H, W)

        # 拼接并解码残差
        combined = torch.cat([img_feat, msg_spatial], dim=1)
        residual = self.residual_decoder(combined)  # [B, 3, H, W]

        # 上采样残差到原始大小
        residual = F.interpolate(residual, size=image.shape[2:], mode='bilinear', align_corners=False)

        # 添加残差
        watermarked = image + 0.1 * residual
        watermarked = torch.clamp(watermarked, 0, 1)

        return watermarked


class StegaStampDecoder(nn.Module):
    """
    类 StegaStamp 解码器

    从含水印图像中提取消息
    """

    def __init__(
        self,
        in_channels: int = 3,
        message_bits: int = 48,
        hidden_dim: int = 256
    ):
        super().__init__()

        # 特征提取器
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # 消息预测
        self.message_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, message_bits)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """提取消息"""
        feat = self.features(image)
        return self.message_predictor(feat)


class StegaStampBaseline:
    """
    StegaStamp 基线模型

    端到端的水印系统
    """

    def __init__(
        self,
        message_bits: int = 48,
        hidden_dim: int = 256,
        device: str = "cuda"
    ):
        self.message_bits = message_bits
        self.device = device

        self.encoder = StegaStampEncoder(message_bits=message_bits, hidden_dim=hidden_dim)
        self.decoder = StegaStampDecoder(message_bits=message_bits, hidden_dim=hidden_dim)

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

    def embed(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """嵌入水印"""
        return self.encoder(image, message)

    def extract(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取水印"""
        logits = self.decoder(image)
        probs = torch.sigmoid(logits)
        return logits, probs

    def train_step(
        self,
        image: torch.Tensor,
        message: torch.Tensor
    ) -> dict:
        """训练步骤"""
        # 嵌入
        watermarked = self.encoder(image, message)

        # 提取
        pred_logits = self.decoder(watermarked)

        # 损失
        loss = F.binary_cross_entropy_with_logits(pred_logits, message)

        return {
            'loss': loss.item(),
            'bit_acc': ((torch.sigmoid(pred_logits) > 0.5) == message).float().mean().item()
        }

    def state_dict(self) -> dict:
        """获取模型状态"""
        return {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict()
        }

    def load_state_dict(self, state_dict: dict):
        """加载模型状态"""
        self.encoder.load_state_dict(state_dict['encoder'])
        self.decoder.load_state_dict(state_dict['decoder'])

    def eval(self):
        """评估模式"""
        self.encoder.eval()
        self.decoder.eval()


def test_stegastamp():
    """测试 StegaStamp 基线"""
    print("\nTesting StegaStamp Baseline...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化
    stegastamp = StegaStampBaseline(message_bits=48, device=device)

    # 测试数据
    image = torch.rand(2, 3, 256, 256, device=device)
    message = torch.randint(0, 2, (2, 48), device=device).float()

    # 嵌入
    watermarked = stegastamp.embed(image, message)
    print(f"  Watermarked shape: {watermarked.shape}")

    # 提取
    logits, probs = stegastamp.extract(watermarked)
    print(f"  Extraction: {probs.shape}")

    # 训练步骤
    metrics = stegastamp.train_step(image, message)
    print(f"  Train step - Loss: {metrics['loss']:.4f}, Acc: {metrics['bit_acc']:.4f}")

    print("✓ StegaStamp Baseline tests passed")


if __name__ == "__main__":
    test_stegastamp()