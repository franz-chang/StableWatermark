"""
盲解码器 (Blind Watermark Decoder)

参考论文 Section III-D 的设计，从可能受到攻击的图像中提取水印消息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FrequencyAnalysisBranch(nn.Module):
    """
    频率分析分支

    提取图像的频率域特征用于水印检测
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()

        # 简化频率分析
        self.analysis = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 下采样
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取频率相关特征"""
        return self.analysis(x)


class MultiScaleFeatureExtractor(nn.Module):
    """
    多尺度特征提取器

    从不同尺度提取特征以捕获局部残留模式
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        # 多尺度分支
        self.scale1 = nn.Sequential(  # 原尺度
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.scale2 = nn.Sequential(  # 1/2 尺度
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.scale4 = nn.Sequential(  # 1/4 尺度
            nn.Conv2d(in_channels, base_channels * 2, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [B, C, H, W]

        Returns:
            features: 多尺度融合特征
        """
        f1 = self.scale1(x)
        f2 = self.scale2(x)

        # 上采样 f2 到 f1 的大小
        f2_up = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)

        # f4 需要先上采样
        f4_up = F.interpolate(f4 := self.scale4(x), size=f2.shape[2:], mode='bilinear', align_corners=False)
        f4_up2 = F.interpolate(f4_up, size=f1.shape[2:], mode='bilinear', align_corners=False)

        # 融合多尺度特征
        combined = torch.cat([f1, f2_up, f4_up2], dim=1)
        return self.fusion(combined)


class BitPredictionHead(nn.Module):
    """
    比特预测头

    输出 48-bit 水印消息的 logit 值
    """

    def __init__(self, input_dim: int = 256, message_bits: int = 48):
        super().__init__()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, message_bits)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征 [B, C, H, W]

        Returns:
            logits: 消息预测 logits [B, message_bits]
        """
        return self.head(features)


class ConfidenceHead(nn.Module):
    """
    置信度头

    输出水印存在的置信度分数
    """

    def __init__(self, input_dim: int = 256):
        super().__init__()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出 [0, 1] 的置信度
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征 [B, C, H, W]

        Returns:
            confidence: 置信度 [B, 1]
        """
        return self.head(features)


class BlindDecoder(nn.Module):
    """
    盲水印解码器

    从可能受到攻击的图像中提取水印消息
    """

    def __init__(
        self,
        in_channels: int = 3,
        message_bits: int = 48,
        base_channels: int = 64
    ):
        super().__init__()

        self.message_bits = message_bits

        # 多尺度特征提取
        self.feature_extractor = MultiScaleFeatureExtractor(
            in_channels=in_channels,
            base_channels=base_channels
        )

        # 频率分析分支
        self.freq_branch = FrequencyAnalysisBranch(
            in_channels=in_channels,
            out_channels=base_channels
        )

        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(base_channels * 4 + base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # 消息预测头
        self.message_head = BitPredictionHead(
            input_dim=base_channels * 4,
            message_bits=message_bits
        )

        # 置信度头
        self.confidence_head = ConfidenceHead(
            input_dim=base_channels * 4
        )

    def forward(
        self,
        image: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: 输入图像 [B, C, H, W]
            return_features: 是否返回中间特征

        Returns:
            (message_logits, confidence) 或 (message_logits, confidence, features)
        """
        # 多尺度特征
        multi_scale_feat = self.feature_extractor(image)

        # 频率特征
        freq_feat = self.freq_branch(image)
        freq_feat_up = F.interpolate(freq_feat, size=multi_scale_feat.shape[2:],
                                      mode='bilinear', align_corners=False)

        # 融合
        combined = torch.cat([multi_scale_feat, freq_feat_up], dim=1)
        fused_feat = self.feature_fusion(combined)

        # 预测
        message_logits = self.message_head(fused_feat)
        confidence = self.confidence_head(fused_feat)

        if return_features:
            return message_logits, confidence, fused_feat

        return message_logits, confidence

    def get_probabilities(self, image: torch.Tensor) -> torch.Tensor:
        """
        获取消息位的概率

        Args:
            image: 输入图像

        Returns:
            probs: [B, message_bits] 概率值
        """
        logits, _ = self(image)
        return torch.sigmoid(logits)


class SimpleBlindDecoder(nn.Module):
    """
    简化版盲解码器

    用于快速实验验证
    """

    def __init__(
        self,
        in_channels: int = 3,
        message_bits: int = 48,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.message_bits = message_bits

        # 特征提取
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2, W/2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4, W/4

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/8, W/8

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # 固定到 4x4
        )

        # 消息预测
        self.message_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, message_bits)
        )

        # 置信度头
        self.confidence_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入图像 [B, C, H, W]

        Returns:
            (message_logits, confidence)
        """
        feat = self.features(x)
        message_logits = self.message_head(feat)
        confidence = self.confidence_head(feat)
        return message_logits, confidence

    def get_probabilities(self, image: torch.Tensor) -> torch.Tensor:
        """获取消息概率"""
        logits, _ = self(image)
        return torch.sigmoid(logits)


def test_blind_decoder():
    """测试盲解码器"""
    print("\nTesting Blind Decoder...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试简化版
    decoder = SimpleBlindDecoder(in_channels=3, message_bits=48, hidden_dim=256)
    decoder = decoder.to(device)

    # 输入
    image = torch.rand(2, 3, 256, 256, device=device)

    # 前向传播
    message_logits, confidence = decoder(image)

    print(f"  Input shape: {image.shape}")
    print(f"  Message logits shape: {message_logits.shape}")
    print(f"  Confidence shape: {confidence.shape}")

    assert message_logits.shape == (2, 48)
    assert confidence.shape == (2, 1)
    assert confidence.min() >= 0 and confidence.max() <= 1

    # 概率测试
    probs = decoder.get_probabilities(image)
    print(f"  Probabilities range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    assert probs.shape == (2, 48)

    # 梯度测试
    message_logits.sum().backward()
    assert any(p.grad is not None for p in decoder.parameters())
    print("  ✓ Gradient flow works")

    print("✓ Blind Decoder tests passed")


if __name__ == "__main__":
    test_blind_decoder()