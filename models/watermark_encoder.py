"""
水印编码器

将秘密消息嵌入到图像特征中
使用 Gumbel-Softmax 采样实现离散水印比特的嵌入
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from modules.gumbel_softmax import MessageEncoder, gumbel_softmax
from modules.attention import CrossAttentionBlock
from modules.conv_blocks import ResBlock, ConvBlock


class WatermarkEncoder(nn.Module):
    """
    水印编码器

    将 48-bit 水印消息嵌入到 U-Net 特征中
    """

    def __init__(
        self,
        feature_dim: int = 1280,
        message_bits: int = 48,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: 输入特征维度
            message_bits: 水印比特数
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout: dropout 概率
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.message_bits = message_bits

        # 消息编码器 (二进制 → one-hot Gumbel)
        self.message_encoder = MessageEncoder(
            message_dim=hidden_dim,
            num_bits=message_bits,
            hidden_dim=hidden_dim
        )

        # 特征投影层
        self.feature_proj = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 消息投影层
        self.message_proj = nn.Sequential(
            nn.Linear(message_bits, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 交叉注意力层 - 消息指导特征修改
        self.cross_attention = CrossAttentionBlock(
            query_dim=hidden_dim,
            context_dim=hidden_dim,
            num_heads=num_heads,
            dim_head=32,
            dropout=dropout
        )

        # 残差块用于特征精炼
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(3)
        ])

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, feature_dim, kernel_size=1)
        )

        # 调制层 - 自适应地混合原始特征和修改后的特征
        self.scale_conv = nn.Conv2d(hidden_dim, feature_dim, kernel_size=1)
        self.shift_conv = nn.Conv2d(hidden_dim, feature_dim, kernel_size=1)

    def embed_message(
        self,
        features: torch.Tensor,
        message: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        将水印嵌入到特征中

        Args:
            features: U-Net 特征 [batch_size, feature_dim, H, W]
            message: 二进制消息 [batch_size, message_bits]
            return_attention: 是否返回注意力图

        Returns:
            watermarked_features: 含水印特征 [batch_size, feature_dim, H, W]
            attention_map: 注意力图 (可选)
        """
        batch_size, _, H, W = features.shape

        # 投影特征
        proj_features = self.feature_proj(features)  # [B, hidden_dim, H, W]

        # 投影消息为 one-hot Gumbel 格式
        message_oh = self.message_encoder(message)  # [B, message_bits]

        # 扩展消息维度用于与特征交互
        message_feat = self.message_proj(message_oh)  # [B, hidden_dim]

        # 重塑为序列格式用于注意力
        features_seq = proj_features.flatten(2).transpose(1, 2)  # [B, H*W, hidden_dim]

        # 消息扩展为 [B, 1, hidden_dim] 用于交叉注意力
        message_seq = message_feat.unsqueeze(1)  # [B, 1, hidden_dim]

        # 交叉注意力
        attended = self.cross_attention(features_seq, message_seq)  # [B, H*W, hidden_dim]

        # 恢复空间维度
        attended = attended.transpose(1, 2).view(batch_size, -1, H, W)  # [B, hidden_dim, H, W]

        # 残差块精炼
        refined = attended
        for block in self.res_blocks:
            refined = block(refined)

        # 自适应调制
        scale = self.scale_conv(refined)
        shift = self.shift_conv(refined)

        # 原始特征与调制后的特征融合
        modulated = scale * features + shift

        # 最终输出
        output = self.output_proj(refined)

        if return_attention:
            attention_map = torch.sigmoid(scale.abs().mean(dim=1, keepdim=True))
            return modulated + 0.1 * output, attention_map

        return modulated + 0.1 * output

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        message: torch.Tensor,
        layer_name: str = "mid_block"
    ) -> Dict[str, torch.Tensor]:
        """
        处理特征字典

        Args:
            features: U-Net 特征字典
            message: 水印消息 [batch_size, message_bits]
            layer_name: 要处理的目标层

        Returns:
            modified_features: 修改后的特征字典
        """
        if layer_name not in features:
            raise ValueError(f"Layer {layer_name} not found in features. Available: {list(features.keys())}")

        feature = features[layer_name]
        watermarked_feature = self.embed_message(feature, message)

        modified_features = features.copy()
        modified_features[layer_name] = watermarked_feature

        return modified_features


class AdaptiveWatermarkEncoder(nn.Module):
    """
    自适应水印编码器

    将水印嵌入到多个尺度的特征中
    """

    def __init__(
        self,
        feature_dims: Dict[str, int] = None,
        message_bits: int = 48,
        hidden_dim: int = 256,
        num_heads: int = 8
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

        # 共享的消息编码器
        self.message_encoder = MessageEncoder(
            message_dim=hidden_dim,
            num_bits=message_bits,
            hidden_dim=hidden_dim
        )

        # 每层的专用编码器
        self.layer_encoders = nn.ModuleDict()
        for layer_name, feat_dim in feature_dims.items():
            self.layer_encoders[layer_name] = WatermarkEncoderBlock(
                feature_dim=feat_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads
            )

    def embed_message_multi_scale(
        self,
        features: Dict[str, torch.Tensor],
        message: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        多尺度嵌入水印

        Args:
            features: 多尺度特征字典
            message: 水印消息

        Returns:
            watermarked_features: 含水印的多尺度特征
        """
        # 编码消息一次 (共享)
        message_oh = self.message_encoder(message)

        watermarked = {}
        for layer_name, feature in features.items():
            if layer_name in self.layer_encoders:
                encoder = self.layer_encoders[layer_name]
                watermarked[layer_name] = encoder(feature, message_oh)

        return watermarked

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        message: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.embed_message_multi_scale(features, message)


class WatermarkEncoderBlock(nn.Module):
    """单个水印嵌入块"""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_heads: int
    ):
        super().__init__()

        self.feature_proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        self.res_blocks = nn.Sequential(
            ResBlock(hidden_dim, hidden_dim * 2),
            ResBlock(hidden_dim, hidden_dim * 2),
        )
        self.output_proj = nn.Conv2d(hidden_dim, feature_dim, kernel_size=1)

        # 调制参数
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        feature: torch.Tensor,
        message_oh: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feature: 输入特征 [B, C, H, W]
            message_oh: one-hot 编码的消息 [B, num_bits]

        Returns:
            output: 处理后的特征 [B, C, H, W]
        """
        proj = self.feature_proj(feature)
        refined = self.res_blocks(proj)
        mod = self.output_proj(refined)

        # 自适应调制 (消息向量作为调制因子)
        msg_scale = message_oh.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        msg_scale = torch.sigmoid(msg_scale)  # [B, 1, 1, 1]

        return feature + self.scale * msg_scale * mod + self.shift * mod


def test_watermark_encoder():
    """测试水印编码器"""
    print("Testing WatermarkEncoder...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试基本编码器
    encoder = WatermarkEncoder(
        feature_dim=1280,
        message_bits=48,
        hidden_dim=256
    ).to(device)

    # 创建模拟特征和消息
    batch_size = 2
    feature = torch.randn(batch_size, 1280, 32, 32, device=device)
    message = (torch.rand(batch_size, 48) > 0.5).float().to(device)

    # 嵌入水印
    watermarked = encoder.embed_message(feature, message)

    print(f"  Input feature shape: {feature.shape}")
    print(f"  Watermarked feature shape: {watermarked.shape}")
    print(f"  Max difference: {(watermarked - feature).abs().max().item():.4f}")

    assert watermarked.shape == feature.shape
    assert not torch.allclose(watermarked, feature)  # 应该不同

    # 测试梯度流
    watermarked.sum().backward()
    grad_exists = any(p.grad is not None for p in encoder.parameters() if p.requires_grad)
    assert grad_exists, "Gradients should flow through encoder"

    print("✓ WatermarkEncoder 测试通过")

    # 测试自适应编码器
    print("Testing AdaptiveWatermarkEncoder...")
    features = {
        "down_block_2": torch.randn(batch_size, 640, 64, 64, device=device),
        "mid_block": torch.randn(batch_size, 1280, 32, 32, device=device),
        "up_block_2": torch.randn(batch_size, 640, 64, 64, device=device)
    }

    adaptive_encoder = AdaptiveWatermarkEncoder().to(device)
    watermarked_features = adaptive_encoder(features, message)

    for k, v in watermarked_features.items():
        assert v.shape == features[k].shape
        print(f"  {k}: {v.shape}")

    print("✓ AdaptiveWatermarkEncoder 测试通过")


if __name__ == "__main__":
    test_watermark_encoder()