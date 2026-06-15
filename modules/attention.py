"""
注意力机制模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionBlock(nn.Module):
    """
    交叉注意力块 - 用于水印嵌入

    将水印消息的表示与图像特征进行交互
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0
    ):
        """
        Args:
            query_dim: 查询维度 (特征维度)
            context_dim: 上下文维度 (消息维度)
            num_heads: 注意力头数
            dim_head: 每个头的维度
            dropout: dropout 概率
        """
        super().__init__()
        inner_dim = dim_head * num_heads
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        # 线性投影层
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像特征 [batch_size, seq_len, query_dim]
            context: 消息表示 [batch_size, msg_len, context_dim]

        Returns:
            output: 注意力输出 [batch_size, seq_len, query_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 投影到 Q, K, V
        q = self.to_q(x)  # [B, seq_len, inner_dim]
        k = self.to_k(context)  # [B, msg_len, inner_dim]
        v = self.to_v(context)  # [B, msg_len, inner_dim]

        # 重塑为多头格式: [B, seq_len, num_heads, dim_head]
        q = q.view(batch_size, seq_len, self.num_heads, self.dim_head)
        k = k.view(batch_size, -1, self.num_heads, self.dim_head)
        v = v.view(batch_size, -1, self.num_heads, self.dim_head)

        # 转置: [B, num_heads, seq_len, dim_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 应用注意力到 V
        out = torch.matmul(attn, v)

        # 合并多头: [B, num_heads, seq_len, dim_head] -> [B, seq_len, inner_dim]
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.inner_dim)

        return self.to_out(out)


class SelfAttentionBlock(nn.Module):
    """
    自注意力块 - 用于特征增强
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch_size, seq_len, dim]

        Returns:
            output: 注意力输出 [batch_size, seq_len, dim]
        """
        batch_size = x.shape[0]

        # LayerNorm + QKV
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)

        # 多头注意力的形状变换
        q = q.view(batch_size, -1, self.num_heads, -1).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, -1).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, -1).transpose(1, 2)

        # 计算注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # 合并多头
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, -1)

        return x + self.to_out(out)


class SpatialAttention(nn.Module):
    """
    空间注意力 - 用于指导水印嵌入位置
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 特征图 [batch_size, channels, height, width]

        Returns:
            attended: 空间加权后的特征 [batch_size, channels, height, width]
        """
        attention_map = self.conv(x)
        return x * attention_map