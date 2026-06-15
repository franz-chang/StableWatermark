"""
卷积块模块
"""

import torch
import torch.nn as nn
from typing import Optional


class ResBlock(nn.Module):
    """
    残差块 - 用于特征提取和增强
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: Optional[int] = None,
        dropout: float = 0.0,
        use_group_norm: bool = True
    ):
        super().__init__()
        hidden_channels = hidden_channels or channels

        self.block = nn.Sequential(
            nn.GroupNorm(32, channels) if use_group_norm else nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.GroupNorm(32, hidden_channels) if use_group_norm else nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpBlock(nn.Module):
    """
    上采样块 - 用于解码器
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_mode: str = "bilinear",
        use_group_norm: bool = True
    ):
        super().__init__()
        self.upsample_mode = upsample_mode

        self.conv = nn.Sequential(
            nn.GroupNorm(32, in_channels) if use_group_norm else nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 上采样
        x = nn.functional.interpolate(
            x,
            scale_factor=2,
            mode=self.upsample_mode,
            align_corners=False if self.upsample_mode == "bilinear" else None
        )

        # 卷积
        x = self.conv(x)

        # 可能的跳跃连接
        if skip is not None:
            # 如果需要，调整 skip 的尺寸
            if x.shape != skip.shape:
                x = nn.functional.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode=self.upsample_mode,
                    align_corners=False if self.upsample_mode == "bilinear" else None
                )
            x = x + skip

        return x


class DownBlock(nn.Module):
    """
    下采样块 - 用于编码器
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_mode: str = "stride",
        use_group_norm: bool = True
    ):
        super().__init__()

        layers = []

        if downsample_mode == "stride":
            layers.extend([
                nn.GroupNorm(32, in_channels) if use_group_norm else nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            ])
        else:  # pool
            layers.extend([
                nn.AvgPool2d(kernel_size=2, stride=2),
            ])
            self.conv = nn.Sequential(
                nn.GroupNorm(32, in_channels) if use_group_norm else nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvBlock(nn.Module):
    """
    基础卷积块
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_group_norm: bool = True,
        activation: str = "relu"
    ):
        super().__init__()

        # GroupNorm: num_groups must divide num_channels
        # Use fewer groups for small channel counts
        if use_group_norm and in_channels >= 32:
            num_groups = min(32, in_channels)
            # Ensure it divides evenly
            while in_channels % num_groups != 0:
                num_groups -= 1
            layers = [nn.GroupNorm(num_groups, in_channels)]
        elif use_group_norm:
            # For small channels (like 3), use InstanceNorm or no norm
            layers = [nn.InstanceNorm2d(in_channels)]
        else:
            layers = [nn.BatchNorm2d(in_channels)]

        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "silu":
            layers.append(nn.SiLU(inplace=True))

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)