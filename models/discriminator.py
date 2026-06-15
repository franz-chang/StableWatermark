"""
判别器模型

用于 GAN 对抗训练，区分真实图像和含水印图像
"""

import torch
import torch.nn as nn
from typing import List, Optional

from modules.conv_blocks import ResBlock, ConvBlock


class Discriminator(nn.Module):
    """
    PatchGAN 判别器

    对图像的每个 patch 进行真假判断
    """

    def __init__(
        self,
        input_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = "instance"  # "instance" or "layer"
    ):
        """
        Args:
            input_channels: 输入通道数
            ndf: 初始卷积核数量
            n_layers: 卷积层数量
            norm_type: 归一化类型
        """
        super().__init__()

        # 归一化函数
        if norm_type == "instance":
            norm_fn = nn.InstanceNorm2d
        elif norm_type == "layer":
            norm_fn = nn.LayerNorm
        else:
            norm_fn = nn.BatchNorm2d

        # 初始卷积层
        layers = [
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # 中间层
        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers.extend([
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1),
                norm_fn(nf),
                nn.LeakyReLU(0.2, inplace=True)
            ])

        # 输出层
        layers.append(
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [batch_size, channels, height, width]

        Returns:
            output: 判别结果 [batch_size, 1, H', W']
        """
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    多尺度判别器

    在不同尺度上进行判别，更有效检测细微差异
    """

    def __init__(
        self,
        input_channels: int = 3,
        ndf: int = 64,
        n_scales: int = 3
    ):
        super().__init__()
        self.n_scales = n_scales

        # 不同尺度的判别器
        self.discriminators = nn.ModuleList([
            Discriminator(input_channels, ndf * (2 ** i))
            for i in range(n_scales)
        ])

        # 池化层用于下采样
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: 输入图像

        Returns:
            outputs: 各尺度的判别结果列表
        """
        outputs = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            outputs.append(disc(x))

        return outputs

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取中间层特征用于特征匹配损失"""
        return self.discriminators[0](x)


class ResidualDiscriminator(nn.Module):
    """
    残差判别器

    使用残差连接增强判别能力
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 64
    ):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 下采样阶段
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 残差块
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim * 4, hidden_dim * 4) for _ in range(3)]
        )

        # 输出
        self.output = nn.Conv2d(hidden_dim * 4, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        return self.output(x)


class UNetDiscriminator(nn.Module):
    """
    UNet 风格的判别器

    编码器-解码器结构，用于获取更丰富的特征
    """

    def __init__(
        self,
        input_channels: int = 3,
        base_dim: int = 64,
        depth: int = 4
    ):
        super().__init__()

        # 编码器
        self.encoder = nn.ModuleList()
        ch = input_channels
        for i in range(depth):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(ch, base_dim * (2 ** i), kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            ch = base_dim * (2 ** i)

        # 中间层
        self.mid = ResBlock(ch, ch)

        # 解码器 (不使用跳跃连接的 UNet)
        self.decoder = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            out_ch = base_dim * (2 ** max(0, i - 1)) if i > 0 else base_dim
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(32, out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            ch = out_ch

        # 最终输出
        self.final = nn.Conv2d(ch, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        # 编码
        for block in self.encoder:
            x = block(x)
            skips.append(x)

        # 中间
        x = self.mid(x)

        # 解码
        for i, block in enumerate(self.decoder):
            x = block(x)

        return self.final(x)


def test_discriminator():
    """测试判别器"""
    print("Testing Discriminator...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 4

    # 测试基本判别器
    disc = Discriminator(input_channels=3, ndf=64, n_layers=3).to(device)
    x = torch.randn(batch_size, 3, 256, 256, device=device)

    output = disc(x)
    print(f"  Input: {x.shape} -> Output: {output.shape}")

    assert output.shape[0] == batch_size
    assert output.shape[1] == 1

    # 测试多尺度判别器
    print("Testing MultiScaleDiscriminator...")
    ms_disc = MultiScaleDiscriminator(n_scales=3).to(device)
    outputs = ms_disc(x)

    for i, out in enumerate(outputs):
        print(f"    Scale {i}: {out.shape}")

    assert len(outputs) == 3

    # 测试残差判别器
    print("Testing ResidualDiscriminator...")
    res_disc = ResidualDiscriminator(input_channels=3, hidden_dim=64).to(device)
    output = res_disc(x)
    print(f"  Input: {x.shape} -> Output: {output.shape}")

    print("✓ All discriminators tested successfully")


if __name__ == "__main__":
    test_discriminator()