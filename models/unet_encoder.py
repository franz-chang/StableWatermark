"""
U-Net 特征提取器

从 Stable Diffusion 的 U-Net 中提取中间层特征
用于水印嵌入和提取
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
try:
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.models.attention_processor import AttnProcessor
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    StableDiffusionPipeline = None
    UNet2DConditionModel = None
import warnings


class UNetEncoder(nn.Module):
    """
    用于提取 U-Net 中间层特征的编码器

    支持从 SD 的不同层提取特征：
    - down_blocks: 下采样阶段的特征
    - mid_block: 中间块特征 (最常用)
    - up_blocks: 上采样阶段的特征
    """

    def __init__(
        self,
        sd_model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        extract_layers: List[str] = None,
        gradient_checkpointing: bool = True
    ):
        """
        Args:
            sd_model_id: Stable Diffusion 模型 ID
            device: 设备 (cuda/cpu)
            extract_layers: 要提取的层名称列表
            gradient_checkpointing: 是否使用梯度检查点
        """
        super().__init__()

        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers library is required for UNetEncoder. "
                "Please install it with: pip install diffusers"
            )

        self.sd_model_id = sd_model_id
        self.device = device

        # 默认提取 mid_block 和 down_blocks 的某些层
        self.extract_layers = extract_layers or ["down_block_2", "mid_block", "up_block_2"]

        # 加载 Pipeline (只用于获取 U-Net)
        print(f"Loading SD model: {sd_model_id}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipe = self.pipe.to(device)

        # 获取 U-Net
        self.unet: UNet2DConditionModel = self.pipe.unet

        # 冻结 U-Net 参数
        for param in self.unet.parameters():
            param.requires_grad = False

        # 注册特征钩子
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        self._register_hooks()

        # 梯度检查点
        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

    def _register_hooks(self):
        """注册用于提取特征的钩子"""
        # 我们需要在 forward 过程中手动提取特征
        # 这里注册一个全局钩子
        pass

    def extract_features(
        self,
        latents: torch.Tensor,
        timestep: int = 500
    ) -> Dict[str, torch.Tensor]:
        """
        从 U-Net 提取特征

        Args:
            latents: 潜在表示 [batch_size, 4, height//8, width//8]
            timestep: 时间步

        Returns:
            features: 特征字典 {layer_name: feature}
        """
        batch_size = latents.shape[0]

        # 准备文本嵌入 (空文本用于无条件生成)
        prompt_embeds = torch.zeros(
            batch_size, 77, 768,
            device=latents.device,
            dtype=latents.dtype
        )

        # 创建时间步张量
        t = torch.tensor([timestep], device=latents.device).repeat(batch_size)

        features = {}

        # 方法1: 通过 hook 提取特征
        # 我们手动执行 U-Net 的各个部分

        # 使用 eval 模式以确保一致性
        with torch.no_grad():
            # 下采样阶段
            sample = latents

            # 输入卷积
            sample = self.unet.conv_in(sample)

            # 下采样块
            down_block_samples = []
            for i, downsample_block in enumerate(self.unet.down_blocks):
                if hasattr(downsample_block, 'downsamplers') and downsample_block.downsamplers:
                    for downsample in downsample_block.downsamplers:
                        sample = downsample(sample)
                down_block_samples.append(sample)

                # 检查是否为要提取的层
                layer_name = f"down_block_{i}"
                if layer_name in self.extract_layers:
                    features[layer_name] = sample.clone()

                # 注意块
                if hasattr(downsample_block, 'attentions'):
                    for attn in downsample_block.attentions:
                        sample = attn(sample, encoder_hidden_states=prompt_embeds).sample

                # 下采样
                sample = downsample_block.downsamplers[0](sample) if downsample_block.downsamplers else sample

            # 中间块
            if "mid_block" in self.extract_layers:
                sample = self.unet.mid_block(sample, prompt_embeds, t)
                features["mid_block"] = sample.clone()
            else:
                sample = self.unet.mid_block(sample, prompt_embeds, t)

            # 上采样块 (只保存跳连接，不完全执行)
            for i, upsample_block in enumerate(self.unet.up_blocks):
                res_samples = []
                for resnet in upsample_block.resnets:
                    res_samples.append(sample)
                    sample = resnet(sample, t)

                layer_name = f"up_block_{i}"
                if layer_name in self.extract_layers:
                    features[layer_name] = sample.clone()

                # 跳跃连接
                if hasattr(upsample_block, 'upsamplers'):
                    sample = upsample_block.upsamplers[0](sample)

        return features

    def get_feature_dim(self, layer_name: str = "mid_block") -> int:
        """获取指定层的特征维度"""
        dims = {
            "down_block_0": 320,
            "down_block_1": 320,
            "down_block_2": 640,
            "down_block_3": 1280,
            "mid_block": 1280,
            "up_block_0": 1280,
            "up_block_1": 1280,
            "up_block_2": 640,
            "up_block_3": 320
        }
        return dims.get(layer_name, 1280)

    def forward(self, latents: torch.Tensor, timestep: int = 500) -> Dict[str, torch.Tensor]:
        """前向传播"""
        return self.extract_features(latents, timestep)

    def __del__(self):
        """清理资源"""
        for hook in self.hooks:
            hook.remove()


class LightweightUNetEncoder(nn.Module):
    """
    轻量级 U-Net 编码器 - 不加载完整的 SD 模型

    只保留用于特征提取的最小结构
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dims: List[int] = [128, 256, 512, 1024],
        out_channels: int = 1280
    ):
        super().__init__()

        self.encoder_blocks = nn.ModuleList()

        ch = in_channels
        for hidden_dim in hidden_dims:
            self.encoder_blocks.append(nn.Sequential(
                nn.Conv2d(ch, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.SiLU(inplace=True)
            ))
            ch = hidden_dim

        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入 [batch_size, 4, H, W]

        Returns:
            features: 特征字典
        """
        features = {}
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            features[f"down_block_{i}"] = x

        features["bottleneck"] = x
        return features


def test_unet_encoder():
    """测试 U-Net 编码器"""
    print("Testing LightweightUNetEncoder...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = LightweightUNetEncoder().to(device)

    # 测试特征提取
    x = torch.randn(2, 4, 64, 64, device=device)
    features = encoder(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Feature keys: {list(features.keys())}")
    for k, v in features.items():
        print(f"    {k}: {v.shape}")

    assert "bottleneck" in features
    assert features["bottleneck"].shape[0] == 2  # batch size

    print("✓ LightweightUNetEncoder 测试通过")

    # 测试完整 U-Net 编码器 (如果 GPU 可用)
    if device == "cuda":
        print("Testing full UNetEncoder (may take a while on first run)...")
        encoder = UNetEncoder(sd_model_id="runwayml/stable-diffusion-v1-5", device=device)
        latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
        features = encoder.extract_features(latents)
        print(f"  Feature keys: {list(features.keys())}")
        print("✓ UNetEncoder 测试通过")


if __name__ == "__main__":
    test_unet_encoder()