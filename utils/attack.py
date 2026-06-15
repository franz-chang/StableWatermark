"""
攻击函数模块

实现多种图像攻击：噪声、滤波、裁剪、旋转、JPEG压缩等
用于测试水印的鲁棒性
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class AttackConfig:
    """攻击配置"""
    gaussian_noise_sigma: float = 0.03
    salt_pepper_prob: float = 0.05
    blur_kernel: int = 5
    crop_scale: float = 0.8
    rotation_degrees: float = 15.0
    jpeg_quality: int = 75
    brightness_factor: float = 1.2
    contrast_factor: float = 1.2


class AttackBase:
    """攻击基类"""

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GaussianNoise(AttackBase):
    """
    高斯噪声攻击

    向图像添加高斯噪声
    """

    def __init__(self, sigma: float = 0.03):
        """
        Args:
            sigma: 噪声标准差 (相对于 [0,1] 范围)
        """
        self.sigma = sigma

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, channels, height, width] 范围 [0, 1]

        Returns:
            attacked: 加噪后的图像
        """
        noise = torch.randn_like(images) * self.sigma
        attacked = torch.clamp(images + noise, 0, 1)
        return attacked


class SaltPepperNoise(AttackBase):
    """
    盐椒噪声攻击
    """

    def __init__(self, prob: float = 0.05):
        """
        Args:
            prob: 噪声概率
        """
        self.prob = prob

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        noise = torch.rand_like(images)
        attacked = torch.where(
            noise < self.prob / 2,
            torch.zeros_like(images),
            torch.where(
                noise > 1 - self.prob / 2,
                torch.ones_like(images),
                images
            )
        )
        return attacked


class GaussianBlur(AttackBase):
    """
    高斯模糊攻击
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 2.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """使用 2D 高斯滤波"""
        batch_size, channels, height, width = images.shape

        if self.kernel_size == 0:
            return images

        # 转换为 numpy 进行滤波
        attacked = []
        for i in range(batch_size):
            img_np = images[i].detach().permute(1, 2, 0).cpu().numpy()  # [H, W, C]

            # 应用高斯模糊
            blurred = cv2.GaussianBlur(img_np, (self.kernel_size, self.kernel_size), self.sigma)

            # 转回 tensor
            attacked.append(torch.from_numpy(blurred).permute(2, 0, 1))

        return torch.stack(attacked).to(images.device)


class MedianBlur(AttackBase):
    """中值滤波攻击"""

    def __init__(self, kernel_size: int = 5):
        self.kernel_size = kernel_size

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        attacked = []
        for i in range(images.shape[0]):
            img_np = images[i].detach().permute(1, 2, 0).cpu().numpy()
            blurred = cv2.medianBlur(img_np, self.kernel_size)
            attacked.append(torch.from_numpy(blurred.copy()).permute(2, 0, 1))

        return torch.stack(attacked).to(images.device)


class CenterCrop(AttackBase):
    """
    中心裁剪攻击

    从中心裁剪图像并调整为原始大小
    """

    def __init__(self, scale: float = 0.8):
        """
        Args:
            scale: 裁剪比例 (0 < scale < 1)
        """
        self.scale = scale

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W]

        Returns:
            attacked: 裁剪并调整大小后的图像
        """
        batch_size, channels, height, width = images.shape

        # 计算裁剪尺寸
        crop_h = int(height * self.scale)
        crop_w = int(width * self.scale)

        # 计算起始位置 (中心)
        start_h = (height - crop_h) // 2
        start_w = (width - crop_w) // 2

        # 裁剪
        cropped = images[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]

        # 调整回原始大小
        attacked = F.interpolate(
            cropped,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )

        return torch.clamp(attacked, 0, 1)


class RandomRotation(AttackBase):
    """随机旋转攻击"""

    def __init__(self, degrees: float = 15.0):
        self.degrees = degrees

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        attacked = []
        for i in range(images.shape[0]):
            img_np = images[i].detach().permute(1, 2, 0).cpu().numpy()
            h, w = img_np.shape[:2]
            center = (w // 2, h // 2)

            angle = np.random.uniform(-self.degrees, self.degrees)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            rotated = cv2.warpAffine(
                img_np, matrix, (w, h),
                borderMode=cv2.BORDER_REFLECT
            )
            attacked.append(torch.from_numpy(rotated.copy()).permute(2, 0, 1))

        return torch.stack(attacked).to(images.device)


class JPEGCompression(AttackBase):
    """
    JPEG 压缩攻击

    通过 JPEG 压缩解压缩模拟质量损失
    """

    def __init__(self, quality: int = 75):
        """
        Args:
            quality: JPEG 质量 (1-100，越低压缩越严重)
        """
        self.quality = quality

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """使用 JPEG 压缩解压缩"""
        attacked = []
        for i in range(images.shape[0]):
            # detach 以避免梯度问题
            img_np = (images[i].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            h, w, c = img_np.shape

            # 编码为 JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            _, encimg = cv2.imencode('.jpg', img_np, encode_param)

            # 解码
            decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

            # 转回 tensor 并归一化
            attacked.append(
                torch.from_numpy(decimg.copy()).permute(2, 0, 1).float() / 255.0
            )

        return torch.stack(attacked).to(images.device)


class BrightnessAdjust(AttackBase):
    """亮度调整攻击"""

    def __init__(self, factor: float = 1.2):
        self.factor = factor

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return torch.clamp(images * self.factor, 0, 1)


class ContrastAdjust(AttackBase):
    """对比度调整攻击"""

    def __init__(self, factor: float = 1.2):
        self.factor = factor

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.mean(dim=(1, 2, 3), keepdim=True)
        return torch.clamp((images - mean) * self.factor + mean, 0, 1)


class CombinedAttack(AttackBase):
    """
    组合攻击

    依次应用多种攻击
    """

    def __init__(self, attacks: List[AttackBase]):
        self.attacks = attacks

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        result = images
        for attack in self.attacks:
            result = attack(result)
        return result


# 预设的攻击组合
def get_attack(name: str, config: AttackConfig = None) -> AttackBase:
    """
    根据名称获取攻击

    Args:
        name: 攻击名称
        config: 攻击配置

    Returns:
        attack: 攻击对象
    """
    if config is None:
        config = AttackConfig()

    attacks = {
        "gaussian_noise": GaussianNoise(config.gaussian_noise_sigma),
        "salt_pepper": SaltPepperNoise(config.salt_pepper_prob),
        "gaussian_blur": GaussianBlur(config.blur_kernel),
        "median_blur": MedianBlur(config.blur_kernel),
        "center_crop": CenterCrop(config.crop_scale),
        "random_rotation": RandomRotation(config.rotation_degrees),
        "jpeg_compression": JPEGCompression(config.jpeg_quality),
        "brightness": BrightnessAdjust(config.brightness_factor),
        "contrast": ContrastAdjust(config.contrast_factor),
        "combined": CombinedAttack([
            GaussianNoise(config.gaussian_noise_sigma),
            JPEGCompression(config.jpeg_quality)
        ]),
        "heavy_combined": CombinedAttack([
            GaussianNoise(config.gaussian_noise_sigma * 1.5),
            GaussianBlur(config.blur_kernel),
            JPEGCompression(max(50, config.jpeg_quality - 20)),
            CenterCrop(config.crop_scale)
        ])
    }

    if name not in attacks:
        raise ValueError(f"Unknown attack: {name}. Available: {list(attacks.keys())}")

    return attacks[name]


def get_all_attacks(config: AttackConfig = None) -> dict:
    """获取所有预定义的攻击"""
    if config is None:
        config = AttackConfig()

    return {
        "gaussian_noise": GaussianNoise(config.gaussian_noise_sigma),
        "gaussian_blur": GaussianBlur(config.blur_kernel),
        "median_blur": MedianBlur(config.blur_kernel),
        "center_crop": CenterCrop(config.crop_scale),
        "random_rotation": RandomRotation(config.rotation_degrees),
        "jpeg_compression": JPEGCompression(config.jpeg_quality),
        "salt_pepper": SaltPepperNoise(config.salt_pepper_prob),
        "brightness": BrightnessAdjust(config.brightness_factor),
        "contrast": ContrastAdjust(config.contrast_factor),
        "combined": CombinedAttack([
            GaussianNoise(config.gaussian_noise_sigma),
            JPEGCompression(config.jpeg_quality)
        ]),
        "heavy_combined": CombinedAttack([
            GaussianNoise(config.gaussian_noise_sigma * 1.5),
            GaussianBlur(config.blur_kernel),
            JPEGCompression(max(50, config.jpeg_quality - 20))
        ])
    }


def test_attacks():
    """测试所有攻击"""
    print("Testing attacks...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2

    # 创建测试图像
    images = torch.rand(batch_size, 3, 256, 256, device=device)

    config = AttackConfig()

    # 测试每种攻击
    for name in [
        "gaussian_noise", "salt_pepper", "gaussian_blur",
        "center_crop", "jpeg_compression", "combined"
    ]:
        attack = get_attack(name, config)
        result = attack(images.clone())

        assert result.shape == images.shape
        assert result.min() >= 0 and result.max() <= 1

        diff = (result - images).abs().mean().item()
        print(f"  {name}: shape={result.shape}, diff={diff:.4f}")

    print("✓ All attacks tested successfully")


if __name__ == "__main__":
    test_attacks()