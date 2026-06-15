"""
数据集加载模块

支持 COCO 数据集和自定义数据集
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import random
from typing import Tuple, Optional, List, Dict, Callable


class WatermarkDataset(Dataset):
    """
    水印数据集

    加载图像并生成随机水印消息
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 512,
        max_samples: Optional[int] = None,
        transform: Optional[Callable] = None,
        message_bits: int = 48,
        mode: str = "train"
    ):
        """
        Args:
            data_root: 数据目录
            image_size: 图像尺寸
            max_samples: 最大样本数
            transform: 图像变换
            message_bits: 水印比特数
            mode: 模式 (train/val/test)
        """
        self.data_root = data_root
        self.image_size = image_size
        self.message_bits = message_bits
        self.mode = mode

        # 获取图像路径
        self.image_paths = self._get_image_paths()

        # 限制样本数
        if max_samples is not None and max_samples < len(self.image_paths):
            random.seed(42)
            self.image_paths = random.sample(self.image_paths, max_samples)

        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def _get_image_paths(self) -> List[str]:
        """获取所有图像路径"""
        image_paths = []

        # 常见的图像格式
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_paths.append(os.path.join(root, file))

        return sorted(image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def generate_message(self, batch_size: int) -> torch.Tensor:
        """
        生成随机水印消息

        Args:
            batch_size: 批次大小

        Returns:
            messages: [batch_size, message_bits] 二进制消息
        """
        # 方法1: 完全随机
        # messages = torch.randint(0, 2, (batch_size, self.message_bits)).float()

        # 方法2: 有意义的哈希值 (可选)
        # 这里使用随机值，实际应用中可改为哈希某些元数据

        return torch.randint(0, 2, (batch_size, self.message_bits)).float()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本

        Returns:
            image: [3, image_size, image_size] 图像张量
            message: [message_bits] 水印消息
        """
        # 加载图像
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # 变换
        image = self.transform(image)

        # 生成消息
        message = self.generate_message(1)[0]

        return image, message


class GeneratedImageDataset(Dataset):
    """
    生成图像数据集 (用于 SD 生成的水印图像)

    图像已经由 Stable Diffusion 生成并保存
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 512,
        transform: Optional[Callable] = None
    ):
        self.data_root = data_root
        self.image_size = image_size
        self.transform = transform

        # 加载图像路径和对应的消息
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        samples = []

        # 查找图像文件和对应的 JSON 元数据
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.endswith(('.jpg', '.png')):
                    image_path = os.path.join(root, file)
                    json_path = image_path.rsplit('.', 1)[0] + '.json'

                    sample = {'image_path': image_path}

                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            meta = json.load(f)
                            sample['message'] = meta.get('message')
                            sample['seed'] = meta.get('seed')

                    samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 获取消息
        if 'message' in sample and sample['message'] is not None:
            message = torch.tensor(sample['message'], dtype=torch.float32)
        else:
            message = torch.zeros(48)  # 默认空消息

        return image, message


class SyntheticDataset(Dataset):
    """
    合成数据集 (用于快速测试)

    直接生成随机图像
    """

    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 256,
        image_channels: int = 3,
        message_bits: int = 48
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.image_channels = image_channels
        self.message_bits = message_bits

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 生成随机图像
        image = torch.rand(self.image_channels, self.image_size, self.image_size)

        # 生成随机消息
        message = torch.randint(0, 2, (self.message_bits,)).float()

        return image, message


def get_dataloader(
    data_root: str,
    batch_size: int = 4,
    image_size: int = 512,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    dataset_type: str = "coco",
    shuffle: bool = True,
    message_bits: int = 48
) -> DataLoader:
    """
    创建数据加载器

    Args:
        data_root: 数据目录 (dataset_type="coco"时可以是目录或预下载路径)
        batch_size: 批次大小
        image_size: 图像尺寸
        num_workers: 数据加载线程数
        max_samples: 最大样本数
        dataset_type: 数据集类型 ("coco", "generated", "synthetic")
        shuffle: 是否打乱
        message_bits: 水印比特数

    Returns:
        dataloader: PyTorch DataLoader
    """
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # 选择数据集类型
    if dataset_type == "synthetic":
        dataset = SyntheticDataset(
            num_samples=max_samples or 1000,
            image_size=image_size,
            message_bits=message_bits
        )
    elif dataset_type == "generated":
        dataset = GeneratedImageDataset(
            data_root=data_root,
            image_size=image_size,
            transform=transform
        )
    else:  # coco 或自定义
        dataset = WatermarkDataset(
            data_root=data_root,
            image_size=image_size,
            max_samples=max_samples,
            transform=transform,
            message_bits=message_bits
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


def download_coco_sample(data_root: str, num_samples: int = 1000):
    """
    下载 COCO 数据集样本 (使用 torchvision)

    需要安装 pycocotools
    """
    try:
        from torchvision.datasets import CocoDetection
        from torchvision import transforms
    except ImportError:
        print("Please install pycocotools: pip install pycocotools")
        return

    from torchvision.utils import save_image
    import glob

    # COCO 验证集 annotation
    ann_file = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    print(f"COCO dataset should be manually downloaded to: {data_root}")
    print("Download from: https://cocodataset.org/")

    # 保存一个空的 data_root 标记文件
    marker_file = os.path.join(data_root, "README.txt")
    with open(marker_file, 'w') as f:
        f.write(f"COCO dataset should be placed in this directory.\n")
        f.write(f"Download from: https://cocodataset.org/\n")


def test_dataset():
    """测试数据集"""
    print("Testing SyntheticDataset...")

    # 使用合成数据集测试
    dataset = SyntheticDataset(num_samples=100, image_size=256, message_bits=48)

    image, message = dataset[0]
    print(f"  Image shape: {image.shape}")
    print(f"  Message shape: {message.shape}")
    print(f"  Message bits: {message[:10]}... (first 10)")

    # 测试 DataLoader
    dataloader = get_dataloader(
        data_root="",
        batch_size=4,
        dataset_type="synthetic",
        shuffle=True,
        num_workers=0  # 避免多进程问题
    )

    batch_images, batch_messages = next(iter(dataloader))
    print(f"  Batch images shape: {batch_images.shape}")
    print(f"  Batch messages shape: {batch_messages.shape}")

    print("✓ Dataset tests passed")


if __name__ == "__main__":
    test_dataset()