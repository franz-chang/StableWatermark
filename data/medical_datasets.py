"""
医疗数据集加载模块

支持多种公开医疗图像数据集的水印嵌入实验

数据集列表:
- ChestX-ray14: NIH 胸部X光数据集
- ISIC: 皮肤病变图像数据集
- DRIVE: 视网膜血管分割数据集
- BrainMRI: 脑肿瘤MRI数据集
- LiTS: 肝脏肿瘤CT数据集
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import zipfile
import tarfile
import urllib.request
import shutil
from typing import Tuple, Optional, List, Dict, Callable, Literal
from dataclasses import dataclass


@dataclass
class MedicalDatasetInfo:
    """医疗数据集信息"""
    name: str
    short_name: str
    modality: str  # X-ray, CT, MRI, Dermoscopy, etc.
    body_part: str  # Chest, Brain, Liver, Retina, Skin
    image_channels: int  # 1 for grayscale, 3 for RGB
    default_size: int
    avg_samples: int
    download_url: str
    description: str
    citation: str


# 数据集元信息注册表
MEDICAL_DATASET_REGISTRY: Dict[str, MedicalDatasetInfo] = {
    "chestxray14": MedicalDatasetInfo(
        name="ChestX-ray14 (NIH ChestX-ray)",
        short_name="ChestX-ray14",
        modality="X-ray",
        body_part="Chest",
        image_channels=1,
        default_size=512,
        avg_samples=112120,
        download_url="https://nihcc.app.box.com/v/ChestXray-NIHCC",
        description="NIH临床胸部X线数据库，包含112,120张前后位胸片，来自30,805名独立患者",
        citation="Wang et al. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks"
    ),
    "isic": MedicalDatasetInfo(
        name="ISIC (Skin Lesion Analysis)",
        short_name="ISIC",
        modality="Dermoscopy",
        body_part="Skin",
        image_channels=3,
        default_size=512,
        avg_samples=25000,
        download_url="https://challenge.isic-archive.com/data/",
        description="皮肤病变黑色素瘤检测数据集，包含25,000+张 dermoscopy 图像",
        citation="Codella et al. Skin Lesion Analysis Toward Melanoma Detection: A Challenge at ISIC 2018"
    ),
    "drive": MedicalDatasetInfo(
        name="DRIVE (Digital Retinal Images)",
        short_name="DRIVE",
        modality="Fundus Photography",
        body_part="Retina",
        image_channels=3,
        default_size=512,
        avg_samples=40,
        download_url="https://drive.grand-challenge.org/",
        description="视网膜血管分割数据集，包含40张眼底图像（用于训练和测试）",
        citation="Staal et al. Ridge-based Vessel Segmentation in Color Images of the Retina"
    ),
    "brainmri": MedicalDatasetInfo(
        name="Brain Tumor MRI",
        short_name="BrainMRI",
        modality="MRI",
        body_part="Brain",
        image_channels=1,
        default_size=512,
        avg_samples=7023,
        download_url="https://www.kaggle.com/datasets/miyaitingchen/brain-tumor-mri-dataset",
        description="脑肿瘤MRI数据集，包含MRI扫描图像，标注为 glioma, meningioma, pituitary",
        citation="Cheng et al. Retrieval of Brain Tumor with Region-specific Diagrammatic Representation"
    ),
    "lits": MedicalDatasetInfo(
        name="LiTS (Liver Tumor Segmentation)",
        short_name="LiTS",
        modality="CT",
        body_part="Liver",
        image_channels=1,
        default_size=512,
        avg_samples=131,
        download_url="https://competitions.codalab.org/competitions/17094",
        description="肝脏肿瘤分割挑战赛数据集，包含131个3D CT扫描中的肝脏和肿瘤分割",
        citation="Bilic et al. The Liver Tumor Segmentation Benchmark (LiTS)"
    ),
    "montgomery": MedicalDatasetInfo(
        name="Montgomery County X-ray",
        short_name="Montgomery",
        modality="X-ray",
        body_part="Chest",
        image_channels=1,
        default_size=512,
        avg_samples=800,
        download_url="https://www.kaggle.com/datasets/darrylhaller/montgomerycounty-xray",
        description="蒙哥马利县肺结核筛查X线数据集，包含胸部X光和左右肺的结节标注",
        citation="Jaeger et al. Automatic tuberculosis screening using chest radiographs"
    ),
    "shenzhen": MedicalDatasetInfo(
        name="Shenzhen Hospital X-ray",
        short_name="Shenzhen",
        modality="X-ray",
        body_part="Chest",
        image_channels=1,
        default_size=512,
        avg_samples=662,
        download_url="https://www.kaggle.com/datasets/darrylhaller/shenzhen-hospital-xray",
        description="深圳医院胸部X光数据集，与蒙哥马利数据集配套用于肺结核研究",
        citation="Jaeger et al. Automatic tuberculosis screening using chest radiographs"
    ),
    "rsnaBreast": MedicalDatasetInfo(
        name="RSNA Breast Cancer Screening",
        short_name="RSNA-BCS",
        modality="Mammography",
        body_part="Breast",
        image_channels=1,
        default_size=512,
        avg_samples=54868,
        download_url="https://www.kaggle.com/competitions/rsna-breast-cancer-detection",
        description="RSNA乳腺癌筛查数据集，包含超过54000张乳腺X线摄影图像",
        citation="LeCun et al. RSNA Mammography Breast Cancer Detection"
    ),
}


class MedicalImageDataset(Dataset):
    """
    通用医疗图像数据集加载器

    支持灰度图像自动转换为RGB (复制通道)
    支持多种数据集格式
    """

    def __init__(
        self,
        data_root: str,
        dataset_type: Literal[
            "chestxray14", "isic", "drive", "brainmri", "lits",
            "montgomery", "shenzhen", "rsnaBreast", "custom"
        ] = "custom",
        image_size: int = 512,
        max_samples: Optional[int] = None,
        transform: Optional[Callable] = None,
        message_bits: int = 48,
        normalize_stats: Optional[Dict[str, float]] = None,  # 自定义归一化参数
        mode: str = "train"
    ):
        """
        Args:
            data_root: 数据目录路径
            dataset_type: 数据集类型
            image_size: 目标图像尺寸
            max_samples: 最大样本数
            transform: 自定义变换
            message_bits: 水印比特数
            normalize_stats: 医疗图像归一化参数 (mean, std)
            mode: 模式 (train/val/test)
        """
        self.data_root = data_root
        self.dataset_type = dataset_type
        self.image_size = image_size
        self.message_bits = message_bits
        self.mode = mode

        # 获取数据集配置
        self.dataset_info = MEDICAL_DATASET_REGISTRY.get(
            dataset_type,
            MedicalDatasetInfo(
                name="Custom", short_name="Custom",
                modality="Unknown", body_part="Unknown",
                image_channels=3, default_size=512,
                avg_samples=0, download_url="",
                description="自定义数据集",
                citation=""
            )
        )

        # 获取图像路径
        self.image_paths = self._get_image_paths()

        # 限制样本数
        if max_samples is not None and max_samples < len(self.image_paths):
            import random
            random.seed(42)
            self.image_paths = random.sample(self.image_paths, max_samples)

        # 设置变换
        self.transform = transform or self._get_default_transform(normalize_stats)

    def _get_default_transform(self, normalize_stats: Optional[Dict[str, float]] = None) -> transforms.Compose:
        """获取默认变换"""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]

        # 灰度图像转换为RGB
        if self.dataset_info.image_channels == 1:
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

        return transforms.Compose(transform_list)

    def _get_image_paths(self) -> List[str]:
        """获取所有图像路径"""
        image_paths = []

        # 基于数据集类型调整搜索策略
        extensions = ['.png', '.jpg', '.jpeg', '.dcm', '.tif', '.tiff']

        if self.dataset_type == "chestxray14":
            # NIH ChestX-ray14 结构: images/000001_01.png
            patterns = ['images/*.png', '*.png']
        elif self.dataset_type == "isic":
            # ISIC 结构: ISIC_*.jpg
            patterns = ['*.jpg', '*.jpeg', 'images/*.jpg']
        elif self.dataset_type == "drive":
            # DRIVE 结构: 1st_manual/*.gif, test/*.tif
            patterns = ['*.tif', '*.png', '*.gif']
        elif self.dataset_type == "brainmri":
            # Brain MRI: 按类别组织 [glioma, meningioma, pituitary, notumor]
            patterns = ['**/*.jpg', '**/*.png']
        elif self.dataset_type == "lits":
            # LiTS: volume 和 segment 配对
            patterns = ['*.png', '*.tif', 'volumes/*.png']
        else:
            patterns = ['*.png', '*.jpg', '*.jpeg']

        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_paths.append(os.path.join(root, file))

        return sorted(image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def generate_message(self, batch_size: int) -> torch.Tensor:
        """生成随机水印消息"""
        return torch.randint(0, 2, (batch_size, self.message_bits)).float()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取样本"""
        image_path = self.image_paths[idx]

        try:
            # 处理不同格式的医学图像
            if image_path.endswith('.dcm'):
                image = self._load_dicom(image_path)
            else:
                image = Image.open(image_path)

                # 特殊处理: 某些医学图像是L模式
                if image.mode == 'L':
                    image = image.convert('RGB')
                elif image.mode != 'RGB':
                    image = image.convert('RGB')

            image = self.transform(image)

        except Exception as e:
            print(f"Warning: Failed to load {image_path}: {e}")
            # 返回空白图像
            image = torch.zeros(3, self.image_size, self.image_size)

        message = self.generate_message(1)[0]

        return image, message

    def _load_dicom(self, path: str) -> Image.Image:
        """加载DICOM格式医学图像"""
        try:
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_voi_lut
        except ImportError:
            raise ImportError("Please install pydicom for DICOM support: pip install pydicom")

        ds = pydicom.dcmread(path)
        arr = apply_voi_lut(ds.pixel_array, ds)

        # 归一化
        arr = arr.astype(float)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        arr = arr.astype(np.uint8)

        return Image.fromarray(arr)


class MultiDomainMedicalDataset(Dataset):
    """
    多领域医疗数据集

    支持同时加载多个医疗数据集，进行跨领域水印鲁棒性测试
    """

    def __init__(
        self,
        data_roots: Dict[str, str],
        image_size: int = 512,
        max_samples_per_domain: Optional[int] = None,
        message_bits: int = 48,
        sample_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            data_roots: 数据集名称到路径的映射
            image_size: 图像尺寸
            max_samples_per_domain: 每个领域最大样本数
            message_bits: 水印比特数
            sample_weights: 采样权重 (用于不均衡数据集)
        """
        self.datasets = {}
        self.domain_names = []
        self.cumulative_sizes = []

        for name, root in data_roots.items():
            if os.path.exists(root):
                dataset = MedicalImageDataset(
                    data_root=root,
                    dataset_type=name,
                    image_size=image_size,
                    max_samples=max_samples_per_domain,
                    message_bits=message_bits
                )
                self.datasets[name] = dataset
                self.domain_names.append(name)
                print(f"  Loaded {name}: {len(dataset)} images")

        # 计算累积大小用于索引
        self.cumulative_sizes = []
        total = 0
        for name in self.domain_names:
            total += len(self.datasets[name])
            self.cumulative_sizes.append(total)

        self.total_size = total

        # 采样权重
        if sample_weights:
            self.sample_weights = sample_weights
        else:
            #均匀采样
            self.sample_weights = {name: 1.0 for name in self.domain_names}

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """获取样本，返回 (图像, 消息, 领域名)"""
        # 确定属于哪个数据集
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                dataset = self.datasets[self.domain_names[i]]
                local_idx = idx if i == 0 else idx - self.cumulative_sizes[i-1]
                break

        image, message = dataset[local_idx]
        return image, message, self.domain_names[i]


def download_medical_dataset(
    dataset_type: str,
    save_dir: str,
    num_samples: Optional[int] = None
) -> str:
    """
    下载医疗数据集

    Args:
        dataset_type: 数据集类型
        save_dir: 保存目录
        num_samples: 最大下载样本数

    Returns:
        dataset_path: 数据集路径
    """
    if dataset_type not in MEDICAL_DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_type}")

    info = MEDICAL_DATASET_REGISTRY[dataset_type]
    dataset_path = os.path.join(save_dir, dataset_type)
    os.makedirs(dataset_path, exist_ok=True)

    print(f"Downloading {info.name}...")
    print(f"URL: {info.download_url}")
    print(f"Save path: {dataset_path}")
    print(f"Please download manually and extract to: {dataset_path}")
    print(f"Note: {info.description}")

    # 创建下载说明文件
    readme_path = os.path.join(dataset_path, "DOWNLOAD_INSTRUCTIONS.txt")
    with open(readme_path, 'w') as f:
        f.write(f"# {info.name}\n\n")
        f.write(f"## Download Instructions\n\n")
        f.write(f"1. Visit: {info.download_url}\n")
        f.write(f"2. Download the dataset\n")
        f.write(f"3. Extract files to this directory\n")
        f.write(f"4. Expected: the images should be in subdirectories or root of this folder\n\n")
        f.write(f"## Dataset Info\n")
        f.write(f"- Modality: {info.modality}\n")
        f.write(f"- Body Part: {info.body_part}\n")
        f.write(f"- Total Samples: ~{info.avg_samples}\n\n")
        f.write(f"## Citation\n")
        f.write(f"{info.citation}\n")

    return dataset_path


def get_medical_dataloader(
    data_root: str,
    dataset_type: str = "custom",
    batch_size: int = 4,
    image_size: int = 512,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    message_bits: int = 48
) -> DataLoader:
    """
    创建医疗数据集 DataLoader

    Args:
        data_root: 数据目录
        dataset_type: 数据集类型
        batch_size: 批次大小
        image_size: 图像尺寸
        num_workers: 数据加载线程数
        max_samples: 最大样本数
        shuffle: 是否打乱
        message_bits: 水印比特数

    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = MedicalImageDataset(
        data_root=data_root,
        dataset_type=dataset_type,
        image_size=image_size,
        max_samples=max_samples,
        message_bits=message_bits
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


def get_multi_domain_dataloader(
    data_roots: Dict[str, str],
    batch_size: int = 4,
    image_size: int = 512,
    num_workers: int = 4,
    max_samples_per_domain: Optional[int] = None,
    shuffle: bool = True,
    message_bits: int = 48
) -> DataLoader:
    """
    创建多领域医疗数据集 DataLoader

    Returns:
        dataloader: DataLoader，返回 (image, message, domain) 元组
    """
    dataset = MultiDomainMedicalDataset(
        data_roots=data_roots,
        image_size=image_size,
        max_samples_per_domain=max_samples_per_domain,
        message_bits=message_bits
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


def print_dataset_info():
    """打印所有支持的数据集信息"""
    print("\n" + "="*70)
    print("Supported Medical Datasets for StableWatermark")
    print("="*70 + "\n")

    for name, info in MEDICAL_DATASET_REGISTRY.items():
        print(f"[{name}]")
        print(f"  Name: {info.name}")
        print(f"  Modality: {info.modality}")
        print(f"  Body Part: {info.body_part}")
        print(f"  Input Channels: {info.image_channels}")
        print(f"  Sample Count: ~{info.avg_samples}")
        print(f"  Description: {info.description}")
        print(f"  Download: {info.download_url}")
        print()


def test_medical_dataset():
    """测试医疗数据集加载"""
    print("Testing Medical Image Dataset Module...")
    print()
    print_dataset_info()
    print("\n✓ Medical dataset module loaded successfully")


if __name__ == "__main__":
    test_medical_dataset()