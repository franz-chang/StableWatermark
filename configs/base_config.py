"""基础配置文件"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """模型配置"""
    # 水印参数
    watermark_bits: int = 48
    watermark_length: int = 48

    # 模型维度
    hidden_dim: int = 320
    message_dim: int = 256
    num_heads: int = 8

    # Gumbel-Softmax
    gumbel_temperature: float = 1.0
    gumbel_hard: bool = True


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础参数
    batch_size: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # 损失权重
    lambda_rec: float = 1.0
    lambda_msg: float = 10.0
    lambda_adv: float = 0.5

    # 调度器
    lr_scheduler: str = "cosine"
    warmup_steps: int = 500

    # 其他
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0
    log_every: int = 50
    save_every: int = 1000


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str = "coco"  # 支持: coco, synthetic, generated, chestxray14, isic, drive, brainmri, lits, montgomery, shenzhen, rsnaBreast
    data_root: str = "./data/coco"
    image_size: int = 512
    max_samples: Optional[int] = 1000  # None 表示使用全部数据
    num_workers: int = 4
    # 医疗数据集专用
    gray_to_rgb: bool = True  # 灰度图像转换为RGB
    normalize_medical: bool = False  # 医学图像特殊归一化


@dataclass
class AttackConfig:
    """攻击配置"""
    attacks: List[str] = field(default_factory=lambda: [
        "gaussian_noise",
        "salt_pepper",
        "gaussianBlur",
        "center_crop",
        "jpeg_compression",
        "combined"
    ])

    # 噪声参数
    gaussian_noise_sigma: float = 0.03
    salt_pepper_prob: float = 0.05

    # 滤波参数
    blur_kernel: int = 5

    # 裁剪参数
    crop_scale: float = 0.8

    # 旋转参数
    rotation_degrees: float = 15.0

    # JPEG参数
    jpeg_quality: int = 75


@dataclass
class ExperimentConfig:
    """实验配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)

    # Stable Diffusion
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda"

    # 输出
    output_dir: str = "./outputs"
    seed: int = 42


def get_config() -> ExperimentConfig:
    """获取默认配置"""
    return ExperimentConfig()


def get_config_str(config: ExperimentConfig) -> str:
    """获取配置字符串"""
    return f"""
Experiment Configuration:
=========================
Model:
  - Watermark bits: {config.model.watermark_bits}
  - Hidden dim: {config.model.hidden_dim}
  - Gumbel temperature: {config.model.gumbel_temperature}

Training:
  - Batch size: {config.training.batch_size}
  - Epochs: {config.training.num_epochs}
  - Learning rate: {config.training.learning_rate}
  - Lambda rec: {config.training.lambda_rec}
  - Lambda msg: {config.training.lambda_msg}
  - Lambda adv: {config.training.lambda_adv}

Dataset:
  - Name: {config.dataset.name}
  - Image size: {config.dataset.image_size}
  - Max samples: {config.dataset.max_samples}

SD Model:
  - Model ID: {config.sd_model_id}
  - Device: {config.device}
"""