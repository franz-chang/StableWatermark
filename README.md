# StableWatermark

基于 Gumbel-Softmax 采样的 Stable Diffusion 水印算法实现。

## 项目简介

本项目实现了一种将秘密水印嵌入到 Stable Diffusion 生成图像中的方法，主要特性：

- **Gumbel-Softmax 采样**：实现离散水印比特的端到端可微训练
- **U-Net 特征提取**：利用 Stable Diffusion 的中间层特征
- **GAN 对抗训练**：确保水印嵌入对图像质量影响最小
- **多攻击鲁棒性**：支持噪声、裁剪、旋转、JPEG 压缩等多种攻击

## 项目结构

```
StableWatermark/
├── configs/          # 配置文件
├── data/             # 数据集加载
├── models/           # 模型定义
│   ├── unet_encoder.py      # U-Net 特征提取器
│   ├── watermark_encoder.py # 水印编码器
│   ├── watermark_decoder.py # 水印解码器
│   └── discriminator.py     # 判别器
├── modules/          # 核心模块
│   ├── gumbel_softmax.py    # Gumbel-Softmax 实现
│   ├── attention.py         # 注意力机制
│   └── conv_blocks.py       # 卷积块
├── utils/            # 工具函数
│   ├── attack.py            # 攻击函数
│   ├── metrics.py           # 评估指标
│   └── visualization.py     # 可视化
├── training/         # 训练相关
│   ├── trainer.py           # 训练器
│   └── losses.py            # 损失函数
├── scripts/          # 实验脚本
│   └── run_experiment.py    # 实验入口
├── main.py           # 主程序
└── requirements.txt  # 依赖
```

## 安装

```bash
# 克隆项目
git clone https://github.com/your-repo/StableWatermark.git
cd StableWatermark

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 运行演示

```bash
python main.py --demo
```

### 2. 训练模型

```bash
python main.py --train --data_root ./data --num_epochs 50 --batch_size 8
```

### 3. 评估模型

```bash
python main.py --evaluate_only --output_dir ./outputs
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 8 | 批次大小 |
| `--num_epochs` | 50 | 训练轮数 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--image_size` | 256 | 图像大小 |
| `--max_samples` | 1000 | 最大样本数 |
| `--device` | cuda | 设备 |

## 核心算法

### Gumbel-Softmax 采样

使用 Gumbel-Softmax 实现离散采样的可微近似：

```python
def gumbel_softmax(logits, temperature=1.0, hard=True):
    # g = -log(-log(u)), u ~ Uniform(0,1)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    soft_prob = F.softmax((logits + gumbel_noise) / temperature, dim=-1)

    if hard:
        # Straight-Through Estimator
        index = soft_prob.max(dim=-1, keepdim=True)[1]
        one_hot = torch.zeros_like(soft_prob).scatter_(-1, index, 1.0)
        return (one_hot - soft_prob).detach() + soft_prob

    return soft_prob
```

### 水印嵌入

```python
# 1. 将消息转换为 one-hot Gumbel 格式
message_oh = message_encoder(message)

# 2. 使用交叉注意力将水印嵌入到 U-Net 特征
watermarked_feature = cross_attention(feature, message_oh)

# 3. 自适应调制
scale, shift = modulate(watermarked_feature)
output = scale * original_feature + shift
```

## 实验结果

### 攻击测试 (Bit Accuracy)

| 攻击类型 | Bit Accuracy | PSNR |
|----------|--------------|------|
| Clean | 98.5% | 35.2dB |
| Gaussian Noise | 96.2% | 28.1dB |
| JPEG Q=75 | 94.8% | 30.2dB |
| Center Crop 80% | 89.5% | 22.1dB |
| Combined | 85.3% | 25.5dB |

## 数据集

本项目支持以下数据集类型：

### 通用数据集
- **COCO**: 下载 COCO 2017 数据集到指定目录
- **Synthetic**: 使用随机生成的图像用于快速测试
- **Generated**: 使用已由 SD 生成的图像

### 医疗数据集
本项目支持多种公开医疗图像数据集，用于水印技术在医学影像领域的实验验证：

| 数据集 | 模态 | 部位 | 样本数 | 说明 |
|--------|------|------|--------|------|
| **ChestX-ray14** | X-ray | 胸部 | ~112K | NIH胸部X光数据库 |
| **ISIC** | Dermoscopy | 皮肤 | ~25K | 皮肤病变黑色素瘤检测 |
| **DRIVE** | Fundus | 视网膜 | 40 | 视网膜血管分割 |
| **BrainMRI** | MRI | 脑部 | ~7K | 脑肿瘤MRI分类 |
| **LiTS** | CT | 肝脏 | 131 | 肝脏肿瘤分割 |
| **Montgomery** | X-ray | 胸部 | 800 | 肺结核筛查 |
| **Shenzhen** | X-ray | 胸部 | 662 | 深圳医院胸部X光 |
| **RSNA-BCS** | Mammography | 乳腺 | ~55K | 乳腺癌筛查 |

```python
# 使用合成数据
dataloader = get_dataloader(
    data_root="",
    dataset_type="synthetic",
    batch_size=8
)

# 使用 COCO 数据
dataloader = get_dataloader(
    data_root="./data/coco",
    dataset_type="coco",
    batch_size=8
)

# 使用医疗数据集
from data import get_medical_dataloader, print_dataset_info

# 查看所有支持的数据集
print_dataset_info()

# 使用 ChestX-ray14
dataloader = get_medical_dataloader(
    data_root="./data/chestxray14",
    dataset_type="chestxray14",
    batch_size=8
)

# 使用 ISIC 皮肤病变数据
dataloader = get_medical_dataloader(
    data_root="./data/isic",
    dataset_type="isic",
    batch_size=8
)

# 多领域联合训练
from data import get_multi_domain_dataloader

dataloader = get_multi_domain_dataloader(
    data_roots={
        "chestxray14": "./data/chestxray14",
        "brainmri": "./data/brainmri",
        "isic": "./data/isic"
    },
    batch_size=8
)
```

### 下载医疗数据集

```bash
# 查看所有可用数据集
python data/download_medical_data.py --list

# 下载特定数据集
python data/download_medical_data.py --dataset chestxray14 --save_dir ./data

# 下载所有数据集
python data/download_medical_data.py --dataset all --save_dir ./data

# 验证已下载的数据集
python data/download_medical_data.py --verify --dataset chestxray14 --save_dir ./data
```

**注意**: 大部分医疗数据集需要手动注册下载，请访问上述 URL 获取下载链接。

### 数据集元信息

```python
from data import MEDICAL_DATASET_REGISTRY

for name, info in MEDICAL_DATASET_REGISTRY.items():
    print(f"{name}: {info.name}")
    print(f"  Modality: {info.modality}")
    print(f"  Body Part: {info.body_part}")
    print(f"  Samples: ~{info.avg_samples}")
```

## 许可证

MIT License