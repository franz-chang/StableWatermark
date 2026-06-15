#!/usr/bin/env python3
"""
医疗数据集下载脚本

自动下载和解压公开医疗数据集用于水印实验

Usage:
    python download_medical_data.py --dataset chestxray14 --save_dir ./data
    python download_medical_data.py --dataset all --save_dir ./data
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
from typing import Optional, Dict, List


# 数据集下载信息
DATASET_DOWNLOADS: Dict[str, Dict] = {
    "chestxray14": {
        "name": "NIH ChestX-ray14",
        "files": [
            {
                "url": "https://无需登录.url",
                "filename": "images.zip",
                "description": "图像文件 (需要手动从Box下载)",
            }
        ],
        "manual_download": True,  # 需要手动注册下载
        "note": "请访问 https://nihcc.app.box.com/v/ChestXray-NIHCC 注册下载"
    },
    "isic": {
        "name": "ISIC 2018",
        "files": [
            {
                "url": "https://isic-archive.com/getArchiveFile?archiveId=1",
                "filename": "isic2018_train.tar",
                "description": "ISIC 2018 Training Data",
                "requires_auth": True
            }
        ],
        "manual_download": True,
        "note": "请访问 https://challenge.isic-archive.com/data/ 下载"
    },
    "drive": {
        "name": "DRIVE",
        "files": [
            {
                "url": "https://drive.grand-challenge.org/site/DRIVE/Download/DRIVE_datasets.zip",
                "filename": "DRIVE_datasets.zip",
                "description": "Complete DRIVE dataset"
            }
        ],
        "manual_download": False,
        "direct_download": True,
        "extract": "images"
    },
    "brainmri": {
        "name": "Brain Tumor MRI",
        "files": [
            {
                "url": "https://raw.githubusercontent.com/s进去了/URL/main/data.zip",
                "filename": "brain_mri.zip"
            }
        ],
        "manual_download": True,
        "note": "请访问 https://www.kaggle.com/datasets/miyaitingchen/brain-tumor-mri-dataset 下载"
    },
    "lits": {
        "name": "LiTS - Liver Tumor Segmentation",
        "files": [
            {
                "url": "https://competitions.codalab.org/MyDotCom/Upload/challenge.zip",
                "filename": "lits.zip"
            }
        ],
        "manual_download": True,
        "note": "请访问 https://competitions.codalab.org/competitions/17094 注册下载"
    },
    "montgomery": {
        "name": "Montgomery County X-ray",
        "files": [
            {
                "url": "https://www.kaggle.com/api/v1/datasets/download/darrylhaller/montgomerycounty-xray",
                "filename": "montgomery.zip"
            }
        ],
        "manual_download": True,
        "note": "请访问 https://www.kaggle.com/datasets/darrylhaller/montgomerycounty-xray 下载"
    },
    "shenzhen": {
        "name": "Shenzhen Hospital X-ray",
        "files": [
            {
                "url": "https://www.kaggle.com/api/v1/datasets/download/darrylhaller/shenzhen-hospital-xray",
                "filename": "shenzhen.zip"
            }
        ],
        "manual_download": True,
        "note": "请访问 https://www.kaggle.com/datasets/darrylhaller/shenzhen-hospital-xray 下载"
    },
    "rsnaBreast": {
        "name": "RSNA Breast Cancer Screening",
        "files": [
            {
                "url": "https://www.kaggle.com/api/v1/competitions/data/download-file?id=rsna-breast-cancer-detection",
                "filename": "rsna_mammo.zip"
            }
        ],
        "manual_download": True,
        "note": "请访问 https://www.kaggle.com/competitions/rsna-breast-cancer-detection 下载"
    }
}


def print_dataset_info():
    """打印所有数据集信息"""
    print("\n" + "=" * 70)
    print("Supported Medical Datasets")
    print("=" * 70 + "\n")

    for key, info in DATASET_DOWNLOADS.items():
        print(f"[{key}]")
        print(f"  Name: {info['name']}")
        if 'note' in info:
            print(f"  Note: {info['note']}")
        else:
            print(f"  Status: Auto-download available")
        print()

    print("-" * 70)
    print("Datasets marked 'auto-download' will be downloaded automatically.")
    print("Others require manual download from the provided links.")
    print("-" * 70 + "\n")


def download_file(url: str, save_path: str, progress: bool = True) -> bool:
    """
    下载文件

    Args:
        url: 下载URL
        save_path: 保存路径
        progress: 是否显示进度

    Returns:
        bool: 是否成功
    """
    try:
        def reporthook(block_num, block_size, total_size):
            if progress:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 // total_size)
                sys.stdout.write(f"\r下载进度: {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, save_path, reporthook if progress else None)
        print()  # 换行
        return True

    except Exception as e:
        print(f"\n下载失败: {e}")
        return False


def extract_archive(archive_path: str, extract_to: str) -> bool:
    """解压归档文件"""
    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith('.tar') or archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"不支持的压缩格式: {archive_path}")
            return False

        return True
    except Exception as e:
        print(f"解压失败: {e}")
        return False


def setup_directories(save_dir: str, dataset_name: str) -> str:
    """创建数据集目录"""
    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir


def download_medical_dataset(
    dataset_name: str,
    save_dir: str,
    force_redownload: bool = False
) -> bool:
    """
    下载单个医疗数据集

    Args:
        dataset_name: 数据集名称
        save_dir: 保存目录
        force_redownload: 强制重新下载

    Returns:
        bool: 是否成功
    """
    if dataset_name not in DATASET_DOWNLOADS:
        print(f"未知数据集: {dataset_name}")
        return False

    info = DATASET_DOWNLOADS[dataset_name]

    # 检查是否需要手动下载
    if info.get('manual_download', False):
        print(f"\n{'='*70}")
        print(f"Manual Download Required: {info['name']}")
        print(f"{'='*70}")
        print(f"\n请手动下载数据集:")
        if 'note' in info:
            print(f"  {info['note']}")
        print(f"\n下载后请解压到: {save_dir}/{dataset_name}/")
        print(f"\n创建说明文件...")
        # 创建说明文件
        readme_path = os.path.join(save_dir, dataset_name, "DOWNLOAD_INSTRUCTIONS.txt")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# {info['name']}\n\n")
            if 'note' in info:
                f.write(f"## Download Instructions\n{info['note']}\n\n")
            if 'files' in info:
                f.write(f"## Expected Files\n")
                for file in info['files']:
                    f.write(f"  - {file.get('description', file.get('url', 'Unknown'))}\n")
        return True

    # 自动下载
    dataset_dir = setup_directories(save_dir, dataset_name)
    print(f"\n准备下载 {info['name']} 到 {dataset_dir}")

    success = True
    for file_info in info.get('files', []):
        filename = file_info['filename']
        file_path = os.path.join(dataset_dir, filename)

        if os.path.exists(file_path) and not force_redownload:
            print(f"文件已存在: {file_path}，跳过下载")
            continue

        if file_info.get('requires_auth', False):
            print(f"需要认证下载: {file_info['url']}")
            print("请手动下载此文件")
            continue

        if 'url' not in file_info:
            continue

        print(f"\n下载: {file_info.get('description', filename)}")
        if download_file(file_info['url'], file_path):
            # 解压
            if 'extract' not in file_info or file_info['extract']:
                print("解压中...")
                extract_archive(file_path, dataset_dir)
                # 清理压缩包
                os.remove(file_path)
                print("解压完成!")
        else:
            success = False

    return success


def download_all(save_dir: str) -> None:
    """下载所有可用的数据集"""
    print("\n开始下载所有医学数据集...")
    print("注意: 大部分数据集需要手动下载，请按提示操作。\n")

    for dataset_name in DATASET_DOWNLOADS.keys():
        print(f"\n{'='*70}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*70}")
        download_medical_dataset(dataset_name, save_dir)

    print("\n" + "="*70)
    print("下载完成!")
    print("="*70)
    print("\n请查看各数据集目录中的 DOWNLOAD_INSTRUCTIONS.txt 文件")
    print("了解需要手动下载的数据集的操作步骤。")


def verify_dataset(dataset_dir: str, dataset_name: str) -> bool:
    """
    验证数据集完整性

    Args:
        dataset_dir: 数据集目录
        dataset_name: 数据集名称

    Returns:
        bool: 是否有效
    """
    if not os.path.exists(dataset_dir):
        return False

    # 统计图像数量
    image_count = 0
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dcm')):
                image_count += 1

    print(f"  发现 {image_count} 张图像")
    return image_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Download medical datasets for StableWatermark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 显示所有可用数据集
  python download_medical_data.py --list

  # 下载单数据集
  python download_medical_data.py --dataset chestxray14 --save_dir ./data

  # 下载所有数据集
  python download_medical_data.py --dataset all --save_dir ./data

  # 验证已下载的数据集
  python download_medical_data.py --verify --dataset chestxray14 --save_dir ./data
        """
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=None,
        help='数据集名称 (chestxray14, isic, drive, brainmri, etc.) 或 "all"'
    )
    parser.add_argument(
        '--save_dir', '-s',
        type=str,
        default='./data',
        help='保存目录'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出所有可用数据集'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='验证已下载的数据集'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新下载'
    )

    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    if args.list:
        print_dataset_info()
        return

    if args.verify:
        if args.dataset:
            dataset_path = os.path.join(args.save_dir, args.dataset)
            print(f"验证数据集: {args.dataset}")
            if verify_dataset(dataset_path, args.dataset):
                print("✓ 数据集验证通过")
            else:
                print("✗ 数据集无效或为空")
        return

    if not args.dataset:
        print("请指定数据集名称，或使用 --list 查看所有可用数据集")
        parser.print_help()
        return

    if args.dataset == "all":
        download_all(args.save_dir)
    else:
        download_medical_dataset(args.dataset, args.save_dir, args.force)


if __name__ == "__main__":
    main()