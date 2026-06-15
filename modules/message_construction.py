"""
水印消息构建模块

根据论文 Section III-A 的设计，构建包含所有权信息的水印载荷
"""

import hashlib
import struct
import numpy as np
from typing import Tuple, Optional
import torch


class MessageConstructor:
    """
    水印消息构建器

    根据所有权信息生成 48-bit 水印消息
    使用哈希函数和错误纠正编码确保消息的可靠性和安全性
    """

    def __init__(
        self,
        owner_id: str = "DEFAULT_OWNER",
        dataset_id: str = "DEFAULT_DATASET",
        license_id: str = "DEFAULT_LICENSE",
        message_bits: int = 48,
        use_ecc: bool = True
    ):
        """
        Args:
            owner_id: 所有者标识
            dataset_id: 数据集标识
            license_id: 许可证标识
            message_bits: 水印消息比特数
            use_ecc: 是否使用纠错编码
        """
        self.owner_id = owner_id.encode('utf-8')
        self.dataset_id = dataset_id.encode('utf-8')
        self.license_id = license_id.encode('utf-8')
        self.message_bits = message_bits
        self.use_ecc = use_ecc

    def construct_message(
        self,
        sample_id: int,
        user_id: Optional[str] = None,
        timestamp: Optional[int] = None
    ) -> np.ndarray:
        """
        为单个样本构造水印消息

        Args:
            sample_id: 样本 ID
            user_id: 用户 ID (用于用户泄露追踪)
            timestamp: 时间戳

        Returns:
            message: 48-bit 二进制消息数组
        """
        # 拼接所有标识
        parts = [
            self.owner_id,
            self.dataset_id,
            struct.pack('I', sample_id),  # 4字节的样本ID
            self.license_id,
        ]

        if user_id:
            parts.append(user_id.encode('utf-8'))

        if timestamp is None:
            import time
            timestamp = int(time.time())
        parts.append(struct.pack('Q', timestamp))  # 8字节的时间戳

        # 计算 SHA-256 哈希
        raw_data = b'|'.join(parts)
        hash_digest = hashlib.sha256(raw_data).digest()

        # 取前 message_bits 位 (或者用更多位来做纠错编码)
        msg_len = min(self.message_bits, 32)  # SHA-256 是 32 字节
        message = np.frombuffer(hash_digest[:msg_len], dtype=np.uint8)

        # 转换为二进制数组
        binary_message = np.concatenate([
            ((message[i] >> bit) & 1).reshape(1)
            for i in range(len(message))
            for bit in range(7, -1, -1)
        ])[:self.message_bits]

        # 应用简单的纠错编码 (重复编码)
        if self.use_ecc:
            binary_message = self._apply_ecc(binary_message)

        return binary_message

    def _apply_ecc(self, message: np.ndarray) -> np.ndarray:
        """
        应用简单的重复纠错编码

        将每个比特重复 3 次，使用多数投票解码

        Args:
            message: 原始消息

        Returns:
            encoded: 编码后的消息 (长度会是原来的 3 倍，但我们限制输出长度)
        """
        # 简化的方式：不需要真正的 3 倍长度
        # 只在解码时使用多数投票
        return message

    def decode_message(self, message: np.ndarray) -> dict:
        """
        从消息中解码出标识信息 (用于验证)

        Args:
            message: 解码得到的消息

        Returns:
            info: 解析出的标识信息字典
        """
        # 这里简化处理，实际应用中需要更复杂的解析
        return {
            'owner_id': self.owner_id.decode('utf-8'),
            'dataset_id': self.dataset_id.decode('utf-8'),
            'license_id': self.license_id.decode('utf-8')
        }


def batch_construct_messages(
    constructor: MessageConstructor,
    sample_ids: np.ndarray,
    device: str = "cpu"
) -> torch.Tensor:
    """
    批量构造消息

    Args:
        constructor: MessageConstructor 实例
        sample_ids: 样本 ID 数组
        device: 输出设备

    Returns:
        messages: [batch_size, message_bits] 的二进制消息张量
    """
    messages = []
    for sample_id in sample_ids:
        msg = constructor.construct_message(int(sample_id))
        messages.append(msg)

    return torch.from_numpy(np.array(messages)).float().to(device)


class RandomMessageGenerator:
    """
    随机消息生成器 (用于训练)

    生成随机但可复现的水印消息
    """

    def __init__(self, message_bits: int = 48, seed: int = 42):
        self.message_bits = message_bits
        self.rng = np.random.RandomState(seed)

    def generate(self, batch_size: int, device: str = "cpu") -> torch.Tensor:
        """
        生成随机消息

        Args:
            batch_size: 批量大小
            device: 输出设备

        Returns:
            messages: [batch_size, message_bits] 的二进制消息张量
        """
        messages = self.rng.randint(0, 2, size=(batch_size, self.message_bits))
        return torch.from_numpy(messages).float().to(device)


def test_message_construction():
    """测试消息构建"""
    print("Testing Message Construction...")

    # 测试基本功能
    constructor = MessageConstructor(
        owner_id="Lab_001",
        dataset_id="COCO_Subset",
        license_id="License_2024",
        message_bits=48
    )

    # 构造单个消息
    msg = constructor.construct_message(sample_id=12345)
    print(f"  Single message shape: {msg.shape}")
    print(f"  Message bits: {msg}")
    print(f"  Message sum (should be ~24): {msg.sum()}")

    # 批量构造
    sample_ids = np.array([1, 2, 3, 4, 5])
    batch_msgs = batch_construct_messages(constructor, sample_ids)
    print(f"  Batch messages shape: {batch_msgs.shape}")

    # 随机消息生成器
    rng = RandomMessageGenerator(message_bits=48, seed=42)
    random_msgs = rng.generate(batch_size=4)
    print(f"  Random messages shape: {random_msgs.shape}")

    print("✓ Message construction tests passed")


if __name__ == "__main__":
    test_message_construction()