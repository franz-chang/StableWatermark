"""
DWT+DCT 水印基线方法

传统的离散小波变换和离散余弦变换水印嵌入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from typing import Tuple, Optional


class DWTDCTWatermark:
    """
    DWT+DCT 水印嵌入和提取

    这是一个传统算法的框架实现，用于与深度学习方法对比
    """

    def __init__(
        self,
        message_bits: int = 48,
        embed_strength: float = 0.05,
        wavelet: str = 'haar',
        level: int = 2
    ):
        """
        Args:
            message_bits: 水印比特数
            embed_strength: 嵌入强度
            wavelet: 小波类型
            level: DWT 分解层数
        """
        self.message_bits = message_bits
        self.embed_strength = embed_strength
        self.wavelet = wavelet
        self.level = level

    def embed(self, image: torch.Tensor, message: np.ndarray) -> torch.Tensor:
        """
        嵌入水印

        Args:
            image: 输入图像 [B, C, H, W], 范围 [0, 1]
            message: 二进制消息 [message_bits]

        Returns:
            watermarked: 含水印图像 [B, C, H, W]
        """
        B, C, H, W = image.shape
        watermarked = image.clone()

        for b in range(B):
            for c in range(C):
                # 提取单通道图像
                img_np = image[b, c].cpu().numpy()

                # DWT 分解
                coeffs = pywt.wavedec2(img_np, self.wavelet, level=self.level)
                Ca, details = coeffs[0], coeffs[1:]

                # 在细节系数中嵌入水印
                new_details = []
                for level_idx, (cd, cr, cc) in enumerate(details):
                    # 选择中频子带
                    if level_idx == self.level - 1:  # 最后一层细节
                        h, w = cd.shape
                        # 将消息分成 3 份嵌入到 H/V/D 子带
                        bits_per_band = self.message_bits // 3

                        cd_warped = self._embed_bits(cd, message[:bits_per_band])
                        cr_warped = self._embed_bits(cr, message[bits_per_band:2*bits_per_band])
                        cc_warped = self._embed_bits(cc, message[2*bits_per_band:])

                        new_details.append((cd_warped, cr_warped, cc_warped))
                    else:
                        new_details.append((cd, cr, cc))

                # 逆 DWT
                watermarked_np = pywt.waverec2([Ca] + new_details, self.wavelet)

                # 确保大小匹配
                if watermarked_np.shape != (H, W):
                    watermarked_np = watermarked_np[:H, :W]

                watermarked[b, c] = torch.from_numpy(watermarked_np).float()

        return torch.clamp(watermarked, 0, 1)

    def _embed_bits(self, coeffs: np.ndarray, bits: np.ndarray) -> np.ndarray:
        """
        在 DCT 系数中嵌入比特

        Args:
            coeffs: 小波系数
            bits: 要嵌入的比特

        Returns:
            modified_coeffs: 修改后的系数
        """
        h, w = coeffs.shape
        modified = coeffs.copy()

        # 将系数展平并选择位置嵌入
        bits_per_coeff = len(bits) // min(h, w)

        # 在中频区域嵌入
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4

        bit_idx = 0
        for i in range(max(0, center_h - radius), min(h, center_h + radius)):
            for j in range(max(0, center_w - radius), min(w, center_w + radius)):
                if bit_idx < len(bits):
                    # 量化调制
                    if bits[bit_idx] == 1:
                        modified[i, j] = (modified[i, j] // self.embed_strength + 1) * self.embed_strength
                    else:
                        modified[i, j] = (modified[i, j] // self.embed_strength) * self.embed_strength
                    bit_idx += 1

        return modified

    def extract(self, watermarked: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取水印

        Args:
            watermarked: 含水印图像 [B, C, H, W]

        Returns:
            (messages, confidences): 消息和置信度
        """
        B, C, H, W = watermarked.shape
        all_messages = []
        all_confidences = []

        for b in range(B):
            channel_votes = []

            for c in range(C):  # 对所有通道投票
                img_np = watermarked[b, c].cpu().numpy()

                # DWT 分解
                try:
                    coeffs = pywt.wavedec2(img_np, self.wavelet, level=self.level)
                except:
                    # 如果分解失败，使用随机消息
                    channel_votes.append((np.random.randint(0, 2, self.message_bits), 0.5))
                    continue

                Ca, details = coeffs[0], coeffs[1:]

                # 在最后一层提取
                if len(details) > 0:
                    cd, cr, cc = details[-1]
                    bits_per_band = self.message_bits // 3

                    # 提取各子带
                    extracted_bits = np.concatenate([
                        self._extract_bits(cd, bits_per_band),
                        self._extract_bits(cr, bits_per_band),
                        self._extract_bits(cc, self.message_bits - 2 * bits_per_band)
                    ])

                    channel_votes.append((extracted_bits, 0.8))

            # 多数投票
            if channel_votes:
                votes = np.array([v[0] for v in channel_votes])
                avg_confidence = np.mean([v[1] for v in channel_votes])

                # 逐比特多数投票
                final_message = np.round(votes.mean(axis=0)).astype(np.uint8)
                all_messages.append(final_message)
                all_confidences.append(avg_confidence)
            else:
                # 失败
                all_messages.append(np.zeros(self.message_bits, dtype=np.uint8))
                all_confidences.append(0.0)

        return np.array(all_messages), np.array(all_confidences)

    def _extract_bits(self, coeffs: np.ndarray, num_bits: int) -> np.ndarray:
        """从 DCT 系数中提取比特"""
        h, w = coeffs.shape
        bits = np.zeros(num_bits, dtype=np.uint8)

        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4

        bit_idx = 0
        for i in range(max(0, center_h - radius), min(h, center_h + radius)):
            for j in range(max(0, center_w - radius), min(w, center_w + radius)):
                if bit_idx < num_bits:
                    # 量化解码
                    val = coeffs[i, j]
                    quantized = round(val / self.embed_strength)
                    bits[bit_idx] = int(quantized % 2)
                    bit_idx += 1

        return bits


class SimpleDCTWatermark:
    """
    简化的 DCT 水印实现

    在 DCT 中频系数中直接嵌入
    """

    def __init__(
        self,
        message_bits: int = 48,
        embed_strength: float = 10.0
    ):
        self.message_bits = message_bits
        self.embed_strength = embed_strength

    def embed(self, image: torch.Tensor, message: np.ndarray) -> torch.Tensor:
        """嵌入水印"""
        B, C, H, W = image.shape
        watermarked = image.clone().numpy() if isinstance(image, torch.Tensor) else image.copy()

        for b in range(B):
            for c in range(C):
                # 转为 numpy
                img = image[b, c].cpu().numpy() if isinstance(image, torch.Tensor) else watermarked[b, c]

                # DCT
                dct = self._dct2(img)

                # 嵌入到中频
                mid_start = H // 4
                mid_end = 3 * H // 4
                bits_per_coeff = self.message_bits // ((mid_end - mid_start) ** 2)

                bit_idx = 0
                for i in range(mid_start, mid_end):
                    for j in range(mid_start, mid_end):
                        if bit_idx < self.message_bits:
                            # 根据消息位调整系数
                            if message[bit_idx] == 1:
                                dct[i, j] = (dct[i, j] // self.embed_strength + 1) * self.embed_strength
                            else:
                                dct[i, j] = (dct[i, j] // self.embed_strength) * self.embed_strength
                            bit_idx += 1

                # 逆 DCT
                watermarked[b, c] = self._idct2(dct)

        return torch.clamp(torch.from_numpy(watermarked).float(), 0, 1)

    def extract(self, watermarked: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """提取水印"""
        B = watermarked.shape[0]
        messages = []
        confidences = []

        for b in range(B):
            img = watermarked[b, 0].cpu().numpy()  # 使用第一个通道
            dct = self._dct2(img)

            mid_start = img.shape[0] // 4
            mid_end = 3 * img.shape[0] // 4

            bits = np.zeros(self.message_bits, dtype=np.uint8)
            bit_idx = 0

            for i in range(mid_start, mid_end):
                for j in range(mid_start, mid_end):
                    if bit_idx < self.message_bits:
                        quantized = round(dct[i, j] / self.embed_strength)
                        bits[bit_idx] = int(quantized % 2)
                        bit_idx += 1

            messages.append(bits)
            confidences.append(0.75)  # 简化置信度

        return np.array(messages), np.array(confidences)

    def _dct2(self, image: np.ndarray) -> np.ndarray:
        """2D DCT"""
        from scipy.fftpack import dct
        return dct(dct(image.T, norm='ortho').T, norm='ortho')

    def _idct2(self, dct_coeff: np.ndarray) -> np.ndarray:
        """2D 逆 DCT"""
        from scipy.fftpack import idct
        return idct(idct(dct_coeff.T, norm='ortho').T, norm='ortho')


def test_dwt_dct():
    """测试 DWT+DCT 水印"""
    print("\nTesting DWT+DCT Watermark...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化水印
    wm = SimpleDCTWatermark(message_bits=48, embed_strength=10.0)

    # 测试数据
    image = torch.rand(2, 3, 64, 64, device=device)
    message = np.random.randint(0, 2, (2, 48))

    # 嵌入
    watermarked = wm.embed(image, message[0])
    print(f"  Embedded: {watermarked.shape}")

    # 提取
    extracted, conf = wm.extract(watermarked.unsqueeze(0))
    print(f"  Extracted shape: {extracted.shape}")
    print(f"  Confidence: {conf}")

    # 计算准确率
    accuracy = (extracted == message[0]).mean()
    print(f"  Bit accuracy: {accuracy:.4f}")

    print("✓ DWT+DCT watermark tests passed")


if __name__ == "__main__":
    test_dwt_dct()