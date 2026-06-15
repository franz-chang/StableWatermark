"""
Gumbel-Softmax 模块

实现可微的离散采样 (Differentiable Relaxation of Discrete Samples)
用于将 48-bit 水印信息嵌入到生成过程中
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax as F_gumbel_softmax


def gumbel_softmax(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = True,
    dim: int = -1,
    eps: float = 1e-20
) -> torch.Tensor:
    """
    Gumbel-Softmax 采样

    Args:
        logits: 原始 logits，形状 [..., num_classes]
        temperature: 温度参数，越小越接近 one-hot
        hard: 是否使用 hard sampling (straight-through estimator)
        dim: 沿着哪个维度进行 softmax
        eps: 数值稳定性参数

    Returns:
        soft_one_hot: Gumbel-Softmax 采样结果

    Example:
        >>> logits = torch.randn(2, 48)  # 2 个样本, 48 个类别
        >>> sample = gumbel_softmax(logits, temperature=1.0, hard=True)
        >>> print(sample.shape)  # torch.Size([2, 48])
    """
    # 生成 Gumbel 噪声: g = -log(-log(u)), u ~ Uniform(0,1)
    # 等价于 F.gumbel_softmax
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)

    # 计算 Gumbel-Softmax: softmax((logits + g) / temperature)
    gumbel_logits = logits + gumbel_noise
    soft_prob = F.softmax(gumbel_logits / temperature, dim=dim)

    if hard:
        # Straight-Through Estimator: 前向传播用 hard sample，反向传播用 soft prob
        # 找到最大概率的位置并设为 1
        index = soft_prob.max(dim=dim, keepdim=True)[1]
        one_hot = torch.zeros_like(soft_prob).scatter_(dim, index, 1.0)
        # detach 使得梯度能够流过 ST estimator
        hard_sample = (one_hot - soft_prob).detach() + soft_prob
        return hard_sample

    return soft_prob


class GumbelSoftmax(nn.Module):
    """
    Gumbel-Softmax 层
    """

    def __init__(
        self,
        num_classes: int,
        temperature: float = 1.0,
        hard: bool = True,
        dim: int = -1
    ):
        """
        Args:
            num_classes: 类别数量 (水印比特数)
            temperature: 初始温度
            hard: 是否使用 hard sampling
            dim: softmax 维度
        """
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.hard = hard
        self.dim = dim

    def set_temperature(self, temperature: float):
        """动态设置温度"""
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 形状 [batch_size, num_classes]

        Returns:
            sample: 形状 [batch_size, num_classes]
        """
        return gumbel_softmax(
            logits,
            temperature=self.temperature,
            hard=self.hard,
            dim=self.dim
        )

    def extra_repr(self) -> str:
        return f"num_classes={self.num_classes}, temperature={self.temperature}, hard={self.hard}"


class MessageEncoder(nn.Module):
    """
    将原始消息转换为 Gumbel-Softmax 格式

    Example:
        >>> encoder = MessageEncoder(message_dim=256, num_bits=48)
        >>> message = torch.randint(0, 2, (2, 48)).float()
        >>> one_hot = encoder(message)  # [2, 48] one-hot
    """

    def __init__(
        self,
        message_dim: int = 256,
        num_bits: int = 48,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.num_bits = num_bits

        # 将二进制消息嵌入到高维空间
        self.embedding = nn.Linear(num_bits, hidden_dim)

        # 消息增强网络
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 转换为 one-hot logits
        self.logits_net = nn.Linear(hidden_dim, num_bits)

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        """
        Args:
            message: 二进制消息 [batch_size, num_bits]

        Returns:
            one_hot: Gumbel-Softmax 采样的 one-hot 向量 [batch_size, num_bits]
        """
        # 嵌入原始消息
        x = self.embedding(message)
        x = self.message_net(x)
        # 输出 logits
        logits = self.logits_net(x)

        # 使用 Gumbel-Softmax
        one_hot = gumbel_softmax(logits, temperature=1.0, hard=True)

        return one_hot


def test_gumbel_softmax():
    """测试 Gumbel-Softmax 功能"""
    print("Testing Gumbel-Softmax...")

    batch_size = 4
    num_bits = 48

    # 随机 logits
    logits = torch.randn(batch_size, num_bits)

    # 测试软采样
    soft = gumbel_softmax(logits, temperature=1.0, hard=False)
    assert soft.shape == (batch_size, num_bits)
    assert torch.allclose(soft.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    # 测试硬采样
    hard = gumbel_softmax(logits, temperature=0.1, hard=True)
    assert hard.shape == (batch_size, num_bits)
    # 硬采样应该是 one-hot
    values, indices = hard.max(dim=1)
    assert torch.allclose(values, torch.ones(batch_size), atol=1e-5)

    # 测试梯度流
    logits.requires_grad_(True)
    sample = gumbel_softmax(logits, temperature=0.5, hard=True)
    loss = sample.sum()
    loss.backward()

    assert logits.grad is not None
    print("✓ Gumbel-Softmax 测试通过")

    # 测试模块
    module = GumbelSoftmax(num_classes=num_bits)
    out = module(logits.detach())
    assert out.shape == (batch_size, num_bits)
    print("✓ GumbelSoftmax 模块测试通过")

    # 测试消息编码器
    encoder = MessageEncoder(message_dim=256, num_bits=num_bits)
    message = torch.randint(0, 2, (batch_size, num_bits)).float()
    one_hot = encoder(message)
    assert one_hot.shape == (batch_size, num_bits)
    print("✓ MessageEncoder 模块测试通过")


if __name__ == "__main__":
    test_gumbel_softmax()