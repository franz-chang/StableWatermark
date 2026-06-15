"""
损失函数模块

包含水印训练所需的各种损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class WatermarkLoss(nn.Module):
    """
    水印综合损失

    包含:
    - 重构损失: 保证图像质量
    - 消息损失: 保证水印提取准确率
    - 对抗损失: 使水印难以被检测
    - 感知损失: 进一步提高图像质量 (可选)
    """

    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_msg: float = 10.0,
        lambda_adv: float = 0.5,
        lambda_perc: float = 0.1,
        use_perceptual_loss: bool = False
    ):
        """
        Args:
            lambda_rec: 重构损失权重
            lambda_msg: 消息损失权重
            lambda_adv: 对抗损失权重
            lambda_perc: 感知损失权重
            use_perceptual_loss: 是否使用感知损失
        """
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_msg = lambda_msg
        self.lambda_adv = lambda_adv
        self.lambda_perc = lambda_perc
        self.use_perceptual_loss = use_perceptual_loss

        # 重构损失: L1 损失对异常值更鲁棒
        self.rec_loss_fn = nn.L1Loss()

        # 消息损失: BCEWithLogits 适合 logits 输出
        self.msg_loss_fn = nn.BCEWithLogitsLoss()

        # 对抗损失
        self.adv_loss_fn = nn.BCEWithLogitsLoss()

        # 感知损失网络 (简化版，使用简单的卷积网络)
        if use_perceptual_loss:
            self.perc_net = PerceptualLossNetwork()
            for param in self.perc_net.parameters():
                param.requires_grad = False

    def compute_rec_loss(
        self,
        watermarked: torch.Tensor,
        original: torch.Tensor
    ) -> torch.Tensor:
        """计算重构损失"""
        return self.rec_loss_fn(watermarked, original)

    def compute_msg_loss(
        self,
        predicted_logits: torch.Tensor,
        target_message: torch.Tensor
    ) -> torch.Tensor:
        """计算消息提取损失"""
        return self.msg_loss_fn(predicted_logits, target_message)

    def compute_adv_loss_for_encoder(
        self,
        discriminator_output: torch.Tensor,
        is_real: bool = True
    ) -> torch.Tensor:
        """
        计算编码器的对抗损失

        Args:
            discriminator_output: 判别器输出
            is_real: 目标是让判别器认为是真图 (True) 还是假图 (False)

        Returns:
            loss: 对抗损失
        """
        if is_real:
            # 编码器想让判别器认为含水印图像是真实的
            target = torch.ones_like(discriminator_output)
        else:
            # 编码器想让判别器认为含水印图像是假的
            target = torch.zeros_like(discriminator_output)

        return self.adv_loss_fn(discriminator_output, target)

    def compute_adv_loss_for_discriminator(
        self,
        real_output: torch.Tensor,
        fake_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算判别器的对抗损失

        Args:
            real_output: 真实图像的判别输出
            fake_output: 含水印图像的判别输出

        Returns:
            (d_loss, d_real_loss, d_fake_loss): 总损失, 真图损失, 假图损失
        """
        # 真图应该被判别为 1
        real_loss = self.adv_loss_fn(real_output, torch.ones_like(real_output))
        # 假图应该被判别为 0
        fake_loss = self.adv_loss_fn(fake_output, torch.zeros_like(fake_output))

        d_loss = real_loss + fake_loss
        return d_loss, real_loss, fake_loss

    def forward(
        self,
        watermarked: torch.Tensor,
        original: torch.Tensor,
        predicted_logits: torch.Tensor,
        target_message: torch.Tensor,
        discriminator_output: Optional[torch.Tensor] = None,
        disc_real_output: Optional[torch.Tensor] = None,
        disc_fake_output: Optional[torch.Tensor] = None
    ) -> dict:
        """
        计算综合损失

        Args:
            watermarked: 含水印图像
            original: 原始图像
            predicted_logits: 解码器预测的 logits
            target_message: 目标水印消息
            discriminator_output: 编码器视角的判别器输出 (可选)
            disc_real_output: 判别器对真图的输出 (可选,用于训练判别器)
            disc_fake_output: 判别器对假图的输出 (可选,用于训练判别器)

        Returns:
            losses: 损失字典
        """
        losses = {}

        # 重构损失
        if self.lambda_rec > 0:
            losses['loss_rec'] = self.compute_rec_loss(watermarked, original)

        # 消息损失
        if self.lambda_msg > 0:
            losses['loss_msg'] = self.compute_msg_loss(predicted_logits, target_message)

        # 对抗损失 (编码器)
        if self.lambda_adv > 0 and discriminator_output is not None:
            losses['loss_adv_encoder'] = self.compute_adv_loss_for_encoder(
                discriminator_output, is_real=True
            )

        # 对抗损失 (判别器)
        if disc_real_output is not None and disc_fake_output is not None:
            losses['loss_adv_discriminator'], _, _ = self.compute_adv_loss_for_discriminator(
                disc_real_output, disc_fake_output
            )

        # 感知损失
        if self.use_perceptual_loss and self.lambda_perc > 0:
            losses['loss_perc'] = F.l1_loss(
                self.perc_net(watermarked),
                self.perc_net(original)
            )

        # 总损失
        total_loss = torch.zeros(1, device=watermarked.device)

        if 'loss_rec' in losses:
            total_loss = total_loss + self.lambda_rec * losses['loss_rec']
        if 'loss_msg' in losses:
            total_loss = total_loss + self.lambda_msg * losses['loss_msg']
        if 'loss_adv_encoder' in losses:
            total_loss = total_loss + self.lambda_adv * losses['loss_adv_encoder']
        if 'loss_perc' in losses:
            total_loss = total_loss + self.lambda_perc * losses['loss_perc']

        losses['loss'] = total_loss

        return losses


class PerceptualLossNetwork(nn.Module):
    """
    简化版感知损失网络

    使用预训练的 VGG 特征距离
    """

    def __init__(self):
        super().__init__()

        # 简化的特征提取器
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class ContrastiveLoss(nn.Module):
    """
    对比损失 (可选)

    用于增强同一消息在不同图像中的嵌入一致性
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        same_message: bool
    ) -> torch.Tensor:
        """
        Args:
            features1: 特征1
            features2: 特征2
            same_message: 是否是相同的消息

        Returns:
            loss: 对比损失
        """
        # 归一化特征
        f1 = F.normalize(features1, dim=-1)
        f2 = F.normalize(features2, dim=-1)

        # 计算相似度
        sim = torch.matmul(f1, f2.T) / self.temperature

        if same_message:
            # 相同消息应该相似
            target = torch.eye(sim.shape[0], device=sim.device)
        else:
            # 不同消息应该不相似
            target = 1 - torch.eye(sim.shape[0], device=sim.device)

        loss = F.binary_cross_entropy_with_logits(sim, target)
        return loss


class DiversityLoss(nn.Module):
    """
    多样性损失

    确保水印对图像的修改分布均匀，不要总是修改相同的位置
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        modifications: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Args:
            modifications: 修改量 [batch_size, channels, height, width]
            batch_size: 批次大小

        Returns:
            loss: 多样性损失 (空间方差的最大值作为惩罚)
        """
        # 计算每个样本的空间方差
        variances = modifications.var(dim=(2, 3)).mean(dim=1)  # [batch_size]

        # 我们想要方差较大 (修改分布广)
        # 但不要太相似 (同一批次应该有不同模式)
        if batch_size > 1:
            # 计算批次内的相似性
            cross_var = modifications.mean(dim=0, keepdim=True).var(dim=(2, 3))
            return -cross_var.mean()
        else:
            return -variances.mean()


def test_losses():
    """测试损失函数"""
    print("Testing losses...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 4
    message_bits = 48

    # 测试数据
    watermarked = torch.rand(batch_size, 3, 256, 256, device=device)
    original = torch.rand(batch_size, 3, 256, 256, device=device)
    target_message = torch.randint(0, 2, (batch_size, message_bits), device=device).float()
    predicted_logits = torch.randn(batch_size, message_bits, device=device)

    disc_real = torch.rand(batch_size, 1, 16, 16, device=device)
    disc_fake = torch.rand(batch_size, 1, 16, 16, device=device)
    disc_output = torch.rand(batch_size, 1, 16, 16, device=device)

    # 测试 WatermarkLoss
    loss_fn = WatermarkLoss(lambda_rec=1.0, lambda_msg=10.0, lambda_adv=0.5)

    losses = loss_fn(
        watermarked, original,
        predicted_logits, target_message,
        discriminator_output=disc_output,
        disc_real_output=disc_real,
        disc_fake_output=disc_fake
    )

    print(f"  Total loss: {losses['loss'].item():.4f}")
    print(f"  Rec loss: {losses.get('loss_rec', 0).item():.4f}")
    print(f"  Msg loss: {losses.get('loss_msg', 0).item():.4f}")
    print(f"  Adv loss (encoder): {losses.get('loss_adv_encoder', 0).item():.4f}")
    print(f"  Adv loss (discriminator): {losses.get('loss_adv_discriminator', 0).item():.4f}")

    # 测试梯度
    losses['loss'].backward()
    print("  Gradients computed successfully")

    print("✓ Loss functions tested successfully")


if __name__ == "__main__":
    test_losses()