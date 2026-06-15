"""
训练器模块

管理水印模型的训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
import os
import json
from datetime import datetime

from models import WatermarkEncoder, WatermarkDecoder, Discriminator
from models.unet_encoder import LightweightUNetEncoder
from utils.metrics import MetricsTracker, calculate_bit_accuracy, calculate_psnr, calculate_ssim
from utils.attack import get_all_attacks, AttackConfig
from training.losses import WatermarkLoss
from .losses import WatermarkLoss as WL


class Trainer:
    """
    水印模型训练器

    管理整个训练流程，包括:
    - 编码器和解码器的联合训练
    - 判别器的对抗训练
    - 日志记录和模型保存
    - 评估和攻击测试
    """

    def __init__(
        self,
        encoder: WatermarkEncoder,
        decoder: WatermarkDecoder,
        discriminator: Optional[Discriminator] = None,
        config: Dict = None,
        device: str = "cuda"
    ):
        """
        Args:
            encoder: 水印编码器
            decoder: 水印解码器
            discriminator: 判别器 (可选)
            config: 配置字典
            device: 设备
        """
        self.device = device
        self.config = config or self._default_config()

        # 模型
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.discriminator = discriminator.to(device) if discriminator else None

        # 损失函数
        self.loss_fn = WatermarkLoss(
            lambda_rec=self.config.get('lambda_rec', 1.0),
            lambda_msg=self.config.get('lambda_msg', 10.0),
            lambda_adv=self.config.get('lambda_adv', 0.5)
        )

        # 优化器
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)

        self.opt_encoder = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr, weight_decay=weight_decay
        )

        if self.discriminator:
            self.opt_discriminator = optim.Adam(
                self.discriminator.parameters(),
                lr=lr, weight_decay=weight_decay
            )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_encoder,
            T_max=self.config.get('num_epochs', 50)
        )

        # 训练历史
        self.history = {
            'loss': [],
            'loss_rec': [],
            'loss_msg': [],
            'loss_adv': [],
            'bit_accuracy': [],
            'psnr': [],
            'ssim': []
        }

        # 当前状态
        self.current_epoch = 0
        self.global_step = 0

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'lambda_rec': 1.0,
            'lambda_msg': 10.0,
            'lambda_adv': 0.5,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'num_epochs': 50,
            'log_every': 50,
            'save_every': 500
        }

    def train_step(
        self,
        images: torch.Tensor,
        messages: torch.Tensor
    ) -> Dict:
        """
        单步训练

        Args:
            images: 原始图像 [batch_size, 3, H, W]
            messages: 水印消息 [batch_size, num_bits]

        Returns:
            metrics: 训练指标
        """
        batch_size = images.shape[0]

        # 编码
        watermarked = self.encoder.embed_message(images, messages)

        # 解码
        predicted_logits = self.decoder(watermarked)

        # 更新编码器和解码器
        self.opt_encoder.zero_grad()

        if self.discriminator:
            # 对抗训练
            disc_output = self.discriminator(watermarked)

            losses = self.loss_fn(
                watermarked, images,
                predicted_logits, messages,
                discriminator_output=disc_output
            )
        else:
            losses = self.loss_fn(
                watermarked, images,
                predicted_logits, messages
            )

        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            max_norm=1.0
        )
        self.opt_encoder.step()

        # 更新判别器
        if self.discriminator and self.global_step % 2 == 0:
            self.opt_discriminator.zero_grad()

            # 真图判别
            real_output = self.discriminator(images)

            # 假图判别 (再次前向传播以获得新的假图)
            with torch.no_grad():
                watermarked_fake = self.encoder.embed_message(images, messages)
            fake_output = self.discriminator(watermarked_fake)

            _, d_real, d_fake = self.loss_fn.compute_adv_loss_for_discriminator(
                real_output, fake_output
            )
            d_loss = d_real + d_fake

            d_loss.backward()
            self.opt_discriminator.step()
        else:
            d_loss = torch.tensor(0.0)

        # 计算指标
        with torch.no_grad():
            predicted_probs = torch.sigmoid(predicted_logits)
            bit_acc = calculate_bit_accuracy(predicted_probs, messages)
            psnr_val = calculate_psnr(watermarked, images)
            ssim_val = calculate_ssim(watermarked, images)

        return {
            'loss': losses['loss'].item(),
            'loss_rec': losses.get('loss_rec', torch.tensor(0.0)).item(),
            'loss_msg': losses.get('loss_msg', torch.tensor(0.0)).item(),
            'loss_adv': d_loss.item(),
            'bit_accuracy': bit_acc,
            'psnr': psnr_val,
            'ssim': ssim_val
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict:
        """
        训练一个 epoch

        Args:
            dataloader: 数据加载器
            epoch: 当前 epoch

        Returns:
            metrics: 平均指标
        """
        self.encoder.train()
        self.decoder.train()

        if self.discriminator:
            self.discriminator.train()

        tracker = MetricsTracker()
        total_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (images, messages) in enumerate(pbar):
            images = images.to(self.device)
            messages = messages.to(self.device)

            metrics = self.train_step(images, messages)
            tracker.update(
                torch.sigmoid(self.decoder(self.encoder.embed_message(images, messages))),
                messages,
                self.encoder.embed_message(images, messages),
                images
            )

            total_steps += 1
            self.global_step += 1

            # 记录历史
            self.history['loss'].append(metrics['loss'])
            self.history['loss_rec'].append(metrics['loss_rec'])
            self.history['loss_msg'].append(metrics['loss_msg'])
            self.history['bit_accuracy'].append(metrics['bit_accuracy'])
            self.history['psnr'].append(metrics['psnr'])

            # 更新进度条
            if batch_idx % self.config.get('log_every', 50) == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'bit_acc': f"{metrics['bit_accuracy']:.3f}",
                    'psnr': f"{metrics['psnr']:.2f}"
                })

        # 更新学习率
        self.scheduler.step()

        return tracker.get_summary()

    def evaluate(
        self,
        dataloader: DataLoader,
        attack_config: Optional[AttackConfig] = None
    ) -> Dict:
        """
        评估模型

        Args:
            dataloader: 数据加载器
            attack_config: 攻击配置

        Returns:
            results: 评估结果
        """
        self.encoder.eval()
        self.decoder.eval()

        results = {'clean': {}, 'attacked': {}}

        # 清洁测试
        clean_tracker = MetricsTracker()

        with torch.no_grad():
            for images, messages in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                messages = messages.to(self.device)

                watermarked = self.encoder.embed_message(images, messages)
                predicted_logits = self.decoder(watermarked)
                predicted_probs = torch.sigmoid(predicted_logits)

                clean_tracker.update(predicted_probs, messages, watermarked, images)

        results['clean'] = clean_tracker.get_summary()

        # 攻击测试
        if attack_config:
            attacks = get_all_attacks(attack_config)

            for attack_name, attack_fn in attacks.items():
                print(f"  Testing attack: {attack_name}")
                attack_tracker = MetricsTracker()

                with torch.no_grad():
                    for images, messages in dataloader:
                        images = images.to(self.device)
                        messages = messages.to(self.device)

                        # 嵌入水印
                        watermarked = self.encoder.embed_message(images, messages)

                        # 应用攻击
                        attacked = attack_fn(watermarked)

                        # 解码
                        predicted_logits = self.decoder(attacked)
                        predicted_probs = torch.sigmoid(predicted_logits)

                        attack_tracker.update(predicted_probs, messages)

                results['attacked'][attack_name] = attack_tracker.get_summary()

        return results

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        include_discriminator: bool = True
    ):
        """
        保存检查点

        Args:
            path: 保存路径
            epoch: 当前 epoch
            include_discriminator: 是否保存判别器
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'opt_encoder_state_dict': self.opt_encoder.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config
        }

        if include_discriminator and self.discriminator:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
            checkpoint['opt_discriminator_state_dict'] = self.opt_discriminator.state_dict()

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.opt_encoder.load_state_dict(checkpoint['opt_encoder_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        if 'discriminator_state_dict' in checkpoint and self.discriminator:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator_state_dict'])

        print(f"Checkpoint loaded from {path}")

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        output_dir: str = "./outputs",
        attack_config: Optional[AttackConfig] = None
    ):
        """
        完整训练流程

        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            num_epochs: 训练轮数
            output_dir: 输出目录
            attack_config: 攻击配置 (用于最终评估)
        """
        os.makedirs(output_dir, exist_ok=True)

        best_val_acc = 0.0

        for epoch in range(self.current_epoch, num_epochs):
            # 训练
            train_metrics = self.train_epoch(train_dataloader, epoch)

            print(f"\nEpoch {epoch} Training Summary:")
            print(f"  Loss: {train_metrics.get('loss', 0):.4f}")
            print(f"  Bit Accuracy: {train_metrics.get('bit_accuracy', 0):.4f}")
            print(f"  PSNR: {train_metrics.get('psnr', 0):.2f} dB")
            print(f"  SSIM: {train_metrics.get('ssim', 0):.4f}")

            # 验证
            if val_dataloader:
                val_results = self.evaluate(val_dataloader, attack_config=None)
                val_acc = val_results['clean'].get('bit_accuracy', 0)

                print(f"\nValidation:")
                print(f"  Bit Accuracy: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_checkpoint(
                        os.path.join(output_dir, "best_model.pt"),
                        epoch
                    )

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt"),
                    epoch
                )

        # 最终评估
        print("\n" + "="*50)
        print("Final Evaluation")

        if val_dataloader:
            final_results = self.evaluate(val_dataloader, attack_config)
            self._save_results(final_results, os.path.join(output_dir, "evaluation_results.json"))

        # 保存训练历史
        with open(os.path.join(output_dir, "training_history.json"), 'w') as f:
            json.dump(self.history, f)

        print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")

    def _save_results(self, results: Dict, path: str):
        """保存评估结果"""
        def convert_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(x) for x in obj]
            else:
                return obj

        results = convert_to_serializable(results)

        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {path}")


def create_trainer(
    feature_dim: int = 1280,
    message_bits: int = 48,
    hidden_dim: int = 256,
    use_discriminator: bool = True,
    device: str = "cuda",
    config: Optional[Dict] = None
) -> Trainer:
    """
    创建训练器

    Args:
        feature_dim: 特征维度
        message_bits: 水印比特数
        hidden_dim: 隐藏层维度
        use_discriminator: 是否使用判别器
        device: 设备
        config: 配置

    Returns:
        trainer: Trainer 实例
    """
    # 创建模型
    encoder = WatermarkEncoder(
        feature_dim=feature_dim,
        message_bits=message_bits,
        hidden_dim=hidden_dim
    )

    decoder = WatermarkDecoder(
        input_channels=3,
        message_bits=message_bits,
        hidden_dim=hidden_dim * 2
    )

    discriminator = None
    if use_discriminator:
        discriminator = Discriminator(input_channels=3, ndf=64, n_layers=3)

    # 创建训练器
    trainer = Trainer(
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        config=config or {},
        device=device
    )

    return trainer


def test_trainer():
    """测试训练器"""
    print("Testing Trainer...")

    from data.dataset import get_dataloader

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建合成数据加载器
    dataloader = get_dataloader(
        data_root="",
        batch_size=4,
        image_size=256,
        dataset_type="synthetic",
        num_workers=0
    )

    # 创建训练器
    trainer = create_trainer(
        feature_dim=1280,
        message_bits=48,
        device=device,
        config={'num_epochs': 2, 'log_every': 10}
    )

    # 训练几个步骤
    print("  Running a few training steps...")
    for i, (images, messages) in enumerate(dataloader):
        if i >= 3:
            break

        images = images.to(device)
        messages = messages.to(device)

        metrics = trainer.train_step(images, messages)
        print(f"    Step {i}: loss={metrics['loss']:.4f}, bit_acc={metrics['bit_accuracy']:.4f}")

    print("✓ Trainer tests passed")


if __name__ == "__main__":
    test_trainer()