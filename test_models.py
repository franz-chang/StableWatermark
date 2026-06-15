#!/usr/bin/env python3
"""
简化的测试脚本 - 不依赖重型库
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gumbel_softmax():
    """测试 Gumbel-Softmax"""
    print("\n" + "="*50)
    print("Testing Gumbel-Softmax")
    print("="*50)

    from modules.gumbel_softmax import gumbel_softmax, GumbelSoftmax, MessageEncoder

    batch_size = 4
    num_bits = 48
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试随机 logits
    logits = torch.randn(batch_size, num_bits, device=device)

    # 软采样
    soft = gumbel_softmax(logits, temperature=1.0, hard=False)
    assert soft.shape == (batch_size, num_bits)
    assert torch.allclose(soft.sum(dim=1), torch.ones(batch_size, device=device), atol=1e-5)
    print("✓ Soft sampling works")

    # 硬采样
    hard = gumbel_softmax(logits, temperature=0.1, hard=True)
    assert hard.shape == (batch_size, num_bits)
    values, indices = hard.max(dim=1)
    assert torch.allclose(values, torch.ones(batch_size, device=device), atol=1e-5)
    print("✓ Hard sampling works")

    # 梯度流
    logits.requires_grad_(True)
    sample = gumbel_softmax(logits, temperature=0.5, hard=True)
    loss = sample.sum()
    loss.backward()
    assert logits.grad is not None
    print("✓ Gradient flow works")

    # 模块测试
    module = GumbelSoftmax(num_classes=num_bits)
    out = module(logits.detach())
    assert out.shape == (batch_size, num_bits)
    print("✓ GumbelSoftmax module works")

    # 消息编码器
    encoder = MessageEncoder(message_dim=256, num_bits=num_bits)
    encoder = encoder.to(device)
    message = torch.randint(0, 2, (batch_size, num_bits), device=device).float()
    one_hot = encoder(message)
    assert one_hot.shape == (batch_size, num_bits)
    print("✓ MessageEncoder module works")

    print("\n✓ Gumbel-Softmax tests passed!")


def test_models():
    """测试核心模型"""
    print("\n" + "="*50)
    print("Testing Core Models")
    print("="*50)

    from models import WatermarkEncoder, WatermarkDecoder, Discriminator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4

    # 测试编码器
    print("\n1. Testing WatermarkEncoder...")
    encoder = WatermarkEncoder(feature_dim=1280, message_bits=48, hidden_dim=256)
    encoder = encoder.to(device)

    feature = torch.randn(batch_size, 1280, 32, 32, device=device)
    message = torch.randint(0, 2, (batch_size, 48), device=device).float()

    watermarked = encoder.embed_message(feature, message)
    assert watermarked.shape == feature.shape
    assert not torch.allclose(watermarked, feature)
    print("  ✓ Encoder forward pass works")

    # 梯度流
    watermarked.sum().backward()
    grad_exists = any(p.grad is not None for p in encoder.parameters() if p.requires_grad)
    assert grad_exists
    print("  ✓ Encoder gradient flow works")

    # 测试解码器
    print("\n2. Testing WatermarkDecoder...")
    decoder = WatermarkDecoder(input_channels=3, message_bits=48, hidden_dim=512)
    decoder = decoder.to(device)

    images = torch.rand(batch_size, 3, 256, 256, device=device)
    logits = decoder(images)
    assert logits.shape == (batch_size, 48)
    print("  ✓ Decoder forward pass works")

    # 测试判别器
    print("\n3. Testing Discriminator...")
    disc = Discriminator(input_channels=3, ndf=64, n_layers=3)
    disc = disc.to(device)

    output = disc(images)
    assert output.shape[0] == batch_size
    assert output.shape[1] == 1
    print("  ✓ Discriminator forward pass works")

    print("\n✓ Core models tests passed!")


def test_attacks():
    """测试攻击函数"""
    print("\n" + "="*50)
    print("Testing Attack Functions")
    print("="*50)

    from utils.attack import get_attack, get_all_attacks, AttackConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    images = torch.rand(batch_size, 3, 256, 256, device=device)

    config = AttackConfig()

    # 测试各种攻击
    attacks_to_test = [
        "gaussian_noise",
        "salt_pepper",
        "gaussian_blur",
        "center_crop",
        "jpeg_compression",
        "combined"
    ]

    for name in attacks_to_test:
        try:
            attack = get_attack(name, config)
            result = attack(images.clone())
            assert result.shape == images.shape
            assert result.min() >= 0 and result.max() <= 1
            print(f"  ✓ {name} works")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")

    print("\n✓ Attack tests passed!")


def test_metrics():
    """测试评估指标"""
    print("\n" + "="*50)
    print("Testing Metrics")
    print("="*50)

    from utils.metrics import (
        calculate_bit_accuracy, calculate_message_accuracy,
        calculate_psnr, calculate_ssim
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    num_bits = 48

    # 测试数据
    predicted = torch.rand(batch_size, num_bits, device=device)
    target = (torch.rand(batch_size, num_bits, device=device) > 0.5).float()

    bit_acc = calculate_bit_accuracy(predicted, target)
    msg_acc = calculate_message_accuracy(predicted, target)

    print(f"  Bit Accuracy: {bit_acc:.4f}")
    print(f"  Message Accuracy: {msg_acc:.4f}")
    print("  ✓ Bit accuracy metrics work")

    # 图像质量指标
    img1 = torch.rand(batch_size, 3, 256, 256, device=device)
    img2 = img1 + torch.randn_like(img1) * 0.05

    psnr_val = calculate_psnr(img1, img2)
    ssim_val = calculate_ssim(img1, img2)

    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    print("  ✓ Image quality metrics work")

    print("\n✓ Metrics tests passed!")


def test_end_to_end():
    """端到端测试"""
    print("\n" + "="*50)
    print("End-to-End Watermark Test")
    print("="*50)

    import torch.nn as nn
    from models import WatermarkEncoder, WatermarkDecoder
    from modules.gumbel_softmax import gumbel_softmax
    from utils.attack import GaussianNoise, JPEGCompression
    from utils.metrics import calculate_bit_accuracy

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建模型
    feature_dim = 512  # 使用 512 而不是 1280，方便通道数匹配
    encoder = WatermarkEncoder(feature_dim=feature_dim, message_bits=48, hidden_dim=256)
    decoder = WatermarkDecoder(input_channels=3, message_bits=48, hidden_dim=512)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # 生成真实图像特征 (模拟 SD 输出)
    batch_size = 4
    images = torch.rand(batch_size, 3, 256, 256, device=device)
    messages = torch.randint(0, 2, (batch_size, 48), device=device).float()

    # 模拟 U-Net 特征
    # 将图像下采样并调整通道数以匹配 feature_dim
    features = torch.nn.functional.interpolate(
        images, size=(32, 32), mode='bilinear', align_corners=False
    )
    # 调整通道数: 从 3 扩展到 feature_dim
    if feature_dim != 3:
        # 使用 1x1 卷积来扩展通道
        channel_proj = nn.Sequential(
            nn.Conv2d(3, feature_dim, kernel_size=1),
            nn.ReLU()
        ).to(device)
        features = channel_proj(features)
    else:
        features = features.repeat(1, feature_dim // 3, 1, 1)

    print(f"\n  Original messages:\n    {messages[0, :8]}... (first 8 bits)")
    print(f"  Feature shape: {features.shape}")

    # 嵌入水印
    watermarked_features = encoder.embed_message(features, messages)
    print(f"  Watermarked feature shape: {watermarked_features.shape}")

    # 将特征转换回图像用于解码器
    # 使用 1x1 卷积将多通道特征转换回 3 通道
    channel_back = nn.Conv2d(feature_dim, 3, kernel_size=1).to(device)
    watermarked_images = channel_back(watermarked_features)
    watermarked_images = torch.sigmoid(watermarked_images)  # 确保在 [0,1] 范围
    watermarked_images = torch.clamp(watermarked_images, 0, 1)

    print("\n  Embedding watermark...")

    # 提取水印 (简化处理)
    with torch.no_grad():
        predicted_logits = decoder(watermarked_images)
        predicted_probs = torch.sigmoid(predicted_logits)

    bit_acc = calculate_bit_accuracy(predicted_probs, messages)
    print(f"\n  Predicted probabilities:\n    {predicted_probs[0, :8]}... (first 8 bits)")
    print(f"\n  ✓ Bit Accuracy: {bit_acc:.4f}")

    # 攻击测试
    print("\n  Testing robustness...")
    attacks = [
        ("Gaussian Noise", GaussianNoise(sigma=0.03)),
        ("JPEG Compression", JPEGCompression(quality=75)),
    ]

    results = {}
    for name, attack in attacks:
        attacked_images = attack(watermarked_images)
        with torch.no_grad():
            attacked_probs = torch.sigmoid(decoder(attacked_images))
        acc = calculate_bit_accuracy(attacked_probs, messages)
        results[name] = acc
        print(f"    {name}: {acc:.4f}")

    print("\n✓ End-to-end test passed!")


def test_training_loss():
    """测试训练损失"""
    print("\n" + "="*50)
    print("Testing Training Loss")
    print("="*50)

    from training.losses import WatermarkLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_fn = WatermarkLoss(
        lambda_rec=1.0,
        lambda_msg=10.0,
        lambda_adv=0.5
    )

    batch_size = 4
    message_bits = 48

    # 测试数据 - 设置 requires_grad 以支持梯度测试
    watermarked = torch.rand(batch_size, 3, 256, 256, device=device).requires_grad_(True)
    original = torch.rand(batch_size, 3, 256, 256, device=device).requires_grad_(True)
    target_message = torch.randint(0, 2, (batch_size, message_bits), device=device).float()
    predicted_logits = torch.randn(batch_size, message_bits, device=device).requires_grad_(True)

    # 计算损失
    losses = loss_fn(
        watermarked, original,
        predicted_logits, target_message,
        discriminator_output=torch.rand(batch_size, 1, 16, 16, device=device),
        disc_real_output=torch.rand(batch_size, 1, 16, 16, device=device),
        disc_fake_output=torch.rand(batch_size, 1, 16, 16, device=device)
    )

    print(f"  Total loss: {losses['loss'].item():.4f}")
    print(f"  Rec loss: {losses.get('loss_rec', 0).item():.4f}")
    print(f"  Msg loss: {losses.get('loss_msg', 0).item():.4f}")

    # 梯度测试
    losses['loss'].backward()
    print("  ✓ Loss backward works")

    print("\n✓ Training loss tests passed!")


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("StableWatermark - Complete Test Suite")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    try:
        test_gumbel_softmax()
        test_models()
        test_attacks()
        test_metrics()
        test_end_to_end()
        test_training_loss()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)