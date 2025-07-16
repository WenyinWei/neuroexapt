#!/usr/bin/env python3
"""
高效vs传统架构对比验证

直接对比参数量、内存使用和前向传播速度
"""

import torch
import torch.nn as nn
import time
import sys
import os

sys.path.append('.')
from neuroexapt.core.model import Network

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())

def measure_memory_usage(model, input_tensor):
    """测量GPU内存使用"""
    torch.cuda.reset_peak_memory_stats()
    model(input_tensor)
    return torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

def measure_speed(model, input_tensor, runs=10):
    """测量前向传播速度"""
    # 预热
    for _ in range(5):
        _ = model(input_tensor)
    
    # 测试
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        _ = model(input_tensor)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / runs * 1000  # ms

def main():
    print("🔍 传统vs高效架构对比验证")
    print("=" * 60)
    
    # 测试配置
    C = 16
    batch_size = 32
    input_tensor = torch.randn(batch_size, 3, 32, 32, device='cuda')
    
    # 创建传统网络（参数量巨大的版本）
    print("📊 创建传统网络...")
    traditional_model = Network(
        C=C,
        num_classes=10,
        layers=6,
        potential_layers=4,
        quiet=True
    ).cuda()
    
    traditional_params = count_parameters(traditional_model)
    
    # 创建简化网络（减少参数的版本）
    print("📊 创建简化网络...")
    simplified_model = Network(
        C=C,
        num_classes=10,
        layers=4,  # 减少层数
        potential_layers=2,  # 减少潜在层数
        quiet=True
    ).cuda()
    
    simplified_params = count_parameters(simplified_model)
    
    print(f"\n📈 参数量对比:")
    print(f"   传统网络: {traditional_params:,} 参数")
    print(f"   简化网络: {simplified_params:,} 参数")
    print(f"   参数减少: {(1 - simplified_params/traditional_params)*100:.1f}%")
    
    # 测试内存使用
    print(f"\n💾 内存使用对比:")
    traditional_memory = measure_memory_usage(traditional_model, input_tensor)
    simplified_memory = measure_memory_usage(simplified_model, input_tensor)
    
    print(f"   传统网络: {traditional_memory:.1f} MB")
    print(f"   简化网络: {simplified_memory:.1f} MB")
    print(f"   内存节省: {(1 - simplified_memory/traditional_memory)*100:.1f}%")
    
    # 测试速度
    print(f"\n⚡ 速度对比:")
    traditional_speed = measure_speed(traditional_model, input_tensor)
    simplified_speed = measure_speed(simplified_model, input_tensor)
    
    print(f"   传统网络: {traditional_speed:.2f} ms")
    print(f"   简化网络: {simplified_speed:.2f} ms")
    print(f"   速度提升: {traditional_speed/simplified_speed:.1f}x")
    
    # 进一步优化建议
    print(f"\n💡 进一步优化策略:")
    print("1. 参数共享: 相同操作跨层共享参数")
    print("2. 动态剪枝: 训练中剪除低权重操作")
    print("3. 渐进搜索: 分阶段增加搜索复杂度")
    print("4. 操作融合: 使用CUDA/Triton融合操作")
    
    # 清理内存
    del traditional_model, simplified_model
    torch.cuda.empty_cache()
    
    print(f"\n🎯 推荐配置 (平衡性能和效率):")
    print("   - layers=4 (instead of 10)")
    print("   - potential_layers=2 (instead of 4)")
    print("   - init_channels=16 (合理起始通道数)")
    print("   - 启用所有CPU优化和Triton加速")

if __name__ == "__main__":
    main() 