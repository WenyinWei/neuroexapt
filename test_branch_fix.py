#!/usr/bin/env python3
"""
测试多分支架构修复
验证grow_width操作后的forward/backward是否正常工作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入修复后的ASO-SE组件
from examples.aso_se_classification import GrowableConvBlock, GrowingNetwork

def test_branch_forward_backward():
    """测试分支前向和反向传播"""
    print("🧪 测试多分支架构的forward/backward...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试块
    block = GrowableConvBlock(block_id=0, in_channels=64, out_channels=128, stride=1).to(device)
    
    # 添加一些分支
    print("添加分支...")
    block.add_branch()
    block.add_branch()
    
    print(f"初始分支数: {len(block.branches)}")
    print(f"初始输出通道: {block.out_channels}")
    
    # 创建测试输入
    x = torch.randn(4, 64, 32, 32, device=device, requires_grad=True)
    
    # 测试初始前向传播
    print("\n1️⃣ 测试初始前向传播...")
    try:
        out1 = block(x)
        print(f"✅ 初始前向传播成功: {x.shape} -> {out1.shape}")
        
        # 测试反向传播
        loss1 = out1.sum()
        loss1.backward()
        print("✅ 初始反向传播成功")
        
    except Exception as e:
        print(f"❌ 初始测试失败: {e}")
        return False
    
    # 清除梯度
    x.grad = None
    
    # 执行通道扩展
    print("\n2️⃣ 执行通道扩展...")
    try:
        new_channels = int(block.out_channels * 1.5)  # 扩展1.5倍
        success = block.expand_channels(new_channels)
        
        if success:
            print(f"✅ 通道扩展成功: {128} -> {new_channels}")
            print(f"新分支数: {len(block.branches)}")
        else:
            print("❌ 通道扩展失败")
            return False
            
    except Exception as e:
        print(f"❌ 通道扩展异常: {e}")
        return False
    
    # 测试扩展后的前向传播
    print("\n3️⃣ 测试扩展后前向传播...")
    try:
        x_new = torch.randn(4, 64, 32, 32, device=device, requires_grad=True)
        out2 = block(x_new)
        print(f"✅ 扩展后前向传播成功: {x_new.shape} -> {out2.shape}")
        
        # 测试反向传播 - 这是关键测试！
        loss2 = out2.sum()
        loss2.backward()
        print("✅ 扩展后反向传播成功 - CUDA错误已修复！")
        
        return True
        
    except Exception as e:
        print(f"❌ 扩展后测试失败: {e}")
        return False

def test_network_growth():
    """测试完整网络的生长"""
    print("\n🌐 测试完整网络生长...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建小型测试网络
    network = GrowingNetwork(
        initial_channels=32,
        num_classes=10,
        initial_depth=3
    ).to(device)
    
    print(f"初始参数数量: {sum(p.numel() for p in network.parameters()):,}")
    
    # 创建测试数据
    x = torch.randn(2, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (2,), device=device)
    
    # 测试初始状态
    print("\n测试初始网络...")
    try:
        logits = network(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        print("✅ 初始网络训练成功")
    except Exception as e:
        print(f"❌ 初始网络失败: {e}")
        return False
    
    # 执行宽度生长
    print("\n执行宽度生长...")
    try:
        success = network.grow_width(expansion_factor=1.5)
        if success:
            print(f"✅ 宽度生长成功")
            print(f"新参数数量: {sum(p.numel() for p in network.parameters()):,}")
        else:
            print("❌ 宽度生长失败")
            return False
    except Exception as e:
        print(f"❌ 宽度生长异常: {e}")
        return False
    
    # 测试生长后的训练
    print("\n测试生长后网络...")
    try:
        network.zero_grad()  # 清除梯度
        x_new = torch.randn(2, 3, 32, 32, device=device)
        y_new = torch.randint(0, 10, (2,), device=device)
        
        logits = network(x_new)
        loss = F.cross_entropy(logits, y_new)
        loss.backward()
        print("✅ 生长后网络训练成功 - 多分支CUDA错误已修复！")
        return True
        
    except Exception as e:
        print(f"❌ 生长后网络失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🔧 多分支架构CUDA错误修复验证")
    print("=" * 60)
    
    # 启用调试模式
    torch.autograd.set_detect_anomaly(True)
    
    # 测试1: 分支级别测试
    test1_success = test_branch_forward_backward()
    
    # 测试2: 网络级别测试
    test2_success = test_network_growth()
    
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    print(f"分支级别测试: {'✅ 通过' if test1_success else '❌ 失败'}")
    print(f"网络级别测试: {'✅ 通过' if test2_success else '❌ 失败'}")
    
    if test1_success and test2_success:
        print("\n🎉 所有测试通过！多分支CUDA错误已成功修复！")
        print("💡 修复要点:")
        print("   1. 使用learnable projection替代F.pad零填充")
        print("   2. 安全的参数迁移和分支重建")
        print("   3. 失败时的优雅降级处理")
        print("   4. 避免在遍历时直接修改列表")
        return True
    else:
        print("\n⚠️ 仍有问题需要进一步调查")
        return False

if __name__ == "__main__":
    main() 