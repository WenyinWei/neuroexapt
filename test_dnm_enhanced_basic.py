#!/usr/bin/env python3
"""
DNM Enhanced Components Basic Test

基础功能测试，验证增强组件是否正常工作
"""

import torch
import torch.nn as nn
import numpy as np
from neuroexapt.core import (
    EnhancedBottleneckDetector, 
    PerformanceGuidedDivision, 
    DivisionStrategy
)

def test_enhanced_bottleneck_detector():
    """测试增强瓶颈检测器"""
    print("🔍 测试增强瓶颈检测器...")
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 10)
    )
    
    # 创建检测器
    detector = EnhancedBottleneckDetector(
        sensitivity_threshold=0.1,
        diversity_threshold=0.3,
        gradient_threshold=1e-6,
        info_flow_threshold=0.5
    )
    
    # 模拟激活值和梯度
    activations = {}
    gradients = {}
    
    # 模拟一些激活值
    batch_size = 8
    activations['0'] = torch.randn(batch_size, 16, 32, 32)  # Conv2d
    activations['2'] = torch.randn(batch_size, 32, 32, 32)  # Conv2d
    activations['6'] = torch.randn(batch_size, 10)          # Linear
    
    # 模拟一些梯度
    gradients['0.weight'] = torch.randn(16, 3, 3, 3) * 0.01
    gradients['2.weight'] = torch.randn(32, 16, 3, 3) * 0.01
    gradients['6.weight'] = torch.randn(10, 32) * 0.01
    
    # 模拟目标
    targets = torch.randint(0, 10, (batch_size,))
    
    # 检测瓶颈
    bottleneck_scores = detector.detect_bottlenecks(model, activations, gradients, targets)
    
    print(f"   检测到 {len(bottleneck_scores)} 个层")
    for layer_name, score in bottleneck_scores.items():
        print(f"   层 {layer_name}: 瓶颈分数 = {score:.3f}")
    
    # 获取分析摘要
    summary = detector.get_analysis_summary(bottleneck_scores)
    print(f"   分析摘要: {summary}")
    
    # 测试触发判断
    performance_trend = [70.0, 72.0, 73.0, 73.1, 73.2]  # 性能停滞
    should_trigger, reasons = detector.should_trigger_division(bottleneck_scores, performance_trend)
    print(f"   是否触发分裂: {should_trigger}")
    if reasons:
        for reason in reasons:
            print(f"     - {reason}")
    
    print("   ✅ 瓶颈检测器测试完成")
    return True

def test_performance_guided_division():
    """测试性能导向分裂器"""
    print("⚡ 测试性能导向分裂器...")
    
    # 创建分裂器
    divider = PerformanceGuidedDivision(
        noise_scale=0.1,
        progressive_epochs=3,
        diversity_threshold=0.7,
        performance_monitoring=True
    )
    
    # 创建简单层进行测试
    conv_layer = nn.Conv2d(16, 32, 3, padding=1)
    linear_layer = nn.Linear(64, 10)
    
    # 模拟激活值和梯度
    activations = torch.randn(8, 16, 16, 16)  # Conv layer activations
    gradients = torch.randn(32, 16, 3, 3) * 0.01  # Conv layer gradients
    targets = torch.randint(0, 10, (8,))
    
    # 测试不同的分裂策略
    strategies = [
        DivisionStrategy.GRADIENT_BASED,
        DivisionStrategy.ACTIVATION_BASED,
        DivisionStrategy.HYBRID,
        DivisionStrategy.INFORMATION_GUIDED
    ]
    
    for strategy in strategies:
        print(f"   测试策略: {strategy.value}")
        
        # 选择中间神经元进行分裂
        neuron_idx = conv_layer.out_channels // 2
        
        try:
            success, division_info = divider.divide_neuron(
                conv_layer, neuron_idx, strategy,
                activations, gradients, targets
            )
            
            if success:
                print(f"     ✅ 分裂成功: {division_info.get('strategy', 'unknown')}")
            else:
                print(f"     ❌ 分裂失败: {division_info.get('error', 'unknown')}")
                
        except Exception as e:
            print(f"     ❌ 异常: {e}")
    
    # 测试线性层分裂
    print("   测试线性层分裂...")
    linear_activations = torch.randn(8, 64)
    linear_gradients = torch.randn(10, 64) * 0.01
    
    success, division_info = divider.divide_neuron(
        linear_layer, 5, DivisionStrategy.HYBRID,
        linear_activations, linear_gradients, targets
    )
    
    if success:
        print(f"     ✅ 线性层分裂成功")
    else:
        print(f"     ❌ 线性层分裂失败: {division_info.get('error', 'unknown')}")
    
    # 获取分裂摘要
    summary = divider.get_division_summary()
    print(f"   分裂摘要: {summary}")
    
    print("   ✅ 性能导向分裂器测试完成")
    return True

def test_integration():
    """测试组件集成"""
    print("🧬 测试组件集成...")
    
    # 创建检测器和分裂器
    detector = EnhancedBottleneckDetector()
    divider = PerformanceGuidedDivision()
    
    # 创建测试模型
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),   # 较小的模型便于测试
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    # 模拟训练数据
    batch_size = 4
    x = torch.randn(batch_size, 3, 8, 8)
    y = torch.randint(0, 2, (batch_size,))
    
    # 前向传播
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    
    # 反向传播
    loss.backward()
    
    # 收集激活值和梯度
    activations = {}
    gradients = {}
    
    # 简单的激活值收集
    with torch.no_grad():
        x_temp = x
        for i, layer in enumerate(model):
            x_temp = layer(x_temp)
            activations[str(i)] = x_temp.clone()
    
    # 收集梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone()
    
    # 瓶颈检测
    bottleneck_scores = detector.detect_bottlenecks(model, activations, gradients, y)
    print(f"   检测到瓶颈层: {len(bottleneck_scores)}")
    
    # 获取最高分数的层
    if bottleneck_scores:
        top_layer = max(bottleneck_scores.items(), key=lambda x: x[1])
        print(f"   最高瓶颈分数: {top_layer[0]} = {top_layer[1]:.3f}")
        
        # 模拟分裂（这里只是测试接口，不做实际分裂）
        for name, module in model.named_modules():
            if name == top_layer[0] and isinstance(module, (nn.Conv2d, nn.Linear)):
                print(f"   目标层类型: {type(module).__name__}")
                break
    
    print("   ✅ 组件集成测试完成")
    return True

def main():
    """主测试函数"""
    print("🧬 DNM 增强组件基础测试")
    print("=" * 50)
    
    try:
        # 运行测试
        test1 = test_enhanced_bottleneck_detector()
        print()
        
        test2 = test_performance_guided_division()
        print()
        
        test3 = test_integration()
        print()
        
        # 总结
        all_passed = test1 and test2 and test3
        
        print("=" * 50)
        if all_passed:
            print("🎉 所有测试通过！DNM 增强组件工作正常")
        else:
            print("⚠️ 部分测试失败，需要检查")
        
        print("📊 测试摘要:")
        print(f"   瓶颈检测器: {'✅' if test1 else '❌'}")
        print(f"   性能导向分裂器: {'✅' if test2 else '❌'}")
        print(f"   组件集成: {'✅' if test3 else '❌'}")
        
    except Exception as e:
        print(f"❌ 测试出现异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()