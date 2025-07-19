#!/usr/bin/env python3
"""
Enhanced DNM Framework Debug Test
测试增强DNM框架的调试输出功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from neuroexapt.core.enhanced_dnm_framework import EnhancedDNMFramework

def create_simple_model():
    """创建简单的测试模型"""
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(), 
        nn.Linear(64, 10)
    )

def generate_mock_data(batch_size=32):
    """生成模拟数据"""
    x = torch.randn(batch_size, 784)
    y = torch.randint(0, 10, (batch_size,))
    return x, y

def collect_activations_and_gradients(model, x, y):
    """收集激活值和梯度用于分析"""
    activations = {}
    gradients = {}
    
    # 添加hook来收集激活值
    hooks = []
    layer_names = []
    
    def make_activation_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # 注册前向hook
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layer_names.append(name)
            hooks.append(module.register_forward_hook(make_activation_hook(name)))
    
    # 前向传播
    model.train()
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    
    # 反向传播
    loss.backward()
    
    # 收集梯度
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and name in layer_names:
            if hasattr(module, 'weight') and module.weight.grad is not None:
                gradients[name] = module.weight.grad.detach().clone()
    
    # 清理hooks
    for hook in hooks:
        hook.remove()
    
    return activations, gradients, loss.item()

def test_enhanced_dnm_with_debug():
    """测试增强DNM框架的调试输出"""
    print("=" * 80)
    print("🧬 增强DNM框架调试测试")
    print("=" * 80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 创建模型
    print("\n📱 创建测试模型...")
    model = create_simple_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 创建DNM框架
    print("\n🧬 初始化增强DNM框架...")
    dnm_framework = EnhancedDNMFramework()
    
    # 模拟训练循环
    print("\n🚀 开始模拟训练...")
    for epoch in range(5):
        print(f"\n" + "="*60)
        print(f"📊 Epoch {epoch + 1}/5")
        print("="*60)
        
        # 生成数据
        x, y = generate_mock_data()
        x, y = x.to(device), y.to(device)
        
        # 收集激活值和梯度
        print(f"\n🔍 收集激活值和梯度...")
        activations, gradients, loss_value = collect_activations_and_gradients(model, x, y)
        
        print(f"📊 训练损失: {loss_value:.4f}")
        print(f"📊 收集到 {len(activations)} 层激活值, {len(gradients)} 层梯度")
        
        # 更新性能历史
        accuracy = np.random.uniform(0.75, 0.95)  # 模拟准确率
        dnm_framework.update_performance_history(accuracy)
        print(f"📊 模拟准确率: {accuracy:.4f}")
        
        # 准备上下文
        context = {
            'epoch': epoch,
            'activations': activations,
            'gradients': gradients,
            'performance_history': dnm_framework.performance_history,
            'loss': loss_value,
            'accuracy': accuracy
        }
        
        # 执行形态发生
        print(f"\n🧬 执行形态发生检查...")
        results = dnm_framework.execute_morphogenesis(model, context)
        
        # 输出结果
        print(f"\n📋 形态发生结果:")
        print(f"  - 模型是否修改: {results['model_modified']}")
        print(f"  - 新增参数: {results['parameters_added']:,}")
        print(f"  - 形态发生类型: {results['morphogenesis_type']}")
        print(f"  - 触发原因数量: {len(results.get('trigger_reasons', []))}")
        
        if results['model_modified']:
            print(f"  - 决策置信度: {results.get('decision_confidence', 0):.3f}")
            print(f"  - 预期改进: {results.get('expected_improvement', 0):.3f}")
            model = results['new_model']
            print(f"✅ 模型已更新！")
        else:
            print(f"❌ 未触发形态发生")
        
        # 优化器步骤
        optimizer.zero_grad()
        
        print(f"\n" + "-"*50)
    
    # 输出最终统计
    print(f"\n" + "="*80)
    print(f"📊 最终统计")
    print("="*80)
    
    summary = dnm_framework.get_morphogenesis_summary()
    print(f"总形态发生事件: {summary['total_events']}")
    print(f"总新增参数: {summary['total_parameters_added']:,}")
    print(f"形态发生类型分布: {summary['morphogenesis_types']}")
    
    if summary['events']:
        print(f"\n详细事件列表:")
        for i, event in enumerate(summary['events'], 1):
            print(f"  {i}. Epoch {event['epoch']}: {event['type']} "
                  f"(参数+{event['parameters_added']:,}, 置信度{event['confidence']:.3f})")
    
    print(f"\n✅ 调试测试完成！")

if __name__ == "__main__":
    test_enhanced_dnm_with_debug()