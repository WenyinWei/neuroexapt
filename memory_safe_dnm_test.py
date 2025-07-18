#!/usr/bin/env python3
"""
内存安全的DNM框架测试
修复了内存爆炸问题，安全地测试增强DNM框架
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc
import psutil
import os
from collections import defaultdict
import sys

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from neuroexapt.core.enhanced_dnm_framework import EnhancedDNMFramework

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }

def print_memory_info(stage=""):
    """打印内存信息"""
    mem = get_memory_usage()
    gpu_mem = ""
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        gpu_mem = f", GPU: {gpu_allocated:.1f}MB allocated, {gpu_cached:.1f}MB cached"
    
    print(f"🧠 {stage} 内存使用: {mem['rss']:.1f}MB ({mem['percent']:.1f}%){gpu_mem}")

def create_simple_model():
    """创建简单的测试模型"""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

def generate_mock_data(batch_size=32, image_size=32):
    """生成模拟数据"""
    x = torch.randn(batch_size, 3, image_size, image_size)
    y = torch.randint(0, 10, (batch_size,))
    return x, y

def collect_activations_and_gradients_safe(model, x, y, max_layers=10):
    """安全地收集激活值和梯度，限制层数和内存使用"""
    activations = {}
    gradients = {}
    
    # 添加hook来收集激活值
    hooks = []
    layer_names = []
    layer_count = 0
    
    def make_activation_hook(name):
        def hook(module, input, output):
            # 内存优化：只保存部分激活值
            if isinstance(output, torch.Tensor):
                # 限制批次大小和特征维度
                reduced_output = output.detach()
                if reduced_output.numel() > 100000:  # 超过10万元素就采样
                    # 保持形状但减少元素
                    if len(reduced_output.shape) == 4:  # Conv层
                        reduced_output = reduced_output[:min(16, reduced_output.shape[0])]  # 限制批次
                    elif len(reduced_output.shape) == 2:  # Linear层
                        reduced_output = reduced_output[:min(32, reduced_output.shape[0])]  # 限制批次
                activations[name] = reduced_output
        return hook
    
    # 注册前向hook - 限制层数
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and layer_count < max_layers:
            layer_names.append(name)
            hooks.append(module.register_forward_hook(make_activation_hook(name)))
            layer_count += 1
    
    print(f"📊 注册了 {len(hooks)} 个hook")
    
    # 前向传播
    model.train()
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    
    print(f"📊 收集到 {len(activations)} 层激活值")
    
    # 反向传播
    loss.backward()
    
    # 收集梯度 - 只收集参数梯度，不是激活值梯度
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and name in layer_names:
            if hasattr(module, 'weight') and module.weight.grad is not None:
                # 内存优化：限制梯度大小
                grad = module.weight.grad.detach().clone()
                if grad.numel() > 100000:  # 超过10万元素就采样
                    # 随机采样梯度
                    flat_grad = grad.flatten()
                    indices = torch.randperm(len(flat_grad))[:100000]
                    sampled_grad = flat_grad[indices].view(-1, 1)  # 重塑为2D
                    gradients[name] = sampled_grad
                else:
                    gradients[name] = grad
    
    print(f"📊 收集到 {len(gradients)} 层梯度")
    
    # 清理hooks
    for hook in hooks:
        hook.remove()
    
    return activations, gradients, loss.item()

def memory_safe_dnm_test():
    """内存安全的DNM测试"""
    print("=" * 80)
    print("🧬 内存安全的增强DNM框架测试")
    print("=" * 80)
    
    print_memory_info("初始")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 创建模型
    print("\n📱 创建测试模型...")
    model = create_simple_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print_memory_info("模型创建后")
    
    # 创建DNM框架
    print("\n🧬 初始化增强DNM框架...")
    dnm_framework = EnhancedDNMFramework()
    
    # 修改配置以降低内存使用
    dnm_framework.config['trigger_interval'] = 2  # 更频繁触发以测试
    dnm_framework.config['max_parameter_growth_ratio'] = 0.1  # 限制参数增长
    
    print_memory_info("DNM框架初始化后")
    
    # 模拟训练循环
    print("\n🚀 开始模拟训练...")
    max_epochs = 3  # 减少epoch数量
    
    for epoch in range(max_epochs):
        print(f"\n" + "="*60)
        print(f"📊 Epoch {epoch + 1}/{max_epochs}")
        print("="*60)
        
        print_memory_info(f"Epoch {epoch+1} 开始")
        
        # 生成数据 - 使用较小的批次和图像
        x, y = generate_mock_data(batch_size=16, image_size=32)  # 减小批次大小
        x, y = x.to(device), y.to(device)
        
        # 收集激活值和梯度
        print(f"\n🔍 收集激活值和梯度...")
        try:
            activations, gradients, loss_value = collect_activations_and_gradients_safe(
                model, x, y, max_layers=8  # 限制分析的层数
            )
            
            print(f"📊 训练损失: {loss_value:.4f}")
            print_memory_info("数据收集后")
            
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
            
            print_memory_info("形态发生后")
            
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
            
            # 强制内存清理
            del activations, gradients, x, y
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print_memory_info(f"Epoch {epoch+1} 结束")
            
        except Exception as e:
            print(f"\n❌ Epoch {epoch+1} 失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 紧急内存清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
        
        print(f"\n" + "-"*50)
    
    # 输出最终统计
    print(f"\n" + "="*80)
    print(f"📊 最终统计")
    print("="*80)
    
    try:
        summary = dnm_framework.get_morphogenesis_summary()
        print(f"总形态发生事件: {summary['total_events']}")
        print(f"总新增参数: {summary['total_parameters_added']:,}")
        print(f"形态发生类型分布: {summary['morphogenesis_types']}")
        
        if summary['events']:
            print(f"\n详细事件列表:")
            for i, event in enumerate(summary['events'], 1):
                print(f"  {i}. Epoch {event['epoch']}: {event['type']} "
                      f"(参数+{event['parameters_added']:,}, 置信度{event['confidence']:.3f})")
    except Exception as e:
        print(f"❌ 统计生成失败: {e}")
    
    print_memory_info("最终")
    print(f"\n✅ 内存安全测试完成！")

if __name__ == "__main__":
    memory_safe_dnm_test()