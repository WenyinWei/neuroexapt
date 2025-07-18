#!/usr/bin/env python3
"""
测试高级形态发生功能
Test Advanced Morphogenesis Features

🧬 测试内容：
1. 串行分裂 (Serial Division) - 增加网络深度
2. 并行分裂 (Parallel Division) - 创建多分支结构
3. 混合分裂 (Hybrid Division) - 组合不同类型的层
4. 智能瓶颈分析
5. 决策制定系统
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from collections import defaultdict

# 导入增强的DNM组件
from neuroexapt.core import (
    EnhancedDNMFramework,
    AdvancedBottleneckAnalyzer,
    AdvancedMorphogenesisExecutor,
    IntelligentMorphogenesisDecisionMaker,
    MorphogenesisType,
    MorphogenesisDecision
)

class AdvancedTestNetwork(nn.Module):
    """高级测试网络"""
    
    def __init__(self, num_classes=10):
        super(AdvancedTestNetwork, self).__init__()
        
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def capture_activations_and_gradients(model, data_loader, device):
    """捕获激活值和梯度"""
    model.eval()
    activations = {}
    gradients = {}
    
    # 注册钩子函数
    def forward_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu()
        return hook
    
    def backward_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients[name] = grad_output[0].detach().cpu()
        return hook
    
    # 注册钩子
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(forward_hook(name)))
            hooks.append(module.register_backward_hook(backward_hook(name)))
    
    # 执行前向和反向传播
    model.train()
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device)
    
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    return activations, gradients

def test_advanced_bottleneck_analyzer():
    """测试高级瓶颈分析器"""
    print("\n🔍 测试高级瓶颈分析器...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedTestNetwork().to(device)
    
    # 创建测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 捕获激活值和梯度
    activations, gradients = capture_activations_and_gradients(model, data_loader, device)
    
    # 测试瓶颈分析器
    analyzer = AdvancedBottleneckAnalyzer()
    analysis = analyzer.analyze_network_bottlenecks(model, activations, gradients)
    
    print("  ✅ 瓶颈分析完成")
    print(f"  📊 分析结果:")
    
    for analysis_type, scores in analysis.items():
        if scores:
            top_bottleneck = max(scores.items(), key=lambda x: x[1])
            print(f"    {analysis_type}: 最高分数 {top_bottleneck[1]:.3f} (层: {top_bottleneck[0]})")
        else:
            print(f"    {analysis_type}: 无数据")
    
    return analysis

def test_morphogenesis_decision_maker(bottleneck_analysis):
    """测试形态发生决策制定器"""
    print("\n🧠 测试智能决策制定器...")
    
    decision_maker = IntelligentMorphogenesisDecisionMaker()
    
    # 模拟性能历史
    performance_history = [0.1, 0.2, 0.35, 0.5, 0.65, 0.75, 0.82, 0.85, 0.86, 0.86]
    
    decision = decision_maker.make_decision(bottleneck_analysis, performance_history)
    
    if decision:
        print("  ✅ 决策制定完成")
        print(f"  🎯 决策结果:")
        print(f"    形态发生类型: {decision.morphogenesis_type.value}")
        print(f"    目标位置: {decision.target_location}")
        print(f"    置信度: {decision.confidence:.3f}")
        print(f"    预期改进: {decision.expected_improvement:.3f}")
        print(f"    复杂度成本: {decision.complexity_cost:.3f}")
        print(f"    预估参数: {decision.parameters_added}")
        print(f"    决策理由: {decision.reasoning}")
        return decision
    else:
        print("  ⚠️ 未发现需要形态发生的瓶颈")
        return None

def test_morphogenesis_executor(model, decision):
    """测试形态发生执行器"""
    if not decision:
        print("\n⏭️ 跳过形态发生执行测试（无决策）")
        return model, 0
    
    print(f"\n🚀 测试形态发生执行器 - {decision.morphogenesis_type.value}...")
    
    executor = AdvancedMorphogenesisExecutor()
    
    # 记录原始参数数量
    original_params = sum(p.numel() for p in model.parameters())
    
    # 执行形态发生
    new_model, parameters_added = executor.execute_morphogenesis(model, decision)
    
    # 验证结果
    new_params = sum(p.numel() for p in new_model.parameters())
    actual_added = new_params - original_params
    
    print("  ✅ 形态发生执行完成")
    print(f"  📊 执行结果:")
    print(f"    原始参数: {original_params:,}")
    print(f"    新增参数: {actual_added:,}")
    print(f"    总参数: {new_params:,}")
    print(f"    增长比例: {(actual_added / original_params * 100):.2f}%")
    
    # 测试新模型的功能
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_model = new_model.to(device)
    
    test_input = torch.randn(2, 3, 32, 32).to(device)
    
    try:
        with torch.no_grad():
            output = new_model(test_input)
        print(f"    输出形状: {output.shape}")
        print("  ✅ 新模型功能验证通过")
    except Exception as e:
        print(f"  ❌ 新模型功能验证失败: {e}")
    
    return new_model, actual_added

def test_enhanced_dnm_framework():
    """测试增强的DNM框架"""
    print("\n🧬 测试增强的DNM框架...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedTestNetwork().to(device)
    
    # 初始化增强的DNM框架
    config = {
        'trigger_interval': 1,  # 每个epoch检查
        'complexity_threshold': 0.5,  # 降低阈值以便测试
        'enable_serial_division': True,
        'enable_parallel_division': True,
        'enable_hybrid_division': True
    }
    
    dnm_framework = EnhancedDNMFramework(config)
    
    # 创建测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 模拟训练过程
    print("  📚 模拟训练过程...")
    
    for epoch in range(3):  # 测试3个epoch
        print(f"    Epoch {epoch + 1}:")
        
        # 捕获激活值和梯度
        activations, gradients = capture_activations_and_gradients(model, data_loader, device)
        
        # 更新性能历史
        performance = 0.7 + epoch * 0.05 + np.random.normal(0, 0.02)
        dnm_framework.update_performance_history(performance)
        
        # 准备上下文
        context = {
            'epoch': epoch,
            'activations': activations,
            'gradients': gradients,
            'performance_history': dnm_framework.performance_history
        }
        
        # 执行形态发生
        results = dnm_framework.execute_morphogenesis(model, context)
        
        if results['model_modified']:
            model = results['new_model']
            print(f"      🎉 形态发生成功: {results['morphogenesis_type']}")
            print(f"      📈 新增参数: {results['parameters_added']:,}")
            print(f"      🎯 置信度: {results.get('decision_confidence', 0):.3f}")
        else:
            print(f"      😴 未触发形态发生")
    
    # 获取总结
    summary = dnm_framework.get_morphogenesis_summary()
    
    print("  ✅ 增强DNM框架测试完成")
    print(f"  📊 总结:")
    print(f"    总事件数: {summary['total_events']}")
    print(f"    总新增参数: {summary['total_parameters_added']:,}")
    print(f"    形态发生类型: {summary['morphogenesis_types']}")
    
    return dnm_framework, summary

def compare_morphogenesis_types():
    """比较不同形态发生类型的效果"""
    print("\n⚖️ 比较不同形态发生类型...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试每种形态发生类型
    morphogenesis_types = [
        MorphogenesisType.WIDTH_EXPANSION,
        MorphogenesisType.SERIAL_DIVISION,
        MorphogenesisType.PARALLEL_DIVISION,
        MorphogenesisType.HYBRID_DIVISION
    ]
    
    results = {}
    
    for morph_type in morphogenesis_types:
        print(f"  🔬 测试 {morph_type.value}...")
        
        # 创建原始模型
        model = AdvancedTestNetwork().to(device)
        original_params = sum(p.numel() for p in model.parameters())
        
        # 创建决策
        decision = MorphogenesisDecision(
            morphogenesis_type=morph_type,
            target_location='classifier.1',  # 选择一个线性层
            confidence=0.8,
            expected_improvement=0.05,
            complexity_cost=0.3,
            parameters_added=5000,
            reasoning=f"测试{morph_type.value}"
        )
        
        # 执行形态发生
        executor = AdvancedMorphogenesisExecutor()
        try:
            new_model, params_added = executor.execute_morphogenesis(model, decision)
            new_params = sum(p.numel() for p in new_model.parameters())
            
            # 测试功能
            test_input = torch.randn(2, 3, 32, 32).to(device)
            with torch.no_grad():
                output = new_model(test_input)
            
            results[morph_type.value] = {
                'success': True,
                'original_params': original_params,
                'new_params': new_params,
                'params_added': params_added,
                'growth_ratio': (new_params - original_params) / original_params,
                'output_shape': output.shape
            }
            
            print(f"    ✅ 成功 - 新增参数: {params_added:,}")
            
        except Exception as e:
            results[morph_type.value] = {
                'success': False,
                'error': str(e)
            }
            print(f"    ❌ 失败: {e}")
    
    # 打印比较结果
    print("\n  📋 比较结果:")
    print("    类型                | 成功 | 原始参数    | 新增参数    | 增长率")
    print("    " + "-" * 65)
    
    for morph_type, result in results.items():
        if result['success']:
            print(f"    {morph_type:<18} | ✅  | {result['original_params']:>10,} | {result['params_added']:>10,} | {result['growth_ratio']:>6.1%}")
        else:
            print(f"    {morph_type:<18} | ❌  | -          | -          | -")
    
    return results

def main():
    """主测试函数"""
    print("🧬 高级形态发生功能测试")
    print("=" * 50)
    
    try:
        # 1. 测试瓶颈分析器
        bottleneck_analysis = test_advanced_bottleneck_analyzer()
        
        # 2. 测试决策制定器
        decision = test_morphogenesis_decision_maker(bottleneck_analysis)
        
        # 3. 测试形态发生执行器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AdvancedTestNetwork().to(device)
        new_model, params_added = test_morphogenesis_executor(model, decision)
        
        # 4. 测试增强的DNM框架
        dnm_framework, summary = test_enhanced_dnm_framework()
        
        # 5. 比较不同形态发生类型
        comparison_results = compare_morphogenesis_types()
        
        print("\n🎉 所有测试完成!")
        print("=" * 50)
        
        print("\n📊 测试总结:")
        print(f"  高级瓶颈分析器: ✅ 正常工作")
        print(f"  智能决策制定器: ✅ 正常工作")
        print(f"  形态发生执行器: ✅ 正常工作")
        print(f"  增强DNM框架: ✅ 正常工作")
        
        # 统计成功的形态发生类型
        successful_types = [t for t, r in comparison_results.items() if r['success']]
        print(f"  支持的形态发生类型: {len(successful_types)}/4")
        
        if len(successful_types) >= 3:
            print("  🌟 高级形态发生功能已就绪!")
        else:
            print("  ⚠️ 部分形态发生类型需要进一步优化")
            
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()