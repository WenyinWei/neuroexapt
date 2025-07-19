#!/usr/bin/env python3
"""
激进形态发生系统演示 - Aggressive Morphogenesis Demo

🚀 展示如何使用新的多点变异系统突破准确率瓶颈
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

def create_accuracy_plateau_scenario():
    """创建准确率停滞场景进行测试"""
    
    # 模拟一个已经接近饱和的性能历史
    performance_history = [
        # 早期快速增长阶段
        0.75, 0.82, 0.87, 0.90, 0.91, 0.92, 0.925, 0.930, 0.932,
        # 接近停滞阶段 - 触发激进模式的关键
        0.934, 0.933, 0.935, 0.934, 0.936, 0.935, 0.937, 0.936, 0.937,
        # 完全停滞阶段 - 需要激进干预
        0.937, 0.936, 0.937, 0.936, 0.937, 0.936, 0.937
    ]
    
    return performance_history

def create_complex_test_model():
    """创建一个复杂的测试模型，模拟实际ResNet架构"""
    
    class ComplexTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # 特征提取部分
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            
            # ResNet风格的特征块
            self.feature_block1 = nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
            
            self.feature_block2 = nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
            
            self.feature_block3 = nn.Sequential(
                nn.Conv2d(256, 512, 3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
            
            # 全局平均池化
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
            # 分类器部分 - 多层结构以提供更多变异目标
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512), 
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10)  # CIFAR-10 类别数
            )
        
        def forward(self, x):
            # 特征提取
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            # 特征块
            x = self.feature_block1(x)
            x = self.feature_block2(x)
            x = self.feature_block3(x)
            
            # 池化和分类
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            
            return x
    
    return ComplexTestModel()

def capture_network_state(model, device='cpu'):
    """捕获网络状态用于瓶颈分析"""
    
    model.eval()
    activations = {}
    gradients = {}
    hooks = []
    
    def make_activation_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    def make_gradient_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients[name] = grad_output[0].detach()
        return hook
    
    # 注册钩子到关键层
    key_layers = [
        ('conv1', model.conv1),
        ('feature_block1.0', model.feature_block1[0]),
        ('feature_block1.3', model.feature_block1[3]),
        ('feature_block2.0', model.feature_block2[0]),
        ('feature_block2.3', model.feature_block2[3]),
        ('feature_block3.0', model.feature_block3[0]),
        ('feature_block3.3', model.feature_block3[3]),
        ('classifier.1', model.classifier[1]),
        ('classifier.4', model.classifier[4]),
        ('classifier.7', model.classifier[7]),
        ('classifier.9', model.classifier[9])
    ]
    
    for name, layer in key_layers:
        hooks.append(layer.register_forward_hook(make_activation_hook(name)))
        hooks.append(layer.register_backward_hook(make_gradient_hook(name)))
    
    # 执行前向和反向传播
    try:
        # 模拟输入
        batch_size = 32
        x = torch.randn(batch_size, 3, 32, 32).to(device)
        targets = torch.randint(0, 10, (batch_size,)).to(device)
        
        # 前向传播
        outputs = model(x)
        
        # 计算损失并反向传播
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        
    finally:
        # 清理钩子
        for hook in hooks:
            hook.remove()
    
    return activations, gradients

def demonstrate_aggressive_morphogenesis():
    """演示激进形态发生系统"""
    
    print("🧬 激进形态发生系统演示")
    print("=" * 60)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    try:
        # 1. 创建测试场景
        print("\n📊 步骤1: 创建准确率停滞场景...")
        performance_history = create_accuracy_plateau_scenario()
        print(f"   性能历史: {len(performance_history)}个数据点")
        print(f"   当前准确率: {performance_history[-1]:.3f}")
        print(f"   最近5个epoch的改进: {max(performance_history[-5:]) - min(performance_history[-5:]):.4f}")
        
        # 2. 创建复杂模型
        print("\n🏗️  步骤2: 创建复杂测试模型...")
        model = create_complex_test_model().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   模型参数量: {total_params:,}")
        
        # 3. 捕获网络状态
        print("\n📈 步骤3: 捕获网络激活和梯度...")
        activations, gradients = capture_network_state(model, device)
        print(f"   捕获激活: {len(activations)}层")
        print(f"   捕获梯度: {len(gradients)}层")
        
        # 4. 初始化增强DNM框架（激进模式）
        print("\n🚀 步骤4: 初始化激进形态发生框架...")
        from neuroexapt.core.enhanced_dnm_framework import EnhancedDNMFramework
        
        # 配置激进模式参数
        aggressive_config = {
            'trigger_interval': 8,
            'enable_aggressive_mode': True,
            'accuracy_plateau_threshold': 0.1,  # 0.1%的改进阈值
            'plateau_detection_window': 5,
            'aggressive_trigger_accuracy': 0.92,  # 92%时激活激进模式
            'max_concurrent_mutations': 3,
            'morphogenesis_budget': 15000  # 更大的参数预算
        }
        
        dnm_framework = EnhancedDNMFramework(config=aggressive_config)
        print("   ✅ 激进DNM框架初始化完成")
        
        # 5. 执行形态发生分析
        print("\n🔬 步骤5: 执行激进形态发生分析...")
        
        # 模拟当前epoch（应该触发激进模式）
        current_epoch = 95
        
        morphogenesis_result = dnm_framework.execute_morphogenesis(
            model=model,
            activations=activations,
            gradients=gradients,
            performance_history=performance_history,
            epoch=current_epoch
        )
        
        # 6. 分析结果
        print("\n📋 步骤6: 分析形态发生结果...")
        print(f"   模型是否修改: {morphogenesis_result['model_modified']}")
        print(f"   新增参数数量: {morphogenesis_result['parameters_added']:,}")
        print(f"   形态发生类型: {morphogenesis_result['morphogenesis_type']}")
        print(f"   触发原因数量: {len(morphogenesis_result['trigger_reasons'])}")
        
        for i, reason in enumerate(morphogenesis_result['trigger_reasons'], 1):
            print(f"     {i}. {reason}")
        
        # 如果是激进模式，显示详细信息
        if 'aggressive_details' in morphogenesis_result:
            details = morphogenesis_result['aggressive_details']
            print(f"\n🎯 激进模式详细信息:")
            print(f"   变异策略: {details['mutation_strategy']}")
            print(f"   目标位置数: {len(details['target_locations'])}")
            print(f"   识别的瓶颈数: {details['bottleneck_count']}")
            print(f"   停滞严重程度: {details['stagnation_severity']:.3f}")
            
            execution_result = details['execution_result']
            print(f"   执行结果: {execution_result['successful_mutations']}/{execution_result['total_mutations']} 成功")
        
        # 7. 验证新模型
        if morphogenesis_result['model_modified']:
            print("\n✅ 步骤7: 验证变异后的模型...")
            new_model = morphogenesis_result['new_model']
            new_total_params = sum(p.numel() for p in new_model.parameters())
            
            print(f"   原始参数量: {total_params:,}")
            print(f"   新模型参数量: {new_total_params:,}")
            print(f"   参数增长: {new_total_params - total_params:,} (+{((new_total_params - total_params) / total_params * 100):.1f}%)")
            
            # 测试模型功能
            test_input = torch.randn(4, 3, 32, 32).to(device)
            with torch.no_grad():
                original_output = model(test_input)
                new_output = new_model(test_input)
            
            print(f"   原始模型输出形状: {original_output.shape}")
            print(f"   新模型输出形状: {new_output.shape}")
            print(f"   输出一致性检查: {'✅通过' if original_output.shape == new_output.shape else '❌失败'}")
        
        print("\n🎉 激进形态发生演示完成!")
        
        return morphogenesis_result
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return None

def test_plateau_detection():
    """测试准确率停滞检测功能"""
    
    print("\n🔍 测试准确率停滞检测...")
    
    try:
        from neuroexapt.core.aggressive_morphogenesis import AggressiveMorphogenesisAnalyzer
        
        analyzer = AggressiveMorphogenesisAnalyzer(
            accuracy_plateau_threshold=0.05,  # 5%改进阈值
            plateau_window=5
        )
        
        # 测试不同的性能历史场景
        test_scenarios = {
            "快速增长": [0.7, 0.75, 0.8, 0.85, 0.9],
            "轻微停滞": [0.92, 0.921, 0.922, 0.923, 0.924],
            "严重停滞": [0.930, 0.931, 0.930, 0.931, 0.930],
            "完全停滞": [0.935, 0.935, 0.935, 0.935, 0.935]
        }
        
        for scenario_name, history in test_scenarios.items():
            is_plateau, severity = analyzer.detect_accuracy_plateau(history)
            print(f"   {scenario_name}: {'🚨停滞' if is_plateau else '✅正常'} (严重程度: {severity:.3f})")
        
        print("   ✅ 停滞检测测试完成")
        
    except Exception as e:
        print(f"   ❌ 停滞检测测试失败: {e}")

def test_bottleneck_signature_analysis():
    """测试瓶颈特征签名分析"""
    
    print("\n🎯 测试瓶颈特征签名分析...")
    
    try:
        from neuroexapt.core.aggressive_morphogenesis import AggressiveMorphogenesisAnalyzer
        
        analyzer = AggressiveMorphogenesisAnalyzer()
        
        # 模拟不同类型的激活和梯度
        test_cases = {
            "正常层": {
                "activation": torch.randn(32, 64, 16, 16) * 0.5,
                "gradient": torch.randn(32, 64, 16, 16) * 0.01
            },
            "饱和层": {
                "activation": torch.ones(32, 64, 16, 16),  # 完全饱和
                "gradient": torch.randn(32, 64, 16, 16) * 0.001  # 很小的梯度
            },
            "死亡层": {
                "activation": torch.zeros(32, 64, 16, 16),  # 死亡神经元
                "gradient": torch.zeros(32, 64, 16, 16)  # 零梯度
            }
        }
        
        activations = {name: data["activation"] for name, data in test_cases.items()}
        gradients = {name: data["gradient"] for name, data in test_cases.items()}
        output_targets = torch.randint(0, 10, (32,))
        
        signatures = analyzer.analyze_reverse_gradient_projection(
            activations, gradients, output_targets
        )
        
        for layer_name, signature in signatures.items():
            print(f"   {layer_name}:")
            print(f"     瓶颈类型: {signature.bottleneck_type}")
            print(f"     严重程度: {signature.severity:.3f}")
            print(f"     参数效率: {signature.parameter_efficiency:.3f}")
        
        print("   ✅ 瓶颈签名分析测试完成")
        
    except Exception as e:
        print(f"   ❌ 瓶颈签名分析测试失败: {e}")

def main():
    """主演示函数"""
    
    print("🧬🚀 激进多点形态发生系统 - 准确率瓶颈突破演示")
    print("=" * 80)
    
    # 基础功能测试
    test_plateau_detection()
    test_bottleneck_signature_analysis()
    
    # 完整系统演示
    print("\n" + "=" * 80)
    result = demonstrate_aggressive_morphogenesis()
    
    if result:
        print("\n📊 演示总结:")
        print(f"   ✅ 激进形态发生系统运行成功")
        print(f"   🎯 演示了准确率停滞检测和多点变异策略")
        print(f"   🔬 验证了反向梯度投影分析功能")
        print(f"   🚀 展示了比传统方法更激进的架构变异能力")
        
        if result['model_modified']:
            print(f"   📈 成功执行了{result['morphogenesis_type']}变异")
            print(f"   💼 新增{result['parameters_added']:,}个参数")
    else:
        print("\n❌ 演示未能完成，请检查错误信息")
    
    print("\n🌟 关键特性:")
    print("   1. 智能停滞检测 - 自动识别准确率饱和状态")
    print("   2. 反向梯度投影 - 从输出反推关键瓶颈层位置")
    print("   3. 多点协调变异 - 同时在多个位置进行架构修改")
    print("   4. 风险评估机制 - 平衡期望改进与变异风险")
    print("   5. 自适应策略选择 - 根据停滞严重程度调整激进程度")
    
    print("\n🎯 应用建议:")
    print("   • 当模型准确率超过92%且连续5个epoch改进<0.1%时自动激活")
    print("   • 优先在分类器层和特征提取层的关键瓶颈位置进行变异")  
    print("   • 使用混合协调策略平衡并行和级联变异的优势")
    print("   • 变异后给模型2-3个epoch的适应期以稳定性能")

if __name__ == "__main__":
    main()