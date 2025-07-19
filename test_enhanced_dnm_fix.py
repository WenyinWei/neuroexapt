#!/usr/bin/env python3
"""
测试增强DNM框架修复 - Test Enhanced DNM Framework Fixes

验证所有修复是否正常工作：
1. 接口兼容性修复
2. Net2Net子网络分析集成
3. Sourcery代码审查建议实现
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append('.')

def test_interface_compatibility():
    """测试接口兼容性"""
    print("🔧 测试接口兼容性...")
    
    try:
        from neuroexapt.core.enhanced_dnm_framework import EnhancedDNMFramework
        
        # 创建测试模型
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        # 创建DNM框架
        dnm_framework = EnhancedDNMFramework()
        
        # 测试新接口
        activations = {'layer1': torch.randn(32, 20)}
        gradients = {'layer1': torch.randn(32, 20)}
        performance_history = [0.7, 0.75, 0.8]
        epoch = 10
        targets = torch.randint(0, 5, (32,))
        
        result = dnm_framework.execute_morphogenesis(
            model, activations, gradients, performance_history, epoch, targets
        )
        
        print(f"   ✅ 新接口测试通过: {result['morphogenesis_type']}")
        
        # 测试老接口兼容性
        context = {
            'activations': activations,
            'gradients': gradients,
            'performance_history': performance_history,
            'epoch': epoch,
            'targets': targets
        }
        
        result_old = dnm_framework.execute_morphogenesis(model, context)
        
        print(f"   ✅ 老接口兼容性测试通过: {result_old['morphogenesis_type']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 接口兼容性测试失败: {e}")
        return False

def test_net2net_integration():
    """测试Net2Net集成"""
    print("\n🧪 测试Net2Net子网络分析集成...")
    
    try:
        from neuroexapt.core.net2net_subnetwork_analyzer import Net2NetSubnetworkAnalyzer
        
        # 创建复杂模型
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
        
        # 创建Net2Net分析器
        analyzer = Net2NetSubnetworkAnalyzer()
        
        # 准备测试数据
        activations = {
            '0': torch.randn(32, 32, 32, 32),  # Conv2d输出
            '2': torch.randn(32, 64, 32, 32),  # Conv2d输出
            '6': torch.randn(32, 10)           # Linear输出
        }
        
        gradients = {
            '0': torch.randn(32, 32, 32, 32),
            '2': torch.randn(32, 64, 32, 32),
            '6': torch.randn(32, 10)
        }
        
        targets = torch.randint(0, 10, (32,))
        current_accuracy = 0.75
        
        # 测试层分析
        result = analyzer.analyze_layer_mutation_potential(
            model, '2', activations, gradients, targets, current_accuracy
        )
        
        print(f"   ✅ Net2Net分析测试通过")
        print(f"      层名: {result['layer_name']}")
        print(f"      建议行动: {result['recommendation']['action']}")
        print(f"      改进潜力: {result['mutation_prediction']['improvement_potential']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Net2Net集成测试失败: {e}")
        import traceback
        print(f"      详细错误: {traceback.format_exc()}")
        return False

def test_aggressive_morphogenesis_with_net2net():
    """测试集成了Net2Net的激进形态发生"""
    print("\n🚀 测试激进形态发生与Net2Net集成...")
    
    try:
        from neuroexapt.core.enhanced_dnm_framework import EnhancedDNMFramework
        
        # 创建模型
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 7)
                self.feature_block1 = nn.Sequential(
                    nn.Conv2d(64, 128, 3),
                    nn.ReLU()
                )
                self.classifier = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10)
                )
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.feature_block1(x)
                x = x.mean(dim=(2, 3))  # Global average pooling
                x = self.classifier(x)
                return x
        
        model = TestModel()
        
        # 激进模式配置
        config = {
            'enable_aggressive_mode': True,
            'accuracy_plateau_threshold': 0.001,
            'aggressive_trigger_accuracy': 0.7,  # 低阈值便于测试
            'max_concurrent_mutations': 2
        }
        
        dnm_framework = EnhancedDNMFramework(config=config)
        
        # 创建高准确率停滞场景
        performance_history = [0.75, 0.752, 0.751, 0.752, 0.751]  # 停滞状态
        
        # 模拟激活和梯度
        activations = {
            'conv1': torch.randn(32, 64, 26, 26),
            'feature_block1.0': torch.randn(32, 128, 24, 24),
            'classifier.0': torch.randn(32, 64),
            'classifier.2': torch.randn(32, 10)
        }
        
        gradients = {
            'conv1': torch.randn(32, 64, 26, 26),
            'feature_block1.0': torch.randn(32, 128, 24, 24),
            'classifier.0': torch.randn(32, 64),
            'classifier.2': torch.randn(32, 10)
        }
        
        targets = torch.randint(0, 10, (32,))
        epoch = 50
        
        # 执行激进形态发生
        result = dnm_framework.execute_morphogenesis(
            model, activations, gradients, performance_history, epoch, targets
        )
        
        print(f"   ✅ 激进形态发生测试通过")
        print(f"      形态发生类型: {result['morphogenesis_type']}")
        print(f"      模型是否修改: {result['model_modified']}")
        print(f"      新增参数: {result['parameters_added']}")
        
        # 检查是否包含Net2Net分析结果
        if 'aggressive_details' in result:
            details = result['aggressive_details']
            if 'net2net_analyses' in details:
                net2net_count = len(details['net2net_analyses'])
                print(f"      Net2Net分析层数: {net2net_count}")
            else:
                print(f"      ⚠️ 未包含Net2Net分析结果")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 激进形态发生测试失败: {e}")
        import traceback
        print(f"      详细错误: {traceback.format_exc()}")
        return False

def test_device_consistency():
    """测试设备一致性修复"""
    print("\n🖥️ 测试设备一致性修复...")
    
    try:
        from neuroexapt.core.performance_guided_division import PerformanceGuidedDivision
        
        # 创建GPU模型（如果可用）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = nn.Linear(10, 5).to(device)
        
        # 创建性能引导分裂器
        divider = PerformanceGuidedDivision()
        
        # 模拟分裂操作
        activations = torch.randn(32, 10).to(device)
        targets = torch.randint(0, 5, (32,)).to(device)
        
        # 这应该不会因为设备不匹配而失败
        result = divider.execute_division(model, model[0], activations, targets)
        
        print(f"   ✅ 设备一致性测试通过")
        print(f"      使用设备: {device}")
        print(f"      分裂成功: {result.get('success', False)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 设备一致性测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧬 增强DNM框架修复验证测试")
    print("=" * 50)
    
    tests = [
        ("接口兼容性", test_interface_compatibility),
        ("Net2Net集成", test_net2net_integration),
        ("激进形态发生+Net2Net", test_aggressive_morphogenesis_with_net2net),
        ("设备一致性", test_device_consistency)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ 测试 {test_name} 出现异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    
    passed_count = 0
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
        if success:
            passed_count += 1
    
    print(f"\n🎯 总体结果: {passed_count}/{len(results)} 个测试通过")
    
    if passed_count == len(results):
        print("🎉 所有修复验证成功！可以继续训练。")
    else:
        print("⚠️ 部分测试失败，建议检查相关模块。")
    
    return passed_count == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)