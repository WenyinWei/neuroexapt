#!/usr/bin/env python3
"""
核心模块测试脚本
Core Module Testing Script

🔧 目标：逐个测试各个模块的导入和基本功能
"""

import sys
import traceback
import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_basic_imports():
    """测试基础导入"""
    print("🔍 测试基础导入...")
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        print("✅ 基础库导入成功")
        return True
    except Exception as e:
        print(f"❌ 基础库导入失败: {e}")
        return False

def test_neuroexapt_core_imports():
    """测试neuroexapt.core导入"""
    print("\n🔍 测试neuroexapt.core导入...")
    
    failed_imports = []
    successful_imports = []
    
    # 测试各个组件
    components_to_test = [
        # 基础组件
        ('MutualInformationEstimator', 'neuroexapt.core.mutual_information_estimator'),
        ('BayesianUncertaintyEstimator', 'neuroexapt.core.bayesian_uncertainty_estimator'),
        ('IntelligentBottleneckDetector', 'neuroexapt.core.intelligent_bottleneck_detector'),
        ('IntelligentMutationPlanner', 'neuroexapt.core.intelligent_mutation_planner'),
        ('AdvancedNet2NetTransfer', 'neuroexapt.core.advanced_net2net_transfer'),
        ('IntelligentArchitectureEvolutionEngine', 'neuroexapt.core.intelligent_architecture_evolution_engine'),
        
        # 配置类
        ('EvolutionConfig', 'neuroexapt.core.intelligent_architecture_evolution_engine'),
        
        # 枚举类
        ('BottleneckType', 'neuroexapt.core.intelligent_bottleneck_detector'),
        ('MutationType', 'neuroexapt.core.intelligent_mutation_planner'),
    ]
    
    for component_name, module_name in components_to_test:
        try:
            module = __import__(module_name, fromlist=[component_name])
            component = getattr(module, component_name)
            successful_imports.append(component_name)
            print(f"  ✅ {component_name}")
        except Exception as e:
            failed_imports.append((component_name, str(e)))
            print(f"  ❌ {component_name}: {e}")
    
    print(f"\n📊 导入结果: {len(successful_imports)} 成功, {len(failed_imports)} 失败")
    
    if failed_imports:
        print("\n❌ 失败的导入:")
        for name, error in failed_imports:
            print(f"  • {name}: {error}")
    
    return len(failed_imports) == 0

def test_simple_model():
    """测试简单模型创建"""
    print("\n🔍 测试简单模型创建...")
    
    try:
        import torch.nn as nn
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.fc = nn.Linear(16 * 32 * 32, 10)
                
            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = TestModel()
        test_input = torch.randn(1, 3, 32, 32)
        output = model(test_input)
        
        print(f"✅ 模型创建成功, 输出形状: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

def test_optimizer_setup():
    """测试优化器设置"""
    print("\n🔍 测试优化器设置...")
    
    try:
        import torch.nn as nn
        import torch.optim as optim
        
        # 创建简单模型
        model = nn.Linear(10, 1)
        
        # 设置优化器
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # 测试训练步骤
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
        
        print("✅ 优化器设置和训练步骤成功")
        return True
        
    except Exception as e:
        print(f"❌ 优化器设置失败: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n🔍 测试数据加载...")
    
    try:
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        
        # 简单的transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # 尝试创建一个小的数据集
        try:
            dataset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform
            )
            loader = DataLoader(dataset, batch_size=4, shuffle=False)
            
            # 获取一个batch
            data_iter = iter(loader)
            batch = next(data_iter)
            data, labels = batch
            
            print(f"✅ 数据加载成功, batch形状: {data.shape}, 标签形状: {labels.shape}")
            return True
            
        except Exception as e:
            print(f"⚠️  CIFAR10下载可能失败: {e}")
            print("📝 创建模拟数据集...")
            
            # 创建模拟数据集
            from torch.utils.data import TensorDataset
            mock_data = torch.randn(100, 3, 32, 32)
            mock_labels = torch.randint(0, 10, (100,))
            mock_dataset = TensorDataset(mock_data, mock_labels)
            mock_loader = DataLoader(mock_dataset, batch_size=4)
            
            batch = next(iter(mock_loader))
            data, labels = batch
            print(f"✅ 模拟数据集创建成功, batch形状: {data.shape}, 标签形状: {labels.shape}")
            return True
            
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False

def test_device_setup():
    """测试设备设置"""
    print("\n🔍 测试设备设置...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ 设备设置成功: {device}")
        
        if device.type == 'cuda':
            print(f"  GPU名称: {torch.cuda.get_device_name()}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU内存: {memory_gb:.1f} GB")
        
        # 测试张量移动到设备
        tensor = torch.randn(10, 10).to(device)
        print(f"  张量设备: {tensor.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ 设备设置失败: {e}")
        return False

def run_diagnostic():
    """运行完整诊断"""
    print("🔧 NeuroExapt 核心模块诊断")
    print("="*60)
    
    tests = [
        ("基础导入", test_basic_imports),
        ("设备设置", test_device_setup),
        ("简单模型", test_simple_model),
        ("优化器设置", test_optimizer_setup),
        ("数据加载", test_data_loading),
        ("NeuroExapt核心导入", test_neuroexapt_core_imports),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{test_name}' 出现异常: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*60)
    print("📊 诊断总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20} {status}")
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过! 环境配置正常")
        return True
    else:
        print("⚠️  部分测试失败，请检查失败的模块")
        
        # 提供修复建议
        print("\n💡 修复建议:")
        for test_name, result in results:
            if not result:
                if "导入" in test_name:
                    print(f"• {test_name}: 检查模块路径和依赖安装")
                elif "数据" in test_name:
                    print(f"• {test_name}: 检查网络连接和存储空间")
                elif "设备" in test_name:
                    print(f"• {test_name}: 检查CUDA安装和GPU驱动")
                else:
                    print(f"• {test_name}: 检查PyTorch安装")
        
        return False

if __name__ == "__main__":
    success = run_diagnostic()
    
    if not success:
        print("\n🔧 如果问题持续存在，请尝试:")
        print("1. 重新安装PyTorch: pip install torch torchvision")
        print("2. 检查Python版本兼容性")
        print("3. 清理Python缓存: python -m pip cache purge")
        print("4. 检查neuroexapt模块是否正确安装")
    
    sys.exit(0 if success else 1)