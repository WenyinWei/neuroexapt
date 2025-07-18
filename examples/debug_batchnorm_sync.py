#!/usr/bin/env python3
"""
🔧 BatchNorm同步调试脚本
专门用于测试和修复DNM框架中的BatchNorm同步问题
"""

import torch
import torch.nn as nn
import logging
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.core.dnm_neuron_division import DNMNeuronDivisionManager

# 配置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class SimpleTestNet(nn.Module):
    """简单的测试网络，模拟ResNet结构"""
    
    def __init__(self):
        super().__init__()
        
        # 简单的stem结构 (类似ResNet)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),  # stem.0
            nn.BatchNorm2d(64),                                       # stem.1
            nn.ReLU(inplace=True)                                     # stem.2
        )
        
        # 简单的层结构
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        
        # 嵌套结构 (类似BasicBlock)
        self.block = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),  # block.0
            nn.BatchNorm2d(256),                                        # block.1
            nn.ReLU(inplace=True)                                       # block.2
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.block(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def debug_batchnorm_finding():
    """调试BatchNorm查找逻辑"""
    
    print("🔧 Debug BatchNorm Finding Logic")
    print("=" * 50)
    
    # 创建测试模型
    model = SimpleTestNet()
    
    # 打印模型结构
    print("\n📋 Model Structure:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            print(f"  {name}: {type(module).__name__}")
    
    # 创建DNM管理器
    dnm_config = {
        'monitoring': {
            'target_layers': ['conv'],
            'min_epoch_before_split': 0,
            'analysis_frequency': 1
        }
    }
    
    division_manager = DNMNeuronDivisionManager(config=dnm_config)
    
    # 测试BatchNorm查找
    test_conv_layers = [
        'stem.0',      # Sequential中的Conv
        'conv1',       # 顶级Conv  
        'block.0'      # 嵌套Sequential中的Conv
    ]
    
    print(f"\n🔍 Testing BatchNorm Finding:")
    for conv_name in test_conv_layers:
        bn_name = division_manager.splitter._find_corresponding_batchnorm(model, conv_name)
        status = "✅ Found" if bn_name else "❌ Not Found"
        print(f"  {conv_name} -> {bn_name} ({status})")

def test_manual_split():
    """测试手动分裂并验证BatchNorm同步"""
    
    print("\n🧬 Testing Manual Conv Split with BatchNorm Sync")
    print("=" * 50)
    
    model = SimpleTestNet()
    
    # 获取原始的stem模块
    stem_conv = model.stem[0]  # stem.0
    stem_bn = model.stem[1]    # stem.1
    
    print(f"\n📊 Before Split:")
    print(f"  Conv channels: {stem_conv.out_channels}")
    print(f"  BatchNorm features: {stem_bn.num_features}")
    
    # 创建DNM管理器并直接调用分裂方法
    dnm_config = {
        'monitoring': {
            'target_layers': ['conv'],
            'min_epoch_before_split': 0,
            'analysis_frequency': 1
        }
    }
    
    division_manager = DNMNeuronDivisionManager(config=dnm_config)
    
    try:
        # 执行分裂: stem.0从64通道分裂3个通道
        split_decisions = {
            'stem.0': [10, 20, 30]  # 分裂第10, 20, 30个通道
        }
        
        total_splits = division_manager._execute_splits(model, split_decisions)
        
        # 检查分裂后的状态
        new_stem_conv = model.stem[0]
        new_stem_bn = model.stem[1]
        
        print(f"\n📈 After Split:")
        print(f"  Conv channels: {new_stem_conv.out_channels}")
        print(f"  BatchNorm features: {new_stem_bn.num_features}")
        print(f"  Total splits executed: {total_splits}")
        
        # 验证模型可以正常前向传播
        test_input = torch.randn(1, 3, 32, 32)
        
        with torch.no_grad():
            output = model(test_input)
            print(f"  Forward pass successful: {output.shape}")
            print(f"  ✅ BatchNorm sync successful!")
        
    except Exception as e:
        print(f"  ❌ Split failed: {e}")
        import traceback
        traceback.print_exc()

def stress_test_multiple_splits():
    """压力测试：多次分裂"""
    
    print("\n💪 Stress Test: Multiple Splits")
    print("=" * 50)
    
    model = SimpleTestNet()
    
    dnm_config = {
        'monitoring': {
            'target_layers': ['conv'],
            'min_epoch_before_split': 0,
            'analysis_frequency': 1
        }
    }
    
    division_manager = DNMNeuronDivisionManager(config=dnm_config)
    
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"Initial parameters: {initial_params:,}")
    
    # 执行多轮分裂
    for round_num in range(3):
        print(f"\n🔄 Split Round {round_num + 1}:")
        
        try:
            # 模拟不同的分裂决策
            if round_num == 0:
                split_decisions = {'stem.0': [5, 15, 25]}
            elif round_num == 1:
                split_decisions = {'conv1': [10, 30, 50]}
            else:
                split_decisions = {'block.0': [20, 40, 60]}
            
            total_splits = division_manager._execute_splits(model, split_decisions)
            
            # 验证模型
            test_input = torch.randn(2, 3, 32, 32)
            with torch.no_grad():
                output = model(test_input)
            
            current_params = sum(p.numel() for p in model.parameters())
            param_growth = (current_params - initial_params) / initial_params * 100
            
            print(f"  ✅ Round {round_num + 1} successful!")
            print(f"  Splits: {total_splits}, Params: {current_params:,} (+{param_growth:.1f}%)")
            
        except Exception as e:
            print(f"  ❌ Round {round_num + 1} failed: {e}")
            break

if __name__ == "__main__":
    print("🔧 DNM BatchNorm Synchronization Debug Suite")
    print("=" * 60)
    
    # 1. 调试BatchNorm查找逻辑
    debug_batchnorm_finding()
    
    # 2. 测试手动分裂
    test_manual_split()
    
    # 3. 压力测试
    stress_test_multiple_splits()
    
    print("\n🎉 Debug completed!")