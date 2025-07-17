#!/usr/bin/env python3
"""
设备问题调试脚本
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuroexapt.core.dnm_framework import DNMFramework
from neuroexapt.core.dnm_neuron_division import AdaptiveNeuronDivision

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def check_model_devices(model, name=""):
    print(f"\n=== {name} 模型设备检查 ===")
    for param_name, param in model.named_parameters():
        print(f"{param_name}: {param.device} (shape: {param.shape})")

def main():
    print("🧬 DNM设备问题调试")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"主设备: {device}")
    
    # 创建模型
    model = SimpleModel().to(device)
    print(f"初始模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    check_model_devices(model, "初始")
    
    # 创建DNM框架
    dnm = DNMFramework(model)
    
    # 执行神经元分裂
    print("\n开始执行神经元分裂...")
    divisor = AdaptiveNeuronDivision()
    
    try:
        new_model, params_added = divisor.execute_division(model, 'classifier.2', 'symmetric', 0.2)
        print(f"分裂成功! 新增参数: {params_added}")
        
        check_model_devices(new_model, "分裂后")
        
        # 测试前向传播
        test_input = torch.randn(2, 3, 32, 32).to(device)
        print(f"测试输入设备: {test_input.device}")
        
        print("执行前向传播...")
        output = new_model(test_input)
        print(f"输出形状: {output.shape}, 设备: {output.device}")
        print("✅ 前向传播成功!")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()