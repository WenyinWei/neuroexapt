#!/usr/bin/env python3
"""
智能触发机制测试脚本
Quick test for intelligent trigger mechanism
"""

import torch
import torch.nn as nn
import numpy as np
from neuroexapt.core import EnhancedDNMFramework

def create_test_model():
    """创建一个简单的测试模型"""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(), 
        nn.Linear(10, 1)
    )

def create_test_data():
    """创建测试用的激活值和梯度"""
    activations = {
        '0': torch.randn(32, 20),  # 第一层激活
        '2': torch.randn(32, 10),  # 第二层激活  
        '4': torch.randn(32, 1),   # 输出层激活
    }
    
    gradients = {
        '0': torch.randn(32, 20) * 0.01,  # 小梯度模拟梯度消失
        '2': torch.randn(32, 10) * 0.1,   # 正常梯度
        '4': torch.randn(32, 1) * 0.05,   # 输出层梯度
    }
    
    return activations, gradients

def test_intelligent_trigger():
    """测试智能触发机制"""
    print("🧠 测试智能触发机制...")
    
    # 创建DNM框架
    config = {
        'trigger_interval': 1,
        'complexity_threshold': 0.3,
        'enable_serial_division': True,
        'enable_parallel_division': True, 
        'enable_hybrid_division': True,
        'max_parameter_growth_ratio': 2.0,
        'enable_intelligent_bottleneck_detection': True,
        'bottleneck_severity_threshold': 0.4,  # 降低阈值便于测试
        'stagnation_threshold': 0.01,
        'net2net_improvement_threshold': 0.2,
        'enable_aggressive_mode': False  # 关闭激进模式专注测试基础功能
    }
    
    dnm_framework = EnhancedDNMFramework(config)
    
    # 创建测试数据
    model = create_test_model()
    activations, gradients = create_test_data()
    
    # 模拟性能停滞的历史
    performance_history_stagnant = [0.7, 0.702, 0.701, 0.703, 0.702]  # 停滞
    performance_history_improving = [0.6, 0.65, 0.7, 0.75, 0.8]      # 改进中
    
    print("\n📊 测试场景1: 性能停滞情况")
    print(f"性能历史: {performance_history_stagnant}")
    
    try:
        should_trigger, reasons = dnm_framework.check_morphogenesis_trigger(
            model, activations, gradients, performance_history_stagnant, epoch=5
        )
        
        print(f"触发结果: {'✅ 触发' if should_trigger else '❌ 未触发'}")
        if reasons:
            print("触发原因:")
            for reason in reasons:
                print(f"  • {reason}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n📊 测试场景2: 性能改进情况")  
    print(f"性能历史: {performance_history_improving}")
    
    try:
        should_trigger, reasons = dnm_framework.check_morphogenesis_trigger(
            model, activations, gradients, performance_history_improving, epoch=5
        )
        
        print(f"触发结果: {'✅ 触发' if should_trigger else '❌ 未触发'}")
        if reasons:
            print("触发原因:")
            for reason in reasons:
                print(f"  • {reason}")
        else:
            print("未触发原因: 性能持续改进中")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 测试完成!")

if __name__ == "__main__":
    test_intelligent_trigger()