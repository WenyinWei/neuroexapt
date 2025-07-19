#!/usr/bin/env python3
"""
智能触发机制测试脚本
Quick test for intelligent trigger mechanism
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from neuroexapt.core.intelligent_dnm_integration import IntelligentDNMCore

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

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
    
    # 配置DNM框架
    config = {
        'trigger_threshold': 0.05,  # 降低阈值增加触发敏感度
        'division_strategies': ['parallel', 'serial'],
        'enable_gradient_tracking': True,
        'enable_aggressive_mode': False  # 关闭激进模式专注测试基础功能
    }
    
    dnm_framework = IntelligentDNMCore()
    
    # 创建测试数据
    model = create_test_model()
    activations, gradients = create_test_data()
    
    # 模拟性能停滞的历史
    performance_history_stagnant = [0.7, 0.702, 0.701, 0.703, 0.702]  # 停滞
    performance_history_improving = [0.6, 0.65, 0.7, 0.75, 0.8]      # 改进中
    
    print("\n📊 测试场景1: 性能停滞情况")
    print(f"性能历史: {performance_history_stagnant}")
    
    try:
        # 构建上下文
        context = {
            'activations': activations,
            'gradients': gradients,
            'performance_history': performance_history_stagnant,
            'current_epoch': 5,
            'stagnation_detected': True
        }
        
        print(f"📋 上下文信息:")
        print(f"  激活数量: {len(activations)}")
        print(f"  梯度数量: {len(gradients)}")
        print(f"  模型层数: {len(list(model.named_modules()))}")
        for name in list(model.named_modules())[:3]:  # 显示前3层
            print(f"    {name}")
        
        result = dnm_framework.enhanced_morphogenesis_execution(model, context)
        
        print(f"✅ 智能分析完成")
        print(f"模型是否修改: {result.get('model_modified', False)}")
        print(f"变异事件: {len(result.get('morphogenesis_events', []))}")
        
        if 'intelligent_analysis' in result:
            analysis = result['intelligent_analysis']
            print(f"候选点发现: {analysis.get('candidates_found', 0)}个")
            print(f"策略评估: {analysis.get('strategies_evaluated', 0)}个")
            print(f"最终决策: {analysis.get('final_decisions', 0)}个")
            print(f"执行置信度: {analysis.get('execution_confidence', 0):.3f}")
            performance_sit = analysis.get('performance_situation', {})
            print(f"性能态势: {performance_sit.get('status', 'unknown')}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n📊 测试场景2: 性能改进情况")  
    print(f"性能历史: {performance_history_improving}")
    
    try:
        # 构建改进情况的上下文
        context = {
            'activations': activations,
            'gradients': gradients,
            'performance_history': performance_history_improving,
            'current_epoch': 5,
            'stagnation_detected': False
        }
        
        result = dnm_framework.enhanced_morphogenesis_execution(model, context)
        
        print(f"✅ 智能分析完成")
        print(f"模型是否修改: {result.get('model_modified', False)}")
        print(f"变异事件: {len(result.get('morphogenesis_events', []))}")
        
        if 'intelligent_analysis' in result:
            analysis = result['intelligent_analysis']
            print(f"候选点发现: {analysis.get('candidates_found', 0)}个")
            print(f"策略评估: {analysis.get('strategies_evaluated', 0)}个")
            print(f"最终决策: {analysis.get('final_decisions', 0)}个")
            print(f"执行置信度: {analysis.get('execution_confidence', 0):.3f}")
            performance_sit = analysis.get('performance_situation', {})
            print(f"性能态势: {performance_sit.get('status', 'unknown')}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 测试完成!")

if __name__ == "__main__":
    test_intelligent_trigger()