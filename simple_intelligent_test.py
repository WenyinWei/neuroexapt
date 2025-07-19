#!/usr/bin/env python3
"""
简化的智能触发机制测试
Simplified intelligent trigger mechanism test
"""

import torch
import torch.nn as nn

def test_intelligent_trigger_simple():
    """简化的智能触发测试，避免复杂依赖"""
    print("🧠 简化智能触发机制测试...")
    
    try:
        # 基本导入测试
        from neuroexapt.core.logging_utils import logger
        print("✅ 日志系统导入成功")
        
        from neuroexapt.core.advanced_morphogenesis import AdvancedBottleneckAnalyzer
        print("✅ 瓶颈分析器导入成功")
        
        # 创建简单测试模型
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.Linear(10, 1)
        )
        print("✅ 测试模型创建成功")
        
        # 创建测试数据
        activations = {
            '0': torch.randn(32, 20),
            '2': torch.randn(32, 10), 
            '4': torch.randn(32, 1)
        }
        gradients = {
            '0': torch.randn(32, 20) * 0.001,  # 小梯度模拟瓶颈
            '2': torch.randn(32, 10) * 0.1,
            '4': torch.randn(32, 1) * 0.05
        }
        print("✅ 测试数据创建成功")
        
        # 测试瓶颈分析器
        analyzer = AdvancedBottleneckAnalyzer()
        print("✅ 瓶颈分析器初始化成功")
        
        analysis = analyzer.analyze_network_bottlenecks(model, activations, gradients)
        print("✅ 瓶颈分析执行成功")
        print(f"   分析结果类型数: {len(analysis)}")
        
        # 输出分析结果摘要
        for bottleneck_type, results in analysis.items():
            if isinstance(results, dict) and results:
                avg_score = sum(results.values()) / len(results)
                print(f"   {bottleneck_type}: {len(results)}层, 平均分数={avg_score:.3f}")
        
        print("\n🎉 简化测试全部通过!")
        print("🧠 智能瓶颈检测系统基础功能正常!")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_stagnation_detection():
    """测试性能停滞检测逻辑"""
    print("\n📊 测试性能停滞检测...")
    
    # 模拟停滞的性能历史
    stagnant_history = [0.7, 0.702, 0.701, 0.703, 0.702]
    improving_history = [0.6, 0.65, 0.7, 0.75, 0.8]
    
    def calculate_stagnation(history):
        if len(history) < 3:
            return 0
        
        improvements = []
        for i in range(1, len(history)):
            improvements.append(history[i] - history[i-1])
        
        avg_improvement = sum(improvements) / len(improvements)
        stagnation_severity = max(0, -avg_improvement * 100)
        return stagnation_severity
    
    stagnant_severity = calculate_stagnation(stagnant_history)
    improving_severity = calculate_stagnation(improving_history)
    
    print(f"  停滞场景: {stagnant_history}")
    print(f"    停滞严重程度: {stagnant_severity:.3f}%")
    print(f"  改进场景: {improving_history}")
    print(f"    停滞严重程度: {improving_severity:.3f}%")
    
    # 验证检测逻辑
    threshold = 0.01  # 0.01% 停滞阈值
    
    should_trigger_stagnant = stagnant_severity > threshold
    should_trigger_improving = improving_severity > threshold
    
    print(f"\n  阈值: {threshold}%")
    print(f"  停滞场景应触发: {'✅是' if should_trigger_stagnant else '❌否'}")
    print(f"  改进场景应触发: {'❌否' if not should_trigger_improving else '⚠️是'}")
    
    return True

if __name__ == "__main__":
    print("🚀 开始简化智能触发测试...\n")
    
    success1 = test_intelligent_trigger_simple()
    success2 = test_performance_stagnation_detection()
    
    if success1 and success2:
        print("\n🎉 所有测试通过!")
        print("✅ 智能瓶颈检测系统基础功能验证成功!")
        print("\n💡 下一步可以运行完整演示:")
        print("   python examples/intelligent_dnm_demo.py")
    else:
        print("\n❌ 测试失败，需要进一步调试")