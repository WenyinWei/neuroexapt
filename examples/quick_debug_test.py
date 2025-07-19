#!/usr/bin/env python3
"""
快速调试功能测试
验证增强DNM框架的调试输出是否正常工作
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import io
import contextlib

def test_debug_printer():
    """测试调试打印器"""
    # print("=" * 60)
    # print("🔍 测试调试打印器")
    # print("=" * 60)

    # 导入并测试主框架调试器
    from neuroexapt.core.enhanced_dnm_framework import debug_printer

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        debug_printer.print_debug("这是一条INFO信息", "INFO")
        debug_printer.print_debug("这是一条SUCCESS信息", "SUCCESS") 
        debug_printer.print_debug("这是一条WARNING信息", "WARNING")
        debug_printer.print_debug("这是一条ERROR信息", "ERROR")
        debug_printer.print_debug("这是一条DEBUG信息", "DEBUG")

        # 测试层次化输出
        debug_printer.enter_section("测试区域")
        debug_printer.print_debug("这是嵌套信息", "INFO")
        debug_printer.enter_section("更深层级")
        debug_printer.print_debug("更深层的信息", "DEBUG")
        debug_printer.exit_section("更深层级")
        debug_printer.exit_section("测试区域")

    output = buf.getvalue()
    # 检查每种日志级别的输出
    assert "这是一条INFO信息" in output
    assert "这是一条SUCCESS信息" in output
    assert "这是一条WARNING信息" in output
    assert "这是一条ERROR信息" in output
    assert "这是一条DEBUG信息" in output
    # 检查层次化输出
    assert "测试区域" in output
    assert "这是嵌套信息" in output
    assert "更深层级" in output
    assert "更深层的信息" in output

    print("\n✅ 调试打印器测试完成！")

def test_simple_morphogenesis():
    """简单的形态发生测试"""
    print("\n" + "=" * 60)
    print("🧬 简单形态发生测试")
    print("=" * 60)
    
    try:
        from neuroexapt.core.enhanced_dnm_framework import EnhancedDNMFramework
        
        # 创建简单模型
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        # 创建DNM框架
        dnm = EnhancedDNMFramework()
        
        # 创建模拟数据
        fake_activations = {
            '0': torch.randn(8, 5),
            '2': torch.randn(8, 2)
        }
        
        fake_gradients = {
            '0': torch.randn(5, 10),
            '2': torch.randn(2, 5)
        }
        
        context = {
            'epoch': 3,  # 触发间隔是3，这样会触发检查
            'activations': fake_activations,
            'gradients': fake_gradients,
            'performance_history': [0.7, 0.8, 0.85, 0.87, 0.88],
            'loss': 1.2,
            'accuracy': 0.88
        }
        
        print("\n🚀 执行形态发生...")
        results = dnm.execute_morphogenesis(model, context)
        
        print(f"\n📊 结果汇总:")
        print(f"  模型修改: {results['model_modified']}")
        print(f"  新增参数: {results['parameters_added']}")
        print(f"  形态发生类型: {results['morphogenesis_type']}")
        
        print("\n✅ 简单形态发生测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_morphogenesis_debug():
    """测试形态发生模块调试输出"""
    print("\n" + "=" * 60)
    print("🔬 形态发生模块调试测试")
    print("=" * 60)
    
    try:
        from neuroexapt.core.advanced_morphogenesis import morpho_debug
        
        morpho_debug.print_debug("测试形态发生模块调试器", "INFO")
        morpho_debug.enter_section("形态发生模块测试")
        morpho_debug.print_debug("嵌套调试信息", "DEBUG")
        morpho_debug.exit_section("形态发生模块测试")
        
        print("\n✅ 形态发生模块调试测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 快速调试功能验证")
    print("=" * 80)
    
    # 测试调试打印器
    test_debug_printer()
    
    # 测试形态发生模块调试
    test_morphogenesis_debug()
    
    # 测试简单形态发生
    test_simple_morphogenesis()
    
    print("\n" + "=" * 80)
    print("🎉 所有调试功能测试完成！")
    print("=" * 80)