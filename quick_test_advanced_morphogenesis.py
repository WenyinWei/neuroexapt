#!/usr/bin/env python3
"""
快速测试高级形态发生功能
"""

import sys
sys.path.append('/workspace')

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入增强的DNM组件
from neuroexapt.core import (
    AdvancedMorphogenesisExecutor,
    MorphogenesisType,
    MorphogenesisDecision
)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.classifier(x)

def test_morphogenesis_types():
    """测试所有形态发生类型"""
    print("🧬 快速测试高级形态发生功能")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    # 测试每种形态发生类型
    morphogenesis_types = [
        MorphogenesisType.WIDTH_EXPANSION,
        MorphogenesisType.SERIAL_DIVISION,
        MorphogenesisType.PARALLEL_DIVISION,
        MorphogenesisType.HYBRID_DIVISION
    ]
    
    results = {}
    
    for morph_type in morphogenesis_types:
        print(f"\n🔬 测试 {morph_type.value}...")
        
        model = SimpleNet().to(device)
        original_params = sum(p.numel() for p in model.parameters())
        
        # 创建决策
        decision = MorphogenesisDecision(
            morphogenesis_type=morph_type,
            target_location='classifier.0',  # 第一个线性层
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
            test_input = torch.randn(4, 256).to(device)
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
            
            print(f"    ✅ 成功")
            print(f"    原始参数: {original_params:,}")
            print(f"    新增参数: {params_added:,}")
            print(f"    总参数: {new_params:,}")
            print(f"    增长率: {results[morph_type.value]['growth_ratio']:.1%}")
            print(f"    输出形状: {output.shape}")
            
        except Exception as e:
            results[morph_type.value] = {
                'success': False,
                'error': str(e)
            }
            print(f"    ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    print(f"\n📊 测试总结:")
    successful_types = [t for t, r in results.items() if r.get('success', False)]
    print(f"  成功的形态发生类型: {len(successful_types)}/4")
    print(f"  支持的类型: {successful_types}")
    
    if len(successful_types) == 4:
        print("  🎉 所有高级形态发生功能正常工作!")
    else:
        print("  ⚠️ 部分形态发生需要进一步调试")
    
    return results

if __name__ == "__main__":
    test_morphogenesis_types()