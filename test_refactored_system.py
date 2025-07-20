#!/usr/bin/env python3
"""
测试重构后的智能DNM系统

验证代码审查问题的解决和收敛监控改进
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加路径以导入我们的模块
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_model():
    """创建测试模型"""
    class TestResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            
            # 特征块
            self.feature_block1 = nn.Sequential(
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128)
            )
            
            self.feature_block2 = nn.Sequential(
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256)
            )
            
            # 分类器
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 10)
            )
            
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.feature_block1(x)
            x = self.feature_block2(x)
            x = self.classifier(x)
            return x
    
    return TestResNet()

def simulate_network_state(model, batch_size=32):
    """模拟网络状态捕获"""
    
    model.eval()
    activations = {}
    gradients = {}
    
    def activation_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().clone()
        return hook
    
    def gradient_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients[name] = grad_output[0].detach().clone()
        return hook
    
    # 注册钩子
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(activation_hook(name)))
            hooks.append(module.register_backward_hook(gradient_hook(name)))
    
    # 模拟前向和反向传播
    x = torch.randn(batch_size, 3, 224, 224)
    y = torch.randint(0, 10, (batch_size,))
    
    # 前向传播
    model.train()
    output = model(x)
    
    # 反向传播
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    
    # 清理钩子
    for hook in hooks:
        hook.remove()
    
    # 模拟性能历史
    performance_history = [
        0.65, 0.68, 0.72, 0.74, 0.76, 0.77, 0.77, 0.78, 0.78, 0.78,  # 性能停滞
        0.78, 0.77, 0.78, 0.78, 0.79, 0.79, 0.79, 0.79, 0.79, 0.80   # 轻微改进
    ]
    
    return {
        'activations': activations,
        'gradients': gradients,
        'performance_history': performance_history,
        'current_epoch': 20,
        'current_loss': float(loss.item())
    }

def test_refactored_bayesian_engine():
    """测试重构后的贝叶斯引擎"""
    
    logger.info("🧪 测试重构后的贝叶斯形态发生引擎")
    
    # 创建模型和上下文
    model = create_test_model()
    context = simulate_network_state(model)
    
    logger.info(f"📊 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"📊 捕获激活: {len(context['activations'])}个")
    logger.info(f"📊 捕获梯度: {len(context['gradients'])}个")
    
    # 测试重构后的贝叶斯引擎
    try:
        from neuroexapt.core.refactored_bayesian_morphogenesis import RefactoredBayesianMorphogenesisEngine
        
        # 创建引擎并设置积极模式
        bayesian_engine = RefactoredBayesianMorphogenesisEngine()
        bayesian_engine.set_aggressive_mode()
        
        logger.info("✅ 重构后的贝叶斯引擎创建成功")
        
        # 执行分析
        result = bayesian_engine.bayesian_morphogenesis_analysis(model, context)
        
        # 检查结果
        optimal_decisions = result.get('optimal_decisions', [])
        execution_plan = result.get('execution_plan', {})
        bayesian_analysis = result.get('bayesian_analysis', {})
        
        logger.info(f"🎯 发现最优决策: {len(optimal_decisions)}个")
        logger.info(f"📋 执行计划: {execution_plan.get('execute', False)}")
        logger.info(f"🎲 决策置信度: {bayesian_analysis.get('decision_confidence', 0):.3f}")
        
        # 详细输出决策信息
        for i, decision in enumerate(optimal_decisions):
            logger.info(f"  决策{i+1}: {decision.get('layer_name', '')} -> {decision.get('mutation_type', '')}")
            logger.info(f"    成功概率: {decision.get('success_probability', 0):.3f}")
            logger.info(f"    期望改进: {decision.get('expected_improvement', 0):.4f}")
            logger.info(f"    期望效用: {decision.get('expected_utility', 0):.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 重构后贝叶斯引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_enhanced_convergence_monitor():
    """测试增强收敛监控器"""
    
    logger.info("🧪 测试增强收敛监控器")
    
    try:
        from neuroexapt.core.enhanced_convergence_monitor import EnhancedConvergenceMonitor
        
        # 创建积极模式的监控器
        monitor = EnhancedConvergenceMonitor(mode='aggressive')
        
        logger.info(f"✅ 增强收敛监控器创建成功 (模式: {monitor.mode})")
        
        # 模拟性能历史测试
        test_scenarios = [
            {
                'name': '早期训练',
                'epochs': [1, 2, 3, 4, 5],
                'performances': [0.60, 0.65, 0.70, 0.72, 0.74],
                'losses': [1.5, 1.3, 1.1, 1.0, 0.9]
            },
            {
                'name': '性能停滞',
                'epochs': [6, 7, 8, 9, 10],
                'performances': [0.74, 0.74, 0.75, 0.74, 0.74],
                'losses': [0.9, 0.9, 0.88, 0.89, 0.9]
            },
            {
                'name': '性能下降',
                'epochs': [11, 12, 13, 14, 15],
                'performances': [0.74, 0.72, 0.70, 0.68, 0.66],
                'losses': [0.9, 1.0, 1.1, 1.2, 1.3]
            }
        ]
        
        for scenario in test_scenarios:
            logger.info(f"\n📊 测试场景: {scenario['name']}")
            
            for epoch, perf, loss in zip(scenario['epochs'], scenario['performances'], scenario['losses']):
                result = monitor.should_allow_morphogenesis(
                    current_epoch=epoch,
                    current_performance=perf,
                    current_loss=loss,
                    gradient_norm=np.random.uniform(0.5, 2.0)
                )
                
                logger.info(f"  Epoch {epoch}: 准确率={perf:.2f}, 损失={loss:.2f}")
                logger.info(f"    允许变异: {result['allow']}")
                logger.info(f"    原因: {result['reason']}")
                logger.info(f"    置信度: {result['confidence']:.2f}")
                
                if result['allow']:
                    logger.info(f"    ✅ 变异被允许: {result.get('suggestion', '')}")
                    break
                else:
                    logger.info(f"    ❌ 变异被阻止: {result.get('suggestion', '')}")
        
        return monitor
        
    except Exception as e:
        logger.error(f"❌ 增强收敛监控器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_integrated_system():
    """测试完整的集成系统"""
    
    logger.info("🧪 测试完整的重构集成系统")
    
    try:
        from neuroexapt.core.intelligent_dnm_integration import IntelligentDNMCore
        
        # 创建集成系统
        dnm_core = IntelligentDNMCore()
        
        logger.info("✅ 智能DNM集成系统创建成功")
        logger.info(f"🚀 当前模式: 积极模式")
        
        # 获取系统状态
        status = dnm_core.get_system_status()
        logger.info(f"📊 系统状态: {status.get('config', {}).get('aggressive_mutation_mode', False)}")
        
        # 创建模型和上下文
        model = create_test_model()
        context = simulate_network_state(model)
        
        # 执行形态发生分析
        result = dnm_core.enhanced_morphogenesis_execution(model, context)
        
        # 分析结果
        model_modified = result.get('model_modified', False)
        morphogenesis_events = result.get('morphogenesis_events', [])
        intelligent_analysis = result.get('intelligent_analysis', {})
        
        logger.info(f"🔧 模型是否修改: {model_modified}")
        logger.info(f"🧬 变异事件数量: {len(morphogenesis_events)}")
        logger.info(f"🧠 智能分析结果:")
        logger.info(f"  候选点发现: {intelligent_analysis.get('candidates_discovered', 0)}个")
        logger.info(f"  策略评估: {intelligent_analysis.get('strategies_evaluated', 0)}个")
        logger.info(f"  最终决策: {intelligent_analysis.get('final_decisions', 0)}个")
        logger.info(f"  执行置信度: {intelligent_analysis.get('execution_confidence', 0):.3f}")
        logger.info(f"  性能态势: {intelligent_analysis.get('performance_trend', 'unknown')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 集成系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_configuration_system():
    """测试配置系统"""
    
    logger.info("🧪 测试配置系统")
    
    try:
        from neuroexapt.core.bayesian_prediction.bayesian_config import BayesianConfigManager
        
        # 创建配置管理器
        config_manager = BayesianConfigManager()
        config = config_manager.get_config()
        
        logger.info("✅ 配置管理器创建成功")
        logger.info(f"📊 默认置信度阈值: {config.dynamic_thresholds['confidence_threshold']}")
        logger.info(f"📊 最小期望改进: {config.dynamic_thresholds['min_expected_improvement']}")
        logger.info(f"📊 蒙特卡罗样本数: {config.mc_samples}")
        
        # 测试积极模式
        config_manager.reset_to_aggressive_mode()
        aggressive_config = config_manager.get_config()
        
        logger.info("🚀 切换到积极模式:")
        logger.info(f"  置信度阈值: {aggressive_config.dynamic_thresholds['confidence_threshold']}")
        logger.info(f"  期望改进阈值: {aggressive_config.dynamic_thresholds['min_expected_improvement']}")
        logger.info(f"  探索奖励: {aggressive_config.utility_params['exploration_bonus']}")
        
        # 测试保守模式
        config_manager.reset_to_conservative_mode()
        conservative_config = config_manager.get_config()
        
        logger.info("🛡️ 切换到保守模式:")
        logger.info(f"  置信度阈值: {conservative_config.dynamic_thresholds['confidence_threshold']}")
        logger.info(f"  期望改进阈值: {conservative_config.dynamic_thresholds['min_expected_improvement']}")
        logger.info(f"  风险厌恶: {conservative_config.utility_params['risk_aversion']}")
        
        return config_manager
        
    except Exception as e:
        logger.error(f"❌ 配置系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主测试函数"""
    
    logger.info("🚀 开始重构系统综合测试")
    logger.info("="*60)
    
    # 测试1: 配置系统
    logger.info("\n" + "="*60)
    logger.info("📁 测试1: 配置系统")
    config_manager = test_configuration_system()
    
    # 测试2: 重构后的贝叶斯引擎
    logger.info("\n" + "="*60)
    logger.info("🧠 测试2: 重构后的贝叶斯引擎")
    bayesian_result = test_refactored_bayesian_engine()
    
    # 测试3: 增强收敛监控器
    logger.info("\n" + "="*60)
    logger.info("⏱️ 测试3: 增强收敛监控器")
    enhanced_monitor = test_enhanced_convergence_monitor()
    
    # 测试4: 完整集成系统
    logger.info("\n" + "="*60)
    logger.info("🔧 测试4: 完整集成系统")
    integrated_result = test_integrated_system()
    
    # 总结
    logger.info("\n" + "="*60)
    logger.info("📋 测试总结")
    logger.info("="*60)
    
    success_count = 0
    total_tests = 4
    
    if config_manager is not None:
        logger.info("✅ 配置系统: 通过")
        success_count += 1
    else:
        logger.info("❌ 配置系统: 失败")
    
    if bayesian_result is not None and bayesian_result.get('optimal_decisions'):
        logger.info("✅ 重构贝叶斯引擎: 通过")
        success_count += 1
    else:
        logger.info("❌ 重构贝叶斯引擎: 失败")
    
    if enhanced_monitor is not None:
        logger.info("✅ 增强收敛监控器: 通过")
        success_count += 1
    else:
        logger.info("❌ 增强收敛监控器: 失败")
    
    if integrated_result is not None:
        logger.info("✅ 完整集成系统: 通过")
        success_count += 1
    else:
        logger.info("❌ 完整集成系统: 失败")
    
    logger.info(f"\n🎯 总体测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        logger.info("🎉 所有测试通过！重构成功解决了代码审查中的问题")
    else:
        logger.info("⚠️ 部分测试失败，需要进一步调试")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)