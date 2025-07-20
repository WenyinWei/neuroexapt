#!/usr/bin/env python3
"""
测试增强贝叶斯形态发生系统

验证新的智能架构变异引擎是否能够：
1. 成功检测到变异候选点
2. 进行贝叶斯推断并生成决策
3. 提供合理的性能改进预测
4. 在线学习和适应
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_model():
    """创建一个测试用的简单ResNet模型"""
    
    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = torch.relu(out)
            return out
    
    class TestResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(3, 2, 1)
            
            # 特征块
            self.feature_block1 = nn.Sequential(
                BasicBlock(64, 128, 2),
                BasicBlock(128, 128, 1)
            )
            self.feature_block2 = nn.Sequential(
                BasicBlock(128, 256, 2),
                BasicBlock(256, 256, 1)
            )
            self.feature_block3 = nn.Sequential(
                BasicBlock(256, 512, 2),
                BasicBlock(512, 512, 1)
            )
            
            # 分类器
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 10)
            )
        
        def forward(self, x):
            x = self.maxpool(torch.relu(self.bn1(self.conv1(x))))
            x = self.feature_block1(x)
            x = self.feature_block2(x)
            x = self.feature_block3(x)
            x = self.avgpool(x)
            x = self.classifier(x)
            return x
    
    return TestResNet()

def simulate_network_state_capture(model: nn.Module, batch_size: int = 32):
    """模拟网络状态捕获（激活值和梯度）"""
    
    model.train()
    
    # 创建模拟输入
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))
    
    # 激活值存储
    activations = {}
    gradients = {}
    
    # 注册前向钩子
    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # 注册反向钩子
    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients[name] = grad_output[0].detach()
        return hook
    
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(save_activation(name)))
            handles.append(module.register_backward_hook(save_gradient(name)))
    
    # 前向传播
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    
    # 反向传播
    loss.backward()
    
    # 清理钩子
    for handle in handles:
        handle.remove()
    
    return activations, gradients, loss.item()

def create_test_context(model: nn.Module, epoch: int = 15):
    """创建测试上下文"""
    
    # 模拟性能历史（显示停滞）
    performance_history = [
        0.72, 0.74, 0.76, 0.78, 0.79, 0.80, 0.805, 0.807, 0.808, 0.808,
        0.8081, 0.8082, 0.8079, 0.8080, 0.8079  # 最近几个epoch停滞
    ]
    
    # 捕获网络状态
    activations, gradients, train_loss = simulate_network_state_capture(model)
    
    context = {
        'epoch': epoch,
        'performance_history': performance_history,
        'train_loss': train_loss,
        'learning_rate': 0.1,
        'activations': activations,
        'gradients': gradients,
        'targets': torch.randint(0, 10, (32,))  # 模拟目标标签
    }
    
    return context

def test_bayesian_morphogenesis_engine():
    """测试增强贝叶斯形态发生引擎"""
    
    logger.info("🧪 开始测试增强贝叶斯形态发生系统")
    
    try:
        # 导入新的贝叶斯引擎
        from neuroexapt.core.enhanced_bayesian_morphogenesis import BayesianMorphogenesisEngine
        
        # 创建测试模型和引擎
        model = create_test_model()
        bayesian_engine = BayesianMorphogenesisEngine()
        
        logger.info(f"✅ 模型创建成功: {sum(p.numel() for p in model.parameters()):,} 参数")
        
        # 启用积极模式
        bayesian_engine.dynamic_thresholds['min_expected_improvement'] = 0.001
        bayesian_engine.dynamic_thresholds['confidence_threshold'] = 0.2
        
        # 创建测试上下文
        context = create_test_context(model)
        logger.info(f"✅ 测试上下文创建成功: {len(context['activations'])}个激活, {len(context['gradients'])}个梯度")
        
        # 执行贝叶斯分析
        logger.info("🚀 开始贝叶斯形态发生分析...")
        result = bayesian_engine.bayesian_morphogenesis_analysis(model, context)
        
        # 分析结果
        logger.info("\n" + "="*60)
        logger.info("📊 贝叶斯分析结果:")
        logger.info("="*60)
        
        bayesian_analysis = result.get('bayesian_analysis', {})
        optimal_decisions = result.get('optimal_decisions', [])
        execution_plan = result.get('execution_plan', {})
        bayesian_insights = result.get('bayesian_insights', {})
        
        logger.info(f"🎯 候选点发现: {bayesian_analysis.get('candidates_found', 0)}个")
        logger.info(f"⭐ 最优决策: {len(optimal_decisions)}个")
        logger.info(f"🎲 决策置信度: {bayesian_analysis.get('decision_confidence', 0.0):.3f}")
        logger.info(f"🚀 是否执行: {'是' if execution_plan.get('execute', False) else '否'}")
        
        if optimal_decisions:
            logger.info(f"\n📋 最优决策详情:")
            for i, decision in enumerate(optimal_decisions[:3]):  # 显示前3个
                logger.info(f"  {i+1}. 目标层: {decision.get('layer_name', 'N/A')}")
                logger.info(f"     变异类型: {decision.get('mutation_type', 'N/A')}")
                logger.info(f"     成功概率: {decision.get('success_probability', 0.0):.3f}")
                logger.info(f"     期望改进: {decision.get('expected_improvement', 0.0):.4f}")
                logger.info(f"     期望效用: {decision.get('expected_utility', 0.0):.4f}")
                logger.info(f"     决策理由: {decision.get('rationale', 'N/A')}")
                logger.info("")
        
        # 测试贝叶斯洞察
        if bayesian_insights:
            logger.info(f"💡 贝叶斯洞察:")
            logger.info(f"   最有前景的变异: {bayesian_insights.get('most_promising_mutation', {}).get('mutation_type', 'N/A')}")
            logger.info(f"   期望性能提升: {bayesian_insights.get('expected_performance_gain', 0.0):.4f}")
            logger.info(f"   风险评估: {bayesian_insights.get('risk_assessment', {}).get('overall_risk', 0.0):.3f}")
        
        # 测试在线学习
        logger.info(f"\n🧠 测试在线学习功能...")
        if optimal_decisions:
            first_decision = optimal_decisions[0]
            
            # 模拟变异成功
            bayesian_engine.update_mutation_outcome(
                mutation_type=first_decision['mutation_type'],
                layer_name=first_decision['layer_name'],
                success=True,
                performance_change=0.015,  # 1.5%的性能改进
                context=context
            )
            
            # 模拟另一个变异失败
            bayesian_engine.update_mutation_outcome(
                mutation_type='width_expansion',
                layer_name='feature_block1.0.conv1',
                success=False,
                performance_change=-0.005,
                context=context
            )
            
            logger.info("✅ 在线学习更新完成")
        
        # 再次分析以验证学习效果
        logger.info(f"\n🔄 验证学习效果...")
        context['epoch'] = 16  # 新的epoch
        context['performance_history'].append(0.823)  # 模拟性能改进
        
        result2 = bayesian_engine.bayesian_morphogenesis_analysis(model, context)
        optimal_decisions2 = result2.get('optimal_decisions', [])
        
        logger.info(f"学习后的决策数量: {len(optimal_decisions2)}")
        if optimal_decisions2:
            logger.info(f"新的最优决策: {optimal_decisions2[0]['mutation_type']} @ {optimal_decisions2[0]['layer_name']}")
        
        # 测试结果验证
        success_metrics = {
            'candidates_found': bayesian_analysis.get('candidates_found', 0) > 0,
            'decisions_generated': len(optimal_decisions) > 0,
            'execution_plan_valid': execution_plan.get('execute', False),
            'confidence_reasonable': bayesian_analysis.get('decision_confidence', 0.0) > 0.1,
            'learning_functional': len(bayesian_engine.mutation_history) > 0
        }
        
        logger.info(f"\n✅ 测试结果验证:")
        for metric, passed in success_metrics.items():
            status = "✅ 通过" if passed else "❌ 失败"
            logger.info(f"   {metric}: {status}")
        
        overall_success = all(success_metrics.values())
        if overall_success:
            logger.info(f"\n🎉 增强贝叶斯形态发生系统测试成功！")
            logger.info(f"系统现在能够更智能地检测变异候选点并做出决策。")
        else:
            logger.warning(f"\n⚠️ 部分测试未通过，需要进一步调试。")
            
        return overall_success, result
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_intelligent_dnm_integration():
    """测试智能DNM集成系统"""
    
    logger.info("\n" + "="*60)
    logger.info("🧪 测试智能DNM集成系统")
    logger.info("="*60)
    
    try:
        from neuroexapt.core.intelligent_dnm_integration import IntelligentDNMCore
        
        # 创建测试模型和DNM核心
        model = create_test_model()
        dnm_core = IntelligentDNMCore()
        
        # 启用积极模式
        dnm_core.enable_aggressive_bayesian_mode()
        
        # 创建测试上下文
        context = create_test_context(model)
        
        # 执行增强形态发生分析
        logger.info("🚀 执行增强形态发生分析...")
        result = dnm_core.enhanced_morphogenesis_execution(model, context)
        
        # 分析集成结果
        logger.info(f"\n📊 集成系统结果:")
        logger.info(f"模型是否修改: {result.get('model_modified', False)}")
        logger.info(f"变异事件数: {len(result.get('morphogenesis_events', []))}")
        logger.info(f"智能分析详情: {result.get('intelligent_analysis', {})}")
        
        # 获取贝叶斯洞察
        insights = dnm_core.get_bayesian_insights()
        logger.info(f"\n💡 贝叶斯引擎状态:")
        logger.info(f"变异历史长度: {insights.get('mutation_history_length', 0)}")
        logger.info(f"性能历史长度: {insights.get('performance_history_length', 0)}")
        logger.info(f"当前阈值: {insights.get('dynamic_thresholds', {})}")
        
        integration_success = (
            len(result.get('morphogenesis_events', [])) > 0 or
            result.get('intelligent_analysis', {}).get('candidates_discovered', 0) > 0
        )
        
        if integration_success:
            logger.info(f"\n✅ 智能DNM集成测试成功！")
        else:
            logger.info(f"\n⚠️ 智能DNM集成未产生变异事件")
            
        return integration_success, result
        
    except Exception as e:
        logger.error(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    logger.info("🚀 开始测试增强贝叶斯形态发生系统")
    
    # 测试贝叶斯引擎
    bayesian_success, bayesian_result = test_bayesian_morphogenesis_engine()
    
    # 测试集成系统
    integration_success, integration_result = test_intelligent_dnm_integration()
    
    # 最终报告
    logger.info("\n" + "="*60)
    logger.info("📊 最终测试报告")
    logger.info("="*60)
    logger.info(f"贝叶斯引擎测试: {'✅ 成功' if bayesian_success else '❌ 失败'}")
    logger.info(f"DNM集成测试: {'✅ 成功' if integration_success else '❌ 失败'}")
    
    if bayesian_success and integration_success:
        logger.info(f"\n🎉 所有测试通过！增强贝叶斯形态发生系统已成功部署。")
        logger.info(f"系统现在具备更强的智能决策能力，能够：")
        logger.info(f"  1. 更积极地检测变异候选点")
        logger.info(f"  2. 基于贝叶斯推断进行智能决策")
        logger.info(f"  3. 预测变异的成功概率和性能改进")
        logger.info(f"  4. 通过在线学习不断优化决策")
    else:
        logger.warning(f"\n⚠️ 部分测试未通过，需要进一步优化系统。")
    
    logger.info(f"\n测试完成。")