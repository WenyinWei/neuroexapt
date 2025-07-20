#!/usr/bin/env python3
"""
增强智能架构进化演示 - 95%准确率目标版本
Enhanced Intelligent Architecture Evolution Demo - 95% Accuracy Target

🎯 核心目标：
1. 在CIFAR-10上达到95%+的准确率
2. 使用Monte Carlo Dropout替代复杂的贝叶斯变分推断
3. 实现完整的贝叶斯决策框架来判断变异价值
4. 集成你提出的理论框架：将变异收益建模为随机变量

🧬 核心理论实现：
ΔI = α·ΔI_MI + β·ΔI_cond + γ·ΔI_uncert - δ·ΔI_cost

基于期望效用最大化的变异决策：
E[U(ΔI)] = E[1 - exp(-λ·ΔI)]

🔧 技术栈：
- Enhanced ResNet with SE-attention
- Monte Carlo Dropout uncertainty estimation
- Bayesian mutation decision framework
- Advanced training techniques (Mixup, CutMix, Label Smoothing)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入新的组件
from neuroexapt.core import (
    IntelligentArchitectureEvolutionEngine,
    EvolutionConfig,
    MutualInformationEstimator,
    IntelligentBottleneckDetector,
    IntelligentMutationPlanner,
    AdvancedNet2NetTransfer
)

# 导入新的不确定性估计器和决策框架
from neuroexapt.core.monte_carlo_uncertainty_estimator import MonteCarloUncertaintyEstimator
from neuroexapt.core.bayesian_mutation_decision import (
    BayesianMutationDecision, MutationEvidence, MutationPrior, MutationDecision
)

# 导入增强的ResNet
from neuroexapt.models.enhanced_resnet import (
    create_enhanced_model, EnhancedTrainingConfig, get_enhanced_transforms,
    LabelSmoothingCrossEntropy, mixup_data, mixup_criterion
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedTrainingModule:
    """
    增强的训练模块 - 目标95%准确率
    
    集成：
    - 增强的ResNet架构
    - 高级数据增强（Mixup, CutMix, RandomErasing）
    - 标签平滑和自适应学习率
    - 混合精度训练
    """
    
    def __init__(self, device, config: EnhancedTrainingConfig = None):
        self.device = device
        self.config = config or EnhancedTrainingConfig()
        
        # 创建增强模型
        self.model = create_enhanced_model(self.config).to(device)
        
        # 设置数据加载器
        self.train_loader, self.test_loader = self._setup_data_loaders()
        
        # 设置优化器和调度器
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.setup_training_components()
        
        # 训练历史
        self.train_history = []
        self.test_history = []
        
        logger.info(f"🚀 Enhanced training module initialized with {self.config.model_type}")
        logger.info(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def _setup_data_loaders(self):
        """设置数据加载器"""
        train_transform, test_transform = get_enhanced_transforms()
        
        # CIFAR-10数据集
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        return train_loader, test_loader
        
    def setup_training_components(self):
        """设置训练组件"""
        # 优化器
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.initial_lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            nesterov=True
        )
        
        # 学习率调度器
        if self.config.lr_schedule == 'cosine_annealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs, eta_min=1e-6
            )
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.config.lr_milestones, 
                gamma=self.config.lr_gamma
            )
        
        # 损失函数
        if self.config.label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(self.config.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        logger.info("✅ Training components setup complete")
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Mixup数据增强
            if self.config.use_mixup and np.random.rand() < 0.5:
                mixed_data, targets_a, targets_b, lam = mixup_data(
                    data, targets, self.config.mixup_alpha
                )
                
                self.optimizer.zero_grad()
                outputs = self.model(mixed_data)
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            else:
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            if not self.config.use_mixup or np.random.rand() >= 0.5:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # 打印进度
            if batch_idx % 100 == 99:
                current_acc = 100. * correct / total if total > 0 else 0
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx + 1}, "
                    f"Loss: {running_loss/(batch_idx + 1):.6f}, "
                    f"Acc: {current_acc:.2f}%"
                )
        
        train_acc = 100. * correct / total if total > 0 else 0
        avg_loss = running_loss / len(self.train_loader)
        
        return avg_loss, train_acc
        
    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                test_loss += self.criterion(outputs, targets).item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total
        
        return test_loss, test_acc
        
    def train_epochs(self, num_epochs):
        """训练多个epoch"""
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch + 1)
            
            # 评估
            test_loss, test_acc = self.evaluate()
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录历史
            self.train_history.append({'loss': train_loss, 'acc': train_acc})
            self.test_history.append({'loss': test_loss, 'acc': test_acc})
            
            # 更新最佳准确率
            if test_acc > best_acc:
                best_acc = test_acc
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            logger.info(
                f"Epoch {epoch + 1}: Train Acc: {train_acc:.2f}%, "
                f"Test Acc: {test_acc:.2f}%, Test Loss: {test_loss:.4f}"
            )
            
            # 早期停止检查
            if test_acc >= 95.0:
                logger.info(f"🎉 Target accuracy of 95% reached! Current: {test_acc:.2f}%")
                break
                
        return best_acc


class EnhancedEvolutionEngine:
    """
    增强的架构进化引擎
    
    集成：
    - Monte Carlo不确定性估计
    - 贝叶斯变异决策框架
    - 基于期望效用的变异选择
    """
    
    def __init__(self, device):
        self.device = device
        
        # 创建组件
        self.mi_estimator = MutualInformationEstimator()
        self.uncertainty_estimator = MonteCarloUncertaintyEstimator(
            n_samples=50, dropout_rate=0.1  # 减少采样次数以加快速度
        )
        self.bottleneck_detector = IntelligentBottleneckDetector(
            mi_estimator=self.mi_estimator,
            uncertainty_estimator=self.uncertainty_estimator
        )
        self.mutation_planner = IntelligentMutationPlanner()
        self.net2net_transfer = AdvancedNet2NetTransfer()
        
        # 贝叶斯决策框架
        self.decision_engine = BayesianMutationDecision(
            alpha=1.5,    # 提高互信息权重
            beta=1.0,     # 条件互信息权重
            gamma=0.8,    # 不确定性权重
            delta=0.3,    # 降低成本权重以鼓励探索
            risk_aversion=1.5  # 中等风险厌恶
        )
        
        logger.info("🧬 Enhanced evolution engine initialized")
        
    def detect_bottlenecks_with_decision(self, model, data_loader):
        """
        检测瓶颈并使用贝叶斯框架做出变异决策
        
        Returns:
            List[Tuple[BottleneckInfo, MutationDecision]]: 瓶颈信息和对应的决策
        """
        logger.info("🔍 开始检测瓶颈...")
        
        # 检测瓶颈
        bottlenecks = self.bottleneck_detector.detect_bottlenecks(
            model, data_loader, self.device
        )
        
        logger.info(f"🎯 检测到 {len(bottlenecks)} 个潜在瓶颈")
        
        # 为每个瓶颈做出变异决策
        bottleneck_decisions = []
        
        for bottleneck in bottlenecks:
            # 构建变异证据
            evidence = MutationEvidence(
                mutual_info_gain=bottleneck.mutual_info,
                cond_mutual_info_gain=bottleneck.conditional_mi,
                uncertainty_reduction=bottleneck.uncertainty,
                transfer_cost=0.05,  # 估计的迁移成本
                bottleneck_severity=bottleneck.severity
            )
            
            # 根据训练阶段调整风险厌恶
            # 这里假设是中期训练
            self.decision_engine.adjust_risk_aversion('middle')
            
            # 做出决策
            decision = self.decision_engine.make_decision(
                evidence, utility_threshold=0.01
            )
            
            bottleneck_decisions.append((bottleneck, decision))
            
            logger.info(
                f"🧠 层 {bottleneck.layer_name}: {decision.reasoning}"
            )
            
        return bottleneck_decisions
        
    def execute_intelligent_mutations(self, model, bottleneck_decisions):
        """
        执行智能变异
        
        Args:
            model: 原始模型
            bottleneck_decisions: 瓶颈决策列表
            
        Returns:
            变异后的模型
        """
        current_model = model
        mutation_count = 0
        
        for bottleneck, decision in bottleneck_decisions:
            if decision.should_mutate and decision.confidence > 0.3:
                try:
                    # 生成变异计划
                    mutation_plans = self.mutation_planner.plan_mutations(
                        [bottleneck], task_type='vision'
                    )
                    
                    if mutation_plans:
                        mutation_plan = mutation_plans[0]
                        
                        # 执行变异
                        mutated_model = self.net2net_transfer.apply_mutation(
                            current_model, mutation_plan
                        )
                        
                        if mutated_model is not None:
                            current_model = mutated_model
                            mutation_count += 1
                            
                            logger.info(
                                f"✅ 成功变异层 {bottleneck.layer_name}: "
                                f"{mutation_plan.mutation_type}"
                            )
                        else:
                            logger.warning(
                                f"❌ 变异失败: {bottleneck.layer_name}"
                            )
                            
                except Exception as e:
                    logger.error(f"变异错误: {bottleneck.layer_name}: {e}")
                    
        logger.info(f"🔄 总共执行了 {mutation_count} 个变异")
        return current_model


def enhanced_intelligent_evolution_demo():
    """
    增强智能架构进化演示主函数
    
    目标：在CIFAR-10上达到95%准确率
    """
    print("="*60)
    print("🚀 增强智能架构进化演示")
    print("🎯 目标：CIFAR-10上95%准确率")
    print("🧬 理论：贝叶斯变异决策 + Monte Carlo不确定性")
    print("="*60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建增强训练配置
    config = EnhancedTrainingConfig()
    config.epochs = 50  # 减少epoch数用于演示
    config.batch_size = 128
    
    # 初始化训练模块
    print("\n📚 初始化增强训练模块...")
    trainer = EnhancedTrainingModule(device, config)
    
    # 基础训练
    print("\n📚 开始基础训练...")
    initial_epochs = 15
    best_acc = trainer.train_epochs(initial_epochs)
    
    print(f"\n🎯 基础训练完成，最佳准确率: {best_acc:.2f}%")
    
    # 如果已经达到95%，直接返回
    if best_acc >= 95.0:
        print("\n🎉 已达到95%准确率目标！")
        return trainer.model
        
    # 智能架构进化
    print("\n🧬 开始智能架构进化...")
    
    evolution_engine = EnhancedEvolutionEngine(device)
    
    max_evolution_rounds = 3
    current_model = trainer.model
    
    for round_idx in range(max_evolution_rounds):
        print(f"\n🔄 进化轮次 {round_idx + 1}/{max_evolution_rounds}")
        
        # 检测瓶颈并做出决策
        bottleneck_decisions = evolution_engine.detect_bottlenecks_with_decision(
            current_model, trainer.test_loader
        )
        
        # 过滤出值得变异的瓶颈
        valuable_mutations = [
            (b, d) for b, d in bottleneck_decisions 
            if d.should_mutate and d.confidence > 0.3
        ]
        
        if not valuable_mutations:
            print("\n🛑 没有发现值得执行的变异，停止进化")
            break
            
        print(f"\n📋 发现 {len(valuable_mutations)} 个值得执行的变异")
        
        # 执行变异
        mutated_model = evolution_engine.execute_intelligent_mutations(
            current_model, valuable_mutations
        )
        
        # 更新训练器的模型
        trainer.model = mutated_model.to(device)
        trainer.setup_training_components()  # 重新设置优化器
        
        # 继续训练变异后的模型
        print(f"\n📚 训练变异后的模型...")
        additional_epochs = 10
        new_best_acc = trainer.train_epochs(additional_epochs)
        
        print(f"\n📊 变异后最佳准确率: {new_best_acc:.2f}%")
        
        # 记录变异结果
        for bottleneck, decision in valuable_mutations:
            actual_gain = (new_best_acc - best_acc) / 100.0
            success = actual_gain > 0
            
            evolution_engine.decision_engine.record_mutation_outcome(
                MutationEvidence(
                    mutual_info_gain=bottleneck.mutual_info,
                    cond_mutual_info_gain=bottleneck.conditional_mi,
                    uncertainty_reduction=bottleneck.uncertainty,
                    transfer_cost=0.05,
                    bottleneck_severity=bottleneck.severity
                ),
                actual_gain,
                success
            )
            
        best_acc = max(best_acc, new_best_acc)
        current_model = trainer.model
        
        # 检查是否达到目标
        if best_acc >= 95.0:
            print(f"\n🎉 达到95%准确率目标！最终准确率: {best_acc:.2f}%")
            break
    
    print(f"\n🏁 进化完成，最终最佳准确率: {best_acc:.2f}%")
    
    # 显示训练历史
    if trainer.test_history:
        final_acc = trainer.test_history[-1]['acc']
        print(f"\n📈 最终测试准确率: {final_acc:.2f}%")
        
        if final_acc >= 95.0:
            print("\n🎊 成功达成95%准确率目标！")
        else:
            print(f"\n📊 距离95%目标还差: {95.0 - final_acc:.2f}%")
            print("💡 建议：增加训练轮数或调整超参数")
    
    return current_model


if __name__ == "__main__":
    try:
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 运行演示
        final_model = enhanced_intelligent_evolution_demo()
        
        print("\n✨ 演示完成！")
        
    except Exception as e:
        logger.error(f"演示过程中遇到错误: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n❌ 演示失败，请检查错误信息")