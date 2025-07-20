"""
共享演示工具模块
Shared Demo Utilities Module

提取多个演示脚本的共同逻辑，减少代码重复
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
import logging
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入核心组件
from neuroexapt.core import (
    IntelligentArchitectureEvolutionEngine,
    EvolutionConfig,
    MutualInformationEstimator,
    IntelligentBottleneckDetector,
    IntelligentMutationPlanner,
    AdvancedNet2NetTransfer,
    MonteCarloUncertaintyEstimator,
    BayesianMutationDecision,
    MutationEvidence,
    MutationPrior,
    MutationDecision,
    MCUncertaintyConfig,
    BayesianDecisionConfig
)

# 导入模型组件
from neuroexapt.models import (
    create_enhanced_model,
    EnhancedTrainingConfig,
    get_enhanced_transforms,
    LabelSmoothingCrossEntropy,
    mixup_data,
    mixup_criterion
)

logger = logging.getLogger(__name__)


@dataclass
class DemoConfig:
    """演示配置"""
    # 设备和基础设置
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    seed: int = 42
    
    # 数据设置
    data_root: str = './data'
    batch_size: int = 128
    num_workers: int = 4
    
    # 训练设置
    initial_epochs: int = 15
    evolution_rounds: int = 3
    additional_epochs_per_round: int = 10
    target_accuracy: float = 95.0
    
    # 模型设置
    model_type: str = 'enhanced_resnet34'
    use_enhanced_features: bool = True
    
    # 进化设置
    use_monte_carlo_uncertainty: bool = True
    use_bayesian_decision: bool = True
    
    # 日志设置
    log_level: str = 'INFO'
    verbose: bool = True


class SharedDataManager:
    """共享数据管理器"""
    
    def __init__(self, config: DemoConfig):
        self.config = config
        self.train_loader = None
        self.test_loader = None
        
    def setup_data_loaders(self):
        """设置数据加载器"""
        if self.config.use_enhanced_features:
            train_transform, test_transform = get_enhanced_transforms()
        else:
            # 基础变换
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                   std=[0.2023, 0.1994, 0.2010])
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                   std=[0.2023, 0.1994, 0.2010])
            ])
        
        # CIFAR-10数据集
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.config.data_root, train=True, download=True, 
            transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.config.data_root, train=False, download=True, 
            transform=test_transform
        )
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=self.config.num_workers, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers, pin_memory=True
        )
        
        logger.info(f"数据加载器设置完成: Train={len(train_dataset)}, Test={len(test_dataset)}")
        return self.train_loader, self.test_loader


class SharedTrainer:
    """共享训练器"""
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 train_loader: DataLoader, test_loader: DataLoader,
                 config: DemoConfig):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        
        # 训练组件
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.setup_training_components()
        
        # 训练历史
        self.train_history = []
        self.test_history = []
        
    def setup_training_components(self):
        """设置训练组件"""
        # 优化器
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-6
        )
        
        # 损失函数
        if self.config.use_enhanced_features:
            self.criterion = LabelSmoothingCrossEntropy(0.1)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # 数据增强
            if (self.config.use_enhanced_features and 
                hasattr(self.config, 'use_mixup') and 
                getattr(self.config, 'use_mixup', False) and 
                np.random.rand() < 0.5):
                mixed_data, targets_a, targets_b, lam = mixup_data(data, targets, 0.2)
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
            if not (self.config.use_enhanced_features and np.random.rand() < 0.5):
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # 打印进度
            if self.config.verbose and batch_idx % 100 == 99:
                current_acc = 100. * correct / total if total > 0 else 0
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx + 1}, "
                    f"Loss: {running_loss/(batch_idx + 1):.6f}, "
                    f"Acc: {current_acc:.2f}%"
                )
        
        train_acc = 100. * correct / total if total > 0 else 0
        avg_loss = running_loss / len(self.train_loader)
        
        return avg_loss, train_acc
        
    def evaluate(self) -> Tuple[float, float]:
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
        
    def train_epochs(self, num_epochs: int) -> float:
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
                
            if self.config.verbose:
                logger.info(
                    f"Epoch {epoch + 1}: Train Acc: {train_acc:.2f}%, "
                    f"Test Acc: {test_acc:.2f}%, Test Loss: {test_loss:.4f}"
                )
                
            # 早期停止检查
            if test_acc >= self.config.target_accuracy:
                logger.info(f"🎉 Target accuracy of {self.config.target_accuracy}% reached! Current: {test_acc:.2f}%")
                break
                
        return best_acc


class SharedEvolutionEngine:
    """共享进化引擎"""
    
    def __init__(self, device: torch.device, config: DemoConfig):
        self.device = device
        self.config = config
        
        # 创建组件
        self.mi_estimator = MutualInformationEstimator()
        
        # 不确定性估计器
        if config.use_monte_carlo_uncertainty:
            mc_config = MCUncertaintyConfig(
                n_samples=30,  # 减少采样数量以提高速度
                dropout_rate=0.1,
                max_batches=3,
                use_wrapper=True
            )
            self.uncertainty_estimator = MonteCarloUncertaintyEstimator(mc_config)
        else:
            from neuroexapt.core import BayesianUncertaintyEstimator
            self.uncertainty_estimator = BayesianUncertaintyEstimator()
            
        self.bottleneck_detector = IntelligentBottleneckDetector(
            mi_estimator=self.mi_estimator,
            uncertainty_estimator=self.uncertainty_estimator
        )
        self.mutation_planner = IntelligentMutationPlanner()
        self.net2net_transfer = AdvancedNet2NetTransfer()
        
        # 贝叶斯决策框架
        if config.use_bayesian_decision:
            decision_config = BayesianDecisionConfig(
                alpha=1.5,
                beta=1.0, 
                gamma=0.8,
                delta=0.3,
                risk_aversion=1.5,
                confidence_threshold=0.2  # 降低阈值以允许更多变异
            )
            self.decision_engine = BayesianMutationDecision(decision_config)
        else:
            self.decision_engine = None
            
        logger.info("🧬 Shared evolution engine initialized")
        
    def detect_and_decide_mutations(self, model: nn.Module, data_loader) -> List[Tuple]:
        """检测瓶颈并做出变异决策"""
        logger.info("🔍 开始检测瓶颈...")
        
        # 检测瓶颈
        bottlenecks = self.bottleneck_detector.detect_bottlenecks(
            model, data_loader, self.device
        )
        
        logger.info(f"🎯 检测到 {len(bottlenecks)} 个潜在瓶颈")
        
        if not self.decision_engine:
            # 简单策略：选择前几个瓶颈
            return [(b, None) for b in bottlenecks[:3]]
            
        # 使用贝叶斯决策
        bottleneck_decisions = []
        
        for bottleneck in bottlenecks:
            # 构建变异证据
            evidence = MutationEvidence(
                mutual_info_gain=bottleneck.mutual_info,
                cond_mutual_info_gain=getattr(bottleneck, 'conditional_mi', 0.0),
                uncertainty_reduction=bottleneck.uncertainty,
                transfer_cost=0.05,
                bottleneck_severity=bottleneck.severity
            )
            
            # 做出决策
            decision = self.decision_engine.make_decision(evidence)
            bottleneck_decisions.append((bottleneck, decision))
            
            if self.config.verbose:
                logger.info(f"🧠 层 {bottleneck.layer_name}: {decision.reasoning}")
                
        return bottleneck_decisions
        
    def execute_mutations(self, model: nn.Module, bottleneck_decisions: List[Tuple]) -> nn.Module:
        """执行变异"""
        current_model = model
        mutation_count = 0
        
        for bottleneck, decision in bottleneck_decisions:
            should_mutate = decision is None or (decision.should_mutate and decision.confidence > 0.2)
            
            if should_mutate:
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
                            
                            if self.config.verbose:
                                logger.info(
                                    f"✅ 成功变异层 {bottleneck.layer_name}: "
                                    f"{mutation_plan.mutation_type}"
                                )
                        else:
                            logger.warning(f"❌ 变异失败: {bottleneck.layer_name}")
                            
                except Exception as e:
                    logger.error(f"变异错误: {bottleneck.layer_name}: {e}")
                    
        logger.info(f"🔄 总共执行了 {mutation_count} 个变异")
        return current_model
        
    def cleanup(self):
        """清理资源"""
        if hasattr(self.uncertainty_estimator, 'cleanup'):
            self.uncertainty_estimator.cleanup()


def setup_demo_environment(config: DemoConfig):
    """设置演示环境"""
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 设置设备
    if config.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.device)
        
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return device


def create_model_from_config(config: DemoConfig) -> nn.Module:
    """根据配置创建模型"""
    if config.use_enhanced_features:
        # 使用增强模型
        model_config = EnhancedTrainingConfig()
        model_config.model_type = config.model_type
        model_config.batch_size = config.batch_size
        return create_enhanced_model(model_config)
    else:
        # 使用基础ResNet
        from torchvision.models import resnet18
        model = resnet18(num_classes=10)
        # 修改第一层以适应CIFAR-10
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model


def run_complete_demo(config: DemoConfig) -> Dict[str, Any]:
    """运行完整的演示流程"""
    print("="*60)
    print("🚀 智能架构进化演示")
    print(f"🎯 目标：CIFAR-10上{config.target_accuracy}%准确率")
    print(f"🧬 Monte Carlo不确定性: {config.use_monte_carlo_uncertainty}")
    print(f"🎲 贝叶斯决策: {config.use_bayesian_decision}")
    print("="*60)
    
    # 设置环境
    device = setup_demo_environment(config)
    logger.info(f"使用设备: {device}")
    
    # 设置数据
    data_manager = SharedDataManager(config)
    train_loader, test_loader = data_manager.setup_data_loaders()
    
    # 创建模型
    model = create_model_from_config(config)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = SharedTrainer(model, device, train_loader, test_loader, config)
    
    # 基础训练
    print("\n📚 开始基础训练...")
    best_acc = trainer.train_epochs(config.initial_epochs)
    print(f"\n🎯 基础训练完成，最佳准确率: {best_acc:.2f}%")
    
    results = {
        'initial_accuracy': best_acc,
        'evolution_rounds': [],
        'final_accuracy': best_acc,
        'target_reached': best_acc >= config.target_accuracy
    }
    
    # 如果已经达到目标，直接返回
    if best_acc >= config.target_accuracy:
        print(f"\n🎉 已达到{config.target_accuracy}%准确率目标！")
        results['final_accuracy'] = best_acc
        return results
        
    # 智能架构进化
    print("\n🧬 开始智能架构进化...")
    evolution_engine = SharedEvolutionEngine(device, config)
    
    current_model = trainer.model
    
    try:
        for round_idx in range(config.evolution_rounds):
            print(f"\n🔄 进化轮次 {round_idx + 1}/{config.evolution_rounds}")
            
            # 检测和决策
            bottleneck_decisions = evolution_engine.detect_and_decide_mutations(
                current_model, test_loader
            )
            
            # 过滤值得变异的瓶颈
            valuable_mutations = [
                (b, d) for b, d in bottleneck_decisions 
                if d is None or (d.should_mutate and d.confidence > 0.2)
            ]
            
            if not valuable_mutations:
                print("\n🛑 没有发现值得执行的变异，停止进化")
                break
                
            print(f"\n📋 发现 {len(valuable_mutations)} 个值得执行的变异")
            
            # 执行变异
            mutated_model = evolution_engine.execute_mutations(
                current_model, valuable_mutations
            )
            
            # 更新训练器
            trainer.model = mutated_model.to(device)
            trainer.setup_training_components()
            
            # 继续训练
            print("\n📚 训练变异后的模型...")
            new_best_acc = trainer.train_epochs(config.additional_epochs_per_round)
            
            print(f"\n📊 变异后最佳准确率: {new_best_acc:.2f}%")
            
            # 记录结果
            round_result = {
                'round': round_idx + 1,
                'mutations_applied': len(valuable_mutations),
                'accuracy_before': best_acc,
                'accuracy_after': new_best_acc,
                'improvement': new_best_acc - best_acc
            }
            results['evolution_rounds'].append(round_result)
            
            best_acc = max(best_acc, new_best_acc)
            current_model = trainer.model
            
            # 检查是否达到目标
            if best_acc >= config.target_accuracy:
                print(f"\n🎉 达到{config.target_accuracy}%准确率目标！最终准确率: {best_acc:.2f}%")
                results['target_reached'] = True
                break
                
    finally:
        # 清理资源
        evolution_engine.cleanup()
    
    results['final_accuracy'] = best_acc
    
    print(f"\n🏁 演示完成，最终最佳准确率: {best_acc:.2f}%")
    
    if results['target_reached']:
        print(f"\n🎊 成功达成{config.target_accuracy}%准确率目标！")
    else:
        print(f"\n📊 距离{config.target_accuracy}%目标还差: {config.target_accuracy - best_acc:.2f}%")
        
    return results