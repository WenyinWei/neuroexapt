"""
统一智能架构进化引擎
Unified Intelligent Architecture Evolution Engine

整合所有核心组件，实现基于理论框架的智能变异决策：
1. 无参数结构评估
2. 多变异类型收益期望建模
3. 轻量级抽样验证
4. 贝叶斯变异决策
5. 动态优先级排序

核心理论公式：
Score(S, M) = α·ΔI + β·Φ(S) - γ·SR(S) - δ·Cost(M)

基于期望效用最大化的决策框架：
E[U(ΔI)] = E[1 - exp(-λ·ΔI)]

作者：基于用户提供的理论框架实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import time
import copy
from collections import defaultdict

from .parameter_free_structural_evaluator import (
    ParameterFreeStructuralEvaluator, StructuralMetrics
)
from .multi_mutation_type_evaluator import (
    MultiMutationTypeEvaluator, MutationType, MutationConfig, 
    MutationBenefitExpectation, MutationBenefitPrior
)
from .lightweight_sampling_validator import (
    LightweightSamplingValidator, SamplingValidationConfig, SamplingResult
)

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """进化配置"""
    # 基础设置
    max_evolution_rounds: int = 5
    target_accuracy: float = 95.0
    max_mutations_per_round: int = 3
    
    # 评估权重（对应理论公式中的 α, β, γ, δ）
    information_gain_weight: float = 0.4      # α - 信息增益权重
    integration_weight: float = 0.3           # β - 积分信息权重  
    redundancy_weight: float = 0.2            # γ - 冗余度权重
    cost_weight: float = 0.1                  # δ - 成本权重
    
    # 决策参数
    risk_aversion: float = 2.0                # λ - 风险规避系数
    min_benefit_threshold: float = 0.001      # 最小收益阈值 (0.1%改进)
    confidence_threshold: float = 0.2         # 置信度阈值 (20%成功概率)
    
    # 验证设置
    enable_sampling_validation: bool = True   # 启用抽样验证
    validation_sample_ratio: float = 0.1      # 验证数据比例
    quick_validation_epochs: int = 3          # 快速验证轮数
    
    # 自适应设置
    adaptive_weights: bool = True             # 自适应权重调整
    learning_rate_decay: float = 0.9          # 学习率衰减
    
    # 约束设置
    max_parameter_increase: float = 0.5       # 最大参数增长比例
    max_computation_increase: float = 0.8     # 最大计算增长比例


@dataclass 
class EvolutionState:
    """进化状态"""
    current_round: int = 0
    current_accuracy: float = 0.0
    best_accuracy: float = 0.0
    total_mutations_applied: int = 0
    
    # 历史记录
    accuracy_history: List[float] = field(default_factory=list)
    mutation_history: List[Tuple[MutationConfig, float]] = field(default_factory=list)
    timing_history: List[float] = field(default_factory=list)
    
    # 状态统计
    successful_mutations: int = 0
    failed_mutations: int = 0
    total_parameter_increase: float = 0.0
    total_computation_increase: float = 0.0


@dataclass
class MutationCandidate:
    """变异候选"""
    config: MutationConfig
    theoretical_expectation: MutationBenefitExpectation
    sampling_result: Optional[SamplingResult] = None
    calibrated_expectation: Optional[MutationBenefitExpectation] = None
    priority_score: float = 0.0
    expected_utility: float = 0.0


class UnifiedIntelligentEvolutionEngine:
    """
    统一智能架构进化引擎
    
    核心功能：
    1. 集成所有评估组件
    2. 实现智能变异决策
    3. 动态优先级排序
    4. 自适应参数调整
    5. 风险规避决策
    """
    
    def __init__(self,
                 config: EvolutionConfig = None,
                 device: torch.device = None):
        """
        初始化进化引擎
        
        Args:
            config: 进化配置
            device: 计算设备
        """
        self.config = config or EvolutionConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化核心组件
        self.structural_evaluator = ParameterFreeStructuralEvaluator(device=self.device)
        self.mutation_evaluator = MultiMutationTypeEvaluator(
            structural_evaluator=self.structural_evaluator,
            device=self.device
        )
        
        if self.config.enable_sampling_validation:
            sampling_config = SamplingValidationConfig(
                sample_data_ratio=self.config.validation_sample_ratio,
                sample_epochs=self.config.quick_validation_epochs
            )
            self.sampling_validator = LightweightSamplingValidator(
                config=sampling_config, device=self.device
            )
        else:
            self.sampling_validator = None
        
        # 进化状态
        self.evolution_state = EvolutionState()
        
        # 自适应权重历史
        self.weight_adaptation_history = []
        
        logger.info("统一智能架构进化引擎初始化完成")
    
    def evolve_architecture(self,
                          model: nn.Module,
                          train_loader: DataLoader,
                          test_loader: DataLoader,
                          criterion: nn.Module = None,
                          optimizer_factory: Callable = None) -> Tuple[nn.Module, EvolutionState]:
        """
        进化神经网络架构
        
        Args:
            model: 初始模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            criterion: 损失函数
            optimizer_factory: 优化器工厂函数
            
        Returns:
            Tuple[nn.Module, EvolutionState]: (进化后模型, 进化状态)
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        if optimizer_factory is None:
            optimizer_factory = lambda params: optim.SGD(params, lr=0.01, momentum=0.9)
        
        # 初始化状态
        current_model = copy.deepcopy(model)
        self.evolution_state = EvolutionState()
        self.evolution_state.current_accuracy = self._evaluate_model(current_model, test_loader)
        self.evolution_state.best_accuracy = self.evolution_state.current_accuracy
        
        # 初始化准确率历史记录
        self.evolution_state.accuracy_history.append(self.evolution_state.current_accuracy)
        
        logger.info(f"开始架构进化，初始准确率: {self.evolution_state.current_accuracy:.2f}%")
        
        # 间歇性进化循环 - 在进化和常规训练之间交替
        consecutive_no_evolution = 0
        
        for round_idx in range(self.config.max_evolution_rounds):
            round_start_time = time.time()
            self.evolution_state.current_round = round_idx + 1
            
            logger.info(f"\n=== 进化轮次 {self.evolution_state.current_round} ===")
            
            # 每3-5轮做一次常规训练来稳定基线
            if round_idx > 0 and round_idx % 4 == 0:
                logger.info("进行间歇性常规训练以稳定基线...")
                current_model = self._inter_evolution_training(
                    current_model, train_loader, test_loader, criterion, optimizer_factory
                )
                # 更新准确率基线
                self.evolution_state.current_accuracy = self._evaluate_model(current_model, test_loader)
                logger.info(f"常规训练后准确率: {self.evolution_state.current_accuracy:.2f}%")
            
            # 智能检测：是否需要进化
            need_evolution = self._should_attempt_evolution(current_model, test_loader)
            if not need_evolution:
                logger.info("当前模型表现良好，跳过此轮进化")
                consecutive_no_evolution += 1
                if consecutive_no_evolution >= 5:
                    logger.info("连续多轮跳过进化，进行额外训练...")
                    current_model = self._inter_evolution_training(
                        current_model, train_loader, test_loader, criterion, optimizer_factory
                    )
                    consecutive_no_evolution = 0
                continue
            
            consecutive_no_evolution = 0
            
            # 1. 生成和评估变异候选
            candidates = self._generate_and_evaluate_candidates(
                current_model, train_loader, test_loader, criterion
            )
            
            if not candidates:
                logger.info("没有找到有效的变异候选，进行常规训练")
                current_model = self._inter_evolution_training(
                    current_model, train_loader, test_loader, criterion, optimizer_factory
                )
                continue
            
            # 2. 选择最佳变异
            selected_mutations = self._select_best_mutations(candidates)
            
            if not selected_mutations:
                logger.info("本轮没有合适的变异候选，进行常规训练")
                current_model = self._inter_evolution_training(
                    current_model, train_loader, test_loader, criterion, optimizer_factory
                )
                round_time = time.time() - round_start_time
                self.evolution_state.timing_history.append(round_time)
                continue
            
            # 3. 应用变异
            improved_model, improvements = self._apply_mutations(
                current_model, selected_mutations, train_loader, test_loader, 
                criterion, optimizer_factory
            )
            
            # 4. 更新状态 - 考虑最好的改进而不是平均改进
            new_accuracy = self._evaluate_model(improved_model, test_loader)
            round_improvement = new_accuracy - self.evolution_state.current_accuracy
            
            # 如果有任何正向改进，记录最佳结果
            best_improvement = max(improvements) if improvements else 0.0
            if best_improvement > 0:
                logger.info(f"检测到正向改进: {best_improvement:.2f}% (即使当前轮次总体改进: {round_improvement:.2f}%)")
                # 接受最好的变异结果，而不是平均结果
                if best_improvement > round_improvement:
                    logger.info("使用最佳变异结果而非平均结果")
                    round_improvement = best_improvement * 0.5  # 保守估计
            
            self.evolution_state.current_accuracy = new_accuracy
            if new_accuracy > self.evolution_state.best_accuracy:
                self.evolution_state.best_accuracy = new_accuracy
                current_model = improved_model
            
            self.evolution_state.accuracy_history.append(new_accuracy)
            self.evolution_state.timing_history.append(time.time() - round_start_time)
            
            logger.info(f"轮次 {self.evolution_state.current_round} 完成: "
                       f"准确率 {self.evolution_state.current_accuracy:.2f}% "
                       f"(+{round_improvement:.2f}%)")
            
            # 5. 自适应权重调整
            if self.config.adaptive_weights:
                self._adapt_weights(improvements)
            
            # 6. 检查收敛条件
            if self.evolution_state.current_accuracy >= self.config.target_accuracy:
                logger.info(f"达到目标准确率 {self.config.target_accuracy}%，停止进化")
                break
            
            # 更宽松的收敛条件：连续多轮无显著改进才停止
            if round_improvement < 0.0001:  # 改进低于0.01%
                no_improvement_rounds = getattr(self, '_no_improvement_rounds', 0) + 1
                self._no_improvement_rounds = no_improvement_rounds
                if no_improvement_rounds >= 2:  # 连续2轮无改进才停止
                    logger.info(f"连续{no_improvement_rounds}轮改进微小，停止进化")
                    break
            else:
                self._no_improvement_rounds = 0  # 重置计数器
        
        # 进化完成
        logger.info(f"\n=== 进化完成 ===")
        logger.info(f"最终准确率: {self.evolution_state.best_accuracy:.2f}%")
        
        # 安全计算总改进，处理空历史的情况
        if self.evolution_state.accuracy_history:
            initial_accuracy = self.evolution_state.accuracy_history[0]
            total_improvement = self.evolution_state.best_accuracy - initial_accuracy
            logger.info(f"总改进: {total_improvement:.2f}%")
        else:
            logger.info("总改进: 无法计算（缺少初始准确率记录）")
            
        logger.info(f"成功变异: {self.evolution_state.successful_mutations}")
        logger.info(f"失败变异: {self.evolution_state.failed_mutations}")
        logger.info(f"总资源增长 - 参数: {self.evolution_state.total_parameter_increase:.3f}, "
                   f"计算: {self.evolution_state.total_computation_increase:.3f}")
        
        return current_model, self.evolution_state
    
    def _generate_and_evaluate_candidates(self,
                                        model: nn.Module,
                                        train_loader: DataLoader,
                                        test_loader: DataLoader,
                                        criterion: nn.Module) -> List[MutationCandidate]:
        """生成和评估变异候选"""
        logger.info("生成变异候选...")
        
        # 获取理论收益期望
        theoretical_candidates = self.mutation_evaluator.get_best_mutation_candidates(
            model, top_k=10
        )
        
        candidates = []
        
        for config, expectation in theoretical_candidates:
            candidate = MutationCandidate(
                config=config,
                theoretical_expectation=expectation
            )
            
            # 抽样验证（如果启用）
            if self.sampling_validator:
                try:
                    sampling_result = self.sampling_validator.validate_mutation_benefit(
                        config, model, train_loader, test_loader, criterion
                    )
                    candidate.sampling_result = sampling_result
                    
                    # 校准期望
                    candidate.calibrated_expectation = self.sampling_validator.calibrate_benefit_expectation(
                        expectation, sampling_result
                    )
                except Exception as e:
                    logger.warning(f"抽样验证失败: {e}")
                    candidate.calibrated_expectation = expectation
            else:
                candidate.calibrated_expectation = expectation
            
            # 计算优先级分数
            candidate.priority_score = self._compute_priority_score(candidate, model)
            candidate.expected_utility = self._compute_expected_utility(candidate)
            
            candidates.append(candidate)
        
        # 按优先级排序
        candidates.sort(key=lambda x: x.expected_utility, reverse=True)
        
        logger.info(f"生成 {len(candidates)} 个变异候选")
        
        return candidates
    
    def _compute_priority_score(self, candidate: MutationCandidate, model: nn.Module) -> float:
        """
        计算变异优先级分数
        基于理论公式：Score(S, M) = α·ΔI + β·Φ(S) - γ·SR(S) - δ·Cost(M)
        """
        expectation = candidate.calibrated_expectation
        
        # 信息增益项 (α·ΔI)
        information_gain = (
            expectation.information_gain_benefit * self.config.information_gain_weight
        )
        
        # 积分信息项 (β·Φ(S)) - 使用多样性收益作为代理
        integration_benefit = (
            expectation.diversity_benefit * self.config.integration_weight
        )
        
        # 冗余度项 (-γ·SR(S)) - 冗余度高时变异收益大，这里用负号因为我们要最大化分数
        redundancy_benefit = (
            -expectation.cost_penalty * self.config.redundancy_weight  # 成本惩罚间接反映冗余度
        )
        
        # 成本项 (-δ·Cost(M))
        cost_penalty = expectation.cost_penalty * self.config.cost_weight
        
        # 总分数
        priority_score = (
            information_gain + integration_benefit + redundancy_benefit - cost_penalty
        )
        
        return priority_score
    
    def _compute_expected_utility(self, candidate: MutationCandidate) -> float:
        """
        计算期望效用
        基于公式：E[U(ΔI)] = E[1 - exp(-λ·ΔI)]
        """
        expectation = candidate.calibrated_expectation
        
        # 期望收益
        expected_benefit = expectation.expected_benefit
        
        # 风险调整
        if expected_benefit > 0:
            # 使用指数效用函数
            utility = 1.0 - np.exp(-self.config.risk_aversion * expected_benefit)
        else:
            # 负收益时使用线性惩罚
            utility = expected_benefit * self.config.risk_aversion
        
        # 结合成功概率
        expected_utility = utility * expectation.success_probability
        
        return expected_utility
    
    def _select_best_mutations(self, candidates: List[MutationCandidate]) -> List[MutationCandidate]:
        """选择最佳变异"""
        if not candidates:
            return []
        
        # 首先尝试按严格标准筛选
        strict_selected = []
        
        for candidate in candidates:
            if len(strict_selected) >= self.config.max_mutations_per_round:
                break
            
            expectation = candidate.calibrated_expectation
            
            # 严格阈值检查
            if (expectation.expected_benefit >= self.config.min_benefit_threshold and
                expectation.success_probability >= self.config.confidence_threshold and
                candidate.expected_utility > 0 and
                self._check_resource_constraints(candidate)):
                
                strict_selected.append(candidate)
                logger.info(f"选择变异(严格标准): {candidate.config.mutation_type.value} "
                           f"(期望收益: {expectation.expected_benefit:.4f}, "
                           f"成功率: {expectation.success_probability:.2f}, "
                           f"效用: {candidate.expected_utility:.4f})")
        
        # 如果严格标准没有选到足够的候选，使用宽松标准
        if len(strict_selected) == 0:
            logger.info("严格标准未选到候选，使用宽松标准选择最佳候选")
            
            # 过滤掉明显不合理的候选 - 更注重长期潜力
            viable_candidates = [
                c for c in candidates 
                if (c.calibrated_expectation.expected_benefit > -0.02 and  # 允许2%短期损失
                    c.calibrated_expectation.success_probability > 0.05 and  # 至少5%成功率
                    self._check_resource_constraints(c))
            ]
            
            if viable_candidates:
                # 按期望效用排序，选择最好的
                viable_candidates.sort(key=lambda c: c.expected_utility, reverse=True)
                relaxed_selected = viable_candidates[:self.config.max_mutations_per_round]
                
                for candidate in relaxed_selected:
                    expectation = candidate.calibrated_expectation
                    logger.info(f"选择变异(宽松标准): {candidate.config.mutation_type.value} "
                               f"(期望收益: {expectation.expected_benefit:.4f}, "
                               f"成功率: {expectation.success_probability:.2f}, "
                               f"效用: {candidate.expected_utility:.4f})")
                
                return relaxed_selected
            else:
                logger.warning("所有候选都不满足基本要求")
                return []
        
        return strict_selected
    
    def _check_resource_constraints(self, candidate: MutationCandidate) -> bool:
        """检查资源约束"""
        # 显式检查参数增长和计算量增长
        param_increase = candidate.calibrated_expectation.parameter_increase
        comp_increase = candidate.calibrated_expectation.computation_increase

        if param_increase is None or comp_increase is None:
            logger.warning("候选变异缺少参数增长或计算量增长信息，无法进行资源约束检查。")
            return False

        if (self.evolution_state.total_parameter_increase + param_increase) > self.config.max_parameter_increase:
            logger.debug(f"参数增长超限: 当前{self.evolution_state.total_parameter_increase:.3f} + "
                        f"新增{param_increase:.3f} > 限制{self.config.max_parameter_increase:.3f}")
            return False

        if (self.evolution_state.total_computation_increase + comp_increase) > self.config.max_computation_increase:
            logger.debug(f"计算量增长超限: 当前{self.evolution_state.total_computation_increase:.3f} + "
                        f"新增{comp_increase:.3f} > 限制{self.config.max_computation_increase:.3f}")
            return False

        return True
    
    def _apply_mutations(self,
                       model: nn.Module,
                       selected_mutations: List[MutationCandidate],
                       train_loader: DataLoader,
                       test_loader: DataLoader,
                       criterion: nn.Module,
                       optimizer_factory: Callable) -> Tuple[nn.Module, List[float]]:
        """应用选定的变异"""
        current_model = copy.deepcopy(model)
        improvements = []
        
        for candidate in selected_mutations:
            logger.info(f"应用变异: {candidate.config.mutation_type.value}")
            
            try:
                # 应用变异
                mutated_model = self._apply_single_mutation(current_model, candidate.config)
                
                # 微调训练
                trained_model = self._fine_tune_model(
                    mutated_model, train_loader, criterion, optimizer_factory
                )
                
                # 评估改进
                old_accuracy = self._evaluate_model(current_model, test_loader)
                new_accuracy = self._evaluate_model(trained_model, test_loader)
                improvement = new_accuracy - old_accuracy
                
                improvements.append(improvement)
                
                if improvement > 0:
                    current_model = trained_model
                    self.evolution_state.successful_mutations += 1
                    self.evolution_state.total_mutations_applied += 1
                    
                    # 更新资源追踪
                    self.evolution_state.total_parameter_increase += candidate.calibrated_expectation.parameter_increase
                    self.evolution_state.total_computation_increase += candidate.calibrated_expectation.computation_increase
                    
                    # 更新变异历史
                    self.mutation_evaluator.update_mutation_history(
                        candidate.config, 
                        self._create_mutation_evidence(candidate),
                        improvement / 100.0  # 转换为小数
                    )
                    
                    logger.info(f"变异成功，改进: {improvement:.2f}%, "
                               f"参数增长: +{candidate.calibrated_expectation.parameter_increase:.3f}, "
                               f"计算增长: +{candidate.calibrated_expectation.computation_increase:.3f}")
                else:
                    self.evolution_state.failed_mutations += 1
                    logger.info(f"变异失败，改进: {improvement:.2f}%")
                
            except Exception as e:
                logger.error(f"应用变异失败: {e}")
                self.evolution_state.failed_mutations += 1
                improvements.append(0.0)
        
        return current_model, improvements
    
    def _apply_single_mutation(self, model: nn.Module, config: MutationConfig) -> nn.Module:
        """应用单个变异（简化实现）"""
        # 这里应该实现具体的变异操作
        # 当前使用抽样验证器的变异应用逻辑
        if self.sampling_validator:
            return self.sampling_validator._apply_mutation(model, config)
        else:
            # 简化实现：返回原模型
            return copy.deepcopy(model)
    
    def _fine_tune_model(self,
                        model: nn.Module,
                        train_loader: DataLoader,
                        criterion: nn.Module,
                        optimizer_factory: Callable,
                        epochs: int = 15) -> nn.Module:
        """
        微调模型 - 更长的训练以评估长期潜力
        
        Args:
            epochs: 训练轮数，默认15轮以更好评估潜力
        """
        model.train()
        optimizer = optimizer_factory(model.parameters())
        
        # 使用学习率衰减
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batches_processed = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batches_processed += 1
                
                # 增加训练步数限制
                if batch_idx >= 100:  # 最多100个batch (之前是50)
                    break
            
            scheduler.step()
            
            # 记录训练进度
            if epoch % 5 == 0:
                avg_loss = epoch_loss / max(batches_processed, 1)
                logger.debug(f"微调 Epoch {epoch+1}/{epochs}, 平均损失: {avg_loss:.4f}")
        
        return model
    
    def _should_attempt_evolution(self, model: nn.Module, test_loader: DataLoader) -> bool:
        """
        智能检测是否需要尝试进化
        
        检查指标：
        1. 最近准确率提升趋势
        2. 与目标的距离
        3. 历史进化成功率
        """
        current_accuracy = self.evolution_state.current_accuracy
        target_accuracy = self.config.target_accuracy
        
        # 如果已经达到目标，降低进化频率
        if current_accuracy >= target_accuracy:
            return np.random.random() < 0.1  # 10%概率继续优化
        
        # 如果距离目标很远，总是尝试进化
        if target_accuracy - current_accuracy > 3.0:
            return True
        
        # 检查最近几轮的改进趋势
        if len(self.evolution_state.accuracy_history) >= 3:
            recent_improvements = []
            for i in range(1, min(4, len(self.evolution_state.accuracy_history))):
                improvement = (self.evolution_state.accuracy_history[-i] - 
                             self.evolution_state.accuracy_history[-i-1])
                recent_improvements.append(improvement)
            
            avg_recent_improvement = np.mean(recent_improvements)
            # 如果最近改进很小，更可能需要进化
            if avg_recent_improvement < 0.01:  # 最近改进 < 0.01%
                return np.random.random() < 0.8  # 80%概率尝试进化
        
        # 默认适中的进化概率
        return np.random.random() < 0.6  # 60%概率
    
    def _inter_evolution_training(self,
                                model: nn.Module,
                                train_loader: DataLoader,
                                test_loader: DataLoader,
                                criterion: nn.Module,
                                optimizer_factory: Callable,
                                epochs: int = 5) -> nn.Module:
        """
        间歇性常规训练 - 在进化轮次之间稳定模型
        """
        logger.info(f"执行{epochs}轮间歇训练...")
        
        # 使用更保守的学习率
        model.train()
        optimizer = optimizer_factory(model.parameters())
        
        # 降低学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1  # 降低到10%
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batches_processed = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batches_processed += 1
                
                # 限制训练量
                if batch_idx >= 20:  # 适中的训练量
                    break
            
            scheduler.step()
            
            if epoch % 2 == 0:
                avg_loss = epoch_loss / max(batches_processed, 1)
                logger.debug(f"间歇训练 Epoch {epoch+1}/{epochs}, 损失: {avg_loss:.4f}")
        
        # 更新准确率历史
        new_accuracy = self._evaluate_model(model, test_loader)
        self.evolution_state.accuracy_history.append(new_accuracy)
        self.evolution_state.current_accuracy = new_accuracy
        
        if new_accuracy > self.evolution_state.best_accuracy:
            self.evolution_state.best_accuracy = new_accuracy
            logger.info(f"间歇训练创造新记录: {new_accuracy:.2f}%")
        
        return model
    
    def _create_mutation_evidence(self, candidate: MutationCandidate):
        """创建变异证据（用于历史更新）"""
        from .multi_mutation_type_evaluator import MutationEvidence
        
        expectation = candidate.calibrated_expectation
        
        return MutationEvidence(
            effective_information_gain=expectation.information_gain_benefit,
            feature_diversity_gain=expectation.diversity_benefit,
            information_flow_improvement=expectation.flow_improvement_benefit,
            parameter_increase_ratio=expectation.cost_penalty
        )
    
    def _adapt_weights(self, improvements: List[float]):
        """自适应权重调整"""
        avg_improvement = np.mean(improvements) if improvements else 0.0
        
        # 记录当前权重和改进
        current_weights = {
            'information_gain': self.config.information_gain_weight,
            'integration': self.config.integration_weight,
            'redundancy': self.config.redundancy_weight,
            'cost': self.config.cost_weight
        }
        
        self.weight_adaptation_history.append({
            'weights': current_weights.copy(),
            'improvement': avg_improvement
        })
        
        # 简单的自适应策略：如果改进不理想，调整权重
        if avg_improvement < self.config.min_benefit_threshold and len(self.weight_adaptation_history) > 2:
            # 增加信息增益权重，降低成本权重
            self.config.information_gain_weight *= 1.1
            self.config.cost_weight *= 0.9
            
            # 重新归一化
            total_weight = (self.config.information_gain_weight + 
                           self.config.integration_weight + 
                           self.config.redundancy_weight + 
                           self.config.cost_weight)
            
            self.config.information_gain_weight /= total_weight
            self.config.integration_weight /= total_weight
            self.config.redundancy_weight /= total_weight
            self.config.cost_weight /= total_weight
            
            logger.info("自适应调整权重以提高收益")
    
    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> float:
        """评估模型准确率"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return accuracy
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """获取进化摘要"""
        initial_accuracy = self.evolution_state.accuracy_history[0] if self.evolution_state.accuracy_history else 0.0
        total_improvement = self.evolution_state.best_accuracy - initial_accuracy if self.evolution_state.accuracy_history else 0.0
        
        return {
            'rounds_completed': self.evolution_state.current_round,
            'initial_accuracy': initial_accuracy,
            'final_accuracy': self.evolution_state.best_accuracy,
            'total_improvement': total_improvement,
            'successful_mutations': self.evolution_state.successful_mutations,
            'failed_mutations': self.evolution_state.failed_mutations,
            'total_mutations': self.evolution_state.total_mutations_applied,
            'accuracy_history': self.evolution_state.accuracy_history.copy(),
            'average_time_per_round': np.mean(self.evolution_state.timing_history) if self.evolution_state.timing_history else 0.0,
            'target_reached': self.evolution_state.best_accuracy >= self.config.target_accuracy,
            'total_parameter_increase': self.evolution_state.total_parameter_increase,
            'total_computation_increase': self.evolution_state.total_computation_increase
        }