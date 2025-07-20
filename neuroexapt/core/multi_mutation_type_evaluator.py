"""
多变异类型收益期望建模器
Multi-Mutation Type Benefit Expectation Modeler

针对不同变异类型的收益期望建模，支持：
1. 串行分裂（Serial Split）- 新增层或多层
2. 并行分裂（Parallel Split）- 异型网络层并行
3. 展宽通道（Channel Widening）- 增加卷积核数量
4. 替换层类型（Layer Type Replacement）- 改变层的类型

基于用户提供的理论框架实现贝叶斯收益期望建模：
p(P|S) = p(S|P) * p(P) / p(S)

作者：基于用户提供的理论框架实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

from .parameter_free_structural_evaluator import ParameterFreeStructuralEvaluator, StructuralMetrics

logger = logging.getLogger(__name__)


class MutationType(Enum):
    """变异类型枚举"""
    SERIAL_SPLIT = "serial_split"           # 串行分裂（新增层）
    PARALLEL_SPLIT = "parallel_split"       # 并行分裂（异型层）
    CHANNEL_WIDENING = "channel_widening"   # 展宽通道
    LAYER_REPLACEMENT = "layer_replacement" # 替换层类型


@dataclass
class MutationConfig:
    """变异配置"""
    mutation_type: MutationType
    target_layer_name: str
    target_layer: nn.Module
    
    # 串行分裂参数
    new_layer_type: Optional[type] = None
    new_layer_count: int = 1
    
    # 并行分裂参数
    parallel_layer_types: Optional[List[type]] = None
    
    # 通道展宽参数
    widening_factor: float = 1.5
    
    # 层替换参数
    replacement_layer_type: Optional[type] = None
    replacement_layer_params: Optional[Dict[str, Any]] = None


@dataclass
class MutationBenefitPrior:
    """变异收益先验分布"""
    # 基础先验参数
    mean_benefit: float = 0.02          # 期望收益（准确率提升）
    std_benefit: float = 0.05           # 收益标准差
    failure_prob: float = 0.3           # 失败概率
    
    # 类型特异性先验
    serial_split_mean: float = 0.025    # 串行分裂期望收益
    parallel_split_mean: float = 0.03   # 并行分裂期望收益
    channel_widening_mean: float = 0.015 # 通道展宽期望收益
    layer_replacement_mean: float = 0.02 # 层替换期望收益
    
    # 成本权重
    computation_cost_weight: float = 0.1
    parameter_cost_weight: float = 0.05
    training_cost_weight: float = 0.08


@dataclass
class MutationEvidence:
    """变异的观测证据"""
    # 结构指标
    effective_information_gain: float = 0.0
    integrated_information_gain: float = 0.0
    redundancy_reduction: float = 0.0
    feature_diversity_gain: float = 0.0
    
    # 信息流指标
    information_flow_improvement: float = 0.0
    gradient_flow_improvement: float = 0.0
    
    # 成本指标
    parameter_increase_ratio: float = 0.0
    computation_increase_ratio: float = 0.0
    
    # 瓶颈指标
    bottleneck_severity: float = 0.0
    bottleneck_resolution_potential: float = 0.0


@dataclass
class MutationBenefitExpectation:
    """变异收益期望"""
    expected_benefit: float = 0.0        # 期望收益
    benefit_variance: float = 0.0        # 收益方差
    success_probability: float = 0.0     # 成功概率
    risk_adjusted_benefit: float = 0.0   # 风险调整收益
    confidence_interval: Tuple[float, float] = (0.0, 0.0)  # 置信区间
    
    # 分解收益
    information_gain_benefit: float = 0.0
    diversity_benefit: float = 0.0
    flow_improvement_benefit: float = 0.0
    cost_penalty: float = 0.0


class MultiMutationTypeEvaluator:
    """
    多变异类型收益期望评估器
    
    核心功能：
    1. 针对不同变异类型计算收益期望
    2. 基于贝叶斯推断建模收益分布
    3. 考虑变异成本和风险
    4. 提供多变异类型的统一评估框架
    """
    
    def __init__(self, 
                 structural_evaluator: ParameterFreeStructuralEvaluator,
                 prior: MutationBenefitPrior = None,
                 device: torch.device = None):
        """
        初始化评估器
        
        Args:
            structural_evaluator: 结构评估器
            prior: 收益先验分布
            device: 计算设备
        """
        self.structural_evaluator = structural_evaluator
        self.prior = prior or MutationBenefitPrior()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 历史变异数据（用于更新先验）
        self.mutation_history: List[Tuple[MutationConfig, MutationEvidence, float]] = []
        
        logger.info("初始化多变异类型收益期望评估器")
    
    def evaluate_mutation_benefit(self, 
                                mutation_config: MutationConfig,
                                current_model: nn.Module) -> MutationBenefitExpectation:
        """
        评估变异的收益期望
        
        Args:
            mutation_config: 变异配置
            current_model: 当前模型
            
        Returns:
            MutationBenefitExpectation: 收益期望评估结果
        """
        # 1. 收集观测证据
        evidence = self._collect_mutation_evidence(mutation_config, current_model)
        
        # 2. 根据变异类型计算收益期望
        benefit_expectation = self._compute_benefit_expectation(
            mutation_config.mutation_type, evidence
        )
        
        # 3. 应用贝叶斯更新（如果有历史数据）
        if self.mutation_history:
            benefit_expectation = self._bayesian_update(
                benefit_expectation, mutation_config.mutation_type, evidence
            )
        
        logger.debug(f"变异 {mutation_config.mutation_type.value} 期望收益: "
                    f"{benefit_expectation.expected_benefit:.4f}")
        
        return benefit_expectation
    
    def _collect_mutation_evidence(self, 
                                 mutation_config: MutationConfig, 
                                 model: nn.Module) -> MutationEvidence:
        """
        收集变异的观测证据
        """
        evidence = MutationEvidence()
        
        # 评估目标层的当前结构指标
        target_layer = mutation_config.target_layer
        current_metrics = self.structural_evaluator.evaluate_layer_structure(target_layer)
        
        # 根据变异类型计算不同的证据
        if mutation_config.mutation_type == MutationType.SERIAL_SPLIT:
            evidence = self._collect_serial_split_evidence(
                mutation_config, current_metrics, model
            )
        elif mutation_config.mutation_type == MutationType.PARALLEL_SPLIT:
            evidence = self._collect_parallel_split_evidence(
                mutation_config, current_metrics, model
            )
        elif mutation_config.mutation_type == MutationType.CHANNEL_WIDENING:
            evidence = self._collect_channel_widening_evidence(
                mutation_config, current_metrics, model
            )
        elif mutation_config.mutation_type == MutationType.LAYER_REPLACEMENT:
            evidence = self._collect_layer_replacement_evidence(
                mutation_config, current_metrics, model
            )
        
        return evidence
    
    def _collect_serial_split_evidence(self,
                                     config: MutationConfig,
                                     current_metrics: StructuralMetrics,
                                     model: nn.Module) -> MutationEvidence:
        """收集串行分裂的证据"""
        evidence = MutationEvidence()
        
        # 计算瓶颈严重程度
        evidence.bottleneck_severity = self._compute_bottleneck_severity(
            config.target_layer, model
        )
        
        # 信息流改进潜力（基于当前层的信息流效率）
        evidence.information_flow_improvement = max(
            0.0, 1.0 - current_metrics.information_flow_efficiency
        ) * 0.8  # 串行分裂对信息流的改进潜力
        
        # 冗余度降低（如果当前层冗余度高，新增层可能有帮助）
        evidence.redundancy_reduction = current_metrics.structural_redundancy * 0.6
        
        # 梯度流改进
        evidence.gradient_flow_improvement = max(
            0.0, 1.0 - current_metrics.gradient_flow_health
        ) * 0.7
        
        # 计算成本
        evidence.parameter_increase_ratio = self._estimate_parameter_increase_serial(
            config.target_layer, config.new_layer_count
        )
        evidence.computation_increase_ratio = evidence.parameter_increase_ratio * 1.2
        
        # 瓶颈解决潜力
        evidence.bottleneck_resolution_potential = (
            evidence.bottleneck_severity * evidence.information_flow_improvement
        )
        
        return evidence
    
    def _collect_parallel_split_evidence(self,
                                       config: MutationConfig,
                                       current_metrics: StructuralMetrics,
                                       model: nn.Module) -> MutationEvidence:
        """收集并行分裂的证据"""
        evidence = MutationEvidence()
        
        # 特征多样性增益（并行分裂的主要优势）
        evidence.feature_diversity_gain = max(
            0.0, 1.0 - current_metrics.feature_diversity
        ) * 0.9
        
        # 有效信息增益（不同类型层的互补性）
        if config.parallel_layer_types:
            # 估计异型层的互补性
            complementarity = self._estimate_layer_complementarity(
                config.target_layer, config.parallel_layer_types
            )
            evidence.effective_information_gain = complementarity * 0.8
        
        # 积分信息增益
        evidence.integrated_information_gain = (
            evidence.feature_diversity_gain * current_metrics.integrated_information
        )
        
        # 计算成本（并行分裂通常成本较高）
        evidence.parameter_increase_ratio = self._estimate_parameter_increase_parallel(
            config.target_layer, len(config.parallel_layer_types or [1])
        )
        evidence.computation_increase_ratio = evidence.parameter_increase_ratio * 1.5
        
        return evidence
    
    def _collect_channel_widening_evidence(self,
                                         config: MutationConfig,
                                         current_metrics: StructuralMetrics,
                                         model: nn.Module) -> MutationEvidence:
        """收集通道展宽的证据"""
        evidence = MutationEvidence()
        
        if isinstance(config.target_layer, nn.Conv2d):
            # 当前通道相关性（高相关性意味着展宽收益大）
            channel_redundancy = current_metrics.structural_redundancy
            
            # 特征多样性增益
            evidence.feature_diversity_gain = channel_redundancy * 0.7
            
            # 有效信息增益（更多通道可以捕捉更多特征）
            evidence.effective_information_gain = (
                channel_redundancy * (config.widening_factor - 1.0) * 0.5
            )
            
            # 成本计算
            current_out_channels = config.target_layer.out_channels
            new_out_channels = int(current_out_channels * config.widening_factor)
            param_ratio = new_out_channels / current_out_channels
            
            evidence.parameter_increase_ratio = param_ratio - 1.0
            evidence.computation_increase_ratio = param_ratio - 1.0
        
        return evidence
    
    def _collect_layer_replacement_evidence(self,
                                          config: MutationConfig,
                                          current_metrics: StructuralMetrics,
                                          model: nn.Module) -> MutationEvidence:
        """收集层替换的证据"""
        evidence = MutationEvidence()
        
        if config.replacement_layer_type:
            # 估计新层类型的改进潜力
            current_nonlinearity = current_metrics.nonlinearity_capacity
            current_gradient_health = current_metrics.gradient_flow_health
            
            # 根据替换层类型估计改进
            new_nonlinearity = self._estimate_layer_nonlinearity(
                config.replacement_layer_type
            )
            new_gradient_health = self._estimate_layer_gradient_health(
                config.replacement_layer_type
            )
            
            # 改进计算
            evidence.effective_information_gain = max(
                0.0, new_nonlinearity - current_nonlinearity
            ) * 0.6
            
            evidence.gradient_flow_improvement = max(
                0.0, new_gradient_health - current_gradient_health
            ) * 0.8
            
            # 替换通常参数数量变化不大
            evidence.parameter_increase_ratio = 0.1  # 小幅增加
            evidence.computation_increase_ratio = 0.05
        
        return evidence
    
    def _compute_benefit_expectation(self,
                                   mutation_type: MutationType,
                                   evidence: MutationEvidence) -> MutationBenefitExpectation:
        """
        基于证据计算收益期望
        """
        expectation = MutationBenefitExpectation()
        
        # 获取类型特异性先验
        type_prior_mean = self._get_type_specific_prior_mean(mutation_type)
        
        # 计算各项收益
        expectation.information_gain_benefit = (
            evidence.effective_information_gain * 0.4 +
            evidence.integrated_information_gain * 0.3 +
            evidence.redundancy_reduction * 0.3
        ) * 0.05  # 缩放到准确率改进范围
        
        expectation.diversity_benefit = evidence.feature_diversity_gain * 0.03
        
        expectation.flow_improvement_benefit = (
            evidence.information_flow_improvement * 0.5 +
            evidence.gradient_flow_improvement * 0.5
        ) * 0.025
        
        # 计算成本惩罚
        expectation.cost_penalty = (
            evidence.parameter_increase_ratio * self.prior.parameter_cost_weight +
            evidence.computation_increase_ratio * self.prior.computation_cost_weight
        )
        
        # 总期望收益
        expectation.expected_benefit = (
            type_prior_mean +
            expectation.information_gain_benefit +
            expectation.diversity_benefit +
            expectation.flow_improvement_benefit -
            expectation.cost_penalty
        )
        
        # 计算方差（基于证据的不确定性）
        uncertainty_factors = [
            abs(evidence.effective_information_gain - 0.5),
            abs(evidence.feature_diversity_gain - 0.5),
            evidence.parameter_increase_ratio * 0.5
        ]
        expectation.benefit_variance = (
            self.prior.std_benefit ** 2 + 
            np.mean(uncertainty_factors) * 0.01
        )
        
        # 成功概率（基于证据强度）
        evidence_strength = (
            expectation.information_gain_benefit +
            expectation.diversity_benefit +
            expectation.flow_improvement_benefit
        ) / 0.1  # 归一化
        
        expectation.success_probability = max(
            0.1, min(0.9, (1.0 - self.prior.failure_prob) * evidence_strength)
        )
        
        # 风险调整收益
        expectation.risk_adjusted_benefit = (
            expectation.expected_benefit * expectation.success_probability
        )
        
        # 置信区间（假设正态分布）
        std_dev = np.sqrt(expectation.benefit_variance)
        expectation.confidence_interval = (
            expectation.expected_benefit - 1.96 * std_dev,
            expectation.expected_benefit + 1.96 * std_dev
        )
        
        return expectation
    
    def _bayesian_update(self,
                        current_expectation: MutationBenefitExpectation,
                        mutation_type: MutationType,
                        evidence: MutationEvidence) -> MutationBenefitExpectation:
        """
        使用历史数据进行贝叶斯更新
        """
        # 获取同类型的历史数据
        same_type_history = [
            (config, hist_evidence, outcome) for config, hist_evidence, outcome in self.mutation_history
            if config.mutation_type == mutation_type
        ]
        
        if not same_type_history:
            return current_expectation
        
        # 计算历史平均收益和方差
        historical_outcomes = [outcome for _, _, outcome in same_type_history]
        hist_mean = np.mean(historical_outcomes)
        hist_var = np.var(historical_outcomes) if len(historical_outcomes) > 1 else self.prior.std_benefit ** 2
        
        # 贝叶斯更新（正态-正态共轭）
        prior_precision = 1.0 / self.prior.std_benefit ** 2
        likelihood_precision = 1.0 / hist_var if hist_var > 0 else prior_precision
        
        posterior_precision = prior_precision + likelihood_precision
        posterior_mean = (
            prior_precision * current_expectation.expected_benefit +
            likelihood_precision * hist_mean
        ) / posterior_precision
        
        posterior_var = 1.0 / posterior_precision
        
        # 更新期望
        updated_expectation = MutationBenefitExpectation(
            expected_benefit=posterior_mean,
            benefit_variance=posterior_var,
            success_probability=current_expectation.success_probability,
            risk_adjusted_benefit=posterior_mean * current_expectation.success_probability,
            confidence_interval=(
                posterior_mean - 1.96 * np.sqrt(posterior_var),
                posterior_mean + 1.96 * np.sqrt(posterior_var)
            ),
            information_gain_benefit=current_expectation.information_gain_benefit,
            diversity_benefit=current_expectation.diversity_benefit,
            flow_improvement_benefit=current_expectation.flow_improvement_benefit,
            cost_penalty=current_expectation.cost_penalty
        )
        
        return updated_expectation
    
    def update_mutation_history(self,
                              mutation_config: MutationConfig,
                              evidence: MutationEvidence,
                              actual_outcome: float):
        """
        更新变异历史记录
        
        Args:
            mutation_config: 变异配置
            evidence: 观测证据
            actual_outcome: 实际结果（准确率改进）
        """
        self.mutation_history.append((mutation_config, evidence, actual_outcome))
        
        # 限制历史记录长度
        if len(self.mutation_history) > 100:
            self.mutation_history = self.mutation_history[-100:]
        
        logger.info(f"更新变异历史：类型={mutation_config.mutation_type.value}, "
                   f"结果={actual_outcome:.4f}")
    
    def get_best_mutation_candidates(self,
                                   model: nn.Module,
                                   top_k: int = 5) -> List[Tuple[MutationConfig, MutationBenefitExpectation]]:
        """
        获取最佳变异候选
        
        Args:
            model: 当前模型
            top_k: 返回前k个候选
            
        Returns:
            List: (变异配置, 收益期望) 的列表，按风险调整收益排序
        """
        candidates = []
        
        # 遍历所有层，生成候选变异
        for layer_name, layer in model.named_modules():
            if len(list(layer.children())) == 0:  # 叶子节点
                # 为每种变异类型生成候选
                layer_candidates = self._generate_layer_mutation_candidates(
                    layer_name, layer
                )
                
                for config in layer_candidates:
                    try:
                        expectation = self.evaluate_mutation_benefit(config, model)
                        candidates.append((config, expectation))
                    except Exception as e:
                        logger.warning(f"评估变异候选失败: {e}")
        
        # 按风险调整收益排序
        candidates.sort(
            key=lambda x: x[1].risk_adjusted_benefit, 
            reverse=True
        )
        
        return candidates[:top_k]
    
    def _generate_layer_mutation_candidates(self,
                                          layer_name: str,
                                          layer: nn.Module) -> List[MutationConfig]:
        """为指定层生成变异候选"""
        candidates = []
        
        # 串行分裂候选
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            candidates.append(MutationConfig(
                mutation_type=MutationType.SERIAL_SPLIT,
                target_layer_name=layer_name,
                target_layer=layer,
                new_layer_type=type(layer),
                new_layer_count=1
            ))
        
        # 并行分裂候选
        if isinstance(layer, nn.Conv2d):
            candidates.append(MutationConfig(
                mutation_type=MutationType.PARALLEL_SPLIT,
                target_layer_name=layer_name,
                target_layer=layer,
                parallel_layer_types=[nn.Conv2d, nn.MaxPool2d]
            ))
        
        # 通道展宽候选
        if isinstance(layer, nn.Conv2d):
            candidates.append(MutationConfig(
                mutation_type=MutationType.CHANNEL_WIDENING,
                target_layer_name=layer_name,
                target_layer=layer,
                widening_factor=1.5
            ))
        
        # 层替换候选
        if isinstance(layer, nn.ReLU):
            candidates.append(MutationConfig(
                mutation_type=MutationType.LAYER_REPLACEMENT,
                target_layer_name=layer_name,
                target_layer=layer,
                replacement_layer_type=nn.GELU
            ))
        
        return candidates
    
    def _get_type_specific_prior_mean(self, mutation_type: MutationType) -> float:
        """获取类型特异性先验均值"""
        type_means = {
            MutationType.SERIAL_SPLIT: self.prior.serial_split_mean,
            MutationType.PARALLEL_SPLIT: self.prior.parallel_split_mean,
            MutationType.CHANNEL_WIDENING: self.prior.channel_widening_mean,
            MutationType.LAYER_REPLACEMENT: self.prior.layer_replacement_mean
        }
        return type_means.get(mutation_type, self.prior.mean_benefit)
    
    # 辅助方法
    def _compute_bottleneck_severity(self, layer: nn.Module, model: nn.Module) -> float:
        """计算瓶颈严重程度"""
        # 简化实现：基于层的相对大小
        try:
            total_params = sum(p.numel() for p in model.parameters())
            layer_params = sum(p.numel() for p in layer.parameters())
            relative_size = layer_params / max(total_params, 1)
            
            # 相对较小的层可能是瓶颈
            return max(0.0, 1.0 - relative_size * 10)
        except:
            return 0.5
    
    def _estimate_parameter_increase_serial(self, layer: nn.Module, new_layer_count: int) -> float:
        """估计串行分裂的参数增长比例"""
        try:
            current_params = sum(p.numel() for p in layer.parameters())
            # 假设新层大小与当前层相似
            estimated_new_params = current_params * new_layer_count * 0.8
            return estimated_new_params / max(current_params, 1)
        except:
            return 0.5
    
    def _estimate_parameter_increase_parallel(self, layer: nn.Module, parallel_count: int) -> float:
        """估计并行分裂的参数增长比例"""
        try:
            current_params = sum(p.numel() for p in layer.parameters())
            # 并行层通常较小
            estimated_new_params = current_params * parallel_count * 0.6
            return estimated_new_params / max(current_params, 1)
        except:
            return 0.3
    
    def _estimate_layer_complementarity(self, current_layer: nn.Module, new_layer_types: List[type]) -> float:
        """估计异型层的互补性"""
        # 简化实现：不同类型的层互补性较高
        if any(layer_type != type(current_layer) for layer_type in new_layer_types):
            return 0.8
        else:
            return 0.3
    
    def _estimate_layer_nonlinearity(self, layer_type: type) -> float:
        """估计层类型的非线性能力"""
        nonlinearity_map = {
            nn.ReLU: 0.8,
            nn.GELU: 0.9,
            nn.Tanh: 0.85,
            nn.Sigmoid: 0.75,
            nn.Conv2d: 0.1,
            nn.Linear: 0.1,
            nn.BatchNorm2d: 0.3,
            nn.MaxPool2d: 0.6
        }
        return nonlinearity_map.get(layer_type, 0.2)
    
    def _estimate_layer_gradient_health(self, layer_type: type) -> float:
        """估计层类型的梯度流健康度"""
        gradient_health_map = {
            nn.ReLU: 0.7,
            nn.GELU: 0.8,
            nn.Tanh: 0.6,
            nn.Sigmoid: 0.5,
            nn.Conv2d: 0.8,
            nn.Linear: 0.8,
            nn.BatchNorm2d: 0.9,
            nn.MaxPool2d: 0.6
        }
        return gradient_health_map.get(layer_type, 0.5)