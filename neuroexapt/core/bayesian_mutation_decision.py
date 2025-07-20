"""
贝叶斯变异决策框架
基于互信息、不确定性和期望效用理论的变异价值评估

核心理论：
1. 将变异收益建模为随机变量 ΔI
2. 结合互信息增益、不确定性降低和迁移成本
3. 使用贝叶斯推断估计收益分布
4. 通过期望效用最大化做出决策

数学框架：
ΔI = α·ΔI_MI + β·ΔI_cond + γ·ΔI_uncert - δ·ΔI_cost
其中：
- ΔI_MI: 互信息增益
- ΔI_cond: 条件互信息增益  
- ΔI_uncert: 不确定性降低
- ΔI_cost: 迁移成本
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class MutationPrior:
    """变异先验分布参数"""
    mean_gain: float = 0.02  # 期望增益
    std_gain: float = 0.05   # 增益标准差
    failure_prob: float = 0.3  # 失败概率
    cost_weight: float = 0.1   # 成本权重


@dataclass
class MutationEvidence:
    """变异的观测证据"""
    mutual_info_gain: float = 0.0      # 互信息增益
    cond_mutual_info_gain: float = 0.0  # 条件互信息增益
    uncertainty_reduction: float = 0.0  # 不确定性降低
    transfer_cost: float = 0.0          # 迁移成本
    bottleneck_severity: float = 0.0    # 瓶颈严重程度


@dataclass
class MutationDecision:
    """变异决策结果"""
    should_mutate: bool                 # 是否应该变异
    expected_utility: float             # 期望效用
    expected_gain: float                # 期望收益
    risk_probability: float             # 风险概率
    confidence: float                   # 决策置信度
    reasoning: str                      # 决策理由


@dataclass
class BayesianDecisionConfig:
    """贝叶斯决策配置参数"""
    # 权重系数
    alpha: float = 1.0          # 互信息权重
    beta: float = 0.8           # 条件互信息权重
    gamma: float = 0.6          # 不确定性权重
    delta: float = 0.4          # 成本权重
    
    # 风险参数
    risk_aversion: float = 2.0  # 风险厌恶系数λ
    
    # 决策阈值
    utility_threshold: float = 0.0      # 效用阈值
    confidence_threshold: float = 0.3   # 置信度阈值
    
    # 先验更新参数
    max_history_size: int = 100         # 最大历史记录数
    history_weight: float = 0.7         # 历史权重在先验更新中的比例
    
    # 数值稳定性参数
    min_std: float = 0.01               # 最小标准差
    min_precision: float = 1e-8         # 最小精度
    
    # 证据可靠性参数
    mi_reliability_scale: float = 0.1           # 互信息可靠性缩放
    uncertainty_reliability_scale: float = 0.05 # 不确定性可靠性缩放


class BayesianMutationDecision:
    """
    贝叶斯变异决策引擎
    
    基于你提出的理论框架：
    1. 建模变异收益的分布
    2. 结合多种信息判据
    3. 计算期望效用
    4. 做出最优决策
    """
    
    def __init__(self, config: BayesianDecisionConfig = None):
        """
        初始化贝叶斯决策引擎
        
        Args:
            config: 配置对象，如果None则使用默认配置
        """
        self.config = config or BayesianDecisionConfig()
        
        # 历史变异经验（用于更新先验）
        self.mutation_history = []
        
    def compute_mutation_gain(self, evidence: MutationEvidence) -> float:
        """
        计算变异的综合收益
        
        Args:
            evidence: 变异证据
            
        Returns:
            综合收益值
        """
        gain = (self.config.alpha * evidence.mutual_info_gain +
                self.config.beta * evidence.cond_mutual_info_gain +
                self.config.gamma * evidence.uncertainty_reduction -
                self.config.delta * evidence.transfer_cost)
                
        # 加入瓶颈严重程度的调制
        gain *= (1.0 + evidence.bottleneck_severity)
        
        return gain
        
    def update_prior_from_history(self, prior: MutationPrior) -> MutationPrior:
        """
        基于历史经验更新先验分布
        
        Args:
            prior: 初始先验
            
        Returns:
            更新后的先验
        """
        if not self.mutation_history:
            return prior
            
        # 计算历史变异的平均收益和标准差
        historical_gains = [item['actual_gain'] for item in self.mutation_history]
        historical_successes = [item['success'] for item in self.mutation_history]
        
        if historical_gains:
            # 贝叶斯更新：结合先验和历史数据
            prior_precision = 1.0 / (prior.std_gain ** 2 + self.config.min_precision)
            data_variance = np.var(historical_gains) + self.config.min_precision
            data_precision = len(historical_gains) / data_variance
            
            # 更新均值（精度加权平均）
            total_precision = prior_precision + data_precision * self.config.history_weight
            updated_mean = (prior_precision * prior.mean_gain + 
                          data_precision * self.config.history_weight * np.mean(historical_gains)) / total_precision
            
            # 更新标准差
            updated_std = np.sqrt(1.0 / total_precision)
            updated_std = max(updated_std, self.config.min_std)  # 避免过度自信
            
            # 更新失败概率
            updated_failure_prob = 1.0 - np.mean(historical_successes)
            updated_failure_prob = max(updated_failure_prob, 0.1)  # 保持最小不确定性
            
            return MutationPrior(
                mean_gain=updated_mean,
                std_gain=updated_std,
                failure_prob=updated_failure_prob,
                cost_weight=prior.cost_weight
            )
            
        return prior
        
    def compute_posterior_distribution(self,
                                     evidence: MutationEvidence,
                                     prior: MutationPrior) -> Tuple[float, float]:
        """
        计算变异收益的后验分布
        
        Args:
            evidence: 观测证据
            prior: 先验分布
            
        Returns:
            (后验均值, 后验标准差)
        """
        # 计算基于证据的收益估计
        evidence_gain = self.compute_mutation_gain(evidence)
        
        # 证据的可靠性（基于互信息的置信度）
        evidence_reliability = min(1.0, 
                                 evidence.mutual_info_gain / self.config.mi_reliability_scale + 
                                 evidence.uncertainty_reduction / self.config.uncertainty_reliability_scale)
        
        # 贝叶斯更新：结合先验和证据
        prior_precision = 1.0 / (prior.std_gain ** 2 + self.config.min_precision)
        evidence_precision = evidence_reliability / (self.config.min_std ** 2)  # 证据精度
        
        # 后验均值
        total_precision = prior_precision + evidence_precision
        posterior_mean = (prior_precision * prior.mean_gain + 
                         evidence_precision * evidence_gain) / total_precision
        
        # 后验标准差
        posterior_std = np.sqrt(1.0 / total_precision)
        posterior_std = max(posterior_std, self.config.min_std)
        
        return posterior_mean, posterior_std
        
    def compute_expected_utility(self,
                               posterior_mean: float,
                               posterior_std: float) -> float:
        """
        计算期望效用（考虑风险厌恶）
        
        Args:
            posterior_mean: 后验均值
            posterior_std: 后验标准差
            
        Returns:
            期望效用值
        """
        # 使用指数效用函数：U(x) = 1 - exp(-λx)
        # 对于正态分布，期望效用有解析解
        
        # 效用函数的期望：E[U(X)] ≈ U(μ) + 0.5 * U''(μ) * σ²
        # 其中 U'(x) = λ*exp(-λx), U''(x) = -λ²*exp(-λx)
        
        lambda_val = self.config.risk_aversion
        
        # 数值稳定性：限制指数函数的输入范围
        exp_input = -lambda_val * posterior_mean
        exp_input = np.clip(exp_input, -50, 50)  # 避免数值溢出
        
        utility_mean = 1.0 - np.exp(exp_input)
        utility_curvature = -lambda_val**2 * np.exp(exp_input)
        
        expected_utility = utility_mean + 0.5 * utility_curvature * (posterior_std**2)
        
        return float(expected_utility)
        
    def compute_risk_probability(self,
                               posterior_mean: float,
                               posterior_std: float) -> float:
        """
        计算变异失败的概率
        
        Args:
            posterior_mean: 后验均值
            posterior_std: 后验标准差
            
        Returns:
            失败概率 P(ΔI < 0)
        """
        if posterior_std <= self.config.min_precision:
            return 0.0 if posterior_mean > 0 else 1.0
            
        # 计算 P(ΔI < 0) = Φ(-μ/σ)
        standardized = -posterior_mean / posterior_std
        failure_prob = stats.norm.cdf(standardized)
        
        return float(np.clip(failure_prob, 0.0, 1.0))
        
    def compute_decision_confidence(self,
                                  expected_utility: float,
                                  posterior_std: float) -> float:
        """
        计算决策置信度
        
        Args:
            expected_utility: 期望效用
            posterior_std: 后验标准差
            
        Returns:
            置信度 [0, 1]
        """
        # 置信度与期望效用和不确定性相关
        # 高期望效用 + 低不确定性 = 高置信度
        
        utility_component = np.tanh(abs(expected_utility) * 10)  # [0, 1]
        uncertainty_component = np.exp(-posterior_std * 20)       # [0, 1]
        
        confidence = 0.6 * utility_component + 0.4 * uncertainty_component
        return float(np.clip(confidence, 0.0, 1.0))
        
    def make_decision(self,
                     evidence: MutationEvidence,
                     prior: MutationPrior = None,
                     utility_threshold: float = None) -> MutationDecision:
        """
        基于贝叶斯框架做出变异决策
        
        Args:
            evidence: 变异证据
            prior: 先验分布，如果None则使用默认
            utility_threshold: 效用阈值，如果None则使用配置值
            
        Returns:
            变异决策结果
        """
        if prior is None:
            prior = MutationPrior()
            
        if utility_threshold is None:
            utility_threshold = self.config.utility_threshold
            
        # 基于历史更新先验
        updated_prior = self.update_prior_from_history(prior)
        
        # 计算后验分布
        posterior_mean, posterior_std = self.compute_posterior_distribution(
            evidence, updated_prior
        )
        
        # 计算期望效用
        expected_utility = self.compute_expected_utility(posterior_mean, posterior_std)
        
        # 计算风险概率
        risk_probability = self.compute_risk_probability(posterior_mean, posterior_std)
        
        # 计算置信度
        confidence = self.compute_decision_confidence(expected_utility, posterior_std)
        
        # 做出决策
        should_mutate = (expected_utility > utility_threshold and 
                        confidence > self.config.confidence_threshold)
        
        # 生成决策理由
        reasoning = self._generate_reasoning(
            evidence, posterior_mean, posterior_std, expected_utility, 
            risk_probability, confidence, should_mutate
        )
        
        decision = MutationDecision(
            should_mutate=should_mutate,
            expected_utility=expected_utility,
            expected_gain=posterior_mean,
            risk_probability=risk_probability,
            confidence=confidence,
            reasoning=reasoning
        )
        
        logger.info(f"变异决策: {decision.should_mutate}, 期望效用: {expected_utility:.4f}, "
                   f"期望收益: {posterior_mean:.4f}, 风险: {risk_probability:.4f}")
        
        return decision
        
    def _generate_reasoning(self,
                          evidence: MutationEvidence,
                          posterior_mean: float,
                          posterior_std: float,
                          expected_utility: float,
                          risk_probability: float,
                          confidence: float,
                          should_mutate: bool) -> str:
        """生成决策理由"""
        
        reasons = []
        
        # 分析证据
        if evidence.mutual_info_gain > 0.05:
            reasons.append(f"强互信息增益({evidence.mutual_info_gain:.3f})")
        elif evidence.mutual_info_gain > 0.01:
            reasons.append(f"中等互信息增益({evidence.mutual_info_gain:.3f})")
        else:
            reasons.append(f"微弱互信息增益({evidence.mutual_info_gain:.3f})")
            
        if evidence.uncertainty_reduction > 0.02:
            reasons.append(f"显著不确定性降低({evidence.uncertainty_reduction:.3f})")
        elif evidence.uncertainty_reduction > 0.005:
            reasons.append(f"适度不确定性降低({evidence.uncertainty_reduction:.3f})")
            
        if evidence.transfer_cost > 0.1:
            reasons.append(f"高迁移成本({evidence.transfer_cost:.3f})")
        elif evidence.transfer_cost > 0.05:
            reasons.append(f"中等迁移成本({evidence.transfer_cost:.3f})")
            
        # 分析后验
        if posterior_std < 0.01:
            reasons.append("高确定性预测")
        elif posterior_std > 0.05:
            reasons.append("高不确定性预测")
            
        # 分析风险
        if risk_probability < 0.1:
            reasons.append("低风险")
        elif risk_probability > 0.5:
            reasons.append("高风险")
            
        decision_reason = "推荐变异" if should_mutate else "不推荐变异"
        
        return f"{decision_reason}: {', '.join(reasons)}. 置信度: {confidence:.2f}"
        
    def record_mutation_outcome(self,
                               evidence: MutationEvidence,
                               actual_gain: float,
                               success: bool):
        """
        记录变异结果，用于更新先验
        
        Args:
            evidence: 原始证据
            actual_gain: 实际收益
            success: 是否成功
        """
        self.mutation_history.append({
            'evidence': evidence,
            'predicted_gain': self.compute_mutation_gain(evidence),
            'actual_gain': actual_gain,
            'success': success,
            'timestamp': len(self.mutation_history)
        })
        
        # 保持历史记录在合理范围内
        if len(self.mutation_history) > self.config.max_history_size:
            # 保留最近的一半记录
            keep_size = self.config.max_history_size // 2
            self.mutation_history = self.mutation_history[-keep_size:]
            
    def adjust_risk_aversion(self, training_stage: str):
        """
        根据训练阶段调整风险厌恶系数
        
        Args:
            training_stage: 'early', 'middle', 'late'
        """
        if training_stage == 'early':
            self.config.risk_aversion = 1.0  # 更激进
        elif training_stage == 'middle':
            self.config.risk_aversion = 2.0  # 中等
        else:  # late
            self.config.risk_aversion = 3.0  # 更保守
            
    def get_statistics(self) -> Dict:
        """获取决策统计信息"""
        if not self.mutation_history:
            return {}
            
        gains = [h['actual_gain'] for h in self.mutation_history]
        successes = [h['success'] for h in self.mutation_history]
        
        return {
            'total_mutations': len(self.mutation_history),
            'success_rate': np.mean(successes),
            'average_gain': np.mean(gains),
            'gain_std': np.std(gains),
            'last_10_success_rate': np.mean(successes[-10:]) if len(successes) >= 10 else None
        }