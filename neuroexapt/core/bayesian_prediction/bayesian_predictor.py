"""
贝叶斯变异收益预测器

基于贝叶斯推断、高斯过程回归和蒙特卡罗采样的变异收益预测
"""

from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import time
import logging
from .prior_knowledge import PriorKnowledgeBase

logger = logging.getLogger(__name__)


class BayesianMutationBenefitPredictor:
    """
    基于贝叶斯推断的变异收益预测器
    
    使用贝叶斯统计、高斯过程回归和蒙特卡罗采样来预测架构变异的期望收益
    """
    
    def __init__(self):
        self.prior_knowledge = PriorKnowledgeBase()
        self._comprehensive_generator = None
        
        self.gp_hyperparams = {
            'length_scale': 1.0,
            'variance': 1.0,
            'noise_variance': 0.01
        }
        self.mc_samples = 1000  # 蒙特卡罗采样数
        
        # 历史变异数据（用于更新先验）
        self.mutation_history = []
    
    @property
    def comprehensive_generator(self):
        """延迟加载综合策略生成器"""
        if self._comprehensive_generator is None:
            from ..mutation_strategies.comprehensive_strategy import ComprehensiveStrategyGenerator
            self._comprehensive_generator = ComprehensiveStrategyGenerator(self.prior_knowledge)
        return self._comprehensive_generator
        
    def predict_mutation_benefit(self, 
                               layer_analysis: Dict[str, Any],
                               mutation_strategy: str,
                               current_accuracy: float,
                               model_complexity: Dict[str, float]) -> Dict[str, Any]:
        """
        使用贝叶斯推断预测变异收益
        
        Args:
            layer_analysis: 层分析结果
            mutation_strategy: 变异策略类型
            current_accuracy: 当前准确率
            model_complexity: 模型复杂度指标
            
        Returns:
            包含期望收益、置信区间、风险评估的预测结果
        """
        logger.debug(f"开始贝叶斯变异收益预测: {mutation_strategy}")
        
        try:
            # 1. 构建特征向量
            feature_vector = self._extract_feature_vector(
                layer_analysis, current_accuracy, model_complexity
            )
            
            # 2. 贝叶斯后验推断
            posterior_params = self._bayesian_posterior_inference(
                feature_vector, mutation_strategy, layer_analysis
            )
            
            # 3. 高斯过程回归预测
            gp_prediction = self._gaussian_process_prediction(
                feature_vector, posterior_params
            )
            
            # 4. 蒙特卡罗期望估计
            mc_estimate = self._monte_carlo_benefit_estimation(
                gp_prediction, mutation_strategy, current_accuracy
            )
            
            # 5. 不确定性量化
            uncertainty_metrics = self._quantify_prediction_uncertainty(
                gp_prediction, mc_estimate, feature_vector
            )
            
            # 6. 风险调整收益
            risk_adjusted_benefit = self._calculate_risk_adjusted_benefit(
                mc_estimate, uncertainty_metrics, mutation_strategy
            )
            
            prediction_result = {
                'expected_accuracy_gain': mc_estimate['expected_gain'],
                'confidence_interval': mc_estimate['confidence_interval'],
                'success_probability': posterior_params['success_probability'],
                'risk_adjusted_benefit': risk_adjusted_benefit,
                'uncertainty_metrics': uncertainty_metrics,
                'bayesian_evidence': posterior_params['evidence'],
                'recommendation_strength': self._calculate_recommendation_strength(
                    risk_adjusted_benefit, uncertainty_metrics
                )
            }
            
            logger.debug(f"贝叶斯预测完成: 期望收益={prediction_result['expected_accuracy_gain']:.4f}")
            return prediction_result
            
        except Exception as e:
            logger.error(f"贝叶斯预测失败: {e}")
            return self._fallback_prediction(mutation_strategy, current_accuracy)

    def predict_comprehensive_mutation_strategy(self,
                                              layer_analysis: Dict[str, Any],
                                              current_accuracy: float,
                                              model: nn.Module,
                                              target_layer_name: str) -> Dict[str, Any]:
        """委托给综合策略生成器"""
        return self.comprehensive_generator.generate_comprehensive_strategy(
            layer_analysis, current_accuracy, model, target_layer_name
        )

    def _extract_feature_vector(self, layer_analysis: Dict[str, Any], 
                              current_accuracy: float,
                              model_complexity: Dict[str, float]) -> np.ndarray:
        """提取用于预测的特征向量"""
        
        # 从层分析中提取关键特征
        mutation_prediction = layer_analysis.get('mutation_prediction', {})
        param_analysis = layer_analysis.get('parameter_analysis', {})
        
        features = [
            current_accuracy,
            mutation_prediction.get('improvement_potential', 0.5),
            param_analysis.get('efficiency_score', 0.5),
            param_analysis.get('utilization_rate', 0.5),
            param_analysis.get('optimization_potential', 0.5),
            model_complexity.get('total_parameters', 1000000) / 1e6,  # 标准化
            model_complexity.get('layer_depth', 10) / 50,  # 标准化
            model_complexity.get('layer_width', 1000) / 5000  # 标准化
        ]
        
        return np.array(features, dtype=np.float32)

    def _bayesian_posterior_inference(self, feature_vector: np.ndarray, 
                                    mutation_strategy: str, 
                                    layer_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """贝叶斯后验推断"""
        
        # 使用先验知识
        prior_params = self.prior_knowledge.get_mutation_prior(mutation_strategy)
        
        # 计算似然
        likelihood = self._calculate_likelihood(feature_vector, layer_analysis)
        
        # 后验计算（简化的贝叶斯更新）
        success_probability = prior_params['success_rate'] * likelihood
        success_probability = np.clip(success_probability, 0.1, 0.9)
        
        # 贝叶斯证据
        evidence = likelihood * prior_params['confidence']
        
        return {
            'success_probability': success_probability,
            'evidence': evidence,
            'likelihood': likelihood,
            'prior_confidence': prior_params['confidence']
        }

    def _calculate_likelihood(self, feature_vector: np.ndarray, 
                            layer_analysis: Dict[str, Any]) -> float:
        """计算似然函数"""
        
        # 基于特征的简单似然模型
        current_acc = feature_vector[0]
        improvement_pot = feature_vector[1]
        
        # 似然与当前准确率和改进潜力相关
        likelihood = (1 - current_acc) * improvement_pot
        return np.clip(likelihood, 0.1, 1.0)

    def _gaussian_process_prediction(self, feature_vector: np.ndarray, 
                                   posterior_params: Dict[str, Any]) -> Dict[str, Any]:
        """高斯过程回归预测"""
        
        # 简化的GP实现
        # 在实际应用中应该使用历史数据训练GP
        
        # 均值预测
        mean_prediction = posterior_params['success_probability'] * 0.05  # 最大5%改进
        
        # 方差预测
        variance_prediction = (1 - posterior_params['evidence']) * 0.01
        
        return {
            'mean_prediction': mean_prediction,
            'variance_prediction': variance_prediction,
            'hyperparameters': self.gp_hyperparams
        }

    def _monte_carlo_benefit_estimation(self, gp_prediction: Dict[str, Any], 
                                      mutation_strategy: str,
                                      current_accuracy: float) -> Dict[str, Any]:
        """蒙特卡罗收益估计"""
        
        mean = gp_prediction['mean_prediction']
        std = np.sqrt(gp_prediction['variance_prediction'])
        
        # 生成MC样本
        samples = np.random.normal(mean, std, self.mc_samples)
        
        # 应用策略特定的变换
        strategy_multipliers = {
            'widening': 1.0,
            'deepening': 0.8,
            'hybrid_expansion': 1.2,
            'aggressive_widening': 1.5,
            'moderate_widening': 1.0
        }
        
        multiplier = strategy_multipliers.get(mutation_strategy, 1.0)
        samples = samples * multiplier
        
        # 确保样本合理性
        samples = np.clip(samples, 0.0, 0.95 - current_accuracy)
        
        # 计算统计量
        expected_gain = np.mean(samples)
        gain_std = np.std(samples)
        
        # 置信区间
        confidence_interval = {
            '68%': (np.percentile(samples, 16), np.percentile(samples, 84)),
            '95%': (np.percentile(samples, 2.5), np.percentile(samples, 97.5)),
            '99%': (np.percentile(samples, 0.5), np.percentile(samples, 99.5))
        }
        
        # 成功概率（获得正收益的概率）
        success_probability = np.mean(samples > 0)
        
        # 风险调整样本
        risk_adjusted_samples = samples * success_probability
        
        return {
            'expected_gain': float(expected_gain),
            'gain_std': float(gain_std),
            'confidence_interval': confidence_interval,
            'success_probability': float(success_probability),
            'samples': risk_adjusted_samples
        }

    def _quantify_prediction_uncertainty(self, gp_prediction: Dict[str, Any],
                                       mc_estimate: Dict[str, Any],
                                       feature_vector: np.ndarray) -> Dict[str, Any]:
        """量化预测不确定性"""
        
        # 认知不确定性（模型不确定性）
        epistemic_uncertainty = gp_prediction['variance_prediction']
        
        # 偶然不确定性（数据噪声）
        aleatoric_uncertainty = mc_estimate['gain_std']**2
        
        # 总不确定性
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # 预测置信度
        prediction_confidence = 1.0 / (1.0 + total_uncertainty)
        
        return {
            'epistemic_uncertainty': float(epistemic_uncertainty),
            'aleatoric_uncertainty': float(aleatoric_uncertainty),
            'total_uncertainty': float(total_uncertainty),
            'prediction_confidence': float(prediction_confidence)
        }

    def _calculate_risk_adjusted_benefit(self, mc_estimate: Dict[str, Any],
                                       uncertainty_metrics: Dict[str, Any],
                                       mutation_strategy: str) -> Dict[str, Any]:
        """计算风险调整后的收益"""
        
        expected_gain = mc_estimate['expected_gain']
        total_uncertainty = uncertainty_metrics['total_uncertainty']
        success_prob = mc_estimate['success_probability']
        
        # 风险调整系数
        risk_aversion_factor = 0.5  # 可调参数
        uncertainty_penalty = risk_aversion_factor * total_uncertainty
        
        # 风险调整收益 = 期望收益 - 不确定性惩罚
        risk_adjusted_gain = expected_gain - uncertainty_penalty
        
        # 夏普比率（收益风险比）
        sharpe_ratio = expected_gain / (mc_estimate['gain_std'] + 1e-8)
        
        # 价值风险（VaR）
        var_95 = mc_estimate['confidence_interval']['95%'][0]  # 5%分位数
        
        # 条件价值风险（CVaR）
        samples = mc_estimate['samples']
        cvar_95 = np.mean(samples[samples <= var_95])
        
        return {
            'risk_adjusted_gain': float(risk_adjusted_gain),
            'sharpe_ratio': float(sharpe_ratio),
            'value_at_risk_95': float(var_95),
            'conditional_var_95': float(cvar_95),
            'risk_reward_score': float(expected_gain / (total_uncertainty + 1e-8))
        }

    def _calculate_recommendation_strength(self, risk_adjusted_benefit: Dict[str, Any],
                                         uncertainty_metrics: Dict[str, Any]) -> str:
        """计算推荐强度"""
        
        gain = risk_adjusted_benefit['risk_adjusted_gain']
        confidence = uncertainty_metrics['prediction_confidence']
        sharpe_ratio = risk_adjusted_benefit['sharpe_ratio']
        
        # 综合评分
        score = gain * confidence * (1 + sharpe_ratio)
        
        if score > 0.02 and confidence > 0.7:
            return "strong_recommend"
        elif score > 0.01 and confidence > 0.5:
            return "recommend"
        elif score > 0.005:
            return "weak_recommend"
        elif score > -0.005:
            return "neutral"
        else:
            return "not_recommend"

    def _calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """计算特征相似性"""
        
        # 使用余弦相似性
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return float(np.clip(similarity, 0.0, 1.0))

    def _fallback_prediction(self, mutation_strategy: str, current_accuracy: float) -> Dict[str, Any]:
        """fallback预测（当贝叶斯预测失败时）"""
        
        # 简单的启发式预测
        base_gain = max(0.01, (0.95 - current_accuracy) * 0.1)
        
        strategy_multipliers = {
            'widening': 0.8,
            'deepening': 0.6,
            'hybrid_expansion': 1.0,
            'aggressive_widening': 1.2,
            'moderate_widening': 1.0
        }
        
        expected_gain = base_gain * strategy_multipliers.get(mutation_strategy, 0.8)
        
        return {
            'expected_accuracy_gain': expected_gain,
            'confidence_interval': {'95%': (0.0, expected_gain * 2)},
            'success_probability': 0.5,
            'risk_adjusted_benefit': {'risk_adjusted_gain': expected_gain * 0.5},
            'uncertainty_metrics': {'prediction_confidence': 0.3},
            'recommendation_strength': "weak_recommend" if expected_gain > 0.005 else "neutral"
        }

    def update_with_mutation_result(self, 
                                  feature_vector: np.ndarray,
                                  mutation_strategy: str,
                                  actual_gain: float,
                                  success: bool):
        """用实际变异结果更新模型"""
        
        self.mutation_history.append({
            'features': feature_vector,
            'strategy': mutation_strategy,
            'actual_gain': actual_gain,
            'success': success,
            'timestamp': time.time()
        })
        
        # 保持历史记录大小
        if len(self.mutation_history) > 100:
            self.mutation_history = self.mutation_history[-100:]
        
        logger.info(f"更新贝叶斯模型: {mutation_strategy}, 实际收益={actual_gain:.4f}")