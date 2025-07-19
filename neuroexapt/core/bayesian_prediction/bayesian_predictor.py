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
            try:
                from ..mutation_strategies.comprehensive_strategy import ComprehensiveStrategyGenerator
                self._comprehensive_generator = ComprehensiveStrategyGenerator(self.prior_knowledge)
            except ImportError as e:
                logger.warning(f"Could not import ComprehensiveStrategyGenerator: {e}")
                # 创建简化版本作为回退
                self._comprehensive_generator = self._create_simple_strategy_generator()
        return self._comprehensive_generator
    
    def _create_simple_strategy_generator(self):
        """创建简化的策略生成器作为回退"""
        class SimpleStrategyGenerator:
            def generate_comprehensive_strategy(self, *args, **kwargs):
                return {
                    'mutation_mode': 'serial_division',
                    'layer_combination_strategy': 'single_layer',
                    'confidence': 0.5,
                    'expected_benefit': 0.01
                }
        return SimpleStrategyGenerator()
        
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
            
            logger.debug(f"贝叶斯预测完成: 期望收益={mc_estimate['expected_gain']:.4f}")
            
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
        return self.comprehensive_generator.predict_comprehensive_mutation_strategy(
            layer_analysis, current_accuracy, model, target_layer_name
        )

    def _extract_feature_vector(self, layer_analysis: Dict[str, Any], 
                              current_accuracy: float,
                              model_complexity: Dict[str, float]) -> np.ndarray:
        """提取用于预测的特征向量"""
        
        # 从层分析中提取关键特征
        mutation_prediction = layer_analysis.get('mutation_prediction', {})
        param_analysis = layer_analysis.get('parameter_space_analysis', {})
        leak_assessment = layer_analysis.get('leak_assessment', {})
        
        features = [
            # 基础准确率和改进空间
            current_accuracy,
            1.0 - current_accuracy,  # 改进空间
            
            # 层特性
            mutation_prediction.get('improvement_potential', 0.0),
            param_analysis.get('efficiency_score', 0.0),
            param_analysis.get('utilization_rate', 0.0),
            
            # 漏点特征
            leak_assessment.get('leak_severity', 0.0),
            1.0 if leak_assessment.get('is_leak_point', False) else 0.0,
            
            # 模型复杂度
            model_complexity.get('total_parameters', 0.0) / 1e6,  # 百万参数为单位
            model_complexity.get('layer_depth', 0.0) / 50.0,      # 归一化深度
            model_complexity.get('layer_width', 0.0) / 1000.0,    # 归一化宽度
            
            # 梯度和激活统计
            mutation_prediction.get('gradient_diversity', 0.0),
            mutation_prediction.get('activation_saturation', 0.5),
        ]
        
        return np.array(features, dtype=np.float32)

    def _bayesian_posterior_inference(self, feature_vector: np.ndarray,
                                    mutation_strategy: str,
                                    layer_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """贝叶斯后验推断"""
        
        # 获取先验参数
        mutation_prior = self.prior_knowledge.get_mutation_prior(mutation_strategy)
        
        # 获取瓶颈类型相关的先验
        leak_assessment = layer_analysis.get('leak_assessment', {})
        leak_type = leak_assessment.get('leak_type', 'general_bottleneck')
        
        bottleneck_prior = self.prior_knowledge.get_bottleneck_response_prior(leak_type)
        strategy_response = bottleneck_prior.get(f"{mutation_strategy}_response", 0.5)
        
        # 贝叶斯更新：根据观测特征更新先验
        observed_evidence = self._calculate_evidence_strength(feature_vector)
        
        # Beta分布的共轭更新
        alpha_posterior = mutation_prior['alpha'] + observed_evidence['positive_evidence']
        beta_posterior = mutation_prior['beta'] + observed_evidence['negative_evidence']
        
        # 计算后验成功概率
        success_probability = alpha_posterior / (alpha_posterior + beta_posterior)
        
        # 贝叶斯证据（边际似然）
        evidence = self._calculate_marginal_likelihood(
            feature_vector, mutation_strategy, strategy_response
        )
        
        return {
            'success_probability': success_probability,
            'alpha_posterior': alpha_posterior,
            'beta_posterior': beta_posterior,
            'evidence': evidence,
            'strategy_response': strategy_response
        }

    def _calculate_evidence_strength(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """计算观测证据强度"""
        
        # 基于特征向量计算支持和反对变异的证据
        current_acc = feature_vector[0]
        improvement_space = feature_vector[1]
        improvement_potential = feature_vector[2]
        leak_severity = feature_vector[5]
        is_leak_point = feature_vector[6]
        
        # 正面证据：支持变异的因素
        positive_evidence = (
            improvement_space * 2.0 +           # 改进空间大
            improvement_potential * 1.5 +       # 改进潜力高
            leak_severity * 2.0 +               # 漏点严重
            is_leak_point * 1.0                 # 确实是漏点
        )
        
        # 负面证据：反对变异的因素
        negative_evidence = (
            current_acc * 1.0 +                 # 当前准确率已经很高
            (1.0 - improvement_potential) * 1.0 # 改进潜力低
        )
        
        return {
            'positive_evidence': max(0.1, positive_evidence),
            'negative_evidence': max(0.1, negative_evidence)
        }

    def _calculate_marginal_likelihood(self, feature_vector: np.ndarray,
                                     mutation_strategy: str,
                                     strategy_response: float) -> float:
        """计算边际似然（贝叶斯证据）"""
        
        # 使用高斯似然函数
        likelihood = 0.0
        
        # 基于特征相似性计算似然
        for historical_mutation in self.mutation_history:
            if historical_mutation['strategy'] == mutation_strategy:
                feature_similarity = self._calculate_feature_similarity(
                    feature_vector, historical_mutation['features']
                )
                
                success = historical_mutation['success']
                likelihood += feature_similarity * (success if success else (1 - success))
        
        # 如果没有历史数据，使用先验响应性
        if likelihood == 0.0:
            likelihood = strategy_response
        
        return float(np.clip(likelihood, 0.01, 0.99))

    def _gaussian_process_prediction(self, feature_vector: np.ndarray,
                                   posterior_params: Dict[str, Any]) -> Dict[str, Any]:
        """高斯过程回归预测"""
        
        # 构建核函数（RBF核）
        def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
            return variance * np.exp(-0.5 * np.sum((x1 - x2)**2) / length_scale**2)
        
        # 如果有历史数据，使用GP回归
        if self.mutation_history:
            # 构建训练数据
            X_train = np.array([m['features'] for m in self.mutation_history])
            y_train = np.array([m['actual_gain'] for m in self.mutation_history])
            
            # 计算核矩阵
            n_train = len(X_train)
            K = np.zeros((n_train, n_train))
            
            for i in range(n_train):
                for j in range(n_train):
                    K[i, j] = rbf_kernel(
                        X_train[i], X_train[j],
                        self.gp_hyperparams['length_scale'],
                        self.gp_hyperparams['variance']
                    )
            
            # 添加噪声项
            K += np.eye(n_train) * self.gp_hyperparams['noise_variance']
            
            # 计算预测
            k_star = np.array([
                rbf_kernel(feature_vector, X_train[i],
                          self.gp_hyperparams['length_scale'],
                          self.gp_hyperparams['variance'])
                for i in range(n_train)
            ])
            
            try:
                K_inv = np.linalg.inv(K)
                mean_pred = k_star.T @ K_inv @ y_train
                
                k_star_star = rbf_kernel(
                    feature_vector, feature_vector,
                    self.gp_hyperparams['length_scale'],
                    self.gp_hyperparams['variance']
                )
                
                var_pred = k_star_star - k_star.T @ K_inv @ k_star
                
            except np.linalg.LinAlgError:
                # 如果矩阵奇异，使用伪逆
                K_pinv = np.linalg.pinv(K)
                mean_pred = k_star.T @ K_pinv @ y_train
                var_pred = self.gp_hyperparams['variance']
        
        else:
            # 没有历史数据时，使用先验均值和方差
            mean_pred = posterior_params['success_probability'] * 0.05  # 假设最大收益5%
            var_pred = self.gp_hyperparams['variance']
        
        return {
            'mean_prediction': float(mean_pred),
            'variance_prediction': float(var_pred),
            'std_prediction': float(np.sqrt(max(var_pred, 0)))
        }

    def _monte_carlo_benefit_estimation(self, gp_prediction: Dict[str, Any],
                                      mutation_strategy: str,
                                      current_accuracy: float) -> Dict[str, Any]:
        """蒙特卡罗期望收益估计"""
        
        mean = gp_prediction['mean_prediction']
        std = gp_prediction['std_prediction']
        
        # 从预测分布中采样
        samples = np.random.normal(mean, std, self.mc_samples)
        
        # 考虑变异策略的风险特性
        strategy_risk_factor = {
            'widening': 0.9,
            'deepening': 0.8,
            'hybrid_expansion': 0.7,
            'aggressive_widening': 0.6
        }.get(mutation_strategy, 0.8)
        
        # 应用风险调整
        risk_adjusted_samples = samples * strategy_risk_factor
        
        # 确保收益不超过理论上限
        max_possible_gain = min(0.95 - current_accuracy, 0.1)  # 最大收益限制
        risk_adjusted_samples = np.clip(risk_adjusted_samples, -0.02, max_possible_gain)
        
        # 计算统计量
        expected_gain = np.mean(risk_adjusted_samples)
        gain_std = np.std(risk_adjusted_samples)
        
        # 置信区间
        confidence_interval = {
            '95%': (
                np.percentile(risk_adjusted_samples, 2.5),
                np.percentile(risk_adjusted_samples, 97.5)
            ),
            '90%': (
                np.percentile(risk_adjusted_samples, 5),
                np.percentile(risk_adjusted_samples, 95)
            ),
            '68%': (
                np.percentile(risk_adjusted_samples, 16),
                np.percentile(risk_adjusted_samples, 84)
            )
        }
        
        # 成功概率（收益为正的概率）
        success_probability = np.mean(risk_adjusted_samples > 0)
        
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
            'aggressive_widening': 1.2
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