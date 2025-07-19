"""
不确定性量化模块

为贝叶斯推断引擎提供高级不确定性量化功能：
1. 认知不确定性（模型不确定性）量化
2. 偶然不确定性（数据噪声）估计  
3. Net2Net参数迁移不确定性分析
4. 多级不确定性传播
5. 置信度校准
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """高级不确定性量化器"""
    
    def __init__(self):
        self.calibration_data = []
        self.net2net_uncertainty_cache = {}
    
    def comprehensive_uncertainty_analysis(self, 
                                         prediction_results: Dict[str, Any],
                                         model_analysis: Dict[str, Any],
                                         net2net_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        综合不确定性分析
        
        Args:
            prediction_results: 预测结果
            model_analysis: 模型分析结果
            net2net_assessment: Net2Net评估结果
            
        Returns:
            详细的不确定性分析报告
        """
        
        try:
            # 1. 分解不确定性来源
            uncertainty_decomposition = self._decompose_uncertainty_sources(
                prediction_results, model_analysis
            )
            
            # 2. Net2Net特定不确定性
            net2net_uncertainty = self._analyze_net2net_uncertainty(
                net2net_assessment, model_analysis
            )
            
            # 3. 贝叶斯不确定性量化
            bayesian_uncertainty = self._bayesian_uncertainty_quantification(
                prediction_results, uncertainty_decomposition
            )
            
            # 4. 预测置信度校准
            calibrated_confidence = self._calibrate_prediction_confidence(
                prediction_results, bayesian_uncertainty
            )
            
            # 5. 不确定性传播分析
            uncertainty_propagation = self._analyze_uncertainty_propagation(
                uncertainty_decomposition, net2net_uncertainty
            )
            
            # 6. 决策不确定性评估
            decision_uncertainty = self._evaluate_decision_uncertainty(
                prediction_results, calibrated_confidence, uncertainty_propagation
            )
            
            comprehensive_analysis = {
                'uncertainty_decomposition': uncertainty_decomposition,
                'net2net_uncertainty': net2net_uncertainty,
                'bayesian_uncertainty': bayesian_uncertainty,
                'calibrated_confidence': calibrated_confidence,
                'uncertainty_propagation': uncertainty_propagation,
                'decision_uncertainty': decision_uncertainty,
                'overall_confidence': self._calculate_overall_confidence(
                    calibrated_confidence, decision_uncertainty
                ),
                'uncertainty_recommendations': self._generate_uncertainty_recommendations(
                    decision_uncertainty, net2net_uncertainty
                )
            }
            
            logger.debug("综合不确定性分析完成")
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"不确定性分析失败: {e}")
            return self._fallback_uncertainty_analysis()
    
    def _decompose_uncertainty_sources(self, 
                                     prediction_results: Dict[str, Any],
                                     model_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """分解不确定性来源"""
        
        # 从预测结果中提取基础不确定性
        epistemic = prediction_results.get('uncertainty_metrics', {}).get('epistemic_uncertainty', 0.1)
        aleatoric = prediction_results.get('uncertainty_metrics', {}).get('aleatoric_uncertainty', 0.1)
        
        # 模型结构不确定性
        model_uncertainty = self._estimate_model_structural_uncertainty(model_analysis)
        
        # 参数不确定性
        parameter_uncertainty = self._estimate_parameter_uncertainty(model_analysis)
        
        # 数据相关不确定性
        data_uncertainty = self._estimate_data_related_uncertainty(model_analysis)
        
        return {
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'model_structural_uncertainty': model_uncertainty,
            'parameter_uncertainty': parameter_uncertainty,
            'data_uncertainty': data_uncertainty,
            'total_decomposed_uncertainty': np.sqrt(
                epistemic**2 + aleatoric**2 + model_uncertainty**2 + 
                parameter_uncertainty**2 + data_uncertainty**2
            )
        }
    
    def _estimate_model_structural_uncertainty(self, model_analysis: Dict[str, Any]) -> float:
        """估计模型结构不确定性"""
        
        # 基于模型复杂度的结构不确定性
        complexity_indicators = []
        
        # 层数深度带来的不确定性
        layer_count = len(model_analysis.get('layer_analyses', {}))
        depth_uncertainty = min(0.1, layer_count / 100.0)
        complexity_indicators.append(depth_uncertainty)
        
        # 参数量带来的不确定性
        total_params = sum(
            analysis.get('layer_size', 0) 
            for analysis in model_analysis.get('layer_analyses', {}).values()
        )
        param_uncertainty = min(0.15, total_params / 10000000.0)
        complexity_indicators.append(param_uncertainty)
        
        # 架构多样性不确定性
        layer_types = set()
        for analysis in model_analysis.get('layer_analyses', {}).values():
            layer_types.add(analysis.get('layer_type', 'unknown'))
        
        diversity_uncertainty = min(0.05, len(layer_types) / 20.0)
        complexity_indicators.append(diversity_uncertainty)
        
        return float(np.mean(complexity_indicators))
    
    def _estimate_parameter_uncertainty(self, model_analysis: Dict[str, Any]) -> float:
        """估计参数不确定性"""
        
        param_uncertainties = []
        
        for layer_name, analysis in model_analysis.get('layer_analyses', {}).items():
            param_analysis = analysis.get('parameter_analysis', {})
            
            # 参数利用率不确定性
            utilization = param_analysis.get('utilization_rate', 0.5)
            util_uncertainty = (1.0 - utilization) * 0.1
            
            # 优化潜力不确定性  
            opt_potential = param_analysis.get('optimization_potential', 0.5)
            opt_uncertainty = opt_potential * 0.08
            
            param_uncertainties.append(util_uncertainty + opt_uncertainty)
        
        return float(np.mean(param_uncertainties)) if param_uncertainties else 0.05
    
    def _estimate_data_related_uncertainty(self, model_analysis: Dict[str, Any]) -> float:
        """估计数据相关不确定性"""
        
        # 基于信息流分析估计数据不确定性
        info_flow = model_analysis.get('information_flow', {})
        flow_efficiency = info_flow.get('flow_efficiency', 0.7)
        
        # 信息流效率低表示数据利用不充分，不确定性高
        data_uncertainty = (1.0 - flow_efficiency) * 0.1
        
        return float(data_uncertainty)
    
    def _analyze_net2net_uncertainty(self, 
                                   net2net_assessment: Dict[str, Any],
                                   model_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """分析Net2Net特定不确定性"""
        
        if not net2net_assessment.get('applicable', False):
            return {
                'function_preserving_uncertainty': 0.0,
                'parameter_transfer_uncertainty': 0.0,
                'smooth_transition_uncertainty': 0.0,
                'total_net2net_uncertainty': 0.0
            }
        
        # 函数保持性不确定性
        methods = net2net_assessment.get('methods', {})
        net2wider_conf = methods.get('net2wider', {}).get('function_preserving_confidence', 0.9)
        function_uncertainty = 1.0 - net2wider_conf
        
        # 参数迁移不确定性
        smooth_transition = methods.get('smooth_transition', {})
        training_stability = smooth_transition.get('training_stability', 0.8)
        transfer_uncertainty = 1.0 - training_stability
        
        # 平滑过渡不确定性
        convergence_accel = smooth_transition.get('convergence_acceleration', 0.7)
        transition_uncertainty = (1.0 - convergence_accel) * 0.5
        
        # 综合Net2Net不确定性
        total_net2net_uncertainty = np.sqrt(
            function_uncertainty**2 + transfer_uncertainty**2 + transition_uncertainty**2
        )
        
        return {
            'function_preserving_uncertainty': float(function_uncertainty),
            'parameter_transfer_uncertainty': float(transfer_uncertainty),
            'smooth_transition_uncertainty': float(transition_uncertainty),
            'total_net2net_uncertainty': float(total_net2net_uncertainty),
            'net2net_confidence_boost': max(0.0, 0.2 - total_net2net_uncertainty)
        }
    
    def _bayesian_uncertainty_quantification(self, 
                                            prediction_results: Dict[str, Any],
                                            uncertainty_decomposition: Dict[str, Any]) -> Dict[str, Any]:
        """贝叶斯不确定性量化"""
        
        # 后验分布参数
        success_prob = prediction_results.get('success_probability', 0.5)
        bayesian_evidence = prediction_results.get('bayesian_evidence', 0.5)
        
        # 贝叶斯置信区间
        confidence_intervals = prediction_results.get('confidence_interval', {})
        ci_95 = confidence_intervals.get('95%', (0.0, 0.02))
        ci_width = ci_95[1] - ci_95[0]
        
        # 后验不确定性
        posterior_uncertainty = ci_width / 4.0  # 近似标准误差
        
        # 证据不确定性
        evidence_uncertainty = (1.0 - bayesian_evidence) * 0.1
        
        # 先验-后验分歧
        prior_posterior_divergence = abs(0.5 - success_prob) * 0.1
        
        return {
            'posterior_uncertainty': float(posterior_uncertainty),
            'evidence_uncertainty': float(evidence_uncertainty),
            'prior_posterior_divergence': float(prior_posterior_divergence),
            'bayesian_total_uncertainty': float(
                posterior_uncertainty + evidence_uncertainty + prior_posterior_divergence
            ),
            'credible_interval_width': float(ci_width)
        }
    
    def _calibrate_prediction_confidence(self, 
                                        prediction_results: Dict[str, Any],
                                        bayesian_uncertainty: Dict[str, Any]) -> Dict[str, Any]:
        """校准预测置信度"""
        
        # 原始置信度
        raw_confidence = prediction_results.get('uncertainty_metrics', {}).get('prediction_confidence', 0.5)
        
        # 贝叶斯校准
        bayesian_total = bayesian_uncertainty['bayesian_total_uncertainty']
        bayesian_calibrated = raw_confidence * (1.0 - bayesian_total)
        
        # 经验校准（基于历史数据）
        empirical_calibrated = self._empirical_confidence_calibration(raw_confidence)
        
        # 组合校准
        final_calibrated = (bayesian_calibrated + empirical_calibrated) / 2.0
        
        # 校准质量评估
        calibration_quality = self._assess_calibration_quality(
            raw_confidence, final_calibrated, bayesian_uncertainty
        )
        
        return {
            'raw_confidence': float(raw_confidence),
            'bayesian_calibrated': float(bayesian_calibrated),
            'empirical_calibrated': float(empirical_calibrated),
            'final_calibrated_confidence': float(final_calibrated),
            'calibration_quality': calibration_quality,
            'confidence_intervals': {
                '90%': (final_calibrated - 0.05, final_calibrated + 0.05),
                '95%': (final_calibrated - 0.1, final_calibrated + 0.1)
            }
        }
    
    def _empirical_confidence_calibration(self, raw_confidence: float) -> float:
        """基于经验数据的置信度校准"""
        
        if not self.calibration_data:
            return raw_confidence  # 没有历史数据时不校准
        
        # 简化的置信度校准（实际应用中应使用更复杂的校准方法）
        # 这里使用线性校准作为示例
        calibration_factor = 0.9  # 通常预测过于自信，需要适当降低
        return raw_confidence * calibration_factor
    
    def _assess_calibration_quality(self, 
                                  raw_confidence: float,
                                  calibrated_confidence: float,
                                  bayesian_uncertainty: Dict[str, Any]) -> Dict[str, Any]:
        """评估校准质量"""
        
        # 校准幅度
        calibration_magnitude = abs(calibrated_confidence - raw_confidence)
        
        # 校准方向
        calibration_direction = "conservative" if calibrated_confidence < raw_confidence else "aggressive"
        
        # 校准稳定性（基于贝叶斯不确定性）
        stability = 1.0 - bayesian_uncertainty['bayesian_total_uncertainty']
        
        return {
            'calibration_magnitude': float(calibration_magnitude),
            'calibration_direction': calibration_direction,
            'calibration_stability': float(stability),
            'calibration_recommended': calibration_magnitude > 0.05
        }
    
    def _analyze_uncertainty_propagation(self, 
                                       uncertainty_decomposition: Dict[str, Any],
                                       net2net_uncertainty: Dict[str, Any]) -> Dict[str, Any]:
        """分析不确定性传播"""
        
        # 不确定性传播矩阵
        base_uncertainties = [
            uncertainty_decomposition['epistemic_uncertainty'],
            uncertainty_decomposition['aleatoric_uncertainty'],
            uncertainty_decomposition['model_structural_uncertainty'],
            uncertainty_decomposition['parameter_uncertainty']
        ]
        
        net2net_uncertainties = [
            net2net_uncertainty['function_preserving_uncertainty'],
            net2net_uncertainty['parameter_transfer_uncertainty'],
            net2net_uncertainty['smooth_transition_uncertainty']
        ]
        
        # 线性传播（保守估计）
        linear_propagation = sum(base_uncertainties) + sum(net2net_uncertainties)
        
        # 平方根传播（独立假设）
        sqrt_propagation = np.sqrt(
            sum(u**2 for u in base_uncertainties) + sum(u**2 for u in net2net_uncertainties)
        )
        
        # 相关性调整传播
        correlation_factor = 0.3  # 假设中等相关性
        correlated_propagation = sqrt_propagation * (1 + correlation_factor)
        
        return {
            'linear_propagation': float(linear_propagation),
            'independent_propagation': float(sqrt_propagation),
            'correlated_propagation': float(correlated_propagation),
            'recommended_propagation': float(correlated_propagation),  # 使用相关性调整版本
            'propagation_confidence': 1.0 - correlated_propagation
        }
    
    def _evaluate_decision_uncertainty(self, 
                                     prediction_results: Dict[str, Any],
                                     calibrated_confidence: Dict[str, Any],
                                     uncertainty_propagation: Dict[str, Any]) -> Dict[str, Any]:
        """评估决策不确定性"""
        
        # 策略选择不确定性
        recommendation_strength = prediction_results.get('recommendation_strength', 'neutral')
        strategy_uncertainty = {
            'strong_recommend': 0.1,
            'recommend': 0.2,
            'weak_recommend': 0.4,
            'neutral': 0.6,
            'not_recommend': 0.3
        }[recommendation_strength]
        
        # 收益预测不确定性
        expected_gain = prediction_results.get('expected_accuracy_gain', 0.01)
        gain_uncertainty = min(0.5, 1.0 / (expected_gain * 100 + 1))  # 收益越小不确定性越大
        
        # 风险评估不确定性
        risk_adjusted = prediction_results.get('risk_adjusted_benefit', {})
        risk_uncertainty = 1.0 - risk_adjusted.get('risk_reward_score', 0.5)
        
        # 传播不确定性影响
        propagation_impact = uncertainty_propagation['recommended_propagation'] * 0.5
        
        # 综合决策不确定性
        total_decision_uncertainty = np.sqrt(
            strategy_uncertainty**2 + gain_uncertainty**2 + 
            risk_uncertainty**2 + propagation_impact**2
        )
        
        return {
            'strategy_selection_uncertainty': float(strategy_uncertainty),
            'gain_prediction_uncertainty': float(gain_uncertainty),
            'risk_assessment_uncertainty': float(risk_uncertainty),
            'propagation_impact': float(propagation_impact),
            'total_decision_uncertainty': float(total_decision_uncertainty),
            'decision_confidence': 1.0 - total_decision_uncertainty,
            'decision_reliability': self._assess_decision_reliability(total_decision_uncertainty)
        }
    
    def _assess_decision_reliability(self, total_uncertainty: float) -> str:
        """评估决策可靠性"""
        if total_uncertainty < 0.2:
            return "high"
        elif total_uncertainty < 0.4:
            return "medium"
        elif total_uncertainty < 0.6:
            return "low"
        else:
            return "very_low"
    
    def _calculate_overall_confidence(self, 
                                    calibrated_confidence: Dict[str, Any],
                                    decision_uncertainty: Dict[str, Any]) -> Dict[str, Any]:
        """计算总体置信度"""
        
        # 校准后的预测置信度
        prediction_confidence = calibrated_confidence['final_calibrated_confidence']
        
        # 决策置信度
        decision_confidence = decision_uncertainty['decision_confidence']
        
        # 加权组合
        overall_confidence = (prediction_confidence * 0.6 + decision_confidence * 0.4)
        
        # 置信度等级
        confidence_level = "high" if overall_confidence > 0.8 else \
                          "medium" if overall_confidence > 0.6 else \
                          "low" if overall_confidence > 0.4 else "very_low"
        
        return {
            'overall_confidence_score': float(overall_confidence),
            'confidence_level': confidence_level,
            'confidence_breakdown': {
                'prediction_component': float(prediction_confidence),
                'decision_component': float(decision_confidence)
            },
            'confidence_stability': self._assess_confidence_stability(
                calibrated_confidence, decision_uncertainty
            )
        }
    
    def _assess_confidence_stability(self, 
                                   calibrated_confidence: Dict[str, Any],
                                   decision_uncertainty: Dict[str, Any]) -> str:
        """评估置信度稳定性"""
        
        calibration_quality = calibrated_confidence['calibration_quality']['calibration_stability']
        decision_reliability = decision_uncertainty['decision_confidence']
        
        stability_score = (calibration_quality + decision_reliability) / 2.0
        
        if stability_score > 0.8:
            return "stable"
        elif stability_score > 0.6:
            return "moderately_stable"
        else:
            return "unstable"
    
    def _generate_uncertainty_recommendations(self, 
                                            decision_uncertainty: Dict[str, Any],
                                            net2net_uncertainty: Dict[str, Any]) -> List[str]:
        """生成不确定性相关建议"""
        
        recommendations = []
        
        # 基于决策不确定性的建议
        if decision_uncertainty['total_decision_uncertainty'] > 0.5:
            recommendations.append("建议收集更多数据以降低决策不确定性")
            recommendations.append("考虑采用ensemble方法提高预测稳定性")
        
        # 基于Net2Net不确定性的建议
        if net2net_uncertainty['total_net2net_uncertainty'] > 0.3:
            recommendations.append("建议进行函数保持性验证以确保Net2Net迁移质量")
            recommendations.append("使用渐进式训练降低参数迁移风险")
        
        # 基于置信度的建议
        if decision_uncertainty['decision_confidence'] < 0.6:
            recommendations.append("建议采用更保守的策略或分阶段执行")
            recommendations.append("增加监控频率，准备回滚方案")
        
        return recommendations
    
    def _fallback_uncertainty_analysis(self) -> Dict[str, Any]:
        """fallback不确定性分析"""
        
        return {
            'uncertainty_decomposition': {
                'total_decomposed_uncertainty': 0.3
            },
            'net2net_uncertainty': {
                'total_net2net_uncertainty': 0.2
            },
            'calibrated_confidence': {
                'final_calibrated_confidence': 0.5
            },
            'decision_uncertainty': {
                'total_decision_uncertainty': 0.4,
                'decision_confidence': 0.6
            },
            'overall_confidence': {
                'overall_confidence_score': 0.5,
                'confidence_level': 'medium'
            },
            'uncertainty_recommendations': [
                "不确定性分析失败，建议采用保守策略"
            ]
        }
    
    def update_calibration_data(self, 
                               predicted_confidence: float,
                               actual_success: bool,
                               prediction_context: Dict[str, Any]):
        """更新校准数据"""
        
        self.calibration_data.append({
            'predicted_confidence': predicted_confidence,
            'actual_success': actual_success,
            'context': prediction_context,
            'timestamp': np.datetime64('now')
        })
        
        # 保持校准数据大小
        if len(self.calibration_data) > 1000:
            self.calibration_data = self.calibration_data[-1000:]
        
        logger.debug(f"更新校准数据: 预测置信度={predicted_confidence:.3f}, 实际成功={actual_success}")


class AdvancedUncertaintyMetrics:
    """高级不确定性指标计算器"""
    
    @staticmethod
    def mutual_information_uncertainty(predictions: np.ndarray, 
                                     targets: np.ndarray) -> float:
        """基于互信息的不确定性计算"""
        
        # 简化的互信息计算
        # 实际应用中应使用更精确的方法
        correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
        if np.isnan(correlation):
            return 1.0
        
        # 互信息近似: -0.5 * log(1 - r^2)
        mi_uncertainty = -0.5 * np.log(1 - correlation**2 + 1e-10)
        return float(np.clip(mi_uncertainty, 0.0, 10.0))
    
    @staticmethod
    def predictive_entropy(probabilities: np.ndarray) -> float:
        """预测熵计算"""
        
        # 避免log(0)
        probs = np.clip(probabilities, 1e-10, 1 - 1e-10)
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)
    
    @staticmethod
    def aleatoric_epistemic_decomposition(predictions: List[np.ndarray]) -> Tuple[float, float]:
        """分解偶然不确定性和认知不确定性"""
        
        if len(predictions) < 2:
            return 0.1, 0.1
        
        predictions_array = np.array(predictions)
        
        # 认知不确定性：预测间的方差
        epistemic = np.var(np.mean(predictions_array, axis=-1))
        
        # 偶然不确定性：每个预测内部的平均方差
        aleatoric = np.mean([np.var(pred) for pred in predictions])
        
        return float(epistemic), float(aleatoric)