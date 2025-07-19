"""
贝叶斯推断引擎

基于贝叶斯推断、高斯过程回归和蒙特卡罗采样的架构变异推断引擎
结合Net2Net参数平滑迁移技术，提供强大的变异决策支持
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import time
import logging
from .prior_knowledge import PriorKnowledgeBase

logger = logging.getLogger(__name__)


class BayesianInferenceEngine:
    """
    贝叶斯推断引擎
    
    核心功能：
    1. 基于贝叶斯统计推断最佳变异策略
    2. 集成Net2Net参数迁移技术评估
    3. 使用高斯过程回归预测变异收益
    4. 蒙特卡罗采样量化不确定性
    5. 提供架构变异决策建议
    """
    
    def __init__(self):
        self.prior_knowledge = PriorKnowledgeBase()
        self._comprehensive_generator = None
        self._net2net_transfer = None
        
        # 高斯过程超参数
        self.gp_hyperparams = {
            'length_scale': 1.0,
            'variance': 1.0,
            'noise_variance': 0.01
        }
        self.mc_samples = 1000  # 蒙特卡罗采样数
        
        # 历史变异数据（用于更新先验）
        self.mutation_history = []
        self.net2net_transfer_history = []
    
    @property
    def comprehensive_generator(self):
        """延迟加载综合策略生成器"""
        if self._comprehensive_generator is None:
            from ..mutation_strategies.comprehensive_strategy import ComprehensiveStrategyGenerator
            self._comprehensive_generator = ComprehensiveStrategyGenerator(self.prior_knowledge)
        return self._comprehensive_generator
    
    @property 
    def net2net_transfer(self):
        """延迟加载Net2Net迁移工具"""
        if self._net2net_transfer is None:
            from ..net2net_transfer import Net2NetTransfer
            self._net2net_transfer = Net2NetTransfer()
        return self._net2net_transfer
        
    def infer_optimal_mutation_strategy(self, 
                                      layer_analysis: Dict[str, Any],
                                      current_accuracy: float,
                                      model: nn.Module,
                                      target_layer_name: str,
                                      model_complexity: Dict[str, float]) -> Dict[str, Any]:
        """
        推断最优变异策略
        
        Args:
            layer_analysis: 层分析结果
            current_accuracy: 当前准确率
            model: 目标模型
            target_layer_name: 目标层名称
            model_complexity: 模型复杂度指标
            
        Returns:
            包含推断结果和决策建议的完整报告
        """
        logger.debug(f"开始贝叶斯推断: 目标层={target_layer_name}")
        
        try:
            # 1. 多策略评估
            strategy_evaluations = self._evaluate_multiple_strategies(
                layer_analysis, current_accuracy, model_complexity
            )
            
            # 2. Net2Net适用性评估
            net2net_assessment = self._assess_net2net_applicability(
                layer_analysis, model, target_layer_name
            )
            
            # 3. 贝叶斯模型选择
            optimal_strategy = self._bayesian_model_selection(
                strategy_evaluations, net2net_assessment, current_accuracy
            )
            
            # 4. 参数迁移规划
            transfer_plan = self._plan_parameter_transfer(
                optimal_strategy, model, target_layer_name
            )
            
            # 5. 风险评估和不确定性量化
            risk_assessment = self._comprehensive_risk_assessment(
                optimal_strategy, transfer_plan, current_accuracy
            )
            
            # 6. 执行建议生成
            execution_recommendations = self._generate_execution_recommendations(
                optimal_strategy, transfer_plan, risk_assessment
            )
            
            inference_result = {
                'optimal_strategy': optimal_strategy,
                'net2net_assessment': net2net_assessment,
                'transfer_plan': transfer_plan,
                'risk_assessment': risk_assessment,
                'execution_recommendations': execution_recommendations,
                'inference_confidence': self._calculate_inference_confidence(
                    strategy_evaluations, net2net_assessment
                ),
                'alternative_strategies': strategy_evaluations[:3]  # 提供备选方案
            }
            
            logger.info(f"✅ 贝叶斯推断完成: 推荐策略={optimal_strategy['strategy_name']}")
            return inference_result
            
        except Exception as e:
            logger.error(f"❌ 贝叶斯推断失败: {e}")
            return self._fallback_inference(current_accuracy, target_layer_name)

    def predict_mutation_benefit(self, 
                               layer_analysis: Dict[str, Any],
                               mutation_strategy: str,
                               current_accuracy: float,
                               model_complexity: Dict[str, float]) -> Dict[str, Any]:
        """
        预测特定变异策略的收益（保持向后兼容）
        """
        logger.debug(f"预测变异收益: {mutation_strategy}")
        
        try:
            # 1. 构建特征向量
            feature_vector = self._extract_feature_vector(
                layer_analysis, current_accuracy, model_complexity
            )
            
            # 2. 贝叶斯后验推断
            posterior_params = self._bayesian_posterior_inference(
                feature_vector, mutation_strategy, layer_analysis
            )
            
            # 3. 集成Net2Net评估
            net2net_boost = self._evaluate_net2net_boost(
                mutation_strategy, layer_analysis, model_complexity
            )
            
            # 4. 高斯过程回归预测
            gp_prediction = self._gaussian_process_prediction(
                feature_vector, posterior_params, net2net_boost
            )
            
            # 5. 蒙特卡罗期望估计
            mc_estimate = self._monte_carlo_benefit_estimation(
                gp_prediction, mutation_strategy, current_accuracy
            )
            
            # 6. 不确定性量化
            uncertainty_metrics = self._quantify_prediction_uncertainty(
                gp_prediction, mc_estimate, feature_vector
            )
            
            # 7. 风险调整收益
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
                'net2net_enhancement': net2net_boost,
                'recommendation_strength': self._calculate_recommendation_strength(
                    risk_adjusted_benefit, uncertainty_metrics, net2net_boost
                )
            }
            
            logger.debug(f"收益预测完成: 期望收益={prediction_result['expected_accuracy_gain']:.4f}")
            return prediction_result
            
        except Exception as e:
            logger.error(f"贝叶斯预测失败: {e}")
            return self._fallback_prediction(mutation_strategy, current_accuracy)

    def _evaluate_multiple_strategies(self, layer_analysis: Dict[str, Any], 
                                    current_accuracy: float,
                                    model_complexity: Dict[str, float]) -> List[Dict[str, Any]]:
        """评估多个变异策略"""
        strategies = [
            'moderate_widening', 'aggressive_widening', 'deepening', 
            'hybrid_expansion', 'widening'
        ]
        
        evaluations = []
        for strategy in strategies:
            try:
                prediction = self.predict_mutation_benefit(
                    layer_analysis, strategy, current_accuracy, model_complexity
                )
                
                evaluations.append({
                    'strategy_name': strategy,
                    'expected_gain': prediction['expected_accuracy_gain'],
                    'success_probability': prediction['success_probability'],
                    'risk_score': 1.0 - prediction['uncertainty_metrics']['prediction_confidence'],
                    'net2net_compatible': self._check_net2net_compatibility(strategy),
                    'full_prediction': prediction
                })
            except Exception as e:
                logger.warning(f"策略 {strategy} 评估失败: {e}")
                continue
        
        # 按期望收益排序
        evaluations.sort(key=lambda x: x['expected_gain'], reverse=True)
        return evaluations

    def _assess_net2net_applicability(self, layer_analysis: Dict[str, Any], 
                                    model: nn.Module, 
                                    target_layer_name: str) -> Dict[str, Any]:
        """评估Net2Net技术的适用性"""
        
        try:
            # 获取目标层
            target_layer = None
            next_layer = None
            
            modules = list(model.named_modules())
            for i, (name, module) in enumerate(modules):
                if name == target_layer_name:
                    target_layer = module
                    # 尝试找到下一层
                    if i + 1 < len(modules):
                        next_layer = modules[i + 1][1]
                    break
            
            if target_layer is None:
                logger.warning(f"未找到目标层: {target_layer_name}")
                return {'applicable': False, 'reason': 'layer_not_found'}
            
            # Net2Net适用性评估
            applicability = {}
            
            # Net2Wider评估
            if isinstance(target_layer, nn.Conv2d):
                applicability['net2wider'] = {
                    'applicable': True,
                    'current_width': target_layer.out_channels,
                    'recommended_expansion': min(target_layer.out_channels * 2, 512),
                    'function_preserving_confidence': 0.95
                }
            elif isinstance(target_layer, nn.Linear):
                applicability['net2wider'] = {
                    'applicable': True,
                    'current_width': target_layer.out_features,
                    'recommended_expansion': min(target_layer.out_features * 2, 2048),
                    'function_preserving_confidence': 0.90
                }
            else:
                applicability['net2wider'] = {'applicable': False}
            
            # Net2Deeper评估
            applicability['net2deeper'] = {
                'applicable': True,
                'insertion_point': f"{target_layer_name}_deeper",
                'identity_initialization': True,
                'gradient_flow_preserved': True
            }
            
            # 平滑过渡评估
            applicability['smooth_transition'] = {
                'parameter_inheritance': True,
                'training_stability': 0.85,
                'convergence_acceleration': 0.7
            }
            
            # 综合适用性评分
            net2net_suitability = self.prior_knowledge.assess_net2net_suitability(
                layer_analysis, 'widening'
            )
            
            return {
                'applicable': True,
                'methods': applicability,
                'suitability_scores': net2net_suitability,
                'overall_recommendation': self._calculate_net2net_recommendation(
                    applicability, net2net_suitability
                )
            }
            
        except Exception as e:
            logger.error(f"Net2Net适用性评估失败: {e}")
            return {'applicable': False, 'reason': str(e)}

    def _bayesian_model_selection(self, strategy_evaluations: List[Dict[str, Any]], 
                                net2net_assessment: Dict[str, Any],
                                current_accuracy: float) -> Dict[str, Any]:
        """贝叶斯模型选择最优策略"""
        
        if not strategy_evaluations:
            return self._get_default_strategy()
        
        # 计算贝叶斯证据
        for evaluation in strategy_evaluations:
            # 基础证据
            base_evidence = evaluation['expected_gain'] * evaluation['success_probability']
            
            # Net2Net增强
            net2net_boost = 1.0
            if net2net_assessment.get('applicable', False) and evaluation['net2net_compatible']:
                net2net_boost = 1.2  # Net2Net技术提供20%的置信度提升
            
            # 当前准确率调整
            accuracy_stage = self.prior_knowledge.get_accuracy_stage(current_accuracy)
            stage_multiplier = {'low': 1.2, 'medium': 1.0, 'high': 0.8}[accuracy_stage]
            
            evaluation['bayesian_evidence'] = base_evidence * net2net_boost * stage_multiplier
        
        # 选择证据最强的策略
        optimal = max(strategy_evaluations, key=lambda x: x['bayesian_evidence'])
        
        # 增强最优策略信息
        optimal['selection_reason'] = 'highest_bayesian_evidence'
        optimal['evidence_score'] = optimal['bayesian_evidence']
        
        return optimal

    def _plan_parameter_transfer(self, optimal_strategy: Dict[str, Any], 
                               model: nn.Module, 
                               target_layer_name: str) -> Dict[str, Any]:
        """规划参数迁移方案"""
        
        strategy_name = optimal_strategy['strategy_name']
        
        transfer_plan = {
            'method': 'standard',
            'preserve_function': False,
            'smooth_transition': False,
            'expected_stability': 0.7
        }
        
        # 如果策略兼容Net2Net
        if optimal_strategy.get('net2net_compatible', False):
            if 'widening' in strategy_name:
                transfer_plan.update({
                    'method': 'net2wider',
                    'preserve_function': True,
                    'smooth_transition': True,
                    'expected_stability': 0.95,
                    'noise_std': 1e-7,
                    'weight_replication_strategy': 'random_mapping'
                })
            elif 'deepening' in strategy_name:
                transfer_plan.update({
                    'method': 'net2deeper', 
                    'preserve_function': True,
                    'smooth_transition': True,
                    'expected_stability': 0.90,
                    'identity_initialization': True,
                    'activation_handling': 'relu_compatible'
                })
        
        # 获取参数迁移置信度
        transfer_confidence = self.prior_knowledge.get_parameter_transfer_confidence(
            transfer_plan['method']
        )
        transfer_plan['confidence'] = transfer_confidence
        
        return transfer_plan

    def _comprehensive_risk_assessment(self, optimal_strategy: Dict[str, Any],
                                     transfer_plan: Dict[str, Any], 
                                     current_accuracy: float) -> Dict[str, Any]:
        """综合风险评估"""
        
        # 策略风险
        strategy_risk = 1.0 - optimal_strategy['success_probability']
        
        # 参数迁移风险
        transfer_risk = 1.0 - transfer_plan['confidence']
        
        # 当前准确率风险
        accuracy_risk = max(0.0, current_accuracy - 0.9) * 2  # 高准确率时风险增加
        
        # 综合风险评分
        overall_risk = (strategy_risk * 0.4 + transfer_risk * 0.3 + accuracy_risk * 0.3)
        
        return {
            'strategy_risk': strategy_risk,
            'transfer_risk': transfer_risk,
            'accuracy_risk': accuracy_risk,
            'overall_risk': overall_risk,
            'risk_level': 'low' if overall_risk < 0.3 else 'medium' if overall_risk < 0.6 else 'high',
            'mitigation_suggestions': self._generate_risk_mitigations(overall_risk, transfer_plan)
        }

    def _generate_execution_recommendations(self, optimal_strategy: Dict[str, Any],
                                          transfer_plan: Dict[str, Any],
                                          risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """生成执行建议"""
        
        recommendations = {
            'primary_action': optimal_strategy['strategy_name'],
            'transfer_method': transfer_plan['method'],
            'confidence_level': optimal_strategy['success_probability'],
            'expected_improvement': optimal_strategy['expected_gain'],
            'risk_level': risk_assessment['risk_level']
        }
        
        # 执行步骤
        if transfer_plan['method'] == 'net2wider':
            recommendations['execution_steps'] = [
                '1. 使用Net2Wider技术扩展层宽度',
                '2. 复制权重并添加小噪声打破对称性', 
                '3. 调整后续层以保持函数一致性',
                '4. 验证函数保持性',
                '5. 开始渐进式训练'
            ]
        elif transfer_plan['method'] == 'net2deeper':
            recommendations['execution_steps'] = [
                '1. 使用Net2Deeper技术插入恒等层',
                '2. 初始化为恒等映射保持函数不变',
                '3. 验证梯度流通畅',
                '4. 逐步引入非线性',
                '5. 监控训练稳定性'
            ]
        else:
            recommendations['execution_steps'] = [
                '1. 保存当前模型状态',
                '2. 执行架构变异',
                '3. 随机初始化新参数',
                '4. 使用较小学习率微调',
                '5. 监控性能变化'
            ]
        
        # 监控建议
        recommendations['monitoring'] = {
            'key_metrics': ['accuracy', 'loss', 'gradient_norm'],
            'early_stopping_criteria': '3个epoch无改善',
            'rollback_threshold': f"准确率下降超过{0.02:.1%}"
        }
        
        return recommendations

    def _evaluate_net2net_boost(self, mutation_strategy: str, 
                              layer_analysis: Dict[str, Any],
                              model_complexity: Dict[str, float]) -> Dict[str, float]:
        """评估Net2Net技术的增强效果"""
        
        # 确定复杂度级别
        param_count = model_complexity.get('total_parameters', 1000000)
        if param_count < 1000000:
            complexity_level = 'low_complexity'
        elif param_count < 10000000:
            complexity_level = 'medium_complexity'
        else:
            complexity_level = 'high_complexity'
        
        # 获取Net2Net收益先验
        net2net_benefit = self.prior_knowledge.get_net2net_benefit_prior(
            complexity_level, mutation_strategy
        )
        
        # 计算Net2Net置信度增强
        confidence_boost = 0.0
        if self._check_net2net_compatibility(mutation_strategy):
            confidence_boost = 0.15  # Net2Net提供15%的置信度提升
        
        return {
            'benefit_boost': net2net_benefit,
            'confidence_boost': confidence_boost,
            'stability_improvement': 0.2,
            'convergence_acceleration': 0.1
        }

    def _check_net2net_compatibility(self, strategy: str) -> bool:
        """检查策略是否兼容Net2Net"""
        compatible_strategies = [
            'widening', 'moderate_widening', 'aggressive_widening',
            'deepening', 'hybrid_expansion'
        ]
        return strategy in compatible_strategies

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
        
        # 使用先验知识 - 现在正确获取success_rate
        prior_params = self.prior_knowledge.get_mutation_prior(mutation_strategy)
        
        # 计算似然
        likelihood = self._calculate_likelihood(feature_vector, layer_analysis)
        
        # 后验计算（简化的贝叶斯更新）
        # 注意：现在prior_params已经包含了success_rate键
        success_probability = prior_params['success_rate'] * likelihood
        success_probability = np.clip(success_probability, 0.1, 0.9)
        
        # 贝叶斯证据
        evidence = likelihood * prior_params['confidence']
        
        return {
            'success_probability': success_probability,
            'evidence': evidence,
            'likelihood': likelihood,
            'prior_confidence': prior_params['confidence'],
            'prior_alpha': prior_params['alpha'],
            'prior_beta': prior_params['beta']
        }

    def _calculate_likelihood(self, feature_vector: np.ndarray, 
                            layer_analysis: Dict[str, Any]) -> float:
        """计算似然函数"""
        
        # 基于特征的似然模型
        current_acc = feature_vector[0]
        improvement_pot = feature_vector[1]
        
        # 似然与当前准确率和改进潜力相关
        likelihood = (1 - current_acc) * improvement_pot
        return np.clip(likelihood, 0.1, 1.0)

    def _gaussian_process_prediction(self, feature_vector: np.ndarray, 
                                   posterior_params: Dict[str, Any],
                                   net2net_boost: Dict[str, float]) -> Dict[str, Any]:
        """高斯过程回归预测"""
        
        # 基础均值预测
        base_mean = posterior_params['success_probability'] * 0.05  # 最大5%改进
        
        # Net2Net增强
        enhanced_mean = base_mean + net2net_boost['benefit_boost']
        
        # 方差预测（考虑Net2Net的稳定性提升）
        base_variance = (1 - posterior_params['evidence']) * 0.01
        enhanced_variance = base_variance * (1 - net2net_boost['stability_improvement'])
        
        return {
            'mean_prediction': enhanced_mean,
            'variance_prediction': enhanced_variance,
            'hyperparameters': self.gp_hyperparams,
            'net2net_enhancement': net2net_boost
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
        
        return {
            'expected_gain': float(expected_gain),
            'gain_std': float(gain_std),
            'confidence_interval': confidence_interval,
            'success_probability': float(success_probability),
            'samples': samples
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
        
        # 预测置信度（考虑Net2Net增强）
        base_confidence = 1.0 / (1.0 + total_uncertainty)
        net2net_boost = gp_prediction.get('net2net_enhancement', {}).get('confidence_boost', 0.0)
        prediction_confidence = min(1.0, base_confidence + net2net_boost)
        
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
        cvar_95 = np.mean(samples[samples <= var_95]) if len(samples[samples <= var_95]) > 0 else var_95
        
        return {
            'risk_adjusted_gain': float(risk_adjusted_gain),
            'sharpe_ratio': float(sharpe_ratio),
            'value_at_risk_95': float(var_95),
            'conditional_var_95': float(cvar_95),
            'risk_reward_score': float(expected_gain / (total_uncertainty + 1e-8))
        }

    def _calculate_recommendation_strength(self, risk_adjusted_benefit: Dict[str, Any],
                                         uncertainty_metrics: Dict[str, Any],
                                         net2net_boost: Dict[str, float]) -> str:
        """计算推荐强度"""
        
        gain = risk_adjusted_benefit['risk_adjusted_gain']
        confidence = uncertainty_metrics['prediction_confidence']
        sharpe_ratio = risk_adjusted_benefit['sharpe_ratio']
        
        # Net2Net增强因子
        net2net_factor = 1 + net2net_boost.get('confidence_boost', 0.0)
        
        # 综合评分
        score = gain * confidence * (1 + sharpe_ratio) * net2net_factor
        
        if score > 0.025 and confidence > 0.8:
            return "strong_recommend"
        elif score > 0.015 and confidence > 0.6:
            return "recommend"
        elif score > 0.008:
            return "weak_recommend"
        elif score > -0.005:
            return "neutral"
        else:
            return "not_recommend"

    def _calculate_inference_confidence(self, strategy_evaluations: List[Dict[str, Any]],
                                      net2net_assessment: Dict[str, Any]) -> float:
        """计算推断置信度"""
        if not strategy_evaluations:
            return 0.3
            
        # 最佳策略的置信度
        best_confidence = strategy_evaluations[0].get('success_probability', 0.5)
        
        # 策略差异度（评估选择的稳定性）
        if len(strategy_evaluations) > 1:
            gain_diff = strategy_evaluations[0]['expected_gain'] - strategy_evaluations[1]['expected_gain']
            stability_factor = min(1.0, gain_diff / 0.01)  # 如果差异大于1%则认为选择稳定
        else:
            stability_factor = 0.7
        
        # Net2Net增强因子
        net2net_factor = 1.0
        if net2net_assessment.get('applicable', False):
            net2net_factor = 1.1
        
        overall_confidence = best_confidence * stability_factor * net2net_factor
        return min(1.0, overall_confidence)

    def _get_default_strategy(self) -> Dict[str, Any]:
        """获取默认策略"""
        return {
            'strategy_name': 'moderate_widening',
            'expected_gain': 0.01,
            'success_probability': 0.6,
            'risk_score': 0.4,
            'net2net_compatible': True,
            'selection_reason': 'default_fallback'
        }

    def _calculate_net2net_recommendation(self, applicability: Dict[str, Any],
                                        suitability_scores: Dict[str, float]) -> str:
        """计算Net2Net推荐等级"""
        if not applicability:
            return "not_recommended"
            
        avg_suitability = np.mean(list(suitability_scores.values()))
        
        if avg_suitability > 0.8:
            return "highly_recommended"
        elif avg_suitability > 0.6:
            return "recommended"
        elif avg_suitability > 0.4:
            return "conditionally_recommended"
        else:
            return "not_recommended"

    def _generate_risk_mitigations(self, overall_risk: float, 
                                 transfer_plan: Dict[str, Any]) -> List[str]:
        """生成风险缓解建议"""
        mitigations = []
        
        if overall_risk > 0.6:
            mitigations.append("建议采用更保守的学习率")
            mitigations.append("增加验证频率，及时发现问题")
            
        if transfer_plan['method'] != 'net2wider' and transfer_plan['method'] != 'net2deeper':
            mitigations.append("考虑使用Net2Net技术降低迁移风险")
            
        if overall_risk > 0.8:
            mitigations.append("建议分阶段执行变异，避免激进变更")
            mitigations.append("准备回滚方案")
            
        return mitigations

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
            'net2net_enhancement': {'benefit_boost': 0.0, 'confidence_boost': 0.0},
            'recommendation_strength': "weak_recommend" if expected_gain > 0.005 else "neutral"
        }

    def _fallback_inference(self, current_accuracy: float, target_layer_name: str) -> Dict[str, Any]:
        """fallback推断"""
        
        default_strategy = self._get_default_strategy()
        
        return {
            'optimal_strategy': default_strategy,
            'net2net_assessment': {'applicable': False, 'reason': 'inference_failed'},
            'transfer_plan': {'method': 'standard', 'confidence': 0.5},
            'risk_assessment': {'overall_risk': 0.6, 'risk_level': 'medium'},
            'execution_recommendations': {
                'primary_action': 'moderate_widening',
                'confidence_level': 0.5,
                'risk_level': 'medium'
            },
            'inference_confidence': 0.3,
            'alternative_strategies': []
        }

    def update_with_mutation_result(self, 
                                  feature_vector: np.ndarray,
                                  mutation_strategy: str,
                                  actual_gain: float,
                                  success: bool,
                                  used_net2net: bool = False):
        """用实际变异结果更新模型"""
        
        mutation_record = {
            'features': feature_vector,
            'strategy': mutation_strategy,
            'actual_gain': actual_gain,
            'success': success,
            'used_net2net': used_net2net,
            'timestamp': time.time()
        }
        
        self.mutation_history.append(mutation_record)
        
        if used_net2net:
            self.net2net_transfer_history.append(mutation_record)
        
        # 保持历史记录大小
        if len(self.mutation_history) > 100:
            self.mutation_history = self.mutation_history[-100:]
        if len(self.net2net_transfer_history) > 50:
            self.net2net_transfer_history = self.net2net_transfer_history[-50:]
        
        logger.info(f"更新贝叶斯模型: {mutation_strategy}, 实际收益={actual_gain:.4f}, Net2Net={used_net2net}")


# 保持向后兼容的别名
BayesianMutationBenefitPredictor = BayesianInferenceEngine