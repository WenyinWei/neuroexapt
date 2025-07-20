"""
增强贝叶斯形态发生引擎

基于贝叶斯推断、高斯过程回归和蒙特卡罗采样的智能架构变异引擎
解决现有系统变异决策过于保守的问题
"""

from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
import logging
from scipy import stats
from scipy.optimize import minimize
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class BayesianMorphogenesisEngine:
    """
    增强贝叶斯形态发生引擎
    
    核心改进：
    1. 贝叶斯网络建模架构变异的成功概率
    2. 高斯过程回归预测性能改进
    3. 蒙特卡罗采样量化不确定性
    4. 期望效用最大化的决策理论
    5. 在线学习更新先验分布
    """
    
    def __init__(self):
        # 贝叶斯先验分布参数
        self.mutation_priors = {
            'width_expansion': {'alpha': 15, 'beta': 5},      # 较高成功率先验
            'depth_expansion': {'alpha': 12, 'beta': 8},      # 中高成功率先验
            'attention_enhancement': {'alpha': 10, 'beta': 10}, # 中等成功率先验
            'residual_connection': {'alpha': 18, 'beta': 2},   # 很高成功率先验
            'batch_norm_insertion': {'alpha': 20, 'beta': 5}, # 很高成功率先验
            'parallel_division': {'alpha': 8, 'beta': 12},    # 中低成功率但高收益
            'serial_division': {'alpha': 12, 'beta': 8},      # 中高成功率先验
            'channel_attention': {'alpha': 10, 'beta': 10},   # 中等成功率先验
            'layer_norm': {'alpha': 16, 'beta': 4},          # 高成功率先验
            'information_enhancement': {'alpha': 9, 'beta': 11} # 中等成功率先验
        }
        
        # 高斯过程超参数
        self.gp_params = {
            'length_scale': 1.0,
            'signal_variance': 1.0,
            'noise_variance': 0.1,
            'mean_function': 0.0
        }
        
        # 历史数据存储
        self.mutation_history = deque(maxlen=100)  # 变异历史
        self.performance_history = deque(maxlen=50)  # 性能历史
        self.architecture_features = deque(maxlen=100)  # 架构特征历史
        
        # 效用函数参数
        self.utility_params = {
            'accuracy_weight': 1.0,        # 准确率权重
            'efficiency_weight': 0.3,      # 效率权重
            'risk_aversion': 0.2,          # 风险厌恶程度
            'exploration_bonus': 0.1       # 探索奖励
        }
        
        # 不确定性量化参数
        self.mc_samples = 500
        self.confidence_levels = [0.68, 0.95, 0.99]
        
        # 动态阈值（更积极）
        self.dynamic_thresholds = {
            'min_expected_improvement': 0.002,   # 最小期望改进（0.2%）
            'max_acceptable_risk': 0.4,         # 最大可接受风险
            'confidence_threshold': 0.3,        # 置信度阈值（更低）
            'exploration_threshold': 0.25       # 探索阈值
        }
    
    def bayesian_morphogenesis_analysis(self, 
                                      model: nn.Module,
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """
        贝叶斯形态发生分析
        
        整合多种贝叶斯方法进行智能决策
        """
        logger.info("🧠 启动增强贝叶斯形态发生分析")
        
        try:
            # 1. 提取架构特征
            arch_features = self._extract_architecture_features(model, context)
            
            # 2. 识别候选变异点（使用更积极的检测）
            candidates = self._aggressive_candidate_detection(model, context, arch_features)
            
            # 3. 贝叶斯变异成功率推断
            success_probabilities = self._bayesian_success_inference(candidates, arch_features)
            
            # 4. 高斯过程性能改进预测
            improvement_predictions = self._gaussian_process_prediction(candidates, arch_features)
            
            # 5. 蒙特卡罗不确定性量化
            uncertainty_analysis = self._monte_carlo_uncertainty(
                candidates, success_probabilities, improvement_predictions
            )
            
            # 6. 期望效用最大化决策
            optimal_decisions = self._expected_utility_maximization(
                candidates, success_probabilities, improvement_predictions, uncertainty_analysis
            )
            
            # 7. 生成执行建议
            execution_plan = self._generate_bayesian_execution_plan(
                optimal_decisions, uncertainty_analysis, context
            )
            
            # 8. 更新历史数据
            self._update_bayesian_history(arch_features, candidates, context)
            
            return {
                'bayesian_analysis': {
                    'candidates_found': len(candidates),
                    'success_probabilities': success_probabilities,
                    'improvement_predictions': improvement_predictions,
                    'uncertainty_analysis': uncertainty_analysis,
                    'decision_confidence': self._calculate_overall_confidence(uncertainty_analysis)
                },
                'optimal_decisions': optimal_decisions,
                'execution_plan': execution_plan,
                'bayesian_insights': {
                    'most_promising_mutation': optimal_decisions[0] if optimal_decisions else None,
                    'expected_performance_gain': self._calculate_expected_gain(optimal_decisions),
                    'risk_assessment': self._comprehensive_risk_assessment(optimal_decisions)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 贝叶斯分析失败: {e}")
            return self._fallback_bayesian_analysis()
    
    def _extract_architecture_features(self, model: nn.Module, context: Dict[str, Any]) -> Dict[str, float]:
        """提取架构特征向量"""
        
        features = {}
        
        # 基础架构统计
        total_params = sum(p.numel() for p in model.parameters())
        features['total_parameters'] = float(total_params)
        features['model_depth'] = len(list(model.modules()))
        
        # 层类型分布
        layer_types = defaultdict(int)
        for module in model.modules():
            layer_types[type(module).__name__] += 1
        
        features['conv_layers'] = layer_types.get('Conv2d', 0)
        features['linear_layers'] = layer_types.get('Linear', 0)
        features['norm_layers'] = layer_types.get('BatchNorm2d', 0) + layer_types.get('LayerNorm', 0)
        features['activation_layers'] = layer_types.get('ReLU', 0) + layer_types.get('GELU', 0)
        
        # 性能指标
        performance_history = context.get('performance_history', [])
        if performance_history:
            features['current_accuracy'] = performance_history[-1]
            features['accuracy_trend'] = self._calculate_trend(performance_history[-5:])
            features['accuracy_variance'] = np.var(performance_history[-10:]) if len(performance_history) >= 10 else 0
        
        # 训练状态
        features['current_epoch'] = context.get('epoch', 0)
        features['train_loss'] = context.get('train_loss', 1.0)
        features['learning_rate'] = context.get('learning_rate', 0.1)
        
        # 激活统计
        activations = context.get('activations', {})
        if activations:
            features['avg_activation_magnitude'] = np.mean([
                torch.mean(torch.abs(act)).item() for act in activations.values()
            ])
            features['activation_sparsity'] = np.mean([
                (act == 0).float().mean().item() for act in activations.values()
            ])
        
        # 梯度统计
        gradients = context.get('gradients', {})
        if gradients:
            features['avg_gradient_norm'] = np.mean([
                torch.norm(grad).item() for grad in gradients.values()
            ])
            features['gradient_variance'] = np.var([
                torch.norm(grad).item() for grad in gradients.values()
            ])
        
        return features
    
    def _aggressive_candidate_detection(self, 
                                      model: nn.Module, 
                                      context: Dict[str, Any],
                                      arch_features: Dict[str, float]) -> List[Dict[str, Any]]:
        """积极的候选点检测（更容易发现候选点）"""
        
        candidates = []
        activations = context.get('activations', {})
        gradients = context.get('gradients', {})
        
        # 降低检测阈值，更容易发现候选点
        bottleneck_threshold = 0.3    # 从0.5降低到0.3
        improvement_threshold = 0.2   # 从0.3降低到0.2
        
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                continue
            
            candidate = {
                'layer_name': name,
                'layer_type': type(module).__name__,
                'module': module,
                'bottleneck_indicators': {},
                'improvement_signals': {},
                'mutation_suitability': {}
            }
            
            # 1. 参数利用率分析（更敏感）
            param_utilization = self._analyze_parameter_utilization(module, activations.get(name))
            candidate['bottleneck_indicators']['parameter_utilization'] = param_utilization
            
            # 2. 信息流效率分析（更敏感）
            if name in activations:
                info_efficiency = self._analyze_information_efficiency(activations[name])
                candidate['bottleneck_indicators']['information_efficiency'] = info_efficiency
            
            # 3. 梯度质量分析（更敏感）
            if name in gradients:
                gradient_quality = self._analyze_gradient_quality(gradients[name])
                candidate['bottleneck_indicators']['gradient_quality'] = gradient_quality
            
            # 4. 架构匹配度分析
            arch_mismatch = self._analyze_architecture_mismatch(module, arch_features)
            candidate['bottleneck_indicators']['architecture_mismatch'] = arch_mismatch
            
            # 综合评分（更容易通过）
            bottleneck_score = np.mean(list(candidate['bottleneck_indicators'].values()))
            
            if bottleneck_score > bottleneck_threshold:
                # 分析变异适用性
                self._analyze_mutation_suitability(candidate, arch_features)
                
                # 计算改进信号
                candidate['improvement_potential'] = min(1.0, bottleneck_score * 1.5)
                candidate['urgency_score'] = self._calculate_urgency_score(candidate, context)
                
                candidates.append(candidate)
                logger.info(f"✅ 发现候选层: {name}, 瓶颈分数: {bottleneck_score:.3f}")
        
        # 即使没有明显瓶颈，也根据性能态势添加探索性候选
        if len(candidates) < 2:
            exploration_candidates = self._generate_exploration_candidates(model, arch_features, context)
            candidates.extend(exploration_candidates)
        
        # 按潜力排序
        candidates.sort(key=lambda x: x['improvement_potential'], reverse=True)
        
        logger.info(f"🎯 积极检测发现: {len(candidates)}个候选点")
        return candidates
    
    def _bayesian_success_inference(self, 
                                  candidates: List[Dict[str, Any]], 
                                  arch_features: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """贝叶斯变异成功率推断"""
        
        success_probs = {}
        
        for candidate in candidates:
            layer_name = candidate['layer_name']
            success_probs[layer_name] = {}
            
            for mutation_type in candidate.get('mutation_suitability', {}):
                # 获取先验分布
                prior = self.mutation_priors.get(mutation_type, {'alpha': 5, 'beta': 5})
                
                # 基于历史数据更新先验
                updated_prior = self._update_prior_with_history(mutation_type, arch_features)
                
                # 计算后验概率
                posterior_prob = self._calculate_posterior_success_probability(
                    updated_prior, candidate, mutation_type, arch_features
                )
                
                success_probs[layer_name][mutation_type] = posterior_prob
        
        return success_probs
    
    def _gaussian_process_prediction(self, 
                                   candidates: List[Dict[str, Any]], 
                                   arch_features: Dict[str, float]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """高斯过程性能改进预测"""
        
        predictions = {}
        
        for candidate in candidates:
            layer_name = candidate['layer_name']
            predictions[layer_name] = {}
            
            for mutation_type in candidate.get('mutation_suitability', {}):
                # 构建特征向量
                feature_vector = self._build_gp_feature_vector(candidate, mutation_type, arch_features)
                
                # 高斯过程预测
                mean_improvement, variance = self._gp_predict_improvement(
                    feature_vector, mutation_type
                )
                
                # 计算置信区间
                std_dev = np.sqrt(variance)
                confidence_intervals = {}
                for confidence in self.confidence_levels:
                    z_score = stats.norm.ppf((1 + confidence) / 2)
                    ci_lower = mean_improvement - z_score * std_dev
                    ci_upper = mean_improvement + z_score * std_dev
                    confidence_intervals[f'{int(confidence*100)}%'] = (ci_lower, ci_upper)
                
                predictions[layer_name][mutation_type] = {
                    'mean_improvement': mean_improvement,
                    'variance': variance,
                    'std_dev': std_dev,
                    'confidence_intervals': confidence_intervals
                }
        
        return predictions
    
    def _monte_carlo_uncertainty(self, 
                               candidates: List[Dict[str, Any]],
                               success_probabilities: Dict[str, Dict[str, float]],
                               improvement_predictions: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """蒙特卡罗不确定性量化"""
        
        mc_results = {}
        
        for candidate in candidates:
            layer_name = candidate['layer_name']
            mc_results[layer_name] = {}
            
            for mutation_type in candidate.get('mutation_suitability', {}):
                success_prob = success_probabilities[layer_name][mutation_type]
                improvement_pred = improvement_predictions[layer_name][mutation_type]
                
                # 蒙特卡罗采样
                mc_samples = []
                for _ in range(self.mc_samples):
                    # 采样成功/失败
                    success = np.random.random() < success_prob
                    
                    if success:
                        # 从预测分布中采样改进值
                        improvement = np.random.normal(
                            improvement_pred['mean_improvement'],
                            improvement_pred['std_dev']
                        )
                    else:
                        # 失败情况下的性能损失
                        improvement = np.random.normal(-0.01, 0.005)  # 小幅性能下降
                    
                    mc_samples.append(improvement)
                
                mc_samples = np.array(mc_samples)
                
                # 统计分析
                mc_results[layer_name][mutation_type] = {
                    'expected_value': np.mean(mc_samples),
                    'variance': np.var(mc_samples),
                    'percentiles': {
                        '5%': np.percentile(mc_samples, 5),
                        '25%': np.percentile(mc_samples, 25),
                        '50%': np.percentile(mc_samples, 50),
                        '75%': np.percentile(mc_samples, 75),
                        '95%': np.percentile(mc_samples, 95)
                    },
                    'probability_positive': np.mean(mc_samples > 0),
                    'value_at_risk_5%': np.percentile(mc_samples, 5),  # VaR
                    'expected_shortfall_5%': np.mean(mc_samples[mc_samples <= np.percentile(mc_samples, 5)])
                }
        
        return mc_results
    
    def _expected_utility_maximization(self,
                                     candidates: List[Dict[str, Any]],
                                     success_probabilities: Dict[str, Dict[str, float]],
                                     improvement_predictions: Dict[str, Dict[str, Dict[str, float]]],
                                     uncertainty_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """期望效用最大化决策"""
        
        decisions = []
        
        for candidate in candidates:
            layer_name = candidate['layer_name']
            
            for mutation_type in candidate.get('mutation_suitability', {}):
                # 计算期望效用
                expected_utility = self._calculate_expected_utility(
                    layer_name, mutation_type, 
                    success_probabilities, improvement_predictions, uncertainty_analysis
                )
                
                # 计算决策置信度
                decision_confidence = self._calculate_decision_confidence(
                    layer_name, mutation_type, uncertainty_analysis
                )
                
                # 检查是否满足决策阈值
                if (expected_utility > self.dynamic_thresholds['min_expected_improvement'] and
                    decision_confidence > self.dynamic_thresholds['confidence_threshold']):
                    
                    decision = {
                        'layer_name': layer_name,
                        'mutation_type': mutation_type,
                        'expected_utility': expected_utility,
                        'decision_confidence': decision_confidence,
                        'success_probability': success_probabilities[layer_name][mutation_type],
                        'expected_improvement': improvement_predictions[layer_name][mutation_type]['mean_improvement'],
                        'risk_metrics': {
                            'value_at_risk': uncertainty_analysis[layer_name][mutation_type]['value_at_risk_5%'],
                            'expected_shortfall': uncertainty_analysis[layer_name][mutation_type]['expected_shortfall_5%'],
                            'probability_positive': uncertainty_analysis[layer_name][mutation_type]['probability_positive']
                        },
                        'rationale': self._generate_decision_rationale(
                            candidate, mutation_type, expected_utility, decision_confidence
                        )
                    }
                    
                    decisions.append(decision)
        
        # 按期望效用排序
        decisions.sort(key=lambda x: x['expected_utility'], reverse=True)
        
        # 选择最优决策（考虑多样性）
        selected_decisions = self._select_diverse_decisions(decisions)
        
        logger.info(f"🎯 期望效用最大化: 选择{len(selected_decisions)}个最优决策")
        return selected_decisions
    
    def _generate_bayesian_execution_plan(self,
                                        optimal_decisions: List[Dict[str, Any]],
                                        uncertainty_analysis: Dict[str, Any],
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """生成贝叶斯执行计划"""
        
        if not optimal_decisions:
            return {
                'execute': False,
                'reason': 'no_viable_decisions_after_bayesian_analysis',
                'recommendations': [
                    'continue_training_with_current_architecture',
                    'adjust_hyperparameters',
                    'consider_data_augmentation'
                ]
            }
        
        primary_decision = optimal_decisions[0]
        
        execution_plan = {
            'execute': True,
            'bayesian_strategy': {
                'primary_decision': primary_decision,
                'expected_improvement': primary_decision['expected_improvement'],
                'success_probability': primary_decision['success_probability'],
                'decision_confidence': primary_decision['decision_confidence']
            },
            'alternative_strategies': optimal_decisions[1:3] if len(optimal_decisions) > 1 else [],
            'risk_management': {
                'monitoring_metrics': [
                    'accuracy_improvement',
                    'loss_reduction', 
                    'gradient_stability',
                    'computational_efficiency'
                ],
                'early_stopping_criteria': [
                    f"accuracy_drop > {abs(primary_decision['risk_metrics']['value_at_risk']):.3f}",
                    'loss_divergence_detected',
                    'gradient_explosion_detected'
                ],
                'rollback_triggers': [
                    f"performance_below_5th_percentile_for_3_epochs",
                    'critical_model_instability'
                ]
            },
            'adaptive_execution': {
                'success_threshold': primary_decision['expected_improvement'] * 0.5,
                'monitoring_frequency': 'every_epoch',
                'adaptation_strategy': 'bayesian_update_with_new_evidence'
            },
            'uncertainty_tracking': {
                'confidence_evolution': 'track_decision_confidence_over_time',
                'posterior_updates': 'update_priors_based_on_results',
                'next_decision_preparation': 'prepare_for_next_bayesian_cycle'
            }
        }
        
        return execution_plan
    
    # === 辅助方法实现 ===
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势斜率"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
    
    def _analyze_parameter_utilization(self, module: nn.Module, activation: Optional[torch.Tensor]) -> float:
        """分析参数利用率（更敏感的检测）"""
        
        base_score = 0.0
        
        if isinstance(module, nn.Conv2d):
            # 通道数相对充分性
            channel_ratio = module.out_channels / max(16, module.in_channels)  # 降低基准
            if channel_ratio < 0.8:  # 更敏感的阈值
                base_score += 0.6
            
            # 卷积核大小适用性
            if module.kernel_size[0] <= 3:
                base_score += 0.3
                
        elif isinstance(module, nn.Linear):
            # 特征数相对充分性
            feature_ratio = module.out_features / max(32, module.in_features)  # 降低基准
            if feature_ratio < 0.5:  # 更敏感的阈值
                base_score += 0.7
        
        # 如果有激活值，分析激活模式
        if activation is not None:
            try:
                # 激活稀疏性
                sparsity = (activation == 0).float().mean().item()
                if sparsity > 0.7:  # 高稀疏性表明参数未充分利用
                    base_score += 0.4
                
                # 激活分布集中度
                flat_act = activation.flatten()
                if len(flat_act) > 0:
                    std_act = torch.std(flat_act).item()
                    if std_act < 0.1:  # 低方差表明信息不足
                        base_score += 0.3
            except:
                pass
        
        return min(1.0, base_score)
    
    def _analyze_information_efficiency(self, activation: torch.Tensor) -> float:
        """分析信息效率（更敏感的检测）"""
        
        try:
            flat_activation = activation.flatten()
            
            # 有效激活比例
            non_zero_ratio = torch.count_nonzero(flat_activation).float() / flat_activation.numel()
            efficiency_loss = 1 - non_zero_ratio
            
            # 动态范围利用
            activation_range = torch.max(flat_activation) - torch.min(flat_activation)
            if activation_range < 1.0:  # 动态范围不足
                efficiency_loss += 0.3
            
            # 信息熵
            hist = torch.histc(flat_activation, bins=20)
            hist_normalized = hist / (hist.sum() + 1e-10)
            entropy = -torch.sum(hist_normalized * torch.log(hist_normalized + 1e-10))
            max_entropy = np.log(20)
            if entropy / max_entropy < 0.5:  # 熵不足
                efficiency_loss += 0.4
            
            return min(1.0, efficiency_loss)
            
        except Exception:
            return 0.5  # 默认中等效率损失
    
    def _analyze_gradient_quality(self, gradient: torch.Tensor) -> float:
        """分析梯度质量（更敏感的检测）"""
        
        try:
            # 梯度范数
            grad_norm = torch.norm(gradient).item()
            quality_loss = 0.0
            
            # 梯度消失检测
            if grad_norm < 1e-5:
                quality_loss += 0.8
            elif grad_norm < 1e-3:
                quality_loss += 0.4
            
            # 梯度爆炸检测
            if grad_norm > 10.0:
                quality_loss += 0.6
            elif grad_norm > 1.0:
                quality_loss += 0.2
            
            # 梯度分布
            grad_std = torch.std(gradient).item()
            grad_mean = torch.mean(torch.abs(gradient)).item()
            
            if grad_std / (grad_mean + 1e-10) > 10:  # 高方差
                quality_loss += 0.3
            
            return min(1.0, quality_loss)
            
        except Exception:
            return 0.4  # 默认中等质量损失
    
    def _analyze_architecture_mismatch(self, module: nn.Module, arch_features: Dict[str, float]) -> float:
        """分析架构不匹配度"""
        
        mismatch_score = 0.0
        
        # 模型复杂度相对于性能的不匹配
        current_accuracy = arch_features.get('current_accuracy', 0.5)
        total_params = arch_features.get('total_parameters', 1000000)
        
        # 参数效率
        param_efficiency = current_accuracy / (total_params / 1000000)  # 每百万参数的准确率
        if param_efficiency < 0.3:  # 参数效率低
            mismatch_score += 0.4
        
        # 深度vs宽度平衡
        depth = arch_features.get('model_depth', 10)
        conv_layers = arch_features.get('conv_layers', 1)
        if conv_layers > 0 and depth / conv_layers > 10:  # 过深相对于宽度
            mismatch_score += 0.3
        
        return min(1.0, mismatch_score)
    
    def _analyze_mutation_suitability(self, candidate: Dict[str, Any], arch_features: Dict[str, float]):
        """分析变异适用性"""
        
        suitability = {}
        layer_type = candidate['layer_type']
        bottlenecks = candidate['bottleneck_indicators']
        
        # 为每种变异类型计算适用性分数
        mutations_to_check = [
            'width_expansion', 'depth_expansion', 'attention_enhancement',
            'residual_connection', 'batch_norm_insertion', 'parallel_division',
            'serial_division', 'channel_attention', 'layer_norm', 'information_enhancement'
        ]
        
        for mutation in mutations_to_check:
            score = self._calculate_mutation_suitability_score(
                mutation, layer_type, bottlenecks, arch_features
            )
            if score > 0.2:  # 较低的阈值
                suitability[mutation] = score
        
        candidate['mutation_suitability'] = suitability
    
    def _calculate_mutation_suitability_score(self, 
                                            mutation: str,
                                            layer_type: str,
                                            bottlenecks: Dict[str, float],
                                            arch_features: Dict[str, float]) -> float:
        """计算特定变异的适用性分数"""
        
        score = 0.0
        
        # 基于层类型的基础适用性
        layer_compatibility = {
            'Conv2d': {
                'width_expansion': 0.8, 'depth_expansion': 0.6, 'attention_enhancement': 0.7,
                'parallel_division': 0.9, 'channel_attention': 0.8
            },
            'Linear': {
                'width_expansion': 0.9, 'depth_expansion': 0.4, 'serial_division': 0.7,
                'batch_norm_insertion': 0.3, 'layer_norm': 0.8
            }
        }
        
        score += layer_compatibility.get(layer_type, {}).get(mutation, 0.5)
        
        # 基于瓶颈类型的适用性
        if bottlenecks.get('parameter_utilization', 0) > 0.4:
            if mutation in ['width_expansion', 'depth_expansion', 'parallel_division']:
                score += 0.3
        
        if bottlenecks.get('information_efficiency', 0) > 0.4:
            if mutation in ['attention_enhancement', 'channel_attention', 'information_enhancement']:
                score += 0.3
        
        if bottlenecks.get('gradient_quality', 0) > 0.4:
            if mutation in ['residual_connection', 'batch_norm_insertion', 'layer_norm']:
                score += 0.3
        
        # 基于架构特征的适用性
        current_accuracy = arch_features.get('current_accuracy', 0.5)
        if current_accuracy < 0.7:  # 准确率较低时，更激进的变异
            if mutation in ['depth_expansion', 'parallel_division', 'attention_enhancement']:
                score += 0.2
        
        return min(1.0, score)
    
    def _calculate_urgency_score(self, candidate: Dict[str, Any], context: Dict[str, Any]) -> float:
        """计算紧急性分数"""
        
        urgency = 0.0
        
        # 基于性能态势
        performance_history = context.get('performance_history', [])
        if len(performance_history) >= 5:
            recent_trend = self._calculate_trend(performance_history[-5:])
            if recent_trend < -0.001:  # 性能下降
                urgency += 0.6
            elif recent_trend < 0.001:  # 性能停滞
                urgency += 0.4
        
        # 基于瓶颈严重程度
        bottleneck_severity = np.mean(list(candidate['bottleneck_indicators'].values()))
        urgency += bottleneck_severity * 0.4
        
        return min(1.0, urgency)
    
    def _generate_exploration_candidates(self, 
                                       model: nn.Module, 
                                       arch_features: Dict[str, float],
                                       context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成探索性候选点"""
        
        exploration_candidates = []
        
        # 如果性能停滞，添加探索性变异
        performance_history = context.get('performance_history', [])
        if len(performance_history) >= 10:
            recent_improvement = performance_history[-1] - performance_history[-10]
            if recent_improvement < 0.01:  # 性能停滞
                
                # 选择一些随机层进行探索性变异
                all_layers = [(name, module) for name, module in model.named_modules() 
                             if isinstance(module, (nn.Conv2d, nn.Linear))]
                
                if all_layers:
                    # 随机选择1-2层
                    selected_layers = np.random.choice(len(all_layers), 
                                                     size=min(2, len(all_layers)), 
                                                     replace=False)
                    
                    for idx in selected_layers:
                        name, module = all_layers[idx]
                        candidate = {
                            'layer_name': name,
                            'layer_type': type(module).__name__,
                            'module': module,
                            'bottleneck_indicators': {'exploration': 0.5},
                            'improvement_signals': {'exploration_driven': 0.6},
                            'improvement_potential': 0.5,
                            'urgency_score': 0.4,
                            'mutation_suitability': {
                                'width_expansion': 0.6,
                                'attention_enhancement': 0.5,
                                'residual_connection': 0.7
                            }
                        }
                        exploration_candidates.append(candidate)
        
        return exploration_candidates
    
    def _update_prior_with_history(self, mutation_type: str, arch_features: Dict[str, float]) -> Dict[str, float]:
        """基于历史数据更新先验"""
        
        prior = self.mutation_priors.get(mutation_type, {'alpha': 5, 'beta': 5}).copy()
        
        # 基于历史变异数据更新
        relevant_history = [h for h in self.mutation_history 
                           if h.get('mutation_type') == mutation_type]
        
        if relevant_history:
            successes = sum(1 for h in relevant_history if h.get('success', False))
            failures = len(relevant_history) - successes
            
            # 贝叶斯更新
            prior['alpha'] += successes
            prior['beta'] += failures
        
        return prior
    
    def _calculate_posterior_success_probability(self,
                                               prior: Dict[str, float],
                                               candidate: Dict[str, Any],
                                               mutation_type: str,
                                               arch_features: Dict[str, float]) -> float:
        """计算后验成功概率"""
        
        # Beta分布的期望值
        alpha = prior['alpha']
        beta = prior['beta']
        base_prob = alpha / (alpha + beta)
        
        # 基于当前情况调整
        adjustment_factors = []
        
        # 瓶颈严重程度调整
        bottleneck_severity = np.mean(list(candidate['bottleneck_indicators'].values()))
        adjustment_factors.append(bottleneck_severity * 0.3)
        
        # 变异适用性调整
        suitability = candidate['mutation_suitability'].get(mutation_type, 0.5)
        adjustment_factors.append(suitability * 0.2)
        
        # 架构特征调整
        current_accuracy = arch_features.get('current_accuracy', 0.5)
        if current_accuracy < 0.6:  # 低准确率时变异更容易成功
            adjustment_factors.append(0.1)
        
        # 综合调整
        total_adjustment = sum(adjustment_factors)
        adjusted_prob = base_prob + total_adjustment
        
        return np.clip(adjusted_prob, 0.01, 0.99)
    
    def _build_gp_feature_vector(self,
                               candidate: Dict[str, Any],
                               mutation_type: str,
                               arch_features: Dict[str, float]) -> np.ndarray:
        """构建高斯过程特征向量"""
        
        features = []
        
        # 架构特征
        features.extend([
            arch_features.get('total_parameters', 0) / 1e6,  # 标准化
            arch_features.get('model_depth', 0) / 100,
            arch_features.get('current_accuracy', 0),
            arch_features.get('accuracy_trend', 0),
            arch_features.get('train_loss', 0)
        ])
        
        # 候选层特征
        features.extend([
            candidate.get('improvement_potential', 0),
            candidate.get('urgency_score', 0),
            np.mean(list(candidate['bottleneck_indicators'].values())),
            candidate['mutation_suitability'].get(mutation_type, 0)
        ])
        
        # 变异类型编码（one-hot）
        mutation_types = ['width_expansion', 'depth_expansion', 'attention_enhancement', 
                         'residual_connection', 'batch_norm_insertion']
        mutation_encoding = [1.0 if mt == mutation_type else 0.0 for mt in mutation_types]
        features.extend(mutation_encoding)
        
        return np.array(features)
    
    def _gp_predict_improvement(self, feature_vector: np.ndarray, mutation_type: str) -> Tuple[float, float]:
        """高斯过程预测改进值"""
        
        # 简化的高斯过程预测（实际应用中可以使用GPyTorch等库）
        # 这里使用基于特征的启发式预测
        
        # 基础改进预期
        base_improvements = {
            'width_expansion': 0.02,
            'depth_expansion': 0.025,
            'attention_enhancement': 0.03,
            'residual_connection': 0.015,
            'batch_norm_insertion': 0.01,
            'parallel_division': 0.035,
            'serial_division': 0.02,
            'channel_attention': 0.025,
            'layer_norm': 0.012,
            'information_enhancement': 0.028
        }
        
        base_improvement = base_improvements.get(mutation_type, 0.015)
        
        # 基于特征调整
        feature_score = np.mean(feature_vector[:4])  # 使用前4个关键特征
        adjustment = feature_score * 0.02
        
        mean_improvement = base_improvement + adjustment
        variance = (mean_improvement * 0.3) ** 2  # 方差为均值的30%
        
        return mean_improvement, variance
    
    def _calculate_expected_utility(self,
                                  layer_name: str,
                                  mutation_type: str,
                                  success_probabilities: Dict[str, Dict[str, float]],
                                  improvement_predictions: Dict[str, Dict[str, Dict[str, float]]],
                                  uncertainty_analysis: Dict[str, Any]) -> float:
        """计算期望效用"""
        
        success_prob = success_probabilities[layer_name][mutation_type]
        expected_improvement = improvement_predictions[layer_name][mutation_type]['mean_improvement']
        mc_analysis = uncertainty_analysis[layer_name][mutation_type]
        
        # 期望收益
        expected_return = success_prob * expected_improvement
        
        # 风险调整
        risk_penalty = (1 - success_prob) * abs(mc_analysis['value_at_risk_5%'])
        
        # 效用计算
        utility = (expected_return * self.utility_params['accuracy_weight'] - 
                  risk_penalty * self.utility_params['risk_aversion'])
        
        # 探索奖励（鼓励尝试新的变异类型）
        if mutation_type not in [h.get('mutation_type') for h in self.mutation_history[-10:]]:
            utility += self.utility_params['exploration_bonus']
        
        return utility
    
    def _calculate_decision_confidence(self,
                                     layer_name: str,
                                     mutation_type: str,
                                     uncertainty_analysis: Dict[str, Any]) -> float:
        """计算决策置信度"""
        
        mc_analysis = uncertainty_analysis[layer_name][mutation_type]
        
        # 基于概率分布的置信度
        prob_positive = mc_analysis['probability_positive']
        
        # 基于不确定性的置信度
        variance = mc_analysis['variance']
        uncertainty_penalty = np.exp(-variance * 10)  # 方差越大，置信度越低
        
        confidence = prob_positive * uncertainty_penalty
        
        return confidence
    
    def _select_diverse_decisions(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """选择多样化的决策"""
        
        if len(decisions) <= 3:
            return decisions
        
        selected = [decisions[0]]  # 选择最优的
        
        for decision in decisions[1:]:
            # 检查多样性
            is_diverse = True
            for selected_decision in selected:
                if (decision['layer_name'] == selected_decision['layer_name'] or
                    decision['mutation_type'] == selected_decision['mutation_type']):
                    is_diverse = False
                    break
            
            if is_diverse and len(selected) < 3:
                selected.append(decision)
        
        return selected
    
    def _generate_decision_rationale(self,
                                   candidate: Dict[str, Any],
                                   mutation_type: str,
                                   expected_utility: float,
                                   decision_confidence: float) -> str:
        """生成决策理由"""
        
        rationale_parts = []
        
        if expected_utility > 0.03:
            rationale_parts.append("高期望效用")
        elif expected_utility > 0.01:
            rationale_parts.append("中等期望效用")
        
        if decision_confidence > 0.7:
            rationale_parts.append("高置信度预测")
        elif decision_confidence > 0.4:
            rationale_parts.append("中等置信度")
        
        bottleneck_severity = np.mean(list(candidate['bottleneck_indicators'].values()))
        if bottleneck_severity > 0.6:
            rationale_parts.append("显著瓶颈检测")
        
        improvement_potential = candidate.get('improvement_potential', 0)
        if improvement_potential > 0.7:
            rationale_parts.append("高改进潜力")
        
        return "; ".join(rationale_parts) if rationale_parts else f"贝叶斯分析推荐{mutation_type}"
    
    def _calculate_overall_confidence(self, uncertainty_analysis: Dict[str, Any]) -> float:
        """计算整体置信度"""
        
        all_confidences = []
        for layer_analysis in uncertainty_analysis.values():
            for mutation_analysis in layer_analysis.values():
                all_confidences.append(mutation_analysis['probability_positive'])
        
        return np.mean(all_confidences) if all_confidences else 0.0
    
    def _calculate_expected_gain(self, optimal_decisions: List[Dict[str, Any]]) -> float:
        """计算期望收益"""
        
        if not optimal_decisions:
            return 0.0
        
        total_gain = sum(d['expected_improvement'] * d['success_probability'] 
                        for d in optimal_decisions)
        return total_gain
    
    def _comprehensive_risk_assessment(self, optimal_decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """综合风险评估"""
        
        if not optimal_decisions:
            return {'overall_risk': 0.0, 'risk_factors': []}
        
        risks = []
        risk_factors = []
        
        for decision in optimal_decisions:
            risk_metrics = decision['risk_metrics']
            risks.append(abs(risk_metrics['value_at_risk']))
            
            if risk_metrics['probability_positive'] < 0.6:
                risk_factors.append(f"低成功概率: {decision['mutation_type']}")
            
            if abs(risk_metrics['expected_shortfall']) > 0.02:
                risk_factors.append(f"高期望损失: {decision['mutation_type']}")
        
        return {
            'overall_risk': np.mean(risks),
            'max_risk': np.max(risks),
            'risk_factors': risk_factors
        }
    
    def _update_bayesian_history(self,
                               arch_features: Dict[str, float],
                               candidates: List[Dict[str, Any]],
                               context: Dict[str, Any]):
        """更新贝叶斯历史"""
        
        history_entry = {
            'timestamp': context.get('epoch', 0),
            'architecture_features': arch_features.copy(),
            'candidates_found': len(candidates),
            'context_summary': {
                'current_accuracy': arch_features.get('current_accuracy', 0),
                'train_loss': context.get('train_loss', 0),
                'learning_rate': context.get('learning_rate', 0)
            }
        }
        
        self.architecture_features.append(arch_features)
        
        # 更新性能历史
        if 'current_accuracy' in arch_features:
            self.performance_history.append(arch_features['current_accuracy'])
    
    def _fallback_bayesian_analysis(self) -> Dict[str, Any]:
        """贝叶斯分析失败的回退"""
        
        return {
            'bayesian_analysis': {
                'candidates_found': 0,
                'success_probabilities': {},
                'improvement_predictions': {},
                'uncertainty_analysis': {},
                'decision_confidence': 0.0
            },
            'optimal_decisions': [],
            'execution_plan': {
                'execute': False,
                'reason': 'bayesian_analysis_failed'
            },
            'bayesian_insights': {
                'most_promising_mutation': None,
                'expected_performance_gain': 0.0,
                'risk_assessment': {'overall_risk': 1.0, 'risk_factors': ['analysis_failure']}
            }
        }
    
    def update_mutation_outcome(self, 
                              mutation_type: str,
                              layer_name: str,
                              success: bool,
                              performance_change: float,
                              context: Dict[str, Any]):
        """更新变异结果，用于在线学习"""
        
        outcome = {
            'mutation_type': mutation_type,
            'layer_name': layer_name,
            'success': success,
            'performance_change': performance_change,
            'timestamp': context.get('epoch', 0),
            'context': context.copy()
        }
        
        self.mutation_history.append(outcome)
        
        # 更新先验分布
        if mutation_type in self.mutation_priors:
            prior = self.mutation_priors[mutation_type]
            if success:
                prior['alpha'] += 1
            else:
                prior['beta'] += 1
        
        logger.info(f"📊 更新变异结果: {mutation_type} @ {layer_name} -> {'成功' if success else '失败'}")