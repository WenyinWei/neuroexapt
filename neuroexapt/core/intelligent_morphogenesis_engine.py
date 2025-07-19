"""
智能形态发生引擎

真正综合的架构变异决策系统，解决组件间配合生硬的问题
核心理念：精准定位变异点，智能选择变异策略
"""

from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class IntelligentMorphogenesisEngine:
    """
    智能形态发生引擎
    
    解决现有问题：
    1. 各组件配合生硬 -> 统一的分析决策流水线
    2. 检测结果全是0 -> 动态阈值和分层检测
    3. 变异点不明确 -> 精准定位和分级推荐
    4. 策略选择简陋 -> 多维度综合决策
    """
    
    def __init__(self):
        # 核心分析组件
        self._layer_analyzer = None
        self._performance_tracker = None
        self._mutation_executor = None
        
        # 动态阈值管理
        self.adaptive_thresholds = {
            'bottleneck_severity': 0.3,        # 动态调整
            'improvement_potential': 0.1,      # 基于历史调整
            'mutation_confidence': 0.6,        # 自适应
            'performance_plateau_ratio': 0.05  # 相对停滞比例
        }
        
        # 分析历史记录
        self.analysis_history = []
        self.mutation_success_rate = {}
        
        # 综合决策权重
        self.decision_weights = {
            'performance_analysis': 0.3,
            'structural_analysis': 0.25,
            'information_flow': 0.2,
            'gradient_analysis': 0.15,
            'historical_success': 0.1
        }
    
    @property
    def layer_analyzer(self):
        """延迟加载层分析器"""
        if self._layer_analyzer is None:
            from .net2net_subnetwork_analyzer import Net2NetSubnetworkAnalyzer
            self._layer_analyzer = Net2NetSubnetworkAnalyzer()
        return self._layer_analyzer
    
    @property
    def performance_tracker(self):
        """延迟加载性能跟踪器"""
        if self._performance_tracker is None:
            from .performance_tracker import PerformanceTracker
            self._performance_tracker = PerformanceTracker()
        return self._performance_tracker
    
    @property
    def mutation_executor(self):
        """延迟加载变异执行器"""
        if self._mutation_executor is None:
            from .mutation_executor import MutationExecutor
            self._mutation_executor = MutationExecutor()
        return self._mutation_executor
    
    def comprehensive_morphogenesis_analysis(self, 
                                           model: nn.Module,
                                           context: Dict[str, Any]) -> Dict[str, Any]:
        """
        综合形态发生分析
        
        设计理念：
        1. 多层次分析：从粗粒度到细粒度
        2. 动态阈值：根据模型状态调整敏感度
        3. 综合评分：多维度信息融合
        4. 精准定位：明确指出变异的具体位置和方式
        """
        
        logger.info("🧠 启动智能形态发生分析")
        
        try:
            # 1. 性能态势分析
            performance_situation = self._analyze_performance_situation(context)
            
            # 2. 架构瓶颈深度挖掘
            structural_bottlenecks = self._deep_structural_analysis(model, context)
            
            # 3. 信息流效率分析
            information_efficiency = self._analyze_information_efficiency(model, context)
            
            # 4. 梯度传播质量分析
            gradient_quality = self._analyze_gradient_propagation(context)
            
            # 5. 动态调整检测阈值
            self._adapt_detection_thresholds(performance_situation, structural_bottlenecks)
            
            # 6. 综合候选变异点识别
            mutation_candidates = self._identify_mutation_candidates(
                model, structural_bottlenecks, information_efficiency, gradient_quality
            )
            
            # 7. 智能变异策略生成
            mutation_strategies = self._generate_intelligent_strategies(
                mutation_candidates, performance_situation, context
            )
            
            # 8. 多维度决策融合
            final_decisions = self._multi_dimensional_decision_fusion(
                performance_situation, mutation_strategies, context
            )
            
            # 9. 执行建议生成
            execution_plan = self._generate_execution_plan(final_decisions, model, context)
            
            # 记录分析历史
            analysis_record = {
                'timestamp': context.get('current_epoch', 0),
                'performance_situation': performance_situation,
                'mutation_candidates_count': len(mutation_candidates),
                'final_decisions_count': len(final_decisions),
                'thresholds_used': self.adaptive_thresholds.copy()
            }
            self.analysis_history.append(analysis_record)
            
            comprehensive_result = {
                'analysis_summary': {
                    'performance_situation': performance_situation,
                    'structural_analysis': {
                        'total_layers_analyzed': len(list(model.named_modules())),
                        'bottlenecks_found': len(structural_bottlenecks),
                        'severity_distribution': self._categorize_bottlenecks(structural_bottlenecks)
                    },
                    'information_efficiency': information_efficiency,
                    'gradient_quality': gradient_quality
                },
                'mutation_candidates': mutation_candidates,
                'mutation_strategies': mutation_strategies,
                'final_decisions': final_decisions,
                'execution_plan': execution_plan,
                'adaptive_thresholds': self.adaptive_thresholds.copy(),
                'analysis_metadata': {
                    'engine_version': '2.0_intelligent',
                    'total_analysis_history': len(self.analysis_history),
                    'dynamic_threshold_adjustment': True
                }
            }
            
            logger.info(f"🎯 智能分析完成: 发现{len(mutation_candidates)}个候选点, {len(final_decisions)}个最终决策")
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"❌ 智能形态发生分析失败: {e}")
            return self._fallback_analysis()
    
    def _analyze_performance_situation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能态势"""
        
        performance_history = context.get('performance_history', [])
        current_accuracy = context.get('current_accuracy', 0.0)
        
        if len(performance_history) < 5:
            return {
                'situation_type': 'insufficient_data',
                'plateau_detected': False,
                'improvement_trend': 'unknown',
                'urgency_level': 'low'
            }
        
        # 性能趋势分析
        recent_accuracies = performance_history[-10:]  # 最近10个epoch
        trend_slope = np.polyfit(range(len(recent_accuracies)), recent_accuracies, 1)[0]
        
        # 停滞检测（更敏感的算法）
        plateau_threshold = self.adaptive_thresholds['performance_plateau_ratio']
        recent_improvement = max(recent_accuracies) - min(recent_accuracies)
        is_plateau = recent_improvement < plateau_threshold
        
        # 波动性分析
        volatility = np.std(recent_accuracies)
        
        # 饱和度评估
        theoretical_max = 0.98  # 假设理论最大值
        saturation_ratio = current_accuracy / theoretical_max
        
        # 综合态势判断
        if saturation_ratio > 0.95:
            situation_type = 'high_saturation'
            urgency_level = 'medium'  # 高饱和时需要精细化变异
        elif is_plateau and trend_slope < 0.001:
            situation_type = 'performance_plateau'
            urgency_level = 'high'
        elif trend_slope < -0.005:
            situation_type = 'performance_decline'
            urgency_level = 'high'
        elif volatility > 0.02:
            situation_type = 'unstable_training'
            urgency_level = 'medium'
        else:
            situation_type = 'normal_training'
            urgency_level = 'low'
        
        return {
            'situation_type': situation_type,
            'plateau_detected': is_plateau,
            'improvement_trend': 'positive' if trend_slope > 0.001 else 'negative' if trend_slope < -0.001 else 'flat',
            'urgency_level': urgency_level,
            'saturation_ratio': saturation_ratio,
            'volatility': volatility,
            'trend_slope': trend_slope,
            'recent_improvement': recent_improvement
        }
    
    def _deep_structural_analysis(self, model: nn.Module, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """深度架构瓶颈分析"""
        
        bottlenecks = []
        activations = context.get('activations', {})
        gradients = context.get('gradients', {})
        
        # 逐层深度分析
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            
            bottleneck_info = {
                'layer_name': name,
                'layer_type': type(module).__name__,
                'bottleneck_scores': {},
                'bottleneck_types': [],
                'improvement_potential': 0.0
            }
            
            # 1. 参数容量分析
            param_capacity_score = self._analyze_parameter_capacity(module)
            bottleneck_info['bottleneck_scores']['parameter_capacity'] = param_capacity_score
            
            # 2. 信息流分析（如果有激活值）
            if name in activations:
                info_flow_score = self._analyze_layer_information_flow(activations[name])
                bottleneck_info['bottleneck_scores']['information_flow'] = info_flow_score
            
            # 3. 梯度质量分析（如果有梯度）
            if name in gradients:
                gradient_score = self._analyze_layer_gradient_quality(gradients[name])
                bottleneck_info['bottleneck_scores']['gradient_quality'] = gradient_score
            
            # 4. 架构效率分析
            arch_efficiency_score = self._analyze_architectural_efficiency(module, context)
            bottleneck_info['bottleneck_scores']['architectural_efficiency'] = arch_efficiency_score
            
            # 综合评分和瓶颈类型判断
            scores = bottleneck_info['bottleneck_scores']
            avg_score = np.mean(list(scores.values()))
            
            # 动态阈值判断
            threshold = self.adaptive_thresholds['bottleneck_severity']
            if avg_score > threshold:
                # 确定瓶颈类型
                if scores.get('parameter_capacity', 0) > threshold:
                    bottleneck_info['bottleneck_types'].append('parameter_constraint')
                if scores.get('information_flow', 0) > threshold:
                    bottleneck_info['bottleneck_types'].append('information_bottleneck')
                if scores.get('gradient_quality', 0) > threshold:
                    bottleneck_info['bottleneck_types'].append('gradient_bottleneck')
                if scores.get('architectural_efficiency', 0) > threshold:
                    bottleneck_info['bottleneck_types'].append('architectural_inefficiency')
                
                # 计算改进潜力
                bottleneck_info['improvement_potential'] = min(1.0, avg_score * 1.2)
                bottlenecks.append(bottleneck_info)
        
        # 按改进潜力排序
        bottlenecks.sort(key=lambda x: x['improvement_potential'], reverse=True)
        
        logger.info(f"🔍 深度结构分析: 发现{len(bottlenecks)}个瓶颈层")
        return bottlenecks
    
    def _analyze_parameter_capacity(self, module: nn.Module) -> float:
        """分析参数容量约束"""
        
        if isinstance(module, nn.Conv2d):
            # 对于卷积层，分析通道数相对于特征复杂度的充分性
            channel_ratio = module.out_channels / max(64, module.in_channels)  # 基准比例
            kernel_efficiency = (module.kernel_size[0] * module.kernel_size[1]) / 9  # 3x3为基准
            
            # 通道数不足或核太小都可能造成容量约束
            capacity_constraint = max(0, 1 - channel_ratio) + max(0, 1 - kernel_efficiency)
            return min(1.0, capacity_constraint / 2)
            
        elif isinstance(module, nn.Linear):
            # 对于线性层，分析特征数相对于输入复杂度的充分性
            feature_ratio = module.out_features / max(128, module.in_features)
            capacity_constraint = max(0, 1 - feature_ratio)
            return min(1.0, capacity_constraint)
        
        return 0.0
    
    def _analyze_layer_information_flow(self, activation: torch.Tensor) -> float:
        """分析层级信息流效率"""
        
        try:
            # 信息熵计算
            flat_activation = activation.flatten()
            
            # 有效信息比例
            non_zero_ratio = torch.count_nonzero(flat_activation).float() / flat_activation.numel()
            
            # 激活分布的均匀性
            hist = torch.histc(flat_activation, bins=50)
            hist_normalized = hist / hist.sum()
            entropy = -torch.sum(hist_normalized * torch.log(hist_normalized + 1e-10))
            max_entropy = np.log(50)  # 50个bin的最大熵
            entropy_ratio = entropy / max_entropy
            
            # 信息流效率 = 1 - 有效性和均匀性的综合
            efficiency_loss = (1 - non_zero_ratio) * 0.6 + (1 - entropy_ratio) * 0.4
            return float(efficiency_loss)
            
        except Exception:
            return 0.5
    
    def _analyze_layer_gradient_quality(self, gradient: torch.Tensor) -> float:
        """分析层级梯度质量"""
        
        try:
            # 梯度范数
            grad_norm = torch.norm(gradient)
            
            # 梯度分布
            grad_std = torch.std(gradient)
            grad_mean = torch.abs(torch.mean(gradient))
            
            # 梯度消失/爆炸检测
            if grad_norm < 1e-7:
                return 0.9  # 严重梯度消失
            elif grad_norm > 100:
                return 0.8  # 梯度爆炸
            
            # 梯度分布质量
            signal_noise_ratio = grad_mean / (grad_std + 1e-10)
            quality_score = 1.0 / (1.0 + signal_noise_ratio)
            
            return float(quality_score)
            
        except Exception:
            return 0.5
    
    def _analyze_architectural_efficiency(self, module: nn.Module, context: Dict[str, Any]) -> float:
        """分析架构效率"""
        
        # 参数利用效率
        param_count = sum(p.numel() for p in module.parameters())
        
        if isinstance(module, nn.Conv2d):
            # FLOPs估算和效率分析
            theoretical_flops = module.out_channels * module.in_channels * np.prod(module.kernel_size)
            efficiency_score = min(1.0, param_count / (theoretical_flops + 1))
        elif isinstance(module, nn.Linear):
            # 线性层的参数密度
            efficiency_score = min(1.0, param_count / (module.in_features * module.out_features + 1))
        else:
            efficiency_score = 0.5
        
        # 返回效率不足程度
        return 1.0 - efficiency_score
    
    def _analyze_information_efficiency(self, model: nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析全局信息效率"""
        
        activations = context.get('activations', {})
        
        if not activations:
            return {'overall_efficiency': 0.5, 'bottleneck_layers': []}
        
        layer_efficiencies = {}
        
        for layer_name, activation in activations.items():
            efficiency = 1.0 - self._analyze_layer_information_flow(activation)
            layer_efficiencies[layer_name] = efficiency
        
        overall_efficiency = np.mean(list(layer_efficiencies.values()))
        
        # 找出效率最低的层
        sorted_layers = sorted(layer_efficiencies.items(), key=lambda x: x[1])
        bottleneck_layers = [layer for layer, eff in sorted_layers[:3] if eff < 0.6]
        
        return {
            'overall_efficiency': overall_efficiency,
            'layer_efficiencies': layer_efficiencies,
            'bottleneck_layers': bottleneck_layers,
            'efficiency_variance': np.var(list(layer_efficiencies.values()))
        }
    
    def _analyze_gradient_propagation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析梯度传播质量"""
        
        gradients = context.get('gradients', {})
        
        if not gradients:
            return {'overall_quality': 0.5, 'problematic_layers': []}
        
        layer_qualities = {}
        
        for layer_name, gradient in gradients.items():
            quality = 1.0 - self._analyze_layer_gradient_quality(gradient)
            layer_qualities[layer_name] = quality
        
        overall_quality = np.mean(list(layer_qualities.values()))
        
        # 找出梯度质量最差的层
        sorted_layers = sorted(layer_qualities.items(), key=lambda x: x[1])
        problematic_layers = [layer for layer, qual in sorted_layers[:3] if qual < 0.5]
        
        return {
            'overall_quality': overall_quality,
            'layer_qualities': layer_qualities,
            'problematic_layers': problematic_layers,
            'quality_variance': np.var(list(layer_qualities.values()))
        }
    
    def _adapt_detection_thresholds(self, performance_situation: Dict[str, Any], 
                                  structural_bottlenecks: List[Dict[str, Any]]):
        """动态调整检测阈值"""
        
        # 根据性能态势调整敏感度
        if performance_situation['situation_type'] == 'high_saturation':
            # 高饱和状态，提高敏感度
            self.adaptive_thresholds['bottleneck_severity'] *= 0.8
            self.adaptive_thresholds['improvement_potential'] *= 0.7
        elif performance_situation['situation_type'] == 'performance_plateau':
            # 停滞状态，中等敏感度
            self.adaptive_thresholds['bottleneck_severity'] *= 0.9
        elif performance_situation['urgency_level'] == 'low':
            # 正常状态，降低敏感度避免过度变异
            self.adaptive_thresholds['bottleneck_severity'] *= 1.1
        
        # 根据历史成功率调整
        if self.analysis_history:
            recent_analyses = self.analysis_history[-5:]
            avg_candidates = np.mean([a['mutation_candidates_count'] for a in recent_analyses])
            avg_decisions = np.mean([a['final_decisions_count'] for a in recent_analyses])
            
            # 如果候选太少，降低阈值
            if avg_candidates < 1:
                self.adaptive_thresholds['bottleneck_severity'] *= 0.8
            # 如果候选太多，提高阈值
            elif avg_candidates > 5:
                self.adaptive_thresholds['bottleneck_severity'] *= 1.2
        
        # 确保阈值在合理范围内
        self.adaptive_thresholds['bottleneck_severity'] = np.clip(
            self.adaptive_thresholds['bottleneck_severity'], 0.1, 0.8
        )
        
        logger.info(f"📊 动态阈值调整: 瓶颈检测阈值={self.adaptive_thresholds['bottleneck_severity']:.3f}")
    
    def _identify_mutation_candidates(self, model: nn.Module,
                                    structural_bottlenecks: List[Dict[str, Any]],
                                    information_efficiency: Dict[str, Any],
                                    gradient_quality: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别变异候选点"""
        
        candidates = []
        
        # 从结构瓶颈中选择候选
        for bottleneck in structural_bottlenecks:
            candidate = {
                'layer_name': bottleneck['layer_name'],
                'layer_type': bottleneck['layer_type'],
                'selection_reasons': ['structural_bottleneck'],
                'bottleneck_types': bottleneck['bottleneck_types'],
                'improvement_potential': bottleneck['improvement_potential'],
                'priority_score': bottleneck['improvement_potential'],
                'recommended_mutations': []
            }
            
            # 根据瓶颈类型推荐变异策略
            if 'parameter_constraint' in bottleneck['bottleneck_types']:
                candidate['recommended_mutations'].append('width_expansion')
            if 'information_bottleneck' in bottleneck['bottleneck_types']:
                candidate['recommended_mutations'].extend(['depth_expansion', 'attention_enhancement'])
            if 'gradient_bottleneck' in bottleneck['bottleneck_types']:
                candidate['recommended_mutations'].extend(['residual_connection', 'batch_norm_insertion'])
            
            candidates.append(candidate)
        
        # 从信息效率分析中补充候选
        for layer_name in information_efficiency.get('bottleneck_layers', []):
            # 避免重复
            if not any(c['layer_name'] == layer_name for c in candidates):
                candidate = {
                    'layer_name': layer_name,
                    'layer_type': 'unknown',
                    'selection_reasons': ['information_inefficiency'],
                    'bottleneck_types': ['information_flow'],
                    'improvement_potential': 0.7,
                    'priority_score': 0.7,
                    'recommended_mutations': ['information_enhancement', 'channel_attention']
                }
                candidates.append(candidate)
        
        # 从梯度质量分析中补充候选
        for layer_name in gradient_quality.get('problematic_layers', []):
            # 避免重复
            if not any(c['layer_name'] == layer_name for c in candidates):
                candidate = {
                    'layer_name': layer_name,
                    'layer_type': 'unknown',
                    'selection_reasons': ['gradient_quality_issue'],
                    'bottleneck_types': ['gradient_propagation'],
                    'improvement_potential': 0.6,
                    'priority_score': 0.6,
                    'recommended_mutations': ['residual_connection', 'layer_norm']
                }
                candidates.append(candidate)
        
        # 按优先级排序
        candidates.sort(key=lambda x: x['priority_score'], reverse=True)
        
        logger.info(f"🎯 识别变异候选: {len(candidates)}个候选点")
        return candidates
    
    def _generate_intelligent_strategies(self, candidates: List[Dict[str, Any]],
                                       performance_situation: Dict[str, Any],
                                       context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成智能变异策略"""
        
        strategies = []
        
        for candidate in candidates:
            for mutation_type in candidate['recommended_mutations']:
                strategy = {
                    'target_layer': candidate['layer_name'],
                    'mutation_type': mutation_type,
                    'rationale': {
                        'bottleneck_types': candidate['bottleneck_types'],
                        'selection_reasons': candidate['selection_reasons'],
                        'improvement_potential': candidate['improvement_potential']
                    },
                    'expected_outcome': self._predict_mutation_outcome(
                        mutation_type, candidate, performance_situation
                    ),
                    'risk_assessment': self._assess_mutation_risk(
                        mutation_type, candidate, context
                    ),
                    'implementation_plan': self._create_implementation_plan(
                        mutation_type, candidate
                    )
                }
                strategies.append(strategy)
        
        return strategies
    
    def _predict_mutation_outcome(self, mutation_type: str, 
                                candidate: Dict[str, Any],
                                performance_situation: Dict[str, Any]) -> Dict[str, Any]:
        """预测变异结果"""
        
        base_improvement = candidate['improvement_potential'] * 0.05  # 最大5%改进
        
        # 根据变异类型调整
        type_multipliers = {
            'width_expansion': 1.0,
            'depth_expansion': 0.8,
            'attention_enhancement': 1.2,
            'residual_connection': 0.9,
            'batch_norm_insertion': 0.7,
            'information_enhancement': 1.1,
            'channel_attention': 1.0,
            'layer_norm': 0.6
        }
        
        adjusted_improvement = base_improvement * type_multipliers.get(mutation_type, 1.0)
        
        # 根据性能态势调整
        if performance_situation['situation_type'] == 'high_saturation':
            adjusted_improvement *= 0.5  # 高饱和时改进较小
        elif performance_situation['situation_type'] == 'performance_plateau':
            adjusted_improvement *= 1.2  # 停滞时改进潜力较大
        
        return {
            'expected_accuracy_improvement': adjusted_improvement,
            'confidence_level': min(0.9, candidate['improvement_potential']),
            'parameter_increase': self._estimate_parameter_increase(mutation_type),
            'computational_overhead': self._estimate_computational_overhead(mutation_type)
        }
    
    def _assess_mutation_risk(self, mutation_type: str, 
                            candidate: Dict[str, Any], 
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """评估变异风险"""
        
        # 基础风险评分
        base_risks = {
            'width_expansion': 0.2,
            'depth_expansion': 0.4,
            'attention_enhancement': 0.3,
            'residual_connection': 0.1,
            'batch_norm_insertion': 0.1,
            'information_enhancement': 0.3,
            'channel_attention': 0.2,
            'layer_norm': 0.1
        }
        
        base_risk = base_risks.get(mutation_type, 0.5)
        
        # 根据历史成功率调整
        historical_success = self.mutation_success_rate.get(mutation_type, 0.5)
        risk_adjustment = 1.0 - historical_success
        
        final_risk = min(1.0, base_risk * (1 + risk_adjustment))
        
        return {
            'overall_risk': final_risk,
            'risk_factors': self._identify_risk_factors(mutation_type, candidate),
            'mitigation_strategies': self._suggest_risk_mitigation(mutation_type),
            'rollback_plan': self._create_rollback_plan(mutation_type)
        }
    
    def _create_implementation_plan(self, mutation_type: str, 
                                  candidate: Dict[str, Any]) -> Dict[str, Any]:
        """创建实施计划"""
        
        return {
            'preparation_steps': self._get_preparation_steps(mutation_type),
            'execution_steps': self._get_execution_steps(mutation_type, candidate),
            'validation_steps': self._get_validation_steps(mutation_type),
            'estimated_time': self._estimate_implementation_time(mutation_type)
        }
    
    def _multi_dimensional_decision_fusion(self, performance_situation: Dict[str, Any],
                                         mutation_strategies: List[Dict[str, Any]],
                                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """多维度决策融合"""
        
        final_decisions = []
        
        for strategy in mutation_strategies:
            # 综合评分
            decision_score = 0.0
            
            # 性能分析权重
            perf_score = strategy['expected_outcome']['expected_accuracy_improvement'] * 10
            decision_score += perf_score * self.decision_weights['performance_analysis']
            
            # 结构分析权重
            struct_score = strategy['rationale']['improvement_potential']
            decision_score += struct_score * self.decision_weights['structural_analysis']
            
            # 风险调整
            risk_penalty = strategy['risk_assessment']['overall_risk']
            decision_score *= (1 - risk_penalty * 0.5)
            
            # 历史成功率权重
            historical_success = self.mutation_success_rate.get(strategy['mutation_type'], 0.5)
            decision_score += historical_success * self.decision_weights['historical_success']
            
            # 只保留高分策略
            confidence_threshold = self.adaptive_thresholds['mutation_confidence']
            if decision_score > confidence_threshold:
                decision = strategy.copy()
                decision['final_score'] = decision_score
                decision['selection_rationale'] = self._generate_selection_rationale(
                    strategy, decision_score, performance_situation
                )
                final_decisions.append(decision)
        
        # 按分数排序，选择最佳策略
        final_decisions.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 根据情况限制数量
        max_decisions = 3 if performance_situation['urgency_level'] == 'high' else 1
        final_decisions = final_decisions[:max_decisions]
        
        logger.info(f"🎯 多维决策融合: {len(final_decisions)}个最终决策")
        return final_decisions
    
    def _generate_execution_plan(self, final_decisions: List[Dict[str, Any]], 
                               model: nn.Module, 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """生成执行计划"""
        
        if not final_decisions:
            return {
                'execute': False,
                'reason': 'no_viable_mutations',
                'recommendations': ['continue_training', 'adjust_hyperparameters']
            }
        
        primary_decision = final_decisions[0]
        
        return {
            'execute': True,
            'primary_mutation': {
                'target_layer': primary_decision['target_layer'],
                'mutation_type': primary_decision['mutation_type'],
                'expected_improvement': primary_decision['expected_outcome']['expected_accuracy_improvement'],
                'confidence': primary_decision['expected_outcome']['confidence_level']
            },
            'alternative_mutations': [
                {
                    'target_layer': d['target_layer'],
                    'mutation_type': d['mutation_type'],
                    'score': d['final_score']
                } for d in final_decisions[1:3]
            ],
            'execution_order': 'sequential',
            'monitoring_plan': {
                'metrics_to_track': ['accuracy', 'loss', 'gradient_norm'],
                'evaluation_frequency': 'every_epoch',
                'success_criteria': f"accuracy_improvement > {primary_decision['expected_outcome']['expected_accuracy_improvement'] * 0.5}"
            },
            'contingency_plan': primary_decision['risk_assessment']['rollback_plan']
        }
    
    # 辅助方法
    def _categorize_bottlenecks(self, bottlenecks: List[Dict[str, Any]]) -> Dict[str, int]:
        """分类瓶颈"""
        categories = defaultdict(int)
        for bottleneck in bottlenecks:
            for btype in bottleneck['bottleneck_types']:
                categories[btype] += 1
        return dict(categories)
    
    def _estimate_parameter_increase(self, mutation_type: str) -> int:
        """估计参数增加量"""
        estimates = {
            'width_expansion': 50000,
            'depth_expansion': 100000,
            'attention_enhancement': 30000,
            'residual_connection': 0,
            'batch_norm_insertion': 100,
            'information_enhancement': 20000,
            'channel_attention': 5000,
            'layer_norm': 200
        }
        return estimates.get(mutation_type, 10000)
    
    def _estimate_computational_overhead(self, mutation_type: str) -> float:
        """估计计算开销"""
        overheads = {
            'width_expansion': 0.2,
            'depth_expansion': 0.3,
            'attention_enhancement': 0.4,
            'residual_connection': 0.05,
            'batch_norm_insertion': 0.02,
            'information_enhancement': 0.15,
            'channel_attention': 0.1,
            'layer_norm': 0.02
        }
        return overheads.get(mutation_type, 0.1)
    
    def _identify_risk_factors(self, mutation_type: str, candidate: Dict[str, Any]) -> List[str]:
        """识别风险因素"""
        risk_factors = []
        
        if mutation_type in ['depth_expansion', 'attention_enhancement']:
            risk_factors.append('increased_overfitting_risk')
        if mutation_type in ['width_expansion', 'information_enhancement']:
            risk_factors.append('computational_overhead')
        if candidate['improvement_potential'] < 0.5:
            risk_factors.append('uncertain_benefit')
        
        return risk_factors
    
    def _suggest_risk_mitigation(self, mutation_type: str) -> List[str]:
        """建议风险缓解措施"""
        mitigations = {
            'width_expansion': ['use_dropout', 'reduce_learning_rate'],
            'depth_expansion': ['use_residual_connections', 'careful_initialization'],
            'attention_enhancement': ['use_attention_dropout', 'layer_norm']
        }
        return mitigations.get(mutation_type, ['monitor_carefully'])
    
    def _create_rollback_plan(self, mutation_type: str) -> Dict[str, Any]:
        """创建回滚计划"""
        return {
            'trigger_conditions': ['accuracy_drop > 2%', 'loss_divergence'],
            'rollback_steps': ['restore_checkpoint', 'adjust_learning_rate'],
            'recovery_strategy': 'conservative_training'
        }
    
    def _get_preparation_steps(self, mutation_type: str) -> List[str]:
        """获取准备步骤"""
        return [
            'create_model_checkpoint',
            'backup_optimizer_state',
            'prepare_mutation_parameters'
        ]
    
    def _get_execution_steps(self, mutation_type: str, candidate: Dict[str, Any]) -> List[str]:
        """获取执行步骤"""
        base_steps = [
            f'locate_target_layer: {candidate["layer_name"]}',
            f'apply_mutation: {mutation_type}',
            'update_model_structure',
            'reinitialize_optimizer',
            'validate_mutation'
        ]
        return base_steps
    
    def _get_validation_steps(self, mutation_type: str) -> List[str]:
        """获取验证步骤"""
        return [
            'check_model_integrity',
            'validate_forward_pass',
            'test_gradient_flow',
            'measure_performance_impact'
        ]
    
    def _estimate_implementation_time(self, mutation_type: str) -> str:
        """估计实施时间"""
        return "1-2 epochs"
    
    def _generate_selection_rationale(self, strategy: Dict[str, Any], 
                                    score: float, 
                                    performance_situation: Dict[str, Any]) -> str:
        """生成选择理由"""
        
        rationale_parts = []
        
        if score > 0.8:
            rationale_parts.append("高置信度改进预期")
        
        if strategy['rationale']['improvement_potential'] > 0.7:
            rationale_parts.append("显著结构改进潜力")
        
        if performance_situation['situation_type'] == 'performance_plateau':
            rationale_parts.append("突破性能瓶颈需要")
        
        if strategy['risk_assessment']['overall_risk'] < 0.3:
            rationale_parts.append("低风险实施")
        
        return "; ".join(rationale_parts) if rationale_parts else "综合评估推荐"
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """fallback分析"""
        return {
            'analysis_summary': {
                'status': 'fallback_mode',
                'structural_analysis': {'bottlenecks_found': 0}
            },
            'mutation_candidates': [],
            'mutation_strategies': [],
            'final_decisions': [],
            'execution_plan': {
                'execute': False,
                'reason': 'analysis_failed'
            }
        }
    
    def update_success_rate(self, mutation_type: str, success: bool):
        """更新变异成功率"""
        if mutation_type not in self.mutation_success_rate:
            self.mutation_success_rate[mutation_type] = 0.5
        
        # 指数移动平均更新
        alpha = 0.1
        current_rate = self.mutation_success_rate[mutation_type]
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
        self.mutation_success_rate[mutation_type] = new_rate
        
        logger.info(f"📊 更新成功率: {mutation_type} = {new_rate:.3f}")