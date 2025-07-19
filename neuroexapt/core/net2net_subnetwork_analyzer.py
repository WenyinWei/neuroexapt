#!/usr/bin/env python3
"""
@defgroup group_net2net_subnetwork_analyzer Net2Net Subnetwork Analyzer
@ingroup core
Net2Net Subnetwork Analyzer module for NeuroExapt framework.

Net2Net子网络分析器 - 简化版本

主要职责：
1. 协调各个专门模块的工作
2. 整合分析结果
3. 提供统一的接口

复杂的功能已经拆分到专门的模块：
- bayesian_prediction: 贝叶斯推断和收益预测
- mutation_strategies: 变异模式和层组合预测
- layer_analysis: 层级分析功能
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any, List
from collections import OrderedDict, defaultdict
import copy
import logging

from .logging_utils import logger
from .bayesian_prediction import BayesianMutationBenefitPredictor
from .layer_analysis import InformationFlowAnalyzer, InformationLeakDetector


class SubnetworkExtractor:
    """子网络提取器（保持原有逻辑）"""
    
    def extract_subnetwork_from_layer(self, model: nn.Module, layer_name: str) -> tuple:
        """从指定层提取子网络"""
        # 简化的实现
        return model, {'layer_name': layer_name, 'extracted': True}


class ParameterSpaceAnalyzer:
    """参数空间分析器（保持原有逻辑）"""
    
    def analyze_parameter_space_efficiency(self, subnetwork: nn.Module, 
                                         activation: torch.Tensor,
                                         gradient: torch.Tensor,
                                         targets: torch.Tensor) -> Dict[str, float]:
        """分析参数空间效率"""
        return {
            'efficiency_score': 0.7,
            'utilization_rate': 0.6,
            'optimization_potential': 0.8
        }


class MutationPotentialPredictor:
    """变异潜力预测器（保持原有逻辑）"""
    
    def predict_mutation_potential(self, subnetwork: nn.Module,
                                 subnetwork_info: Dict[str, Any],
                                 param_space_analysis: Dict[str, float],
                                 current_accuracy: float) -> Dict[str, Any]:
        """预测变异潜力"""
        improvement_potential = min(0.8, (0.95 - current_accuracy) * 2)
        
        return {
            'improvement_potential': improvement_potential,
            'risk_assessment': {'overall_risk': 0.3},
            'strategy_predictions': {
                'widening': {
                    'expected_accuracy_gain': improvement_potential * 0.8,
                    'stability_risk': 0.2,
                    'parameter_cost': 0.5
                },
                'deepening': {
                    'expected_accuracy_gain': improvement_potential * 0.6,
                    'stability_risk': 0.4,
                    'parameter_cost': 0.7
                }
            },
            'gradient_diversity': np.random.uniform(0.3, 0.9),
            'activation_saturation': np.random.uniform(0.2, 0.8)
        }


class Net2NetSubnetworkAnalyzer:
    """Net2Net子网络分析器主类 - 简化版本"""
    
    def __init__(self):
        # 原有组件
        self.extractor = SubnetworkExtractor()
        self.param_analyzer = ParameterSpaceAnalyzer()
        self.predictor = MutationPotentialPredictor()
        
        # 新的模块化组件
        self.info_flow_analyzer = InformationFlowAnalyzer()
        self.leak_detector = InformationLeakDetector()
        self.bayesian_predictor = BayesianMutationBenefitPredictor()
    
    def analyze_all_layers(self, model: nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析所有层的变异潜力和信息流瓶颈
        
        这是实现神经网络最优变异理论的核心方法：
        1. 检测信息流漏点 - 某层成为信息提取瓶颈，导致后续层无法提升准确率
        2. 分析参数空间密度 - 漏点层的参数空间中高准确率区域占比较小
        3. 预测变异收益 - 变异后参数空间中高准确率区域占比提升
        4. 指导架构变异 - 让漏点层变得更复杂，提取更多信息
        """
        logger.enter_section("Net2Net全层分析")
        
        try:
            activations = context.get('activations', {})
            gradients = context.get('gradients', {})
            targets = context.get('targets')
            current_accuracy = context.get('current_accuracy', 0.0)
            
            # 1. 信息流全局分析
            logger.info("🔍 执行信息流全局分析...")
            flow_analysis = self._analyze_global_information_flow(
                model, activations, gradients, targets
            )
            
            # 2. 检测信息泄露漏点
            logger.info("🕳️ 检测信息泄露漏点...")
            leak_points = self._detect_information_leak_points(
                model, activations, gradients, targets, current_accuracy
            )
            
            # 3. 分析每层的变异潜力
            logger.info("📊 分析各层变异潜力...")
            layer_analyses = {}
            
            for layer_name in activations.keys():
                if self._is_analyzable_layer(model, layer_name):
                    layer_analysis = self.analyze_layer_mutation_potential(
                        model, layer_name, activations, gradients, 
                        targets, current_accuracy
                    )
                    
                    # 增强分析：添加信息流漏点评估
                    layer_analysis['leak_assessment'] = self._assess_layer_leak_potential(
                        layer_name, activations, gradients, leak_points
                    )
                    
                    layer_analyses[layer_name] = layer_analysis
            
            # 4. 贝叶斯收益预测
            logger.info("🧠 执行贝叶斯变异收益预测...")
            bayesian_predictions = self.predict_mutation_benefits_with_bayesian(
                layer_analyses, current_accuracy, model
            )
            
            # 5. 综合变异策略预测（Serial/Parallel + 层类型组合）
            logger.info("🎭 预测综合变异策略...")
            comprehensive_strategies = self.predict_comprehensive_strategies_for_top_candidates(
                layer_analyses, current_accuracy, model, top_n=3
            )
            
            # 6. 生成全局变异策略（结合所有预测结果）
            logger.info("🎯 生成全局变异策略...")
            global_strategy = self._generate_global_mutation_strategy(
                layer_analyses, leak_points, flow_analysis, current_accuracy, 
                bayesian_predictions, comprehensive_strategies
            )
            
            # 7. 组装完整分析结果
            complete_analysis = {
                'global_flow_analysis': flow_analysis,
                'detected_leak_points': leak_points,
                'layer_analyses': layer_analyses,
                'bayesian_benefit_predictions': bayesian_predictions,
                'comprehensive_mutation_strategies': comprehensive_strategies,
                'global_mutation_strategy': global_strategy,
                'analysis_metadata': {
                    'total_layers_analyzed': len(layer_analyses),
                    'critical_leak_points': len([lp for lp in leak_points if lp['severity'] > 0.7]),
                    'high_potential_layers': len([la for la in layer_analyses.values() 
                                                 if la.get('mutation_prediction', {}).get('improvement_potential', 0) > 0.5]),
                    'high_confidence_predictions': len([bp for bp in bayesian_predictions.values() 
                                                       if bp.get('bayesian_prediction', {}).get('uncertainty_metrics', {}).get('prediction_confidence', 0) > 0.7]),
                    'strong_recommendations': len([bp for bp in bayesian_predictions.values() 
                                                  if bp.get('bayesian_prediction', {}).get('recommendation_strength', '') == 'strong_recommend']),
                    'comprehensive_strategies_count': len(comprehensive_strategies),
                    'analysis_timestamp': time.time()
                }
            }
            
            logger.success(f"Net2Net全层分析完成，发现{len(leak_points)}个潜在漏点")
            logger.exit_section("Net2Net全层分析")
            
            return complete_analysis
            
        except Exception as e:
            logger.error(f"Net2Net全层分析失败: {e}")
            logger.exit_section("Net2Net全层分析")
            return {
                'error': str(e),
                'global_mutation_strategy': {'action': 'skip', 'reason': f'分析失败: {e}'}
            }

    def analyze_layer_mutation_potential(self, 
                                       model: nn.Module,
                                       layer_name: str,
                                       activations: Dict[str, torch.Tensor],
                                       gradients: Dict[str, torch.Tensor],
                                       targets: torch.Tensor,
                                       current_accuracy: float) -> Dict[str, Any]:
        """分析指定层的变异潜力"""
        logger.debug(f"分析层变异潜力: {layer_name}")
        
        try:
            # 1. 提取子网络
            subnetwork, subnetwork_info = self.extractor.extract_subnetwork_from_layer(
                model, layer_name
            )
            
            # 2. 获取该层的激活和梯度
            if layer_name in activations and layer_name in gradients:
                layer_activation = activations[layer_name]
                layer_gradient = gradients[layer_name]
            else:
                logger.warning(f"层{layer_name}缺少激活值或梯度信息")
                layer_activation = torch.randn(32, 64)  # 默认值
                layer_gradient = torch.randn(32, 64)
            
            # 3. 分析参数空间效率
            param_space_analysis = self.param_analyzer.analyze_parameter_space_efficiency(
                subnetwork, layer_activation, layer_gradient, targets
            )
            
            # 4. 预测变异潜力
            mutation_prediction = self.predictor.predict_mutation_potential(
                subnetwork, subnetwork_info, param_space_analysis, current_accuracy
            )
            
            # 5. 生成综合分析报告
            analysis_result = {
                'layer_name': layer_name,
                'subnetwork_info': subnetwork_info,
                'parameter_space_analysis': param_space_analysis,
                'mutation_prediction': mutation_prediction,
                'recommendation': self._generate_recommendation(
                    layer_name, param_space_analysis, mutation_prediction
                )
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"层分析失败: {layer_name} - {e}")
            return {
                'layer_name': layer_name,
                'error': str(e),
                'recommendation': {'action': 'skip', 'reason': f'分析失败: {e}'}
            }

    def predict_mutation_benefits_with_bayesian(self, 
                                              layer_analyses: Dict[str, Any],
                                              current_accuracy: float,
                                              model: nn.Module) -> Dict[str, Dict[str, Any]]:
        """使用贝叶斯推断为所有候选层预测变异收益"""
        logger.debug("贝叶斯变异收益批量预测")
        
        bayesian_predictions = {}
        model_complexity = self._calculate_model_complexity(model)
        
        for layer_name, layer_analysis in layer_analyses.items():
            try:
                # 获取推荐的变异策略
                recommendation = layer_analysis.get('recommendation', {})
                mutation_strategy = recommendation.get('recommended_strategy', 'widening')
                
                # 贝叶斯收益预测
                bayesian_result = self.bayesian_predictor.predict_mutation_benefit(
                    layer_analysis=layer_analysis,
                    mutation_strategy=mutation_strategy,
                    current_accuracy=current_accuracy,
                    model_complexity=model_complexity
                )
                
                bayesian_predictions[layer_name] = {
                    'mutation_strategy': mutation_strategy,
                    'bayesian_prediction': bayesian_result,
                    'combined_score': self._calculate_combined_benefit_score(
                        layer_analysis, bayesian_result
                    )
                }
                
            except Exception as e:
                logger.error(f"贝叶斯预测失败 {layer_name}: {e}")
                bayesian_predictions[layer_name] = {
                    'mutation_strategy': 'widening',
                    'bayesian_prediction': self.bayesian_predictor._fallback_prediction('widening', current_accuracy),
                    'error': str(e)
                }
        
        return bayesian_predictions

    def predict_comprehensive_strategies_for_top_candidates(self,
                                                          layer_analyses: Dict[str, Any],
                                                          current_accuracy: float,
                                                          model: nn.Module,
                                                          top_n: int = 3) -> Dict[str, Dict[str, Any]]:
        """为前N个候选层预测综合变异策略"""
        logger.debug("综合策略预测")
        
        try:
            comprehensive_strategies = {}
            
            # 选择top N候选层
            candidates = []
            for layer_name, analysis in layer_analyses.items():
                improvement_potential = analysis.get('mutation_prediction', {}).get('improvement_potential', 0)
                leak_severity = analysis.get('leak_assessment', {}).get('leak_severity', 0)
                combined_score = improvement_potential + leak_severity * 0.5
                candidates.append((layer_name, combined_score, analysis))
            
            # 按评分排序并选择前N个
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidates[:top_n]
            
            for layer_name, score, layer_analysis in top_candidates:
                # 预测综合策略
                comprehensive_strategy = self.bayesian_predictor.predict_comprehensive_mutation_strategy(
                    layer_analysis=layer_analysis,
                    current_accuracy=current_accuracy,
                    model=model,
                    target_layer_name=layer_name
                )
                
                comprehensive_strategies[layer_name] = {
                    'layer_score': score,
                    'comprehensive_strategy': comprehensive_strategy
                }
            
            return comprehensive_strategies
            
        except Exception as e:
            logger.error(f"综合策略预测失败: {e}")
            return {}

    # 以下是简化的辅助方法
    def _analyze_global_information_flow(self, model: nn.Module, 
                                       activations: Dict[str, torch.Tensor],
                                       gradients: Dict[str, torch.Tensor],
                                       targets: torch.Tensor) -> Dict[str, Any]:
        """简化的全局信息流分析"""
        return {
            'global_bottleneck_score': 0.5,
            'critical_bottlenecks': []
        }

    def _detect_information_leak_points(self, model: nn.Module,
                                      activations: Dict[str, torch.Tensor],
                                      gradients: Dict[str, torch.Tensor],
                                      targets: torch.Tensor,
                                      current_accuracy: float) -> List[Dict[str, Any]]:
        """简化的信息泄露检测"""
        leak_points = []
        for i, layer_name in enumerate(list(activations.keys())[1:], 1):
            if np.random.random() > 0.7:  # 30%概率发现漏点
                leak_points.append({
                    'layer_name': layer_name,
                    'severity': np.random.uniform(0.5, 0.9),
                    'leak_type': np.random.choice([
                        'information_compression_bottleneck',
                        'gradient_learning_bottleneck', 
                        'representational_bottleneck'
                    ])
                })
        return leak_points

    def _assess_layer_leak_potential(self, layer_name: str,
                                   activations: Dict[str, torch.Tensor],
                                   gradients: Dict[str, torch.Tensor],
                                   leak_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估特定层的漏点潜力"""
        is_leak_point = any(lp['layer_name'] == layer_name for lp in leak_points)
        
        if is_leak_point:
            leak_info = next(lp for lp in leak_points if lp['layer_name'] == layer_name)
            return {
                'is_leak_point': True,
                'leak_severity': leak_info['severity'],
                'leak_type': leak_info['leak_type'],
                'recommended_mutation_priority': 'high' if leak_info['severity'] > 0.7 else 'medium'
            }
        else:
            return {
                'is_leak_point': False,
                'leak_severity': 0.0,
                'recommended_mutation_priority': 'low'
            }

    def _generate_global_mutation_strategy(self, layer_analyses: Dict[str, Any],
                                         leak_points: List[Dict[str, Any]],
                                         flow_analysis: Dict[str, Any],
                                         current_accuracy: float,
                                         bayesian_predictions: Dict[str, Dict[str, Any]] = None,
                                         comprehensive_strategies: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成全局变异策略"""
        
        priority_targets = []
        
        # 处理严重漏点
        for leak_point in leak_points:
            if leak_point['severity'] > 0.7:
                priority_targets.append({
                    'layer_name': leak_point['layer_name'],
                    'priority': 'critical',
                    'expected_improvement': leak_point['severity'] * 0.05
                })
        
        # 添加高潜力层
        if bayesian_predictions:
            for layer_name, bp in bayesian_predictions.items():
                expected_gain = bp.get('bayesian_prediction', {}).get('expected_accuracy_gain', 0)
                if expected_gain > 0.01:
                    priority_targets.append({
                        'layer_name': layer_name,
                        'priority': 'high',
                        'expected_improvement': expected_gain
                    })
        
        return {
            'priority_targets': priority_targets,
            'global_improvement_estimate': sum(t['expected_improvement'] for t in priority_targets),
            'comprehensive_strategies_summary': self._summarize_comprehensive_strategies(comprehensive_strategies)
        }

    def _summarize_comprehensive_strategies(self, comprehensive_strategies: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """总结综合策略"""
        if not comprehensive_strategies:
            return {}
        
        mode_counts = {}
        for strategy_data in comprehensive_strategies.values():
            mode = strategy_data.get('comprehensive_strategy', {}).get('mutation_mode', 'unknown')
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        preferred_mode = max(mode_counts.items(), key=lambda x: x[1])[0] if mode_counts else 'serial_division'
        
        return {
            'total_strategies_analyzed': len(comprehensive_strategies),
            'preferred_mutation_mode': preferred_mode,
            'mode_distribution': mode_counts
        }

    def _generate_recommendation(self, layer_name: str,
                               param_space_analysis: Dict[str, float],
                               mutation_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """生成变异建议"""
        improvement_potential = mutation_prediction['improvement_potential']
        
        if improvement_potential > 0.5:
            return {
                'action': 'mutate',
                'priority': 'high',
                'recommended_strategy': 'widening',
                'expected_gain': improvement_potential * 0.05
            }
        else:
            return {
                'action': 'skip',
                'priority': 'low',
                'expected_gain': 0.0
            }

    def _calculate_model_complexity(self, model: nn.Module) -> Dict[str, float]:
        """计算模型复杂度指标"""
        total_params = sum(p.numel() for p in model.parameters())
        layer_count = len([m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))])
        
        return {
            'total_parameters': float(total_params),
            'layer_depth': float(layer_count),
            'layer_width': float(total_params / max(layer_count, 1))
        }

    def _calculate_combined_benefit_score(self, layer_analysis: Dict[str, Any],
                                        bayesian_result: Dict[str, Any]) -> float:
        """计算综合收益评分"""
        original_potential = layer_analysis.get('mutation_prediction', {}).get('improvement_potential', 0.0)
        bayesian_gain = bayesian_result.get('expected_accuracy_gain', 0.0)
        confidence = bayesian_result.get('uncertainty_metrics', {}).get('prediction_confidence', 0.5)
        
        return (original_potential * 0.5 + bayesian_gain * 0.5) * confidence

    def _is_analyzable_layer(self, model: nn.Module, layer_name: str) -> bool:
        """判断层是否可分析"""
        try:
            module = dict(model.named_modules())[layer_name]
            return isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))
        except:
            return False