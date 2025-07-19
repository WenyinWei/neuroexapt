#!/usr/bin/env python3
"""
Net2Net子网络分析器 - 解决循环依赖问题

核心思想：分析网络中的信息流漏点和瓶颈，指导动态架构变异

主要功能：
1. 子网络特征提取和分析
2. 信息流瓶颈检测和量化  
3. 参数空间密度分析
4. 变异潜力预测和收益估计
5. 变异策略推荐和执行指导
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import OrderedDict, defaultdict
import copy

from .logging_utils import logger
# 延迟导入以避免循环依赖
# from .bayesian_prediction import BayesianMutationBenefitPredictor
from .layer_analysis import InformationFlowAnalyzer, InformationLeakDetector


class SubnetworkExtractor:
    """子网络提取器（保持原有逻辑）"""
    
    def __init__(self):
        self.extracted_subnets = {}
        
    def extract_key_subnetworks(self, model: nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        """提取关键子网络"""
        subnets = {}
        
        # 基于层类型分组
        layer_groups = self._group_layers_by_type(model)
        
        # 基于信息流分组
        flow_groups = self._group_by_information_flow(model, context)
        
        # 基于变异潜力分组
        mutation_groups = self._group_by_mutation_potential(model, context)
        
        subnets['layer_type_groups'] = layer_groups
        subnets['information_flow_groups'] = flow_groups
        subnets['mutation_potential_groups'] = mutation_groups
        
        return subnets
    
    def _group_layers_by_type(self, model: nn.Module) -> Dict[str, List[str]]:
        """按层类型分组"""
        groups = defaultdict(list)
        for name, module in model.named_modules():
            if name:  # 跳过root module
                module_type = type(module).__name__
                groups[module_type].append(name)
        return dict(groups)
    
    def _group_by_information_flow(self, model: nn.Module, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """基于信息流特征分组"""
        # 简化实现
        groups = {'high_flow': [], 'medium_flow': [], 'low_flow': []}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 简单启发式分组
                if hasattr(module, 'out_channels'):
                    if module.out_channels > 256:
                        groups['high_flow'].append(name)
                    elif module.out_channels > 64:
                        groups['medium_flow'].append(name)
                    else:
                        groups['low_flow'].append(name)
        
        return groups
    
    def _group_by_mutation_potential(self, model: nn.Module, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """基于变异潜力分组"""
        groups = {'high_potential': [], 'medium_potential': [], 'low_potential': []}
        
        # 简化的启发式分组
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 10000:
                    groups['high_potential'].append(name)
                elif param_count > 1000:
                    groups['medium_potential'].append(name)
                else:
                    groups['low_potential'].append(name)
        
        return groups


class ParameterSpaceAnalyzer:
    """参数空间分析器（保持原有逻辑）"""
    
    def __init__(self):
        self.density_cache = {}
    
    def analyze_parameter_space_density(self, 
                                      layer_name: str,
                                      layer_module: nn.Module,
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """分析参数空间密度"""
        
        # 获取参数统计
        param_stats = self._get_parameter_statistics(layer_module)
        
        # 估计密度分布
        density_info = self._estimate_density_distribution(param_stats, context)
        
        # 分析高准确率区域
        high_acc_regions = self._analyze_high_accuracy_regions(density_info, context)
        
        return {
            'layer_name': layer_name,
            'parameter_count': param_stats['total_params'],
            'parameter_distribution': param_stats['distribution'],
            'density_estimation': density_info,
            'high_accuracy_regions': high_acc_regions,
            'mutation_readiness': self._calculate_mutation_readiness(high_acc_regions)
        }
    
    def _get_parameter_statistics(self, module: nn.Module) -> Dict[str, Any]:
        """获取参数统计信息"""
        stats = {
            'total_params': sum(p.numel() for p in module.parameters()),
            'trainable_params': sum(p.numel() for p in module.parameters() if p.requires_grad),
            'distribution': {}
        }
        
        for name, param in module.named_parameters():
            if param.requires_grad:
                stats['distribution'][name] = {
                    'shape': list(param.shape),
                    'mean': float(param.data.mean()),
                    'std': float(param.data.std()),
                    'min': float(param.data.min()),
                    'max': float(param.data.max())
                }
        
        return stats
    
    def _estimate_density_distribution(self, param_stats: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """估计密度分布"""
        return {
            'estimated_density': 0.7,  # 简化实现
            'confidence': 0.8,
            'distribution_type': 'gaussian_mixture'
        }
    
    def _analyze_high_accuracy_regions(self, density_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """分析高准确率区域"""
        return {
            'region_proportion': 0.3,  # 简化实现
            'peak_density': 0.9,
            'connectivity': 0.6
        }
    
    def _calculate_mutation_readiness(self, high_acc_regions: Dict[str, Any]) -> float:
        """计算变异准备度"""
        return 1.0 - high_acc_regions['region_proportion']


class MutationPotentialPredictor:
    """变异潜力预测器（保持原有逻辑）"""
    
    def __init__(self):
        self.prediction_cache = {}
    
    def predict_mutation_potential(self, 
                                 layer_analysis: Dict[str, Any],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """预测变异潜力"""
        
        # 基于参数空间分析预测
        param_potential = self._predict_from_parameter_space(layer_analysis, context)
        
        # 基于信息流分析预测
        flow_potential = self._predict_from_information_flow(layer_analysis, context)
        
        # 综合预测
        combined_potential = self._combine_predictions(param_potential, flow_potential)
        
        return {
            'parameter_space_potential': param_potential,
            'information_flow_potential': flow_potential,
            'combined_potential': combined_potential,
            'confidence': self._calculate_prediction_confidence(combined_potential),
            'recommended_mutations': self._recommend_mutations(combined_potential)
        }
    
    def _predict_from_parameter_space(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> float:
        """基于参数空间预测潜力"""
        if 'mutation_readiness' in analysis:
            return analysis['mutation_readiness']
        return 0.5  # 默认值
    
    def _predict_from_information_flow(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> float:
        """基于信息流预测潜力"""
        # 简化实现
        return 0.6
    
    def _combine_predictions(self, param_potential: float, flow_potential: float) -> float:
        """组合预测结果"""
        return 0.6 * param_potential + 0.4 * flow_potential
    
    def _calculate_prediction_confidence(self, potential: float) -> float:
        """计算预测置信度"""
        return min(0.9, max(0.1, abs(potential - 0.5) * 2))
    
    def _recommend_mutations(self, potential: float) -> List[str]:
        """推荐变异策略"""
        recommendations = []
        if potential > 0.7:
            recommendations.append('aggressive_widening')
        elif potential > 0.5:
            recommendations.append('moderate_widening')
        else:
            recommendations.append('conservative_mutation')
        return recommendations


class Net2NetSubnetworkAnalyzer:
    """Net2Net子网络分析器主类 - 使用延迟加载避免循环依赖"""
    
    def __init__(self):
        # 原有组件
        self.extractor = SubnetworkExtractor()
        self.param_analyzer = ParameterSpaceAnalyzer()
        self.predictor = MutationPotentialPredictor()
        
        # 新的模块化组件
        self.info_flow_analyzer = InformationFlowAnalyzer()
        self.leak_detector = InformationLeakDetector()
        
        # 延迟加载贝叶斯预测器
        self._bayesian_predictor = None
    
    @property
    def bayesian_predictor(self):
        """延迟加载贝叶斯预测器"""
        if self._bayesian_predictor is None:
            try:
                from .bayesian_prediction import BayesianMutationBenefitPredictor
                self._bayesian_predictor = BayesianMutationBenefitPredictor()
            except ImportError as e:
                logger.warning(f"Could not import BayesianMutationBenefitPredictor: {e}")
                # 使用简化的预测器作为回退
                self._bayesian_predictor = self._create_simple_predictor()
        return self._bayesian_predictor
    
    def _create_simple_predictor(self):
        """创建简化的预测器作为回退"""
        class SimpleBayesianPredictor:
            def predict_mutation_benefit(self, layer_analysis, mutation_strategy, current_accuracy, model_complexity):
                return {
                    'expected_accuracy_gain': 0.01,
                    'confidence_interval': {'95%': (0.0, 0.02)},
                    'success_probability': 0.6,
                    'risk_adjusted_benefit': {'risk_adjusted_gain': 0.005},
                    'uncertainty_metrics': {'prediction_confidence': 0.4},
                    'recommendation_strength': "weak_recommend"
                }
        return SimpleBayesianPredictor()
    
    def analyze_all_layers(self, model: nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析所有层的变异潜力和信息流瓶颈
        
        这是实现神经网络最优变异理论的核心方法：
        1. 检测信息流漏点 - 某层成为信息提取瓶颈，导致后续层无法提升准确率
        2. 分析参数空间密度 - 漏点层的参数空间中高准确率区域占比较小
        3. 预测变异收益 - 变异后参数空间中高准确率区域占比提升
        4. 指导架构变异 - 让漏点层变得更复杂，提取更多信息
        """
        
        logger.info("🔍 开始全层分析...")
        
        try:
            # 1. 提取关键子网络
            subnetworks = self.extractor.extract_key_subnetworks(model, context)
            
            # 2. 信息流分析
            info_flow_analysis = self._analyze_information_flow_comprehensive(model, context)
            
            # 3. 逐层详细分析
            layer_analyses = self._analyze_layers_detailed(model, context)
            
            # 4. 瓶颈检测和量化
            bottleneck_analysis = self._detect_and_quantify_bottlenecks(layer_analyses, info_flow_analysis)
            
            # 5. 变异收益预测
            mutation_predictions = self._predict_mutation_benefits(layer_analyses, context)
            
            # 6. 生成变异建议
            mutation_recommendations = self._generate_mutation_recommendations(
                bottleneck_analysis, mutation_predictions, context
            )
            
            comprehensive_analysis = {
                'subnetworks': subnetworks,
                'information_flow': info_flow_analysis,
                'layer_analyses': layer_analyses,
                'bottleneck_analysis': bottleneck_analysis,
                'mutation_predictions': mutation_predictions,
                'recommendations': mutation_recommendations,
                'analysis_metadata': {
                    'timestamp': context.get('timestamp', 'unknown'),
                    'model_size': sum(p.numel() for p in model.parameters()),
                    'analysis_depth': 'comprehensive'
                }
            }
            
            logger.info(f"✅ 全层分析完成，发现 {len(bottleneck_analysis.get('detected_bottlenecks', []))} 个瓶颈")
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"❌ 全层分析失败: {e}")
            return self._fallback_analysis(model, context)
    
    def _analyze_information_flow_comprehensive(self, model: nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        """综合信息流分析"""
        try:
            # 使用信息流分析器
            flow_analysis = self.info_flow_analyzer.analyze_information_flow(model, context)
            
            # 使用泄漏检测器
            leak_analysis = self.leak_detector.detect_information_leaks(model, context)
            
            return {
                'flow_patterns': flow_analysis,
                'leak_detection': leak_analysis,
                'flow_efficiency': self._calculate_flow_efficiency(flow_analysis, leak_analysis)
            }
        except Exception as e:
            logger.warning(f"信息流分析失败: {e}")
            return {'flow_patterns': {}, 'leak_detection': {}, 'flow_efficiency': 0.5}
    
    def _analyze_layers_detailed(self, model: nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        """详细的逐层分析"""
        layer_analyses = {}
        
        for name, module in model.named_modules():
            if self._is_analyzable_layer(module):
                try:
                    # 参数空间分析
                    param_analysis = self.param_analyzer.analyze_parameter_space_density(name, module, context)
                    
                    # 变异潜力预测
                    mutation_potential = self.predictor.predict_mutation_potential(param_analysis, context)
                    
                    layer_analyses[name] = {
                        'parameter_analysis': param_analysis,
                        'mutation_potential': mutation_potential,
                        'layer_type': type(module).__name__,
                        'layer_size': sum(p.numel() for p in module.parameters())
                    }
                except Exception as e:
                    logger.warning(f"层 {name} 分析失败: {e}")
                    continue
        
        return layer_analyses
    
    def _is_analyzable_layer(self, module: nn.Module) -> bool:
        """判断层是否可分析"""
        return isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))
    
    def _detect_and_quantify_bottlenecks(self, layer_analyses: Dict[str, Any], info_flow: Dict[str, Any]) -> Dict[str, Any]:
        """检测和量化瓶颈"""
        bottlenecks = []
        
        for layer_name, analysis in layer_analyses.items():
            # 基于变异潜力判断瓶颈
            potential = analysis.get('mutation_potential', {}).get('combined_potential', 0.5)
            readiness = analysis.get('parameter_analysis', {}).get('mutation_readiness', 0.5)
            
            if potential > 0.7 and readiness > 0.6:
                bottleneck_score = (potential + readiness) / 2
                bottlenecks.append({
                    'layer_name': layer_name,
                    'bottleneck_score': bottleneck_score,
                    'bottleneck_type': 'parameter_space_constraint',
                    'severity': 'high' if bottleneck_score > 0.8 else 'medium'
                })
        
        return {
            'detected_bottlenecks': bottlenecks,
            'bottleneck_count': len(bottlenecks),
            'average_severity': np.mean([b['bottleneck_score'] for b in bottlenecks]) if bottlenecks else 0.0
        }
    
    def _predict_mutation_benefits(self, layer_analyses: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """预测变异收益"""
        predictions = {}
        
        current_accuracy = context.get('current_accuracy', 0.8)
        model_complexity = context.get('model_complexity', {'parameters': 1000000})
        
        for layer_name, analysis in layer_analyses.items():
            try:
                # 使用贝叶斯预测器（延迟加载）
                layer_prediction = self.bayesian_predictor.predict_mutation_benefit(
                    layer_analysis=analysis,
                    mutation_strategy='moderate_widening',
                    current_accuracy=current_accuracy,
                    model_complexity=model_complexity
                )
                predictions[layer_name] = layer_prediction
            except Exception as e:
                logger.warning(f"层 {layer_name} 变异收益预测失败: {e}")
                # 使用简化预测
                predictions[layer_name] = {
                    'expected_accuracy_gain': 0.005,
                    'confidence_interval': {'95%': (0.0, 0.01)},
                    'recommendation_strength': 'weak_recommend'
                }
        
        return predictions
    
    def _generate_mutation_recommendations(self, bottleneck_analysis: Dict[str, Any], 
                                         mutation_predictions: Dict[str, Any], 
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """生成变异建议"""
        recommendations = []
        
        # 基于瓶颈分析生成建议
        for bottleneck in bottleneck_analysis.get('detected_bottlenecks', []):
            layer_name = bottleneck['layer_name']
            if layer_name in mutation_predictions:
                pred = mutation_predictions[layer_name]
                if pred.get('recommendation_strength') in ['recommend', 'strong_recommend']:
                    recommendations.append({
                        'layer_name': layer_name,
                        'mutation_type': 'widening',
                        'priority': bottleneck['bottleneck_score'],
                        'expected_gain': pred.get('expected_accuracy_gain', 0.01),
                        'confidence': pred.get('uncertainty_metrics', {}).get('prediction_confidence', 0.5)
                    })
        
        # 排序建议
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return {
            'mutations': recommendations[:5],  # 最多推荐5个变异
            'total_candidates': len(recommendations),
            'average_expected_gain': np.mean([r['expected_gain'] for r in recommendations]) if recommendations else 0.0
        }
    
    def _calculate_flow_efficiency(self, flow_analysis: Dict[str, Any], leak_analysis: Dict[str, Any]) -> float:
        """计算信息流效率"""
        # 简化实现
        return 0.7
    
    def _fallback_analysis(self, model: nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        """回退分析"""
        logger.info("使用回退分析模式")
        return {
            'subnetworks': {},
            'information_flow': {},
            'layer_analyses': {},
            'bottleneck_analysis': {'detected_bottlenecks': []},
            'mutation_predictions': {},
            'recommendations': {'mutations': []},
            'analysis_metadata': {'analysis_depth': 'fallback'}
        }