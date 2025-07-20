"""
贝叶斯候选点检测器

专门负责发现和评估变异候选点
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BayesianCandidateDetector:
    """贝叶斯候选点检测器"""
    
    def __init__(self, config=None):
        from .bayesian_config import BayesianConfig
        self.config = config if config else BayesianConfig()
        
    def detect_candidates(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测变异候选点"""
        
        candidates = []
        
        # 1. 基于激活的候选点检测
        activation_candidates = self._detect_activation_candidates(features)
        candidates.extend(activation_candidates)
        
        # 2. 基于梯度的候选点检测
        gradient_candidates = self._detect_gradient_candidates(features)
        candidates.extend(gradient_candidates)
        
        # 3. 基于架构的候选点检测
        architecture_candidates = self._detect_architecture_candidates(features)
        candidates.extend(architecture_candidates)
        
        # 4. 基于性能的候选点检测
        performance_candidates = self._detect_performance_candidates(features)
        candidates.extend(performance_candidates)
        
        # 5. 去重和优先级排序
        unique_candidates = self._deduplicate_and_prioritize(candidates)
        
        logger.info(f"🔍 候选点检测完成: 发现{len(unique_candidates)}个候选点")
        
        return unique_candidates
    
    def _detect_activation_candidates(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于激活的候选点检测"""
        
        activation_features = features.get('activation_features', {})
        if not activation_features.get('available'):
            return []
        
        candidates = []
        layer_features = activation_features.get('layer_features', {})
        global_features = activation_features.get('global_features', {})
        
        # 检测低激活层（可能的瓶颈）
        avg_activation = global_features.get('avg_activation', 0)
        for layer_name, layer_data in layer_features.items():
            layer_mean = layer_data.get('mean', 0)
            
            # 激活过低的层
            if layer_mean < avg_activation * 0.3:
                candidates.append({
                    'layer_name': layer_name,
                    'detection_method': 'low_activation',
                    'priority': 0.7,
                    'rationale': f'激活过低 ({layer_mean:.4f})',
                    'suggested_mutations': ['width_expansion', 'attention_enhancement'],
                    'confidence': 0.6
                })
            
            # 激活饱和的层（可能需要正则化）
            sparsity = layer_data.get('zeros_ratio', 0)
            if sparsity > 0.8:
                candidates.append({
                    'layer_name': layer_name,
                    'detection_method': 'high_sparsity',
                    'priority': 0.6,
                    'rationale': f'激活稀疏度过高 ({sparsity:.4f})',
                    'suggested_mutations': ['batch_norm_insertion', 'layer_norm'],
                    'confidence': 0.5
                })
        
        return candidates
    
    def _detect_gradient_candidates(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于梯度的候选点检测"""
        
        gradient_features = features.get('gradient_features', {})
        if not gradient_features.get('available'):
            return []
        
        candidates = []
        layer_features = gradient_features.get('layer_features', {})
        global_features = gradient_features.get('global_features', {})
        
        avg_grad_norm = global_features.get('avg_grad_norm', 0)
        
        for layer_name, layer_data in layer_features.items():
            grad_norm = layer_data.get('norm', 0)
            
            # 梯度消失
            if grad_norm < avg_grad_norm * 0.1 and avg_grad_norm > 0:
                candidates.append({
                    'layer_name': layer_name,
                    'detection_method': 'gradient_vanishing',
                    'priority': 0.8,
                    'rationale': f'梯度消失 (norm: {grad_norm:.6f})',
                    'suggested_mutations': ['residual_connection', 'batch_norm_insertion'],
                    'confidence': 0.7
                })
            
            # 梯度爆炸
            elif grad_norm > avg_grad_norm * 10 and avg_grad_norm > 0:
                candidates.append({
                    'layer_name': layer_name,
                    'detection_method': 'gradient_explosion',
                    'priority': 0.9,
                    'rationale': f'梯度爆炸 (norm: {grad_norm:.6f})',
                    'suggested_mutations': ['layer_norm', 'batch_norm_insertion'],
                    'confidence': 0.8
                })
        
        return candidates
    
    def _detect_architecture_candidates(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于架构的候选点检测"""
        
        architecture_info = features.get('architecture_info', {})
        layer_relationship = features.get('layer_relationship_features', {})
        
        candidates = []
        
        # 检测参数不平衡
        layer_info = architecture_info.get('layer_info', [])
        if layer_info:
            param_counts = [info['param_count'] for info in layer_info]
            avg_params = np.mean(param_counts)
            
            for info in layer_info:
                # 参数过少的层
                if info['param_count'] < avg_params * 0.1 and avg_params > 0:
                    candidates.append({
                        'layer_name': info['name'],
                        'detection_method': 'low_parameter_count',
                        'priority': 0.5,
                        'rationale': f'参数量过少 ({info["param_count"]})',
                        'suggested_mutations': ['width_expansion'],
                        'confidence': 0.4
                    })
        
        # 检测连接复杂度
        complexity = layer_relationship.get('connection_complexity', 0)
        skip_connections = layer_relationship.get('skip_connections', 0)
        
        if skip_connections == 0 and len(layer_info) > 5:
            # 缺少跳跃连接的深层网络
            # 选择中间层添加残差连接
            middle_layers = layer_info[len(layer_info)//3:2*len(layer_info)//3]
            for info in middle_layers:
                candidates.append({
                    'layer_name': info['name'],
                    'detection_method': 'missing_skip_connections',
                    'priority': 0.6,
                    'rationale': '深层网络缺少跳跃连接',
                    'suggested_mutations': ['residual_connection'],
                    'confidence': 0.5
                })
        
        return candidates
    
    def _detect_performance_candidates(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于性能的候选点检测"""
        
        performance_features = features.get('performance_features', {})
        if not performance_features.get('available'):
            return []
        
        candidates = []
        
        # 检测性能停滞
        trend = performance_features.get('short_term_trend', 0)
        improvement_ratio = performance_features.get('improvement_ratio', 1)
        
        if abs(trend) < 0.001 and improvement_ratio < 0.3:
            # 性能停滞，建议探索性变异
            architecture_info = features.get('architecture_info', {})
            layer_info = architecture_info.get('layer_info', [])
            
            if layer_info:
                # 随机选择一些层进行探索性变异
                import random
                selected_layers = random.sample(layer_info, min(3, len(layer_info)))
                
                for info in selected_layers:
                    candidates.append({
                        'layer_name': info['name'],
                        'detection_method': 'performance_stagnation',
                        'priority': 0.4,
                        'rationale': '性能停滞，建议探索性变异',
                        'suggested_mutations': ['attention_enhancement', 'channel_attention'],
                        'confidence': 0.3
                    })
        
        # 检测性能下降
        elif trend < -0.005:
            # 性能下降，需要紧急修复
            candidates.append({
                'layer_name': 'global',  # 全局性问题
                'detection_method': 'performance_degradation',
                'priority': 0.9,
                'rationale': f'性能下降趋势 ({trend:.6f})',
                'suggested_mutations': ['batch_norm_insertion', 'layer_norm'],
                'confidence': 0.7
            })
        
        return candidates
    
    def _deduplicate_and_prioritize(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重和优先级排序"""
        
        # 按层名去重，保留优先级最高的
        layer_candidates = {}
        
        for candidate in candidates:
            layer_name = candidate['layer_name']
            priority = candidate.get('priority', 0)
            
            if (layer_name not in layer_candidates or 
                priority > layer_candidates[layer_name].get('priority', 0)):
                layer_candidates[layer_name] = candidate
        
        # 转换为列表并按优先级排序
        unique_candidates = list(layer_candidates.values())
        unique_candidates.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        # 添加全局排名
        for i, candidate in enumerate(unique_candidates):
            candidate['rank'] = i + 1
            candidate['total_candidates'] = len(unique_candidates)
        
        return unique_candidates
    
    def evaluate_candidate_quality(self, candidate: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """评估候选点质量"""
        
        base_confidence = candidate.get('confidence', 0.5)
        
        # 质量评估因子
        quality_factors = []
        
        # 1. 检测方法置信度
        detection_method = candidate.get('detection_method', '')
        method_confidence = {
            'gradient_vanishing': 0.8,
            'gradient_explosion': 0.9,
            'low_activation': 0.6,
            'high_sparsity': 0.5,
            'missing_skip_connections': 0.6,
            'performance_degradation': 0.8,
            'performance_stagnation': 0.3,
            'low_parameter_count': 0.4
        }
        
        quality_factors.append(method_confidence.get(detection_method, 0.5))
        
        # 2. 优先级权重
        priority = candidate.get('priority', 0)
        quality_factors.append(priority)
        
        # 3. 数据可用性权重
        feature_summary = features.get('feature_summary', {})
        data_quality = 0.5
        if feature_summary.get('has_gradients'):
            data_quality += 0.2
        if feature_summary.get('has_activations'):
            data_quality += 0.2
        if feature_summary.get('has_performance_history'):
            data_quality += 0.1
        
        quality_factors.append(data_quality)
        
        # 计算综合质量分数
        quality_score = np.mean(quality_factors)
        adjusted_confidence = base_confidence * quality_score
        
        return {
            'quality_score': quality_score,
            'adjusted_confidence': adjusted_confidence,
            'quality_factors': quality_factors,
            'data_quality': data_quality,
            'recommendation': 'high_quality' if quality_score > 0.7 else 'medium_quality' if quality_score > 0.5 else 'low_quality'
        }