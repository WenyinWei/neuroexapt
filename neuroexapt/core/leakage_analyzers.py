"""
信息泄漏分析器子模块
将InformationLeakageDetector拆分为专注的分析器组件
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class EntropyAnalyzer:
    """信息熵分析器"""
    
    def analyze_information_entropy(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """分析各层的信息熵"""
        entropy_metrics = {}
        
        for layer_name, activation in activations.items():
            if activation is None or activation.numel() == 0:
                continue
                
            try:
                # 将激活值转换为概率分布
                if len(activation.shape) > 2:
                    flat_activation = activation.flatten(2).mean(dim=2)
                else:
                    flat_activation = activation
                
                # 计算每个通道的熵
                channel_entropies = []
                for channel_idx in range(flat_activation.shape[1]):
                    channel_data = flat_activation[:, channel_idx]
                    
                    # 将数据离散化为概率分布
                    hist, _ = torch.histogram(channel_data, bins=50, density=True)
                    hist = hist + 1e-8
                    hist = hist / hist.sum()
                    
                    # 计算熵
                    entropy = -torch.sum(hist * torch.log2(hist + 1e-8))
                    channel_entropies.append(entropy.item())
                
                avg_entropy = np.mean(channel_entropies)
                entropy_std = np.std(channel_entropies)
                info_loss_score = max(0, 1 - avg_entropy / 4.0)
                
                entropy_metrics[layer_name] = {
                    'average_entropy': avg_entropy,
                    'entropy_std': entropy_std,
                    'channel_entropies': channel_entropies,
                    'information_loss_score': info_loss_score,
                    'is_information_bottleneck': info_loss_score > 0.6
                }
                
            except Exception as e:
                logger.warning(f"层 {layer_name} 熵计算失败: {e}")
                continue
        
        return entropy_metrics


class DiversityAnalyzer:
    """特征多样性分析器"""
    
    def analyze_feature_diversity(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """分析特征多样性"""
        diversity_metrics = {}
        
        for layer_name, activation in activations.items():
            if activation is None or activation.numel() == 0:
                continue
                
            try:
                if len(activation.shape) > 2:
                    flat_activation = activation.flatten(2).mean(dim=2)
                else:
                    flat_activation = activation
                
                if flat_activation.shape[1] > 1:
                    correlation_matrix = torch.corrcoef(flat_activation.T)
                    mask = ~torch.eye(correlation_matrix.shape[0], dtype=bool)
                    avg_correlation = torch.abs(correlation_matrix[mask]).mean().item()
                    redundancy_score = min(1.0, avg_correlation * 2)
                    
                    # 计算有效秩
                    try:
                        _, s, _ = torch.svd(flat_activation.T)
                        normalized_s = s / s.max()
                        effective_rank = (normalized_s > 0.01).sum().item() / s.shape[0]
                    except:
                        effective_rank = 1.0
                    
                    diversity_metrics[layer_name] = {
                        'average_correlation': avg_correlation,
                        'redundancy_score': redundancy_score,
                        'effective_rank': effective_rank,
                        'feature_collapse': redundancy_score > 0.7,
                        'diversity_loss': 1 - effective_rank
                    }
                    
            except Exception as e:
                logger.warning(f"层 {layer_name} 多样性分析失败: {e}")
                continue
        
        return diversity_metrics


class GradientFlowAnalyzer:
    """梯度流分析器"""
    
    def analyze_gradient_flow(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """分析梯度流阻塞"""
        gradient_metrics = {}
        
        for layer_name, gradient in gradients.items():
            if gradient is None or gradient.numel() == 0:
                continue
                
            try:
                grad_norm = torch.norm(gradient).item()
                grad_var = torch.var(gradient).item()
                
                vanishing_score = max(0, 1 - grad_norm / 0.1)
                exploding_score = max(0, (grad_norm - 10.0) / 10.0)
                health_score = 1 - max(vanishing_score, exploding_score)
                
                gradient_metrics[layer_name] = {
                    'gradient_norm': grad_norm,
                    'gradient_variance': grad_var,
                    'vanishing_score': vanishing_score,
                    'exploding_score': exploding_score,
                    'health_score': health_score,
                    'is_gradient_blocked': vanishing_score > 0.8 or exploding_score > 0.5
                }
                
            except Exception as e:
                logger.warning(f"层 {layer_name} 梯度分析失败: {e}")
                continue
        
        return gradient_metrics


class InformationFlowAnalyzer:
    """信息流分析器"""
    
    def analyze_information_flow(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """分析层间信息传递效率"""
        layer_names = list(activations.keys())
        flow_metrics = {}
        
        for i in range(len(layer_names) - 1):
            current_layer = layer_names[i]
            next_layer = layer_names[i + 1]
            
            current_activation = activations[current_layer]
            next_activation = activations[next_layer]
            
            if current_activation is None or next_activation is None:
                continue
                
            try:
                current_info = self._compute_information_content(current_activation)
                next_info = self._compute_information_content(next_activation)
                
                information_retention = next_info / max(current_info, 1e-8)
                information_loss = max(0, 1 - information_retention)
                
                flow_metrics[f"{current_layer}->{next_layer}"] = {
                    'current_info': current_info,
                    'next_info': next_info,
                    'retention_rate': information_retention,
                    'loss_rate': information_loss,
                    'is_bottleneck': information_loss > 0.3
                }
                
            except Exception as e:
                logger.warning(f"信息流分析失败 {current_layer}->{next_layer}: {e}")
                continue
        
        return flow_metrics
    
    def _compute_information_content(self, activation: torch.Tensor) -> float:
        """计算激活的信息含量"""
        try:
            if len(activation.shape) > 2:
                flat_activation = activation.flatten(2).mean(dim=2)
            else:
                flat_activation = activation
            
            feature_vars = torch.var(flat_activation, dim=0)
            mean_var = torch.mean(feature_vars).item()
            return mean_var
        except:
            return 0.0


class RepresentationQualityAnalyzer:
    """表示质量分析器"""
    
    def analyze_representation_quality(self, 
                                     activations: Dict[str, torch.Tensor],
                                     targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """分析表示质量"""
        quality_metrics = {}
        
        for layer_name, activation in activations.items():
            if activation is None or activation.numel() == 0:
                continue
                
            try:
                if len(activation.shape) > 2:
                    flat_activation = activation.flatten(2).mean(dim=2)
                else:
                    flat_activation = activation
                
                # 激活分布健康度
                activation_mean = torch.mean(flat_activation, dim=0)
                activation_std = torch.std(flat_activation, dim=0)
                
                dead_neurons = (activation_std < 1e-6).sum().item()
                dead_ratio = dead_neurons / activation_std.shape[0]
                
                saturated_neurons = (torch.abs(activation_mean) > 2.0).sum().item()
                saturated_ratio = saturated_neurons / activation_mean.shape[0]
                
                # 表示分离度
                if targets is not None and flat_activation.shape[0] > 1:
                    try:
                        unique_targets = torch.unique(targets)
                        if len(unique_targets) > 1:
                            inter_class_distance = self._compute_inter_class_distance(
                                flat_activation, targets, unique_targets
                            )
                        else:
                            inter_class_distance = 0.0
                    except:
                        inter_class_distance = 0.0
                else:
                    inter_class_distance = 0.0
                
                quality_score = (
                    (1 - dead_ratio) * 0.4 +
                    (1 - saturated_ratio) * 0.3 +
                    min(1.0, inter_class_distance) * 0.3
                )
                
                quality_metrics[layer_name] = {
                    'dead_neuron_ratio': dead_ratio,
                    'saturated_neuron_ratio': saturated_ratio,
                    'inter_class_distance': inter_class_distance,
                    'quality_score': quality_score,
                    'needs_repair': quality_score < 0.6
                }
                
            except Exception as e:
                logger.warning(f"层 {layer_name} 质量分析失败: {e}")
                continue
        
        return quality_metrics
    
    def _compute_inter_class_distance(self, 
                                    activation: torch.Tensor,
                                    targets: torch.Tensor,
                                    unique_targets: torch.Tensor) -> float:
        """计算类别间表示距离"""
        try:
            class_centers = []
            for target in unique_targets:
                mask = targets == target
                if mask.sum() > 0:
                    center = activation[mask].mean(dim=0)
                    class_centers.append(center)
            
            if len(class_centers) < 2:
                return 0.0
            
            distances = []
            for i in range(len(class_centers)):
                for j in range(i + 1, len(class_centers)):
                    dist = torch.norm(class_centers[i] - class_centers[j]).item()
                    distances.append(dist)
            
            return np.mean(distances) if distances else 0.0
        except:
            return 0.0


class LeakagePointIdentifier:
    """泄漏点识别器"""
    
    def identify_critical_leakage_points(self, 
                                       entropy_analysis: Dict,
                                       diversity_analysis: Dict,
                                       gradient_flow_analysis: Dict,
                                       information_flow_analysis: Dict,
                                       representation_quality: Dict) -> List[Dict[str, Any]]:
        """识别关键的信息泄漏点"""
        leakage_points = []
        layer_problems = defaultdict(list)
        
        # 从各个分析中收集问题
        self._collect_entropy_problems(layer_problems, entropy_analysis)
        self._collect_diversity_problems(layer_problems, diversity_analysis)
        self._collect_gradient_problems(layer_problems, gradient_flow_analysis)
        self._collect_quality_problems(layer_problems, representation_quality)
        
        # 综合评估并排序
        for layer_name, problems in layer_problems.items():
            if problems:
                total_severity = sum(p['severity'] for p in problems)
                avg_severity = total_severity / len(problems)
                
                leakage_point = {
                    'layer_name': layer_name,
                    'problems': problems,
                    'total_severity': total_severity,
                    'average_severity': avg_severity,
                    'problem_count': len(problems),
                    'priority': self._compute_repair_priority(layer_name, problems)
                }
                
                leakage_points.append(leakage_point)
        
        # 按优先级排序
        leakage_points.sort(key=lambda x: x['priority'], reverse=True)
        return leakage_points
    
    def _collect_entropy_problems(self, layer_problems: defaultdict, entropy_analysis: Dict):
        """收集信息熵问题"""
        for layer_name, metrics in entropy_analysis.items():
            if metrics.get('is_information_bottleneck', False):
                layer_problems[layer_name].append({
                    'type': 'information_bottleneck',
                    'severity': metrics['information_loss_score'],
                    'description': f"信息熵过低 ({metrics['average_entropy']:.2f})"
                })
    
    def _collect_diversity_problems(self, layer_problems: defaultdict, diversity_analysis: Dict):
        """收集特征多样性问题"""
        for layer_name, metrics in diversity_analysis.items():
            if metrics.get('feature_collapse', False):
                layer_problems[layer_name].append({
                    'type': 'feature_collapse',
                    'severity': metrics['redundancy_score'],
                    'description': f"特征冗余度过高 ({metrics['redundancy_score']:.2f})"
                })
    
    def _collect_gradient_problems(self, layer_problems: defaultdict, gradient_flow_analysis: Dict):
        """收集梯度流问题"""
        for layer_name, metrics in gradient_flow_analysis.items():
            if metrics.get('is_gradient_blocked', False):
                layer_problems[layer_name].append({
                    'type': 'gradient_blocked',
                    'severity': 1 - metrics['health_score'],
                    'description': f"梯度流异常 (健康度: {metrics['health_score']:.2f})"
                })
    
    def _collect_quality_problems(self, layer_problems: defaultdict, representation_quality: Dict):
        """收集表示质量问题"""
        for layer_name, metrics in representation_quality.items():
            if metrics.get('needs_repair', False):
                layer_problems[layer_name].append({
                    'type': 'representation_degradation',
                    'severity': 1 - metrics['quality_score'],
                    'description': f"表示质量低下 (评分: {metrics['quality_score']:.2f})"
                })
    
    def _compute_repair_priority(self, layer_name: str, problems: List[Dict]) -> float:
        """计算修复优先级"""
        severity_score = sum(p['severity'] for p in problems)
        
        type_weights = {
            'information_bottleneck': 1.0,
            'feature_collapse': 0.9,
            'gradient_blocked': 0.8,
            'representation_degradation': 0.7
        }
        
        weighted_severity = sum(
            p['severity'] * type_weights.get(p['type'], 0.5) 
            for p in problems
        )
        
        # 层位置权重
        if 'classifier' in layer_name:
            position_weight = 1.2
        elif any(x in layer_name for x in ['feature', 'conv', 'block']):
            position_weight = 1.0
        else:
            position_weight = 0.8
        
        return weighted_severity * position_weight