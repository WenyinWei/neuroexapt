"""
信息泄漏检测器
精准识别网络中信息丢失和特征泄漏的关键层
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import defaultdict
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InformationLeakageDetector:
    """
    信息泄漏检测器
    
    核心思想：
    1. 通过信息论指标检测信息丢失
    2. 分析特征表示的质量和多样性
    3. 识别梯度流阻塞和信息瓶颈
    4. 定位真正需要变异的关键层
    """
    
    def __init__(self):
        self.activation_cache = {}
        self.gradient_cache = {}
        self.information_metrics = {}
        
    def detect_information_leakage(self, 
                                  model: nn.Module,
                                  activations: Dict[str, torch.Tensor],
                                  gradients: Dict[str, torch.Tensor],
                                  targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        检测信息泄漏和特征丢失
        
        Returns:
            Dict包含泄漏点分析和修复建议
        """
        
        logger.info("🔍 开始信息泄漏检测分析...")
        
        # 1. 信息熵分析
        entropy_analysis = self._analyze_information_entropy(activations)
        
        # 2. 特征多样性分析  
        diversity_analysis = self._analyze_feature_diversity(activations)
        
        # 3. 梯度流分析
        gradient_flow_analysis = self._analyze_gradient_flow(gradients)
        
        # 4. 层间信息传递分析
        information_flow_analysis = self._analyze_information_flow(activations)
        
        # 5. 表示质量分析
        representation_quality = self._analyze_representation_quality(activations, targets)
        
        # 6. 综合分析找出真正的泄漏点
        leakage_points = self._identify_critical_leakage_points(
            entropy_analysis, diversity_analysis, gradient_flow_analysis, 
            information_flow_analysis, representation_quality
        )
        
        # 7. 生成精准的修复建议
        repair_suggestions = self._generate_targeted_repair_suggestions(leakage_points)
        
        return {
            'leakage_points': leakage_points,
            'repair_suggestions': repair_suggestions,
            'detailed_analysis': {
                'entropy_analysis': entropy_analysis,
                'diversity_analysis': diversity_analysis,
                'gradient_flow_analysis': gradient_flow_analysis,
                'information_flow_analysis': information_flow_analysis,
                'representation_quality': representation_quality
            },
            'summary': self._generate_leakage_summary(leakage_points)
        }
    
    def _analyze_information_entropy(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """分析各层的信息熵"""
        
        entropy_metrics = {}
        
        for layer_name, activation in activations.items():
            if activation is None or activation.numel() == 0:
                continue
                
            try:
                # 将激活值转换为概率分布
                if len(activation.shape) > 2:
                    # 对于卷积层，计算空间维度的平均
                    flat_activation = activation.flatten(2).mean(dim=2)  # [batch, channels]
                else:
                    flat_activation = activation
                
                # 计算每个通道的熵
                channel_entropies = []
                for channel_idx in range(flat_activation.shape[1]):
                    channel_data = flat_activation[:, channel_idx]
                    
                    # 将数据离散化为概率分布
                    hist, _ = torch.histogram(channel_data, bins=50, density=True)
                    hist = hist + 1e-8  # 避免零概率
                    hist = hist / hist.sum()
                    
                    # 计算熵
                    entropy = -torch.sum(hist * torch.log2(hist + 1e-8))
                    channel_entropies.append(entropy.item())
                
                avg_entropy = np.mean(channel_entropies)
                entropy_std = np.std(channel_entropies)
                
                # 检测信息丢失
                info_loss_score = max(0, 1 - avg_entropy / 4.0)  # 4bit作为参考
                
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
    
    def _analyze_feature_diversity(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """分析特征多样性"""
        
        diversity_metrics = {}
        
        for layer_name, activation in activations.items():
            if activation is None or activation.numel() == 0:
                continue
                
            try:
                if len(activation.shape) > 2:
                    # 对于卷积层
                    flat_activation = activation.flatten(2).mean(dim=2)
                else:
                    flat_activation = activation
                
                # 计算特征相关性矩阵
                if flat_activation.shape[1] > 1:
                    correlation_matrix = torch.corrcoef(flat_activation.T)
                    
                    # 计算平均相关性（排除对角线）
                    mask = ~torch.eye(correlation_matrix.shape[0], dtype=bool)
                    avg_correlation = torch.abs(correlation_matrix[mask]).mean().item()
                    
                    # 计算特征冗余度
                    redundancy_score = min(1.0, avg_correlation * 2)
                    
                    # 计算有效秩（特征独立性指标）
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
    
    def _analyze_gradient_flow(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """分析梯度流阻塞"""
        
        gradient_metrics = {}
        
        for layer_name, gradient in gradients.items():
            if gradient is None or gradient.numel() == 0:
                continue
                
            try:
                # 计算梯度范数
                grad_norm = torch.norm(gradient).item()
                
                # 计算梯度方差（衡量梯度多样性）
                grad_var = torch.var(gradient).item()
                
                # 检测梯度消失
                vanishing_threshold = 1e-6
                vanishing_score = max(0, 1 - grad_norm / 0.1)  # 0.1作为健康梯度阈值
                
                # 检测梯度爆炸
                exploding_threshold = 10.0
                exploding_score = max(0, (grad_norm - exploding_threshold) / exploding_threshold)
                
                # 梯度健康度
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
    
    def _analyze_information_flow(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
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
                # 计算信息传递效率
                current_info = self._compute_information_content(current_activation)
                next_info = self._compute_information_content(next_activation)
                
                # 信息保持率
                information_retention = next_info / max(current_info, 1e-8)
                
                # 信息丢失率
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
            
            # 使用特征的方差作为信息含量的代理
            feature_vars = torch.var(flat_activation, dim=0)
            mean_var = torch.mean(feature_vars).item()
            
            return mean_var
            
        except:
            return 0.0
    
    def _analyze_representation_quality(self, 
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
                
                # 1. 激活分布健康度
                activation_mean = torch.mean(flat_activation, dim=0)
                activation_std = torch.std(flat_activation, dim=0)
                
                # 检测死神经元
                dead_neurons = (activation_std < 1e-6).sum().item()
                dead_ratio = dead_neurons / activation_std.shape[0]
                
                # 检测饱和神经元
                saturated_neurons = (torch.abs(activation_mean) > 2.0).sum().item()
                saturated_ratio = saturated_neurons / activation_mean.shape[0]
                
                # 2. 表示分离度
                if targets is not None and flat_activation.shape[0] > 1:
                    try:
                        # 计算不同类别间的表示距离
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
                
                # 综合质量评分
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
            
            # 计算类别中心间的平均距离
            distances = []
            for i in range(len(class_centers)):
                for j in range(i + 1, len(class_centers)):
                    dist = torch.norm(class_centers[i] - class_centers[j]).item()
                    distances.append(dist)
            
            return np.mean(distances) if distances else 0.0
            
        except:
            return 0.0
    
    def _identify_critical_leakage_points(self, 
                                        entropy_analysis: Dict,
                                        diversity_analysis: Dict,
                                        gradient_flow_analysis: Dict,
                                        information_flow_analysis: Dict,
                                        representation_quality: Dict) -> List[Dict[str, Any]]:
        """识别关键的信息泄漏点"""
        
        leakage_points = []
        
        # 收集所有层的问题
        layer_problems = defaultdict(list)
        
        # 1. 从熵分析中识别信息瓶颈
        for layer_name, metrics in entropy_analysis.items():
            if metrics.get('is_information_bottleneck', False):
                layer_problems[layer_name].append({
                    'type': 'information_bottleneck',
                    'severity': metrics['information_loss_score'],
                    'description': f"信息熵过低 ({metrics['average_entropy']:.2f})"
                })
        
        # 2. 从多样性分析中识别特征坍塌
        for layer_name, metrics in diversity_analysis.items():
            if metrics.get('feature_collapse', False):
                layer_problems[layer_name].append({
                    'type': 'feature_collapse',
                    'severity': metrics['redundancy_score'],
                    'description': f"特征冗余度过高 ({metrics['redundancy_score']:.2f})"
                })
        
        # 3. 从梯度分析中识别梯度阻塞
        for layer_name, metrics in gradient_flow_analysis.items():
            if metrics.get('is_gradient_blocked', False):
                layer_problems[layer_name].append({
                    'type': 'gradient_blocked',
                    'severity': 1 - metrics['health_score'],
                    'description': f"梯度流异常 (健康度: {metrics['health_score']:.2f})"
                })
        
        # 4. 从表示质量中识别表示退化
        for layer_name, metrics in representation_quality.items():
            if metrics.get('needs_repair', False):
                layer_problems[layer_name].append({
                    'type': 'representation_degradation',
                    'severity': 1 - metrics['quality_score'],
                    'description': f"表示质量低下 (评分: {metrics['quality_score']:.2f})"
                })
        
        # 5. 综合评估并排序
        for layer_name, problems in layer_problems.items():
            if problems:
                # 计算综合严重性
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
    
    def _compute_repair_priority(self, layer_name: str, problems: List[Dict]) -> float:
        """计算修复优先级"""
        
        # 基础严重性
        severity_score = sum(p['severity'] for p in problems)
        
        # 问题类型权重
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
        
        # 层位置权重（中间层更关键）
        if 'classifier' in layer_name:
            position_weight = 1.2  # 分类器层很重要
        elif any(x in layer_name for x in ['feature', 'conv', 'block']):
            position_weight = 1.0  # 特征层正常权重
        else:
            position_weight = 0.8  # 其他层
        
        priority = weighted_severity * position_weight
        
        return priority
    
    def _generate_targeted_repair_suggestions(self, leakage_points: List[Dict]) -> List[Dict[str, Any]]:
        """生成针对性的修复建议"""
        
        suggestions = []
        
        for point in leakage_points:
            layer_name = point['layer_name']
            problems = point['problems']
            
            # 根据问题类型生成具体建议
            repair_actions = []
            
            for problem in problems:
                if problem['type'] == 'information_bottleneck':
                    repair_actions.extend([
                        'width_expansion',  # 增加通道数
                        'residual_connection',  # 添加跳跃连接
                        'attention_enhancement'  # 增强信息流
                    ])
                elif problem['type'] == 'feature_collapse':
                    repair_actions.extend([
                        'depth_expansion',  # 增加层深度
                        'parallel_division',  # 并行分支
                        'batch_norm_insertion'  # 规范化
                    ])
                elif problem['type'] == 'gradient_blocked':
                    repair_actions.extend([
                        'residual_connection',  # 改善梯度流
                        'batch_norm_insertion',  # 稳定训练
                        'layer_norm'  # 层归一化
                    ])
                elif problem['type'] == 'representation_degradation':
                    repair_actions.extend([
                        'width_expansion',  # 增强表示能力
                        'attention_enhancement',  # 注意力机制
                        'information_enhancement'  # 信息增强
                    ])
            
            # 去重并排序
            unique_actions = list(set(repair_actions))
            
            suggestion = {
                'layer_name': layer_name,
                'priority': point['priority'],
                'recommended_actions': unique_actions,
                'primary_action': unique_actions[0] if unique_actions else 'width_expansion',
                'rationale': f"检测到{len(problems)}个问题: " + 
                           ', '.join(p['description'] for p in problems),
                'expected_improvement': min(1.0, point['total_severity'] * 0.5)
            }
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_leakage_summary(self, leakage_points: List[Dict]) -> Dict[str, Any]:
        """生成泄漏分析摘要"""
        
        if not leakage_points:
            return {
                'total_leakage_points': 0,
                'average_severity': 0.0,
                'most_critical_layer': None,
                'summary': "未检测到明显的信息泄漏问题"
            }
        
        total_points = len(leakage_points)
        avg_severity = np.mean([p['average_severity'] for p in leakage_points])
        most_critical = leakage_points[0] if leakage_points else None
        
        # 问题类型统计
        problem_types = defaultdict(int)
        for point in leakage_points:
            for problem in point['problems']:
                problem_types[problem['type']] += 1
        
        summary_text = f"检测到{total_points}个泄漏点，平均严重性{avg_severity:.2f}。"
        if most_critical:
            summary_text += f"最关键层: {most_critical['layer_name']}"
        
        return {
            'total_leakage_points': total_points,
            'average_severity': avg_severity,
            'most_critical_layer': most_critical['layer_name'] if most_critical else None,
            'problem_type_distribution': dict(problem_types),
            'summary': summary_text
        }