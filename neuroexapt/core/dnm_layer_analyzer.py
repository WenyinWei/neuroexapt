#!/usr/bin/env python3
"""
"""
defgroup group_dnm_layer_analyzer Dnm Layer Analyzer
ingroup core
Dnm Layer Analyzer module for NeuroExapt framework.
"""


DNM Layer Performance Analyzer - 层级性能分析器

🎯 核心功能：
1. 逐层性能瓶颈识别
2. 特征表示质量评估  
3. 梯度流分析
4. 信息传递效率评估
5. 智能分裂位置推荐

🧬 分析维度：
- 信息论指标（熵、互信息）
- 梯度健康度
- 特征分离度
- 激活饱和度
- 学习效率

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LayerPerformanceAnalyzer:
    """层级性能分析器"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_metrics_history = defaultdict(lambda: deque(maxlen=10))
        self.performance_baseline = {}
        self.critical_layers = set()
        
    def analyze_all_layers(self, activations: Dict[str, torch.Tensor], 
                          gradients: Dict[str, torch.Tensor],
                          targets: torch.Tensor,
                          current_accuracy: float) -> Dict[str, Dict[str, float]]:
        """分析所有层的性能指标"""
        
        layer_analysis = {}
        
        for layer_name in activations.keys():
            if layer_name in gradients and gradients[layer_name] is not None:
                analysis = self._analyze_single_layer(
                    layer_name, 
                    activations[layer_name], 
                    gradients[layer_name],
                    targets,
                    current_accuracy
                )
                layer_analysis[layer_name] = analysis
                
                # 更新历史记录
                self.layer_metrics_history[layer_name].append(analysis)
        
        return layer_analysis
    
    def _analyze_single_layer(self, layer_name: str, 
                             activation: torch.Tensor,
                             gradient: torch.Tensor, 
                             targets: torch.Tensor,
                             current_accuracy: float) -> Dict[str, float]:
        """分析单个层的性能"""
        
        metrics = {}
        
        # 1. 信息论分析
        metrics.update(self._compute_information_metrics(activation))
        
        # 2. 梯度健康度分析  
        metrics.update(self._compute_gradient_health(gradient))
        
        # 3. 特征表示质量
        metrics.update(self._compute_representation_quality(activation, targets))
        
        # 4. 学习效率分析
        metrics.update(self._compute_learning_efficiency(layer_name, current_accuracy))
        
        # 5. 计算综合瓶颈分数
        metrics['bottleneck_score'] = self._compute_bottleneck_score(metrics)
        
        return metrics
    
    def _compute_information_metrics(self, activation: torch.Tensor) -> Dict[str, float]:
        """计算信息论指标"""
        metrics = {}
        
        # 展平激活值
        act_flat = activation.view(activation.size(0), -1)
        
        if act_flat.size(1) == 0:
            return {'entropy': 0.0, 'mutual_info_proxy': 0.0, 'information_flow': 0.0}
        
        # 计算激活熵
        try:
            # 归一化激活值
            act_norm = F.softmax(act_flat, dim=-1) + 1e-8
            entropy = -torch.sum(act_norm * torch.log(act_norm), dim=-1).mean().item()
            metrics['entropy'] = entropy
        except:
            metrics['entropy'] = 0.0
        
        # 计算特征间相关性（互信息代理）
        try:
            if act_flat.size(1) > 1:
                correlation_matrix = torch.corrcoef(act_flat.T)
                avg_correlation = torch.mean(torch.abs(correlation_matrix - torch.eye(correlation_matrix.size(0), device=correlation_matrix.device))).item()
                metrics['mutual_info_proxy'] = avg_correlation
            else:
                metrics['mutual_info_proxy'] = 0.0
        except:
            metrics['mutual_info_proxy'] = 0.0
        
        # 信息流指标（激活值的标准差）
        metrics['information_flow'] = torch.std(act_flat).item()
        
        return metrics
    
    def _compute_gradient_health(self, gradient: torch.Tensor) -> Dict[str, float]:
        """计算梯度健康度"""
        metrics = {}
        
        grad_flat = gradient.view(-1)
        
        # 梯度范数
        metrics['gradient_norm'] = torch.norm(grad_flat).item()
        
        # 梯度稳定性（标准差与均值的比）
        grad_mean = torch.mean(torch.abs(grad_flat)).item()
        grad_std = torch.std(grad_flat).item()
        metrics['gradient_stability'] = grad_std / (grad_mean + 1e-8)
        
        # 梯度饱和度（接近0的梯度比例）
        near_zero_ratio = torch.sum(torch.abs(grad_flat) < 1e-6).float() / len(grad_flat)
        metrics['gradient_saturation'] = near_zero_ratio.item()
        
        # 梯度健康分数
        if metrics['gradient_norm'] < 1e-8:
            health_score = 0.0  # 梯度消失
        elif metrics['gradient_norm'] > 100:
            health_score = 0.2  # 梯度爆炸
        else:
            health_score = 1.0 / (1.0 + metrics['gradient_stability'])
        
        metrics['gradient_health'] = health_score
        
        return metrics
    
    def _compute_representation_quality(self, activation: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """计算特征表示质量"""
        metrics = {}
        
        act_flat = activation.view(activation.size(0), -1)
        
        if act_flat.size(1) < 2 or len(torch.unique(targets)) < 2:
            return {
                'feature_separability': 0.0,
                'activation_diversity': 0.0,
                'representation_efficiency': 0.0
            }
        
        # 特征可分离性（使用类内/类间距离比）
        try:
            separability = self._compute_feature_separability(act_flat, targets)
            metrics['feature_separability'] = separability
        except:
            metrics['feature_separability'] = 0.0
        
        # 激活多样性
        activation_diversity = self._compute_activation_diversity(act_flat)
        metrics['activation_diversity'] = activation_diversity
        
        # 表示效率（信息密度）
        representation_efficiency = self._compute_representation_efficiency(act_flat)
        metrics['representation_efficiency'] = representation_efficiency
        
        return metrics
    
    def _compute_feature_separability(self, features: torch.Tensor, targets: torch.Tensor) -> float:
        """计算特征可分离性"""
        unique_labels = torch.unique(targets)
        
        if len(unique_labels) < 2:
            return 0.0
        
        # 计算类内距离和类间距离
        intra_class_distances = []
        inter_class_distances = []
        
        for label in unique_labels:
            mask = targets == label
            if torch.sum(mask) < 2:
                continue
                
            class_features = features[mask]
            class_center = torch.mean(class_features, dim=0)
            
            # 类内距离
            intra_dist = torch.mean(torch.norm(class_features - class_center, dim=1)).item()
            intra_class_distances.append(intra_dist)
            
            # 类间距离
            for other_label in unique_labels:
                if other_label <= label:
                    continue
                other_mask = targets == other_label
                if torch.sum(other_mask) == 0:
                    continue
                    
                other_center = torch.mean(features[other_mask], dim=0)
                inter_dist = torch.norm(class_center - other_center).item()
                inter_class_distances.append(inter_dist)
        
        if not intra_class_distances or not inter_class_distances:
            return 0.0
        
        avg_intra = np.mean(intra_class_distances)
        avg_inter = np.mean(inter_class_distances)
        
        # 可分离性 = 类间距离 / 类内距离
        separability = avg_inter / (avg_intra + 1e-8)
        return min(separability, 10.0)  # 限制上界
    
    def _compute_activation_diversity(self, activation: torch.Tensor) -> float:
        """计算激活多样性"""
        # 使用激活值的方差作为多样性指标
        variance_per_neuron = torch.var(activation, dim=0)
        avg_variance = torch.mean(variance_per_neuron).item()
        
        # 计算激活模式的多样性（不同样本间的差异）
        if activation.size(0) > 1:
            pairwise_distances = torch.pdist(activation)
            avg_pairwise_distance = torch.mean(pairwise_distances).item()
        else:
            avg_pairwise_distance = 0.0
        
        # 综合多样性分数
        diversity = np.sqrt(avg_variance * avg_pairwise_distance)
        return diversity
    
    def _compute_representation_efficiency(self, activation: torch.Tensor) -> float:
        """计算表示效率"""
        # 有效维度 vs 总维度
        variance_per_neuron = torch.var(activation, dim=0)
        active_neurons = torch.sum(variance_per_neuron > 1e-6).item()
        total_neurons = activation.size(1)
        
        efficiency = active_neurons / (total_neurons + 1e-8)
        return efficiency
    
    def _compute_learning_efficiency(self, layer_name: str, current_accuracy: float) -> Dict[str, float]:
        """计算学习效率"""
        metrics = {}
        
        history = self.layer_metrics_history[layer_name]
        
        if len(history) < 3:
            metrics['learning_rate'] = 0.0
            metrics['improvement_trend'] = 0.0
            return metrics
        
        # 计算最近几个epoch的改进趋势
        recent_accuracies = [current_accuracy] + [h.get('accuracy_impact', 0) for h in list(history)[-3:]]
        
        if len(recent_accuracies) > 1:
            # 计算改进趋势
            improvements = np.diff(recent_accuracies)
            avg_improvement = np.mean(improvements)
            metrics['improvement_trend'] = avg_improvement
            
            # 学习率（改进的一致性）
            consistency = 1.0 - np.std(improvements)
            metrics['learning_rate'] = max(0.0, consistency)
        else:
            metrics['learning_rate'] = 0.0
            metrics['improvement_trend'] = 0.0
        
        return metrics
    
    def _compute_bottleneck_score(self, metrics: Dict[str, float]) -> float:
        """计算综合瓶颈分数（越高越需要改进）"""
        
        # 权重配置
        weights = {
            'gradient_health': 0.25,      # 梯度健康度（越低越需要改进）
            'feature_separability': 0.20,  # 特征可分离性（越低越需要改进）
            'representation_efficiency': 0.15,  # 表示效率
            'information_flow': 0.15,     # 信息流
            'learning_rate': 0.15,        # 学习效率
            'activation_diversity': 0.10   # 激活多样性
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                
                # 将所有指标转换为"需要改进"的分数（越高越需要改进）
                if metric in ['gradient_health', 'feature_separability', 'representation_efficiency', 
                             'information_flow', 'learning_rate', 'activation_diversity']:
                    # 这些指标越低越需要改进
                    improvement_need = 1.0 - min(1.0, value)
                else:
                    improvement_need = min(1.0, value)
                
                score += improvement_need * weight
                total_weight += weight
        
        if total_weight > 0:
            score /= total_weight
        
        return score
    
    def recommend_optimal_layers(self, layer_analysis: Dict[str, Dict[str, float]], 
                                top_k: int = 3) -> List[Tuple[str, float, str]]:
        """推荐最需要改进的层"""
        
        recommendations = []
        
        for layer_name, metrics in layer_analysis.items():
            bottleneck_score = metrics.get('bottleneck_score', 0.0)
            
            # 分析主要问题
            main_issue = self._identify_main_issue(metrics)
            
            recommendations.append((layer_name, bottleneck_score, main_issue))
        
        # 按瓶颈分数排序
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:top_k]
    
    def _identify_main_issue(self, metrics: Dict[str, float]) -> str:
        """识别层的主要问题"""
        
        issues = []
        
        # 梯度问题
        if metrics.get('gradient_health', 1.0) < 0.3:
            if metrics.get('gradient_norm', 0) < 1e-6:
                issues.append("梯度消失")
            elif metrics.get('gradient_norm', 0) > 100:
                issues.append("梯度爆炸")
            else:
                issues.append("梯度不稳定")
        
        # 特征表示问题
        if metrics.get('feature_separability', 1.0) < 0.5:
            issues.append("特征分离度低")
        
        # 表示效率问题
        if metrics.get('representation_efficiency', 1.0) < 0.3:
            issues.append("表示效率低")
        
        # 信息流问题
        if metrics.get('information_flow', 1.0) < 0.1:
            issues.append("信息流受阻")
        
        # 学习效率问题
        if metrics.get('learning_rate', 1.0) < 0.2:
            issues.append("学习效率低")
        
        if not issues:
            return "性能良好"
        
        return " + ".join(issues[:2])  # 最多显示两个主要问题

class SmartLayerSelector:
    """智能层选择器"""
    
    def __init__(self, analyzer: LayerPerformanceAnalyzer):
        self.analyzer = analyzer
        self.selection_history = deque(maxlen=20)
        
    def select_optimal_division_layers(self, layer_analysis: Dict[str, Dict[str, float]], 
                                     max_selections: int = 2) -> List[Tuple[str, float, str]]:
        """智能选择最优分裂层"""
        
        # 获取推荐层
        recommendations = self.analyzer.recommend_optimal_layers(layer_analysis, top_k=10)
        
        # 过滤掉最近已经处理过的层
        filtered_recommendations = []
        recent_layers = set(h['layer'] for h in list(self.selection_history)[-5:])
        
        for layer_name, score, issue in recommendations:
            # 避免过度分裂同一层
            if layer_name not in recent_layers or score > 0.8:  # 高分数可以重复处理
                filtered_recommendations.append((layer_name, score, issue))
        
        # 选择多样化的层
        selected = self._select_diverse_layers(filtered_recommendations, max_selections)
        
        # 记录选择历史
        for layer_name, score, issue in selected:
            self.selection_history.append({
                'layer': layer_name,
                'score': score,
                'issue': issue,
                'epoch': len(self.selection_history)
            })
        
        return selected
    
    def _select_diverse_layers(self, recommendations: List[Tuple[str, float, str]], 
                              max_selections: int) -> List[Tuple[str, float, str]]:
        """选择多样化的层（避免都在同一个模块）"""
        
        if len(recommendations) <= max_selections:
            return recommendations
        
        selected = []
        used_modules = set()
        
        # 优先选择不同模块的层
        for layer_name, score, issue in recommendations:
            if len(selected) >= max_selections:
                break
                
            # 提取模块名（如 'features', 'classifier'）
            module_name = layer_name.split('.')[0] if '.' in layer_name else layer_name
            
            if module_name not in used_modules or len(selected) == 0:
                selected.append((layer_name, score, issue))
                used_modules.add(module_name)
        
        # 如果还没选够，继续选择高分数的层
        while len(selected) < max_selections and len(selected) < len(recommendations):
            for layer_name, score, issue in recommendations:
                if (layer_name, score, issue) not in selected:
                    selected.append((layer_name, score, issue))
                    break
        
        return selected