#!/usr/bin/env python3
"""
"""
defgroup group_enhanced_bottleneck_detector Enhanced Bottleneck Detector
ingroup core
Enhanced Bottleneck Detector module for NeuroExapt framework.
"""


Enhanced Bottleneck Detector for DNM Framework

增强的瓶颈检测器，实现多维度网络瓶颈分析
基于梯度方差、激活多样性、信息流等指标进行综合评估
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EnhancedBottleneckDetector:
    """增强的瓶颈检测器
    
    基于多维度指标检测网络中的性能瓶颈：
    1. 梯度方差分析 - 检测梯度传播瓶颈
    2. 激活多样性 - 评估神经元激活的多样性
    3. 信息流分析 - 基于信息论的层重要性
    4. 层贡献度 - 通过扰动分析层的重要性
    5. 性能敏感度 - 评估层对最终性能的影响
    """
    
    def __init__(self, 
                 sensitivity_threshold: float = 0.1,
                 diversity_threshold: float = 0.3,
                 gradient_threshold: float = 1e-6,
                 info_flow_threshold: float = 0.5):
        """初始化瓶颈检测器
        
        Args:
            sensitivity_threshold: 性能敏感度阈值
            diversity_threshold: 激活多样性阈值
            gradient_threshold: 梯度方差阈值
            info_flow_threshold: 信息流阈值
        """
        self.sensitivity_threshold = sensitivity_threshold
        self.diversity_threshold = diversity_threshold
        self.gradient_threshold = gradient_threshold
        self.info_flow_threshold = info_flow_threshold
        
        # 历史数据缓存
        self.gradient_history = defaultdict(list)
        self.activation_history = defaultdict(list)
        self.performance_history = []
        
        # 评估指标权重
        self.metric_weights = {
            'gradient_variance': 0.25,
            'activation_diversity': 0.20,
            'information_flow': 0.25,
            'layer_contribution': 0.20,
            'performance_sensitivity': 0.10
        }
    
    def detect_bottlenecks(self, 
                          model: nn.Module, 
                          activations: Dict[str, torch.Tensor],
                          gradients: Dict[str, torch.Tensor],
                          targets: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """检测网络瓶颈
        
        Args:
            model: 神经网络模型
            activations: 层激活值字典
            gradients: 层梯度字典
            targets: 目标值（可选）
            
        Returns:
            Dict[str, float]: 层名称到瓶颈分数的映射
        """
        bottleneck_scores = {}
        
        # 获取所有评估的层
        layer_names = self._get_evaluable_layers(model)
        
        for layer_name in layer_names:
            try:
                # 计算各维度指标
                gradient_score = self._compute_gradient_variance_score(layer_name, gradients)
                diversity_score = self._compute_activation_diversity_score(layer_name, activations)
                info_flow_score = self._compute_information_flow_score(layer_name, activations, targets)
                contribution_score = self._compute_layer_contribution_score(layer_name, model, activations)
                sensitivity_score = self._compute_performance_sensitivity_score(layer_name, gradients)
                
                # 加权综合评分
                total_score = (
                    self.metric_weights['gradient_variance'] * gradient_score +
                    self.metric_weights['activation_diversity'] * diversity_score +
                    self.metric_weights['information_flow'] * info_flow_score +
                    self.metric_weights['layer_contribution'] * contribution_score +
                    self.metric_weights['performance_sensitivity'] * sensitivity_score
                )
                
                bottleneck_scores[layer_name] = total_score
                
                logger.debug(f"Layer {layer_name}: grad={gradient_score:.3f}, "
                           f"div={diversity_score:.3f}, info={info_flow_score:.3f}, "
                           f"contrib={contribution_score:.3f}, sens={sensitivity_score:.3f}, "
                           f"total={total_score:.3f}")
                
            except Exception as e:
                logger.warning(f"Error evaluating layer {layer_name}: {e}")
                bottleneck_scores[layer_name] = 0.0
        
        return bottleneck_scores
    
    def _get_evaluable_layers(self, model: nn.Module) -> List[str]:
        """获取可评估的层名称列表"""
        evaluable_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                # 排除太浅或太深的层
                if len(name.split('.')) >= 2:
                    evaluable_layers.append(name)
        return evaluable_layers
    
    def _compute_gradient_variance_score(self, 
                                       layer_name: str, 
                                       gradients: Dict[str, torch.Tensor]) -> float:
        """计算梯度方差评分
        
        梯度方差低表示该层可能是瓶颈
        """
        if layer_name not in gradients:
            return 0.0
        
        grad = gradients[layer_name]
        if grad is None or grad.numel() == 0:
            # Assign a high score to indicate a potential bottleneck due to a dead/disconnected layer
            return 1.0
        
        # 计算梯度方差
        grad_var = torch.var(grad).item()
        
        # 更新历史记录
        self.gradient_history[layer_name].append(grad_var)
        if len(self.gradient_history[layer_name]) > 10:
            self.gradient_history[layer_name].pop(0)
        
        # 计算相对方差（相对于历史平均值）
        if len(self.gradient_history[layer_name]) > 1:
            historical_mean = np.mean(self.gradient_history[layer_name][:-1])
            if historical_mean > 0:
                relative_var = grad_var / historical_mean
            else:
                relative_var = 1.0
        else:
            relative_var = 1.0
        
        # 低梯度方差 = 高瓶颈分数
        if grad_var < self.gradient_threshold:
            return 1.0
        else:
            return max(0.0, 1.0 - np.log10(grad_var + 1e-10) / 5.0)
    
    def _compute_activation_diversity_score(self, 
                                          layer_name: str,
                                          activations: Dict[str, torch.Tensor]) -> float:
        """计算激活多样性评分
        
        激活多样性低表示神经元功能相似，可能需要分裂
        """
        if layer_name not in activations:
            return 0.0
        
        activation = activations[layer_name]
        if activation is None or activation.numel() == 0:
            return 0.0
        
        # 计算激活值的多样性
        if len(activation.shape) >= 4:  # Conv层 (N, C, H, W)
            # 在空间维度上平均，获得 (N, C)
            activation_flat = activation.mean(dim=(2, 3))
        elif len(activation.shape) == 2:  # 全连接层 (N, F)
            activation_flat = activation
        else:
            return 0.0
        
        # 计算神经元间的相关性
        if activation_flat.shape[1] > 1 and activation_flat.shape[0] >= 2:  # 至少需要2个样本
            try:
                # 检查批次大小，torch.corrcoef 在小批次时可能不稳定
                if activation_flat.shape[0] < 10:
                    # 对于小批次，使用更稳定的方法计算相关性
                    correlations = []
                    for i in range(activation_flat.shape[1]):
                        for j in range(i+1, activation_flat.shape[1]):
                            corr = torch.corrcoef(torch.stack([activation_flat[:, i], activation_flat[:, j]]))[0, 1]
                            if not torch.isnan(corr):
                                correlations.append(abs(corr.item()))
                    
                    mean_correlation = np.mean(correlations) if correlations else 0.0
                else:
                    correlation_matrix = torch.corrcoef(activation_flat.T)
                    # 去除对角线元素
                    mask = ~torch.eye(correlation_matrix.shape[0], dtype=torch.bool)
                    correlations = correlation_matrix[mask]
                    
                    # 过滤掉NaN值
                    valid_correlations = correlations[~torch.isnan(correlations)]
                    mean_correlation = torch.abs(valid_correlations).mean().item() if len(valid_correlations) > 0 else 0.0
                
                diversity_score = mean_correlation  # 高相关性 = 高瓶颈分数
            except Exception as e:
                # 如果相关系数计算失败，返回默认值
                diversity_score = 0.0
        else:
            diversity_score = 0.0
        
        return min(1.0, max(0.0, diversity_score))
    
    def _compute_information_flow_score(self, 
                                      layer_name: str,
                                      activations: Dict[str, torch.Tensor],
                                      targets: Optional[torch.Tensor]) -> float:
        """计算信息流评分
        
        基于信息论的方法评估层的信息处理能力
        """
        if layer_name not in activations or targets is None:
            return 0.0
        
        activation = activations[layer_name]
        if activation is None or activation.numel() == 0:
            return 0.0
        
        try:
            # 简化的互信息估计
            if len(activation.shape) >= 4:
                activation_flat = activation.mean(dim=(2, 3))
            elif len(activation.shape) == 2:
                activation_flat = activation
            else:
                return 0.0
            
            # 计算激活值的熵
            activation_entropy = self._estimate_entropy(activation_flat)
            
            # 简单的信息流评分：低熵表示信息瓶颈
            info_score = 1.0 - min(1.0, activation_entropy / 5.0)  # 归一化到[0,1]
            
            return max(0.0, info_score)
            
        except Exception as e:
            logger.warning(f"Error computing information flow for {layer_name}: {e}")
            return 0.0
    
    def _estimate_entropy(self, tensor: torch.Tensor) -> float:
        """估计张量的熵"""
        try:
            # 简单的熵估计方法
            tensor_flat = tensor.flatten()
            
            # 使用直方图估计概率分布
            hist = torch.histc(tensor_flat, bins=50)
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # 移除零概率
            
            # 计算熵
            entropy = -torch.sum(hist * torch.log2(hist + 1e-10)).item()
            return entropy
            
        except Exception:
            return 1.0  # 默认值
    
    def _compute_layer_contribution_score(self, 
                                        layer_name: str,
                                        model: nn.Module,
                                        activations: Dict[str, torch.Tensor]) -> float:
        """计算层贡献度评分
        
        通过扰动分析评估层对网络输出的贡献
        """
        if layer_name not in activations:
            return 0.0
        
        activation = activations[layer_name]
        if activation is None or activation.numel() == 0:
            return 0.0
        
        # 简化的贡献度评估：基于激活值的幅度
        activation_magnitude = torch.norm(activation).item()
        
        # 归一化处理
        if activation_magnitude > 0:
            # 将幅度转换为贡献度评分
            contribution_score = min(1.0, activation_magnitude / 100.0)
        else:
            contribution_score = 1.0  # 零激活可能表示瓶颈
        
        return 1.0 - contribution_score  # 低贡献 = 高瓶颈分数
    
    def _compute_performance_sensitivity_score(self, 
                                             layer_name: str,
                                             gradients: Dict[str, torch.Tensor]) -> float:
        """计算性能敏感度评分
        
        基于梯度幅度评估层对性能的敏感度
        """
        if layer_name not in gradients:
            return 0.0
        
        grad = gradients[layer_name]
        if grad is None or grad.numel() == 0:
            return 0.0
        
        # 计算梯度幅度
        grad_magnitude = torch.norm(grad).item()
        
        # 低梯度幅度可能表示该层对性能不敏感，可能是瓶颈
        if grad_magnitude < self.gradient_threshold:
            return 1.0
        else:
            return max(0.0, 1.0 - np.log10(grad_magnitude + 1e-10) / 3.0)
    
    def update_performance_history(self, performance: float):
        """更新性能历史记录"""
        self.performance_history.append(performance)
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)
    
    def get_top_bottlenecks(self, 
                           bottleneck_scores: Dict[str, float], 
                           top_k: int = 3) -> List[Tuple[str, float]]:
        """获取得分最高的瓶颈层"""
        sorted_scores = sorted(bottleneck_scores.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
        return sorted_scores[:top_k]
    
    def should_trigger_division(self, 
                              bottleneck_scores: Dict[str, float],
                              performance_trend: List[float]) -> Tuple[bool, List[str]]:
        """判断是否应该触发神经元分裂
        
        Args:
            bottleneck_scores: 瓶颈评分字典
            performance_trend: 最近的性能趋势
            
        Returns:
            Tuple[bool, List[str]]: (是否触发, 触发原因列表)
        """
        reasons = []
        
        # 检查是否有高瓶颈分数的层
        max_score = max(bottleneck_scores.values()) if bottleneck_scores else 0.0
        if max_score > 0.7:
            reasons.append(f"高瓶颈分数检测: {max_score:.3f}")
        
        # 检查性能停滞
        if len(performance_trend) >= 3:
            recent_improvement = performance_trend[-1] - performance_trend[-3]
            if recent_improvement < 0.01:
                reasons.append(f"性能停滞: 最近改善 {recent_improvement:.3f}")
        
        # 检查多个层的瓶颈
        high_score_layers = [name for name, score in bottleneck_scores.items() if score > 0.5]
        if len(high_score_layers) >= 2:
            reasons.append(f"多层瓶颈: {len(high_score_layers)} 层")
        
        should_trigger = len(reasons) > 0
        return should_trigger, reasons
    
    def get_analysis_summary(self, bottleneck_scores: Dict[str, float]) -> Dict[str, Any]:
        """获取分析摘要"""
        if not bottleneck_scores:
            return {}
        
        scores = list(bottleneck_scores.values())
        return {
            'total_layers': len(bottleneck_scores),
            'max_score': max(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'high_score_layers': len([s for s in scores if s > 0.5]),
            'top_bottlenecks': self.get_top_bottlenecks(bottleneck_scores, 3)
        }