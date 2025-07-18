#!/usr/bin/env python3
"""
Performance-Guided Neuron Division for DNM Framework

基于性能导向的神经元分裂策略，实现精准、高效的神经元分裂
结合梯度信息、激活模式和性能反馈，确保分裂的有效性
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class DivisionStrategy(Enum):
    """分裂策略枚举"""
    GRADIENT_BASED = "gradient_based"
    ACTIVATION_BASED = "activation_based" 
    HYBRID = "hybrid"
    INFORMATION_GUIDED = "information_guided"


class PerformanceGuidedDivision:
    """基于性能导向的神经元分裂器
    
    提供多种分裂策略：
    1. 梯度导向分裂 - 基于梯度信息选择分裂位置
    2. 激活导向分裂 - 基于激活模式选择分裂位置
    3. 混合策略 - 综合考虑多种因素
    4. 信息导向分裂 - 基于信息论指标
    """
    
    def __init__(self, 
                 noise_scale: float = 0.1,
                 progressive_epochs: int = 5,
                 diversity_threshold: float = 0.8,
                 performance_monitoring: bool = True):
        """初始化分裂器
        
        Args:
            noise_scale: 权重初始化噪声强度
            progressive_epochs: 渐进式激活的周期数
            diversity_threshold: 多样性阈值
            performance_monitoring: 是否启用性能监控
        """
        self.noise_scale = noise_scale
        self.progressive_epochs = progressive_epochs
        self.diversity_threshold = diversity_threshold
        self.performance_monitoring = performance_monitoring
        
        # 分裂历史记录
        self.division_history = []
        self.performance_before_division = []
        self.performance_after_division = []
        
        # 渐进式激活状态
        self.progressive_weights = {}
        self.current_epoch = 0
    
    def divide_neuron(self,
                     layer: nn.Module,
                     neuron_idx: int,
                     strategy: DivisionStrategy = DivisionStrategy.HYBRID,
                     activations: Optional[torch.Tensor] = None,
                     gradients: Optional[torch.Tensor] = None,
                     targets: Optional[torch.Tensor] = None) -> Tuple[bool, Dict[str, Any]]:
        """执行神经元分裂
        
        Args:
            layer: 目标层
            neuron_idx: 神经元索引
            strategy: 分裂策略
            activations: 激活值
            gradients: 梯度信息
            targets: 目标值
            
        Returns:
            Tuple[bool, Dict]: (是否成功, 分裂信息)
        """
        try:
            if strategy == DivisionStrategy.GRADIENT_BASED:
                return self._gradient_guided_division(layer, neuron_idx, gradients)
            elif strategy == DivisionStrategy.ACTIVATION_BASED:
                return self._activation_guided_division(layer, neuron_idx, activations)
            elif strategy == DivisionStrategy.INFORMATION_GUIDED:
                return self._information_guided_division(layer, neuron_idx, activations, targets)
            else:  # HYBRID
                return self._hybrid_division_strategy(layer, neuron_idx, activations, gradients, targets)
                
        except Exception as e:
            logger.error(f"Error in neuron division: {e}")
            return False, {'error': str(e)}
    
    def _gradient_guided_division(self,
                                layer: nn.Module,
                                neuron_idx: int,
                                gradients: Optional[torch.Tensor]) -> Tuple[bool, Dict[str, Any]]:
        """基于梯度的神经元分裂"""
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            return False, {'error': 'Unsupported layer type for gradient-guided division'}
        
        if gradients is None:
            return False, {'error': 'No gradient information available'}
        
        try:
            # 获取原始权重
            original_weight = layer.weight.data.clone()
            
            if isinstance(layer, nn.Conv2d):
                success, info = self._divide_conv_neuron_gradient(layer, neuron_idx, gradients, original_weight)
            else:  # nn.Linear
                success, info = self._divide_linear_neuron_gradient(layer, neuron_idx, gradients, original_weight)
            
            if success:
                self._record_division('gradient_based', layer, neuron_idx, info)
            
            return success, info
            
        except Exception as e:
            logger.error(f"Error in gradient-guided division: {e}")
            return False, {'error': str(e)}
    
    def _activation_guided_division(self,
                                  layer: nn.Module,
                                  neuron_idx: int,
                                  activations: Optional[torch.Tensor]) -> Tuple[bool, Dict[str, Any]]:
        """基于激活的神经元分裂"""
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            return False, {'error': 'Unsupported layer type for activation-guided division'}
        
        if activations is None:
            return False, {'error': 'No activation information available'}
        
        try:
            # 分析激活模式
            activation_pattern = self._analyze_activation_pattern(activations, neuron_idx)
            
            # 获取原始权重
            original_weight = layer.weight.data.clone()
            
            if isinstance(layer, nn.Conv2d):
                success, info = self._divide_conv_neuron_gradient(layer, neuron_idx, None, original_weight)
                info['activation_pattern'] = activation_pattern
            else:  # nn.Linear
                success, info = self._divide_linear_neuron_gradient(layer, neuron_idx, None, original_weight)
                info['activation_pattern'] = activation_pattern
            
            if success:
                self._record_division('activation_based', layer, neuron_idx, info)
            
            return success, info
            
        except Exception as e:
            logger.error(f"Error in activation-guided division: {e}")
            return False, {'error': str(e)}
    
    def _information_guided_division(self,
                                   layer: nn.Module,
                                   neuron_idx: int,
                                   activations: Optional[torch.Tensor],
                                   targets: Optional[torch.Tensor]) -> Tuple[bool, Dict[str, Any]]:
        """基于信息论的神经元分裂"""
        if activations is None or targets is None:
            return False, {'error': 'Missing activation or target information'}
        
        try:
            # 计算信息论指标
            mutual_info = self._estimate_mutual_information(activations, targets, neuron_idx)
            entropy = self._estimate_neuron_entropy(activations, neuron_idx)
            
            # 基于信息论指导分裂
            if mutual_info > 0.1 and entropy > 0.5:  # 高信息量神经元
                return self._high_information_division(layer, neuron_idx, mutual_info, entropy)
            else:  # 低信息量神经元
                return self._low_information_division(layer, neuron_idx, mutual_info, entropy)
                
        except Exception as e:
            logger.error(f"Error in information-guided division: {e}")
            return False, {'error': str(e)}
    
    def _high_information_division(self, layer: nn.Module, neuron_idx: int, mutual_info: float, entropy: float) -> Tuple[bool, Dict[str, Any]]:
        """高信息量神经元的分裂策略"""
        # 对于高信息量神经元，使用保守策略
        return self._conservative_division(layer, neuron_idx, mutual_info + entropy)
    
    def _low_information_division(self, layer: nn.Module, neuron_idx: int, mutual_info: float, entropy: float) -> Tuple[bool, Dict[str, Any]]:
        """低信息量神经元的分裂策略"""
        # 对于低信息量神经元，使用激进策略
        return self._aggressive_division(layer, neuron_idx, mutual_info + entropy)
    
    def _hybrid_division_strategy(self,
                                layer: nn.Module,
                                neuron_idx: int,
                                activations: Optional[torch.Tensor],
                                gradients: Optional[torch.Tensor],
                                targets: Optional[torch.Tensor]) -> Tuple[bool, Dict[str, Any]]:
        """混合分裂策略"""
        try:
            # 收集所有可用信息
            gradient_score = self._evaluate_gradient_importance(gradients, neuron_idx) if gradients is not None else 0.0
            activation_score = self._evaluate_activation_importance(activations, neuron_idx) if activations is not None else 0.0
            info_score = self._evaluate_information_importance(activations, targets, neuron_idx) if activations is not None and targets is not None else 0.0
            
            # 加权组合评分
            total_score = 0.4 * gradient_score + 0.3 * activation_score + 0.3 * info_score
            
            # 根据综合评分选择分裂策略
            if total_score > 0.7:
                # 高重要性神经元 - 使用保守分裂
                return self._conservative_division(layer, neuron_idx, total_score)
            elif total_score > 0.3:
                # 中等重要性神经元 - 使用标准分裂
                return self._standard_division(layer, neuron_idx, total_score)
            else:
                # 低重要性神经元 - 使用激进分裂
                return self._aggressive_division(layer, neuron_idx, total_score)
                
        except Exception as e:
            logger.error(f"Error in hybrid division strategy: {e}")
            return False, {'error': str(e)}
    
    def _divide_conv_neuron_gradient(self,
                                   layer: nn.Conv2d,
                                   neuron_idx: int,
                                   gradients: torch.Tensor,
                                   original_weight: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """基于梯度分裂卷积神经元"""
        out_channels, in_channels, kernel_h, kernel_w = layer.weight.shape
        
        if neuron_idx >= out_channels:
            return False, {'error': f'Neuron index {neuron_idx} out of range for {out_channels} channels'}
        
        # 创建新的层
        new_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels + 1,
            kernel_size=(kernel_h, kernel_w),
            stride=layer.stride,
            padding=layer.padding,
            bias=layer.bias is not None
        )
        
        # 复制原始权重
        new_layer.weight.data[:out_channels] = original_weight
        
        # 基于梯度信息初始化新神经元
        parent_grad = gradients[neuron_idx] if gradients.dim() > 0 else gradients
        if isinstance(parent_grad, torch.Tensor) and parent_grad.numel() > 0:
            # 添加梯度导向的变化
            noise = torch.randn_like(original_weight[neuron_idx]) * self.noise_scale
            gradient_direction = torch.sign(parent_grad.mean()) if parent_grad.numel() > 0 else 1.0
            new_layer.weight.data[out_channels] = original_weight[neuron_idx] + gradient_direction * noise
        else:
            # 默认噪声初始化
            new_layer.weight.data[out_channels] = original_weight[neuron_idx] + torch.randn_like(original_weight[neuron_idx]) * self.noise_scale
        
        # 处理偏置
        if layer.bias is not None:
            new_layer.bias.data[:out_channels] = layer.bias.data
            new_layer.bias.data[out_channels] = layer.bias.data[neuron_idx] + (torch.randn(1) * self.noise_scale).item()
        
        # 替换层
        self._replace_layer_in_model(layer, new_layer)
        
        return True, {
            'strategy': 'gradient_based',
            'original_channels': out_channels,
            'new_channels': out_channels + 1,
            'neuron_idx': neuron_idx,
            'noise_scale': self.noise_scale
        }
    
    def _divide_linear_neuron_gradient(self,
                                     layer: nn.Linear,
                                     neuron_idx: int,
                                     gradients: torch.Tensor,
                                     original_weight: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """基于梯度分裂线性神经元"""
        out_features, in_features = layer.weight.shape
        
        if neuron_idx >= out_features:
            return False, {'error': f'Neuron index {neuron_idx} out of range for {out_features} features'}
        
        # 创建新的层
        new_layer = nn.Linear(in_features, out_features + 1, bias=layer.bias is not None)
        
        # 复制原始权重
        new_layer.weight.data[:out_features] = original_weight
        
        # 基于梯度信息初始化新神经元
        parent_grad = gradients[neuron_idx] if gradients.dim() > 0 else gradients
        if isinstance(parent_grad, torch.Tensor) and parent_grad.numel() > 0:
            gradient_direction = torch.sign(parent_grad.mean()) if parent_grad.numel() > 0 else 1.0
            noise = torch.randn_like(original_weight[neuron_idx]) * self.noise_scale
            new_layer.weight.data[out_features] = original_weight[neuron_idx] + gradient_direction * noise
        else:
            new_layer.weight.data[out_features] = original_weight[neuron_idx] + torch.randn_like(original_weight[neuron_idx]) * self.noise_scale
        
        # 处理偏置
        if layer.bias is not None:
            new_layer.bias.data[:out_features] = layer.bias.data
            new_layer.bias.data[out_features] = layer.bias.data[neuron_idx] + torch.randn(1) * self.noise_scale
        
        return True, {
            'strategy': 'gradient_based',
            'original_features': out_features,
            'new_features': out_features + 1,
            'neuron_idx': neuron_idx,
            'new_layer': new_layer
        }
    
    def _analyze_activation_pattern(self, activations: torch.Tensor, neuron_idx: int) -> Dict[str, float]:
        """分析激活模式"""
        if activations.dim() == 4:  # Conv layer: (N, C, H, W)
            if neuron_idx < activations.shape[1]:
                neuron_activations = activations[:, neuron_idx, :, :]
            else:
                neuron_activations = activations.mean(dim=1)
        elif activations.dim() == 2:  # Linear layer: (N, F)
            if neuron_idx < activations.shape[1]:
                neuron_activations = activations[:, neuron_idx]
            else:
                neuron_activations = activations.mean(dim=1)
        else:
            neuron_activations = activations.flatten()
        
        return {
            'mean': neuron_activations.mean().item(),
            'std': neuron_activations.std().item(),
            'sparsity': (neuron_activations == 0).float().mean().item(),
            'max_activation': neuron_activations.max().item(),
            'min_activation': neuron_activations.min().item()
        }
    
    def _estimate_mutual_information(self, activations: torch.Tensor, targets: torch.Tensor, neuron_idx: int) -> float:
        """估计互信息"""
        try:
            # 简化的互信息估计
            if activations.dim() == 4 and neuron_idx < activations.shape[1]:
                neuron_activations = activations[:, neuron_idx, :, :].mean(dim=(1, 2))
            elif activations.dim() == 2 and neuron_idx < activations.shape[1]:
                neuron_activations = activations[:, neuron_idx]
            else:
                return 0.0
            
                        # 使用相关系数作为互信息的近似
            # 注意：相关系数仅能捕捉线性关系，作为互信息的代理可能会导致误导性结论。
            # 更健壮的信息传递估计方法（如基于分箱的互信息估计、sklearn 的 mutual_info_regression/mutual_info_classif 等）可用于更准确的评估。
            # 具体实现可根据需求替换此处相关系数的计算。
            if targets.dim() > 1:
                targets_flat = targets.argmax(dim=1).float()
            else:
                targets_flat = targets.float()

            correlation = torch.corrcoef(torch.stack([neuron_activations, targets_flat]))[0, 1]
            return abs(correlation.item()) if not torch.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _estimate_neuron_entropy(self, activations: torch.Tensor, neuron_idx: int) -> float:
        """估计神经元熵"""
        try:
            if activations.dim() == 4 and neuron_idx < activations.shape[1]:
                neuron_activations = activations[:, neuron_idx, :, :].flatten()
            elif activations.dim() == 2 and neuron_idx < activations.shape[1]:
                neuron_activations = activations[:, neuron_idx]
            else:
                return 0.0
            
            # 使用直方图估计熵
            hist = torch.histc(neuron_activations, bins=20)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            
            entropy = -torch.sum(hist * torch.log2(hist + 1e-10)).item()
            return entropy
            
        except Exception:
            return 0.0
    
    def _evaluate_gradient_importance(self, gradients: torch.Tensor, neuron_idx: int) -> float:
        """评估梯度重要性"""
        if gradients is None:
            return 0.0
        
        try:
            if gradients.dim() > 0 and neuron_idx < gradients.shape[0]:
                grad_magnitude = torch.norm(gradients[neuron_idx]).item()
            else:
                grad_magnitude = torch.norm(gradients).item()
            
            # 归一化到 [0, 1]
            return min(1.0, grad_magnitude / 10.0)
        except Exception:
            return 0.0
    
    def _evaluate_activation_importance(self, activations: torch.Tensor, neuron_idx: int) -> float:
        """评估激活重要性"""
        if activations is None:
            return 0.0
        
        try:
            pattern = self._analyze_activation_pattern(activations, neuron_idx)
            
            # 基于激活统计计算重要性
            sparsity_score = 1.0 - pattern['sparsity']  # 低稀疏性 = 高重要性
            variance_score = min(1.0, pattern['std'] / 5.0)  # 高方差 = 高重要性
            magnitude_score = min(1.0, abs(pattern['mean']) / 5.0)  # 高幅度 = 高重要性
            
            return (sparsity_score + variance_score + magnitude_score) / 3.0
        except Exception:
            return 0.0
    
    def _evaluate_information_importance(self, activations: torch.Tensor, targets: torch.Tensor, neuron_idx: int) -> float:
        """评估信息重要性"""
        if activations is None or targets is None:
            return 0.0
        
        try:
            mutual_info = self._estimate_mutual_information(activations, targets, neuron_idx)
            entropy = self._estimate_neuron_entropy(activations, neuron_idx)
            
            # 结合互信息和熵计算重要性
            return min(1.0, (mutual_info + entropy / 5.0) / 2.0)
        except Exception:
            return 0.0
    
    def _conservative_division(self, layer: nn.Module, neuron_idx: int, score: float) -> Tuple[bool, Dict[str, Any]]:
        """保守分裂策略 - 用于高重要性神经元"""
        # 使用更小的噪声，保持功能稳定性
        original_noise = self.noise_scale
        self.noise_scale = original_noise * 0.5
        
        try:
            if isinstance(layer, nn.Conv2d):
                success, info = self._divide_conv_neuron_gradient(layer, neuron_idx, None, layer.weight.data.clone())
            else:
                success, info = self._divide_linear_neuron_gradient(layer, neuron_idx, None, layer.weight.data.clone())
            
            info['division_type'] = 'conservative'
            info['importance_score'] = score
            return success, info
        finally:
            self.noise_scale = original_noise
    
    def _standard_division(self, layer: nn.Module, neuron_idx: int, score: float) -> Tuple[bool, Dict[str, Any]]:
        """标准分裂策略"""
        try:
            if isinstance(layer, nn.Conv2d):
                success, info = self._divide_conv_neuron_gradient(layer, neuron_idx, None, layer.weight.data.clone())
            else:
                success, info = self._divide_linear_neuron_gradient(layer, neuron_idx, None, layer.weight.data.clone())
            
            info['division_type'] = 'standard'
            info['importance_score'] = score
            return success, info
        except Exception as e:
            return False, {'error': str(e)}
    
    def _aggressive_division(self, layer: nn.Module, neuron_idx: int, score: float) -> Tuple[bool, Dict[str, Any]]:
        """激进分裂策略 - 用于低重要性神经元"""
        # 使用更大的噪声，促进功能分化
        original_noise = self.noise_scale
        self.noise_scale = original_noise * 1.5
        
        try:
            if isinstance(layer, nn.Conv2d):
                success, info = self._divide_conv_neuron_gradient(layer, neuron_idx, None, layer.weight.data.clone())
            else:
                success, info = self._divide_linear_neuron_gradient(layer, neuron_idx, None, layer.weight.data.clone())
            
            info['division_type'] = 'aggressive'
            info['importance_score'] = score
            return success, info
        finally:
            self.noise_scale = original_noise
    
    def _replace_layer_in_model(self, old_layer: nn.Module, new_layer: nn.Module):
        """在模型中替换层 - 这需要在调用者中实现具体的替换逻辑"""
        # 这个方法应该由调用者重写或提供回调
        pass
    
    def _record_division(self, strategy: str, layer: nn.Module, neuron_idx: int, info: Dict[str, Any]):
        """记录分裂历史"""
        self.division_history.append({
            'strategy': strategy,
            'layer_type': type(layer).__name__,
            'neuron_idx': neuron_idx,
            'timestamp': self.current_epoch,
            'info': info
        })
    
    def update_epoch(self, epoch: int):
        """更新当前训练周期"""
        self.current_epoch = epoch
    
    def get_division_summary(self) -> Dict[str, Any]:
        """获取分裂摘要信息"""
        if not self.division_history:
            return {'total_divisions': 0}
        
        strategies = [d['strategy'] for d in self.division_history]
        layer_types = [d['layer_type'] for d in self.division_history]
        
        return {
            'total_divisions': len(self.division_history),
            'strategies_used': {s: strategies.count(s) for s in set(strategies)},
            'layer_types_affected': {lt: layer_types.count(lt) for lt in set(layer_types)},
            'average_performance_gain': np.mean(self.performance_after_division) - np.mean(self.performance_before_division) if self.performance_after_division else 0.0
        }