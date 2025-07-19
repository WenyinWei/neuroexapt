#!/usr/bin/env python3
"""
"""
\defgroup group_dnm_neuron_division Dnm Neuron Division
\ingroup core
Dnm Neuron Division module for NeuroExapt framework.
"""


DNM Neuron Division Module - 神经元分裂专用模块

🧬 核心功能：
1. 智能识别分裂时机
2. 执行不同类型的神经元分裂
3. 保持网络功能性
4. 优化参数初始化

🎯 目标：实现真正有效的神经元增长和网络扩展
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import copy
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

class NeuronDivisionStrategies:
    """神经元分裂策略集合"""
    
    @staticmethod
    def symmetric_division(original_weights: torch.Tensor, division_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """对称分裂：将一个神经元分裂为两个相似的神经元"""
        device = original_weights.device
        dtype = original_weights.dtype
        noise_scale = torch.std(original_weights) * 0.1
        
        # 第一个神经元：保持大部分原始权重
        neuron1 = original_weights + torch.normal(0, noise_scale, size=original_weights.shape, device=device, dtype=dtype)
        
        # 第二个神经元：稍微不同的权重
        neuron2 = original_weights * division_ratio + torch.normal(0, noise_scale, size=original_weights.shape, device=device, dtype=dtype)
        
        return neuron1, neuron2
    
    @staticmethod
    def asymmetric_division(original_weights: torch.Tensor, specialization_factor: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
        """非对称分裂：创建专门化的神经元"""
        std_dev = torch.std(original_weights)
        
        # 主神经元：保持大部分功能
        main_neuron = original_weights * (1.0 + specialization_factor)
        
        # 专门化神经元：关注特定模式
        specialized_weights = torch.zeros_like(original_weights)
        # 只保留最重要的连接
        threshold = torch.quantile(torch.abs(original_weights), 0.7)
        mask = torch.abs(original_weights) > threshold
        specialized_weights[mask] = original_weights[mask] * (1.0 + specialization_factor)
        
        return main_neuron, specialized_weights
    
    @staticmethod
    def functional_division(original_weights: torch.Tensor, activation_pattern: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """功能分裂：基于激活模式进行分裂"""
        if activation_pattern is not None:
            # 基于激活模式分割权重
            high_activation_mask = activation_pattern > torch.median(activation_pattern)
            
            # 高激活神经元
            high_act_neuron = original_weights.clone()
            high_act_neuron[~high_activation_mask] *= 0.3
            
            # 低激活神经元  
            low_act_neuron = original_weights.clone()
            low_act_neuron[high_activation_mask] *= 0.3
            
            return high_act_neuron, low_act_neuron
        else:
            # 随机功能分割
            mask = torch.rand_like(original_weights) > 0.5
            neuron1 = original_weights.clone()
            neuron2 = original_weights.clone()
            
            neuron1[~mask] *= 0.2
            neuron2[mask] *= 0.2
            
            return neuron1, neuron2

class AdaptiveNeuronDivision:
    """自适应神经元分裂器"""
    
    def __init__(self):
        self.division_history = defaultdict(list)
        self.performance_tracker = {}
        
    def execute_division(self, model: nn.Module, layer_name: str, 
                        division_strategy: str = 'adaptive',
                        target_expansion: float = 0.2) -> Tuple[nn.Module, int]:
        """执行神经元分裂"""
        
        # 获取原始设备
        original_device = next(model.parameters()).device
        
        # 深拷贝模型并确保在正确设备上
        new_model = copy.deepcopy(model).to(original_device)
        
        # 找到目标层
        target_layer = self._find_layer(new_model, layer_name)
        if target_layer is None:
            logger.warning(f"未找到层: {layer_name}")
            return model, 0
            
        # 根据层类型执行分裂
        if isinstance(target_layer, nn.Linear):
            return self._divide_linear_layer(new_model, layer_name, target_layer, 
                                           division_strategy, target_expansion)
        elif isinstance(target_layer, nn.Conv2d):
            return self._divide_conv_layer(new_model, layer_name, target_layer,
                                         division_strategy, target_expansion)
        else:
            logger.warning(f"不支持的层类型: {type(target_layer)}")
            return model, 0
    
    def _find_layer(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """查找指定层"""
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def _divide_linear_layer(self, model: nn.Module, layer_name: str, layer: nn.Linear,
                           division_strategy: str, target_expansion: float) -> Tuple[nn.Module, int]:
        """分裂全连接层"""
        
        original_out_features = layer.out_features
        expansion_size = max(1, int(original_out_features * target_expansion))
        new_out_features = original_out_features + expansion_size
        
        # 获取原始设备和数据类型
        device = layer.weight.device
        dtype = layer.weight.dtype
        
        # 创建新的权重和偏置张量（确保在正确的设备上）
        new_weight = torch.zeros(new_out_features, layer.in_features, dtype=dtype, device=device)
        new_bias = torch.zeros(new_out_features, dtype=dtype, device=device) if layer.bias is not None else None
        
        # 复制原始权重
        new_weight[:original_out_features] = layer.weight.data
        if new_bias is not None:
            new_bias[:original_out_features] = layer.bias.data
            
        # 选择分裂策略
        strategy_func = self._get_division_strategy(division_strategy)
        
        # 执行神经元分裂
        neurons_to_divide = self._select_neurons_for_division(layer, expansion_size)
        
        for i, neuron_idx in enumerate(neurons_to_divide):
            if i >= expansion_size:
                break
                
            original_weights = layer.weight.data[neuron_idx]
            original_bias = layer.bias.data[neuron_idx] if layer.bias is not None else torch.tensor(0.0, device=device)
            
            # 执行分裂
            if division_strategy == 'symmetric':
                new_weights, _ = strategy_func(original_weights)
                new_weight[original_out_features + i] = new_weights
            elif division_strategy == 'asymmetric':
                _, specialized_weights = strategy_func(original_weights)
                new_weight[original_out_features + i] = specialized_weights
            else:  # adaptive
                new_weights, _ = self._adaptive_division(original_weights, layer_name, neuron_idx)
                new_weight[original_out_features + i] = new_weights
                
            # 设置偏置
            if new_bias is not None:
                new_bias[original_out_features + i] = original_bias * 0.9
        
        # 更新层参数（确保在正确设备上）
        layer.out_features = new_out_features
        # 确保参数在正确设备上并且requires_grad=True
        new_weight_param = nn.Parameter(new_weight.to(device).detach().requires_grad_(True))
        layer.weight = new_weight_param
        if layer.bias is not None:
            new_bias_param = nn.Parameter(new_bias.to(device).detach().requires_grad_(True))
            layer.bias = new_bias_param
            
        # 更新下一层的输入维度（如果存在且不是最后一层）
        if not self._is_final_layer(model, layer_name):
            self._update_next_layer_input(model, layer_name, expansion_size)
        
        # 记录分裂历史
        self.division_history[layer_name].append({
            'expansion_size': expansion_size,
            'strategy': division_strategy,
            'neurons_divided': neurons_to_divide
        })
        
        logger.info(f"Linear层分裂完成: {layer_name}, 新增神经元: {expansion_size}")
        return model, expansion_size * (layer.in_features + 1)
    
    def _divide_conv_layer(self, model: nn.Module, layer_name: str, layer: nn.Conv2d,
                          division_strategy: str, target_expansion: float) -> Tuple[nn.Module, int]:
        """分裂卷积层"""
        
        original_out_channels = layer.out_channels
        expansion_size = max(1, int(original_out_channels * target_expansion))
        new_out_channels = original_out_channels + expansion_size
        
        # 获取原始设备
        device = layer.weight.device
        
        # 创建新的卷积层
        new_conv = nn.Conv2d(
            layer.in_channels,
            new_out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
            layer.bias is not None,
            layer.padding_mode
        ).to(device)  # 确保在正确的设备上
        
        # 复制原始权重
        with torch.no_grad():
            new_conv.weight.data[:original_out_channels] = layer.weight.data
            if layer.bias is not None:
                new_conv.bias.data[:original_out_channels] = layer.bias.data
        
        # 执行通道分裂
        channels_to_divide = self._select_channels_for_division(layer, expansion_size)
        strategy_func = self._get_division_strategy(division_strategy)
        
        for i, channel_idx in enumerate(channels_to_divide):
            if i >= expansion_size:
                break
                
            original_kernel = layer.weight.data[channel_idx]
            original_bias = layer.bias.data[channel_idx] if layer.bias is not None else torch.tensor(0.0, device=device)
            
            # 分裂卷积核
            if division_strategy == 'symmetric':
                new_kernel, _ = self._divide_conv_kernel(original_kernel, 'symmetric')
            elif division_strategy == 'asymmetric':
                _, new_kernel = self._divide_conv_kernel(original_kernel, 'asymmetric')
            else:  # adaptive
                new_kernel, _ = self._adaptive_conv_division(original_kernel, layer_name, channel_idx)
                
            new_conv.weight.data[original_out_channels + i] = new_kernel
            if new_conv.bias is not None:
                new_conv.bias.data[original_out_channels + i] = original_bias * 0.9
        
        # 替换层
        self._replace_layer(model, layer_name, new_conv)
        
        # 更新下一层的输入通道数（如果不是最后一层）
        if not self._is_final_layer(model, layer_name):
            self._update_next_conv_layer_input(model, layer_name, expansion_size)
        
        # 记录分裂历史
        self.division_history[layer_name].append({
            'expansion_size': expansion_size,
            'strategy': division_strategy,
            'channels_divided': channels_to_divide
        })
        
        param_increase = expansion_size * layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
        logger.info(f"Conv层分裂完成: {layer_name}, 新增通道: {expansion_size}")
        return model, param_increase
    
    def _get_division_strategy(self, strategy_name: str):
        """获取分裂策略函数"""
        strategies = {
            'symmetric': NeuronDivisionStrategies.symmetric_division,
            'asymmetric': NeuronDivisionStrategies.asymmetric_division,
            'functional': NeuronDivisionStrategies.functional_division
        }
        return strategies.get(strategy_name, NeuronDivisionStrategies.symmetric_division)
    
    def _select_neurons_for_division(self, layer: nn.Linear, num_divisions: int) -> List[int]:
        """选择要分裂的神经元"""
        weights = layer.weight.data
        
        # 计算每个神经元的重要性分数
        importance_scores = []
        for i in range(weights.size(0)):
            neuron_weights = weights[i]
            
            # 综合多个指标
            weight_variance = torch.var(neuron_weights).item()
            weight_norm = torch.norm(neuron_weights).item()
            weight_sparsity = (torch.abs(neuron_weights) < 0.01).float().mean().item()
            
            # 高方差、适中范数、低稀疏性的神经元适合分裂
            score = weight_variance * (1.0 - weight_sparsity) * min(weight_norm, 1.0)
            importance_scores.append((i, score))
        
        # 选择得分最高的神经元
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in importance_scores[:num_divisions]]
    
    def _select_channels_for_division(self, layer: nn.Conv2d, num_divisions: int) -> List[int]:
        """选择要分裂的卷积通道"""
        weights = layer.weight.data
        
        importance_scores = []
        for i in range(weights.size(0)):
            channel_weights = weights[i]
            
            # 计算通道重要性
            weight_energy = torch.sum(channel_weights ** 2).item()
            weight_diversity = torch.std(channel_weights).item()
            
            score = weight_energy * weight_diversity
            importance_scores.append((i, score))
        
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in importance_scores[:num_divisions]]
    
    def _adaptive_division(self, original_weights: torch.Tensor, layer_name: str, neuron_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """自适应分裂策略"""
        # 根据历史表现选择最佳策略
        history = self.division_history.get(layer_name, [])
        
        if len(history) < 3:
            # 初期使用对称分裂
            return NeuronDivisionStrategies.symmetric_division(original_weights)
        else:
            # 基于历史表现选择策略
            # 这里简化为随机选择，实际应该基于性能反馈
            strategy = np.random.choice(['symmetric', 'asymmetric', 'functional'])
            func = self._get_division_strategy(strategy)
            return func(original_weights)
    
    def _adaptive_conv_division(self, original_kernel: torch.Tensor, layer_name: str, channel_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """自适应卷积分裂"""
        return self._divide_conv_kernel(original_kernel, 'symmetric')
    
    def _divide_conv_kernel(self, kernel: torch.Tensor, strategy: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """分裂卷积核"""
        device = kernel.device
        dtype = kernel.dtype
        
        if strategy == 'symmetric':
            noise = torch.normal(0, torch.std(kernel) * 0.1, size=kernel.shape, device=device, dtype=dtype)
            kernel1 = kernel + noise
            kernel2 = kernel - noise
            return kernel1, kernel2
        elif strategy == 'asymmetric':
            # 创建专门化的核
            kernel1 = kernel * 1.1
            kernel2 = kernel * 0.5
            # 在kernel2中增强边缘检测
            if kernel.size(-1) >= 3 and kernel.size(-2) >= 3:
                edge_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                         dtype=dtype, device=device)
                kernel2[:, :, :3, :3] += edge_kernel.unsqueeze(0).unsqueeze(0) * 0.1
            return kernel1, kernel2
        else:
            flattened = kernel.view(-1)
            result1, result2 = NeuronDivisionStrategies.symmetric_division(flattened)
            return result1.view(kernel.shape), result2.view(kernel.shape)
    
    def _update_next_layer_input(self, model: nn.Module, current_layer_name: str, expansion_size: int):
        """更新下一层的输入维度"""
        layer_names = [name for name, _ in model.named_modules()]
        
        try:
            current_idx = layer_names.index(current_layer_name)
            if current_idx + 1 < len(layer_names):
                next_layer_name = layer_names[current_idx + 1]
                next_layer = self._find_layer(model, next_layer_name)
                
                if isinstance(next_layer, nn.Linear):
                    old_in_features = next_layer.in_features
                    new_in_features = old_in_features + expansion_size
                    
                    # 获取设备信息
                    device = next_layer.weight.device
                    dtype = next_layer.weight.dtype
                    
                    # 创建新的权重矩阵
                    new_weight = torch.zeros(next_layer.out_features, new_in_features, dtype=dtype, device=device)
                    new_weight[:, :old_in_features] = next_layer.weight.data
                    
                    # 初始化新的连接权重
                    with torch.no_grad():
                        nn.init.normal_(new_weight[:, old_in_features:], mean=0, std=0.01)
                    
                    next_layer.in_features = new_in_features
                    next_layer.weight = nn.Parameter(new_weight)
                    
                    logger.info(f"更新下一层输入维度: {next_layer_name}, {old_in_features} -> {new_in_features}")
                    
        except (ValueError, IndexError):
            logger.warning(f"无法找到层 {current_layer_name} 的下一层")
    
    def _update_next_conv_layer_input(self, model: nn.Module, current_layer_name: str, expansion_size: int):
        """更新下一个卷积层的输入通道数"""
        # 寻找下一个线性层或卷积层
        found_current = False
        
        for name, module in model.named_modules():
            if found_current and isinstance(module, (nn.Linear, nn.Conv2d)):
                if isinstance(module, nn.Conv2d):
                    # 更新卷积层输入通道
                    old_in_channels = module.in_channels
                    new_in_channels = old_in_channels + expansion_size
                    
                    # 获取设备信息
                    device = module.weight.device
                    
                    new_conv = nn.Conv2d(
                        new_in_channels,
                        module.out_channels,
                        module.kernel_size,
                        module.stride,
                        module.padding,
                        module.dilation,
                        module.groups,
                        module.bias is not None,
                        module.padding_mode
                    ).to(device)  # 确保在正确的设备上
                    
                    # 复制权重并扩展
                    with torch.no_grad():
                        new_conv.weight.data[:, :old_in_channels] = module.weight.data
                        if module.bias is not None:
                            new_conv.bias.data = module.bias.data
                            
                        # 初始化新的输入通道
                        nn.init.kaiming_normal_(new_conv.weight.data[:, old_in_channels:])
                    
                    self._replace_layer(model, name, new_conv)
                    logger.info(f"更新Conv层输入通道: {name}, {old_in_channels} -> {new_in_channels}")
                break
                
            if name == current_layer_name:
                found_current = True
    
    def _is_final_layer(self, model: nn.Module, layer_name: str) -> bool:
        """检查是否为最后一层"""
        layer_names = [name for name, module in model.named_modules() 
                      if isinstance(module, (nn.Linear, nn.Conv2d)) and name != '']
        
        if not layer_names:
            return True
            
        # 特殊处理：如果是分类器的输出层，则认为是最后一层
        if 'classifier' in layer_name:
            # 检查是否是分类器中的最后一个Linear层
            parts = layer_name.split('.')
            if len(parts) >= 2:
                try:
                    layer_idx = int(parts[-1])
                    # 对于我们的分类器结构，第6层（索引6）是最后的Linear层
                    if layer_idx == 6:
                        return True
                except ValueError:
                    pass
            
        # 找到当前层在列表中的位置
        try:
            current_idx = layer_names.index(layer_name)
            return current_idx == len(layer_names) - 1
        except ValueError:
            return True  # 如果找不到，假设是最后一层
    
    def _replace_layer(self, model: nn.Module, layer_name: str, new_layer: nn.Module):
        """替换模型中的层"""
        parts = layer_name.split('.')
        
        if len(parts) == 1:
            setattr(model, layer_name, new_layer)
        else:
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_layer)
    
    def get_division_statistics(self) -> Dict[str, Any]:
        """获取分裂统计信息"""
        total_divisions = sum(len(history) for history in self.division_history.values())
        
        strategy_counts = defaultdict(int)
        total_expansions = 0
        
        for layer_name, history in self.division_history.items():
            for event in history:
                strategy_counts[event['strategy']] += 1
                total_expansions += event['expansion_size']
        
        return {
            'total_division_events': total_divisions,
            'total_neurons_added': total_expansions,
            'strategy_usage': dict(strategy_counts),
            'layers_modified': list(self.division_history.keys())
        }

