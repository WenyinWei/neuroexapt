#!/usr/bin/env python3
"""
Dynamic Neural Morphogenesis - 神经元分裂模块

基于信息熵的神经元动态分裂机制：
1. 实时监控每个神经元的信息承载量
2. 识别信息过载的高熵神经元
3. 执行智能分裂，继承权重并添加适应性变异
4. 支持CNN和全连接层的分裂操作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class NeuronInformationAnalyzer:
    """神经元信息分析器 - 计算信息熵和负载"""
    
    def __init__(self, bins=32, smoothing_factor=1e-8):
        self.bins = bins
        self.smoothing_factor = smoothing_factor
        self.activation_history = defaultdict(list)
        
    def analyze_activation_entropy(self, activations: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        分析激活的信息熵
        
        Args:
            activations: 神经元激活值 [batch_size, channels, ...] 或 [batch_size, neurons]
            layer_name: 层名称
            
        Returns:
            每个神经元/通道的信息熵
        """
        if len(activations.shape) == 4:  # Conv2D层
            return self._analyze_conv_entropy(activations)
        elif len(activations.shape) == 2:  # Linear层
            return self._analyze_linear_entropy(activations)
        else:
            logger.warning(f"Unsupported activation shape: {activations.shape}")
            return torch.zeros(activations.shape[1])
    
    def _analyze_conv_entropy(self, activations: torch.Tensor) -> torch.Tensor:
        """分析卷积层的通道熵"""
        B, C, H, W = activations.shape
        channel_entropies = []
        
        for c in range(C):
            # 获取该通道的所有激活值
            channel_data = activations[:, c, :, :].reshape(-1)
            
            # 计算信息熵
            entropy = self._calculate_entropy(channel_data)
            channel_entropies.append(entropy)
        
        return torch.tensor(channel_entropies, device=activations.device)
    
    def _analyze_linear_entropy(self, activations: torch.Tensor) -> torch.Tensor:
        """分析全连接层的神经元熵"""
        B, N = activations.shape
        neuron_entropies = []
        
        for n in range(N):
            # 获取该神经元的所有激活值
            neuron_data = activations[:, n]
            
            # 计算信息熵
            entropy = self._calculate_entropy(neuron_data)
            neuron_entropies.append(entropy)
        
        return torch.tensor(neuron_entropies, device=activations.device)
    
    def _calculate_entropy(self, data: torch.Tensor) -> float:
        """
        计算数据的信息熵
        
        信息熵公式: H(X) = -Σ p(x) * log2(p(x))
        高熵表示信息分布均匀，低熵表示信息集中
        """
        if len(data) == 0:
            return 0.0
        
        # 数据预处理
        data = data.detach().cpu().float()
        
        # 处理常数数据
        if torch.std(data) < self.smoothing_factor:
            return 0.0
        
        # 数据归一化到[0,1]范围
        data_min, data_max = torch.min(data), torch.max(data)
        if data_max - data_min > self.smoothing_factor:
            normalized_data = (data - data_min) / (data_max - data_min)
        else:
            normalized_data = data
        
        # 创建直方图
        hist = torch.histc(normalized_data, bins=self.bins, min=0.0, max=1.0)
        
        # 计算概率分布
        prob = hist / (hist.sum() + self.smoothing_factor)
        prob = prob[prob > 0]  # 只考虑非零概率
        
        # 计算信息熵
        if len(prob) > 1:
            entropy = -torch.sum(prob * torch.log2(prob + self.smoothing_factor))
            return entropy.item()
        else:
            return 0.0
    
    def calculate_information_load(self, activations: torch.Tensor) -> Dict[str, float]:
        """
        计算神经元的信息负载指标
        
        返回多维度的信息负载分析：
        - entropy: 信息熵
        - variance: 方差 (信息散布程度)
        - sparsity: 稀疏度 (非零激活比例)
        - dynamic_range: 动态范围 (最大值-最小值)
        """
        data = activations.detach().cpu().float()
        
        # 基础统计
        entropy = self._calculate_entropy(data)
        variance = torch.var(data).item()
        mean_abs = torch.mean(torch.abs(data)).item()
        
        # 稀疏度分析
        threshold = mean_abs * 0.1  # 10%平均值作为激活阈值
        active_ratio = torch.mean((torch.abs(data) > threshold).float()).item()
        
        # 动态范围
        dynamic_range = (torch.max(data) - torch.min(data)).item()
        
        # 信息密度 (结合熵和方差)
        information_density = entropy * math.sqrt(variance + self.smoothing_factor)
        
        return {
            'entropy': entropy,
            'variance': variance,
            'sparsity': 1.0 - active_ratio,  # 稀疏度 = 1 - 激活比例
            'dynamic_range': dynamic_range,
            'information_density': information_density,
            'overload_score': self._calculate_overload_score(entropy, variance, active_ratio)
        }
    
    def _calculate_overload_score(self, entropy: float, variance: float, active_ratio: float) -> float:
        """
        计算神经元过载评分
        
        综合考虑：
        - 高熵 (信息复杂)
        - 高方差 (激活不稳定)  
        - 高激活率 (神经元繁忙)
        """
        # 标准化各项指标
        normalized_entropy = min(entropy / math.log2(self.bins), 1.0)
        normalized_variance = min(variance / 10.0, 1.0)  # 假设方差上限为10
        
        # 加权综合评分
        overload_score = (
            0.5 * normalized_entropy +      # 信息复杂度权重50%
            0.3 * normalized_variance +     # 激活不稳定性权重30%
            0.2 * active_ratio             # 神经元繁忙度权重20%
        )
        
        return min(overload_score, 1.0)


class IntelligentNeuronSplitter:
    """智能神经元分裂器"""
    
    def __init__(self, 
                 entropy_threshold: float = 0.7,
                 overload_threshold: float = 0.6,
                 split_probability: float = 0.4,
                 max_splits_per_layer: int = 3,
                 inheritance_noise: float = 0.1):
        
        self.entropy_threshold = entropy_threshold
        self.overload_threshold = overload_threshold
        self.split_probability = split_probability
        self.max_splits_per_layer = max_splits_per_layer
        self.inheritance_noise = inheritance_noise
        
        self.analyzer = NeuronInformationAnalyzer()
        self.split_history = defaultdict(list)
        
    def decide_split_candidates(self, 
                              activations: torch.Tensor, 
                              layer_name: str) -> List[int]:
        """
        决定需要分裂的神经元候选
        
        Args:
            activations: 神经元激活值
            layer_name: 层名称
            
        Returns:
            需要分裂的神经元/通道索引列表
        """
        split_candidates = []
        
        if len(activations.shape) == 4:  # Conv层
            num_channels = activations.shape[1]
            
            for c in range(num_channels):
                channel_data = activations[:, c, :, :]
                info_load = self.analyzer.calculate_information_load(channel_data)
                
                # 判断是否需要分裂
                if self._should_split_neuron(info_load, c, layer_name):
                    split_candidates.append(c)
                    
        elif len(activations.shape) == 2:  # Linear层
            num_neurons = activations.shape[1]
            
            for n in range(num_neurons):
                neuron_data = activations[:, n]
                info_load = self.analyzer.calculate_information_load(neuron_data)
                
                # 判断是否需要分裂
                if self._should_split_neuron(info_load, n, layer_name):
                    split_candidates.append(n)
        
        # 限制分裂数量
        if len(split_candidates) > self.max_splits_per_layer:
            # 按照过载评分排序，选择最需要分裂的
            candidate_scores = []
            for idx in split_candidates:
                if len(activations.shape) == 4:
                    data = activations[:, idx, :, :]
                else:
                    data = activations[:, idx]
                
                info_load = self.analyzer.calculate_information_load(data)
                candidate_scores.append((idx, info_load['overload_score']))
            
            # 排序并选择前N个
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            split_candidates = [idx for idx, _ in candidate_scores[:self.max_splits_per_layer]]
        
        # 记录分裂历史
        if split_candidates:
            self.split_history[layer_name].extend(split_candidates)
            logger.info(f"Layer {layer_name}: Selected {len(split_candidates)} neurons for splitting")
        
        return split_candidates
    
    def _should_split_neuron(self, info_load: Dict[str, float], neuron_idx: int, layer_name: str) -> bool:
        """判断神经元是否应该分裂"""
        
        # 条件1: 信息熵超过阈值
        entropy_condition = info_load['entropy'] > self.entropy_threshold
        
        # 条件2: 过载评分超过阈值
        overload_condition = info_load['overload_score'] > self.overload_threshold
        
        # 条件3: 随机概率
        random_condition = torch.rand(1).item() < self.split_probability
        
        # 条件4: 避免重复分裂同一神经元
        recent_splits = self.split_history.get(layer_name, [])
        not_recently_split = neuron_idx not in recent_splits[-10:]  # 最近10次分裂中不包含
        
        # 综合判断
        should_split = (entropy_condition and overload_condition and 
                       random_condition and not_recently_split)
        
        if should_split:
            logger.debug(f"Neuron {neuron_idx} in {layer_name} marked for splitting: "
                        f"entropy={info_load['entropy']:.3f}, "
                        f"overload={info_load['overload_score']:.3f}")
        
        return should_split
    
    def execute_conv_split(self, conv_layer: nn.Conv2d, split_indices: List[int]) -> nn.Conv2d:
        """执行卷积层通道分裂"""
        if not split_indices:
            return conv_layer
        
        # 创建新的卷积层
        new_out_channels = conv_layer.out_channels + len(split_indices)
        new_conv = nn.Conv2d(
            in_channels=conv_layer.in_channels,
            out_channels=new_out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=conv_layer.bias is not None,
            padding_mode=conv_layer.padding_mode
        ).to(conv_layer.weight.device)
        
        # 权重继承和分裂
        with torch.no_grad():
            # 复制原始权重
            new_conv.weight[:conv_layer.out_channels] = conv_layer.weight.data
            if conv_layer.bias is not None:
                new_conv.bias[:conv_layer.out_channels] = conv_layer.bias.data
            
            # 为分裂的通道初始化权重
            for i, split_idx in enumerate(split_indices):
                new_idx = conv_layer.out_channels + i
                
                # 继承父通道权重 + 自适应噪声
                parent_weight = conv_layer.weight.data[split_idx]
                noise_scale = self.inheritance_noise * torch.std(parent_weight)
                noise = torch.randn_like(parent_weight) * noise_scale
                
                new_conv.weight[new_idx] = parent_weight + noise
                
                if conv_layer.bias is not None:
                    parent_bias = conv_layer.bias.data[split_idx]
                    bias_noise = torch.randn(1, device=parent_bias.device) * noise_scale
                    new_conv.bias[new_idx] = parent_bias + bias_noise
        
        logger.info(f"Conv layer split: {conv_layer.out_channels} -> {new_out_channels} channels")
        return new_conv
    
    def execute_linear_split(self, linear_layer: nn.Linear, split_indices: List[int]) -> nn.Linear:
        """执行全连接层神经元分裂"""
        if not split_indices:
            return linear_layer
        
        # 创建新的全连接层
        new_out_features = linear_layer.out_features + len(split_indices)
        new_linear = nn.Linear(
            in_features=linear_layer.in_features,
            out_features=new_out_features,
            bias=linear_layer.bias is not None
        ).to(linear_layer.weight.device)
        
        # 权重继承和分裂
        with torch.no_grad():
            # 复制原始权重
            new_linear.weight[:linear_layer.out_features] = linear_layer.weight.data
            if linear_layer.bias is not None:
                new_linear.bias[:linear_layer.out_features] = linear_layer.bias.data
            
            # 为分裂的神经元初始化权重
            for i, split_idx in enumerate(split_indices):
                new_idx = linear_layer.out_features + i
                
                # 继承父神经元权重 + 自适应噪声
                parent_weight = linear_layer.weight.data[split_idx]
                noise_scale = self.inheritance_noise * torch.std(parent_weight)
                noise = torch.randn_like(parent_weight) * noise_scale
                
                new_linear.weight[new_idx] = parent_weight + noise
                
                if linear_layer.bias is not None:
                    parent_bias = linear_layer.bias.data[split_idx]
                    bias_noise = torch.randn(1, device=parent_bias.device) * noise_scale
                    new_linear.bias[new_idx] = parent_bias + bias_noise
        
        logger.info(f"Linear layer split: {linear_layer.out_features} -> {new_out_features} neurons")
        return new_linear


class DNMNeuronDivision:
    """DNM神经元分裂主控制器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.splitter = IntelligentNeuronSplitter(**self.config['splitter'])
        self.activation_hooks = {}
        self.split_statistics = defaultdict(int)
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'splitter': {
                'entropy_threshold': 0.7,
                'overload_threshold': 0.6,
                'split_probability': 0.4,
                'max_splits_per_layer': 3,
                'inheritance_noise': 0.1
            },
            'monitoring': {
                'target_layers': ['conv', 'linear'],  # 监控的层类型
                'analysis_frequency': 5,  # 每5个epoch分析一次
                'min_epoch_before_split': 10  # 最小训练epoch后才开始分裂
            }
        }
    
    def register_model_hooks(self, model: nn.Module) -> None:
        """为模型注册激活监控hooks"""
        self.activation_cache = {}
        hooks = []
        
        def create_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activation_cache[name] = output.detach().clone()
            return hook_fn
        
        for name, module in model.named_modules():
            if self._should_monitor_layer(module, name):
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
                logger.debug(f"Registered hook for layer: {name}")
        
        self.activation_hooks[id(model)] = hooks
    
    def remove_model_hooks(self, model: nn.Module) -> None:
        """移除模型的hooks"""
        model_id = id(model)
        if model_id in self.activation_hooks:
            for hook in self.activation_hooks[model_id]:
                hook.remove()
            del self.activation_hooks[model_id]
            logger.debug("Removed all activation hooks")
    
    def _should_monitor_layer(self, module: nn.Module, name: str) -> bool:
        """判断是否应该监控该层"""
        target_types = {
            'conv': nn.Conv2d,
            'linear': nn.Linear
        }
        
        for layer_type in self.config['monitoring']['target_layers']:
            if isinstance(module, target_types.get(layer_type)):
                return True
        return False
    
    def analyze_and_split(self, model: nn.Module, epoch: int) -> Dict[str, Any]:
        """分析模型并执行神经元分裂"""
        
        # 检查是否满足分裂条件
        if epoch < self.config['monitoring']['min_epoch_before_split']:
            return {'splits_executed': 0, 'message': 'Too early for splitting'}
        
        if epoch % self.config['monitoring']['analysis_frequency'] != 0:
            return {'splits_executed': 0, 'message': 'Not analysis epoch'}
        
        split_decisions = {}
        
        # 分析各层的激活并决定分裂策略
        for layer_name, activations in self.activation_cache.items():
            try:
                split_candidates = self.splitter.decide_split_candidates(
                    activations, layer_name
                )
                
                if split_candidates:
                    split_decisions[layer_name] = split_candidates
                    logger.info(f"Layer {layer_name}: {len(split_candidates)} neurons marked for splitting")
                    
            except Exception as e:
                logger.warning(f"Failed to analyze layer {layer_name}: {e}")
                continue
        
        # 执行分裂操作
        splits_executed = self._execute_splits(model, split_decisions)
        
        # 更新统计信息
        for layer_name, candidates in split_decisions.items():
            self.split_statistics[layer_name] += len(candidates)
        
        # 清理激活缓存
        self.activation_cache.clear()
        
        result = {
            'splits_executed': splits_executed,
            'split_decisions': split_decisions,
            'total_splits_per_layer': dict(self.split_statistics),
            'message': f'Successfully executed {splits_executed} splits'
        }
        
        logger.info(f"DNM Neuron Division completed: {splits_executed} splits executed")
        return result
    
    def _execute_splits(self, model: nn.Module, split_decisions: Dict[str, List[int]]) -> int:
        """执行具体的分裂操作"""
        total_splits = 0
        
        for layer_name, split_indices in split_decisions.items():
            try:
                # 获取目标层
                target_module = self._get_module_by_name(model, layer_name)
                if target_module is None:
                    logger.warning(f"Could not find module: {layer_name}")
                    continue
                
                # 根据层类型执行分裂
                if isinstance(target_module, nn.Conv2d):
                    new_module = self.splitter.execute_conv_split(target_module, split_indices)
                elif isinstance(target_module, nn.Linear):
                    new_module = self.splitter.execute_linear_split(target_module, split_indices)
                else:
                    logger.warning(f"Unsupported layer type for splitting: {type(target_module)}")
                    continue
                
                # 替换模型中的层
                self._replace_module_in_model(model, layer_name, new_module)
                total_splits += len(split_indices)
                
                logger.info(f"Successfully split layer {layer_name}: {len(split_indices)} new neurons/channels")
                
            except Exception as e:
                logger.error(f"Failed to split layer {layer_name}: {e}")
                continue
        
        return total_splits
    
    def _get_module_by_name(self, model: nn.Module, module_name: str) -> Optional[nn.Module]:
        """根据名称获取模块"""
        try:
            return model.get_submodule(module_name)
        except AttributeError:
            # 兼容老版本PyTorch
            parts = module_name.split('.')
            module = model
            for part in parts:
                module = getattr(module, part)
            return module
        except Exception:
            return None
    
    def _replace_module_in_model(self, model: nn.Module, module_name: str, new_module: nn.Module) -> None:
        """在模型中替换模块"""
        parts = module_name.split('.')
        parent = model
        
        # 找到父模块
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # 替换目标模块
        setattr(parent, parts[-1], new_module)
    
    def get_split_summary(self) -> Dict[str, Any]:
        """获取分裂操作的总结"""
        return {
            'total_layers_split': len(self.split_statistics),
            'splits_per_layer': dict(self.split_statistics),
            'total_splits': sum(self.split_statistics.values()),
            'config': self.config
        }


# 使用示例和测试函数
def test_neuron_division():
    """测试神经元分裂功能"""
    print("🧬 Testing DNM Neuron Division")
    
    # 创建测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = TestModel()
    dnm_division = DNMNeuronDivision()
    
    # 注册hooks
    dnm_division.register_model_hooks(model)
    
    # 模拟前向传播
    dummy_input = torch.randn(16, 3, 32, 32)
    with torch.no_grad():
        _ = model(dummy_input)
    
    # 执行分裂分析
    result = dnm_division.analyze_and_split(model, epoch=15)
    
    print(f"Split result: {result}")
    print(f"Split summary: {dnm_division.get_split_summary()}")
    
    # 清理
    dnm_division.remove_model_hooks(model)
    
    return model, result


if __name__ == "__main__":
    test_neuron_division()