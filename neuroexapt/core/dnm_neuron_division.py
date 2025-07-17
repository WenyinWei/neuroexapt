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
        # 安全处理激活张量
        if not isinstance(activations, torch.Tensor) or activations.numel() == 0:
            return torch.zeros(1, device=activations.device if isinstance(activations, torch.Tensor) else torch.device('cpu'))
        
        # 确保激活张量是连续的
        activations = activations.contiguous()
        
        if len(activations.shape) == 4:  # Conv2D层
            return self._analyze_conv_entropy(activations)
        elif len(activations.shape) == 2:  # Linear层
            return self._analyze_linear_entropy(activations)
        elif len(activations.shape) == 3:  # 可能是展平后的卷积层
            # 尝试将其视为2D进行处理
            reshaped = activations.view(activations.shape[0], -1)
            return self._analyze_linear_entropy(reshaped)
        else:
            logger.warning(f"Unsupported activation shape: {activations.shape} for layer {layer_name}")
            # 返回安全的默认值
            return torch.zeros(activations.shape[1] if len(activations.shape) > 1 else 1, 
                             device=activations.device)
    
    def _analyze_conv_entropy(self, activations: torch.Tensor) -> torch.Tensor:
        """分析卷积层的通道熵"""
        if len(activations.shape) < 4:
            # 处理不规则的激活形状
            return torch.zeros(activations.shape[1] if len(activations.shape) > 1 else 1, device=activations.device)
        
        B, C, H, W = activations.shape
        channel_entropies = []
        
        for c in range(C):
            try:
                # 安全地获取该通道的所有激活值
                channel_data = activations[:, c, :, :].contiguous().view(-1)
                
                # 计算信息熵
                entropy = self._calculate_entropy(channel_data)
                channel_entropies.append(entropy)
            except Exception as e:
                # 如果出现错误，使用默认熵值
                channel_entropies.append(0.0)
        
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
        
        # 导入Net2Net变换器
        try:
            from .dnm_net2net import Net2NetTransformer, DNMArchitectureMutator
            self.net2net_transformer = Net2NetTransformer(noise_scale=inheritance_noise)
            self.architecture_mutator = DNMArchitectureMutator(self.net2net_transformer)
        except ImportError:
            logger.warning("Net2Net transformer not available, using simple splitting")
            self.net2net_transformer = None
            self.architecture_mutator = None
        
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
                
                # 🔧 关键修复：同步更新相关BatchNorm层和下游层
                if isinstance(target_module, nn.Conv2d):
                    self._sync_batchnorm_after_conv_split(model, layer_name, target_module.out_channels, new_module.out_channels, split_indices)
                    # 🚀 新增：级联更新下游Conv层的输入通道
                    self._sync_downstream_conv_input_channels(model, layer_name, target_module.out_channels, new_module.out_channels, split_indices)
                    # 🎯 最终修复：级联更新下游Linear层的输入特征
                    self._sync_downstream_linear_input_features(model, layer_name, target_module.out_channels, new_module.out_channels, split_indices)
                    # 🔗 残差连接修复：更新ResidualBlock的shortcut层
                    self._sync_residual_shortcut_channels(model, layer_name, target_module.out_channels, new_module.out_channels, split_indices)
                
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
    
    def _sync_batchnorm_after_conv_split(self, model: nn.Module, conv_layer_name: str, 
                                        old_channels: int, new_channels: int, split_indices: List[int]) -> None:
        """
        🔧 关键修复：Conv层分裂后同步相关BatchNorm层
        
        这是最容易忽略但极其重要的步骤！
        当Conv层通道数改变时，对应的BatchNorm层必须同步更新：
        - num_features
        - running_mean
        - running_var  
        - weight (gamma)
        - bias (beta)
        """
        # 查找对应的BatchNorm层
        bn_layer_name = self._find_corresponding_batchnorm(model, conv_layer_name)
        if not bn_layer_name:
            logger.warning(f"No corresponding BatchNorm found for {conv_layer_name}")
            return
        
        try:
            bn_module = self._get_module_by_name(model, bn_layer_name)
            if not isinstance(bn_module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                return
            
            logger.info(f"Syncing BatchNorm {bn_layer_name}: {old_channels} -> {new_channels} features")
            
            # 创建新的BatchNorm层
            if isinstance(bn_module, nn.BatchNorm2d):
                new_bn = nn.BatchNorm2d(
                    num_features=new_channels,
                    eps=bn_module.eps,
                    momentum=bn_module.momentum,
                    affine=bn_module.affine,
                    track_running_stats=bn_module.track_running_stats
                ).to(bn_module.weight.device if bn_module.weight is not None else 'cpu')
            else:  # BatchNorm1d
                new_bn = nn.BatchNorm1d(
                    num_features=new_channels,
                    eps=bn_module.eps,
                    momentum=bn_module.momentum,
                    affine=bn_module.affine,
                    track_running_stats=bn_module.track_running_stats
                ).to(bn_module.weight.device if bn_module.weight is not None else 'cpu')
            
            # 继承原始参数
            with torch.no_grad():
                if bn_module.affine:
                    # 复制原始weight (gamma) 和 bias (beta)
                    new_bn.weight[:old_channels] = bn_module.weight.data
                    new_bn.bias[:old_channels] = bn_module.bias.data
                    
                    # 为新通道初始化参数
                    for i, split_idx in enumerate(split_indices):
                        new_idx = old_channels + i
                        # gamma继承父通道值
                        new_bn.weight[new_idx] = bn_module.weight.data[split_idx]
                        # beta继承父通道值
                        new_bn.bias[new_idx] = bn_module.bias.data[split_idx]
                
                if bn_module.track_running_stats:
                    # 复制running_mean和running_var
                    new_bn.running_mean[:old_channels] = bn_module.running_mean
                    new_bn.running_var[:old_channels] = bn_module.running_var
                    new_bn.num_batches_tracked = bn_module.num_batches_tracked
                    
                    # 为新通道初始化running stats
                    for i, split_idx in enumerate(split_indices):
                        new_idx = old_channels + i
                        new_bn.running_mean[new_idx] = bn_module.running_mean[split_idx]
                        new_bn.running_var[new_idx] = bn_module.running_var[split_idx]
            
            # 替换BatchNorm层
            self._replace_module_in_model(model, bn_layer_name, new_bn)
            logger.info(f"✅ BatchNorm {bn_layer_name} successfully synced!")
            
        except Exception as e:
            logger.error(f"Failed to sync BatchNorm {bn_layer_name}: {e}")
    
    def _sync_downstream_conv_input_channels(self, model: nn.Module, conv_layer_name: str,
                                           old_out_channels: int, new_out_channels: int, 
                                           split_indices: List[int]) -> None:
        """
        🚀 级联同步：更新下游Conv层的输入通道
        
        当一个Conv层的输出通道增加时，所有以它为输入的Conv层都需要相应更新输入通道
        这是解决 "weight of size [69, 64, 3, 3], expected input to have 64 channels, but got 69" 的关键！
        """
        logger.debug(f"🔍 Finding downstream Conv layers for {conv_layer_name}")
        
        # 查找所有可能受影响的下游Conv层
        downstream_conv_layers = self._find_downstream_conv_layers(model, conv_layer_name)
        
        for downstream_name in downstream_conv_layers:
            try:
                downstream_conv = self._get_module_by_name(model, downstream_name)
                if not isinstance(downstream_conv, nn.Conv2d):
                    continue
                
                # 检查输入通道是否匹配
                if downstream_conv.in_channels == old_out_channels:
                    logger.info(f"🔄 Updating downstream Conv {downstream_name}: in_channels {old_out_channels} -> {new_out_channels}")
                    
                    # 创建新的Conv层，扩展输入通道
                    new_downstream_conv = self._expand_conv_input_channels(
                        downstream_conv, old_out_channels, new_out_channels, split_indices
                    )
                    
                    # 替换模型中的层
                    self._replace_module_in_model(model, downstream_name, new_downstream_conv)
                    logger.info(f"✅ Successfully updated downstream Conv {downstream_name}")
                    
            except Exception as e:
                logger.error(f"Failed to update downstream Conv {downstream_name}: {e}")
    
    def _find_downstream_conv_layers(self, model: nn.Module, conv_layer_name: str) -> List[str]:
        """查找可能受影响的下游Conv层"""
        downstream_layers = []
        
        # 简单的启发式方法：查找后续的Conv层
        conv_parts = conv_layer_name.split('.')
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and name != conv_layer_name:
                name_parts = name.split('.')
                
                # 检查是否为序列中的下一层
                if self._is_likely_downstream_layer(conv_parts, name_parts):
                    downstream_layers.append(name)
        
        logger.debug(f"Found potential downstream Conv layers: {downstream_layers}")
        return downstream_layers
    
    def _is_likely_downstream_layer(self, upstream_parts: List[str], downstream_parts: List[str]) -> bool:
        """判断是否为下游层"""
        # 🔧 修复：正确识别跨block的连接模式
        
        # stem.0 -> block1.main_path.0 或 block1.shortcut.0
        if upstream_parts[0] == 'stem' and len(downstream_parts) >= 2:
            if downstream_parts[0] == 'block1':
                return True
        
        # stem.0 -> layer1.0 这种模式（旧的命名）
        if len(upstream_parts) == 2 and len(downstream_parts) == 2:
            if upstream_parts[0] == 'stem' and downstream_parts[0] == 'layer1':
                return True
        
        # block间的连接: block1 -> block2, block2 -> block3, etc.
        if len(upstream_parts) >= 2 and len(downstream_parts) >= 2:
            if upstream_parts[0].startswith('block') and downstream_parts[0].startswith('block'):
                try:
                    up_block_num = int(upstream_parts[0].replace('block', ''))
                    down_block_num = int(downstream_parts[0].replace('block', ''))
                    # 连续的block
                    if down_block_num == up_block_num + 1:
                        return True
                except ValueError:
                    pass
        
        # Sequential层内的连接: block1.main_path.0 -> block1.main_path.3 (跳过BN和ReLU)
        if len(upstream_parts) == len(downstream_parts) and len(upstream_parts) >= 3:
            # 同一个模块内的后续层
            if upstream_parts[:-1] == downstream_parts[:-1]:
                try:
                    up_idx = int(upstream_parts[-1])
                    down_idx = int(downstream_parts[-1])
                    # 考虑中间可能有BN和ReLU，所以允许间隔
                    if down_idx > up_idx and down_idx - up_idx <= 6:
                        return True
                except ValueError:
                    pass
        
        return False
    
    def _expand_conv_input_channels(self, conv_layer: nn.Conv2d, old_in_channels: int, 
                                  new_in_channels: int, split_indices: List[int]) -> nn.Conv2d:
        """扩展Conv层的输入通道"""
        # 创建新的Conv层
        new_conv = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=conv_layer.bias is not None,
            padding_mode=conv_layer.padding_mode
        ).to(conv_layer.weight.device)
        
        # 权重继承策略
        with torch.no_grad():
            # 复制原始权重 [out_channels, in_channels, kernel_h, kernel_w]
            new_conv.weight[:, :old_in_channels, :, :] = conv_layer.weight.data
            
            # 为新的输入通道初始化权重（继承自分裂的父通道）
            for i, split_idx in enumerate(split_indices):
                new_in_idx = old_in_channels + i
                # 继承父通道的权重
                new_conv.weight[:, new_in_idx, :, :] = conv_layer.weight.data[:, split_idx, :, :]
            
            # 复制bias
            if conv_layer.bias is not None:
                new_conv.bias.data = conv_layer.bias.data
        
        return new_conv
    
    def _sync_downstream_linear_input_features(self, model: nn.Module, conv_layer_name: str,
                                             old_out_channels: int, new_out_channels: int,
                                             split_indices: List[int]) -> None:
        """
        🎯 最终修复：更新下游Linear层的输入特征数
        
        当最后一个Conv层通道增加时，后续的Linear层(classifier)需要相应更新输入特征数
        解决: "mat1 and mat2 shapes cannot be multiplied (4x69 and 64x15)"
        """
        logger.debug(f"🔍 Finding downstream Linear layers for {conv_layer_name}")
        
        # 查找所有Linear层
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # 检查输入特征是否匹配（考虑可能通过Global Average Pooling）
                if module.in_features == old_out_channels:
                    logger.info(f"🔄 Updating downstream Linear {name}: in_features {old_out_channels} -> {new_out_channels}")
                    
                    try:
                        # 创建新的Linear层，扩展输入特征
                        new_linear = self._expand_linear_input_features(
                            module, old_out_channels, new_out_channels, split_indices
                        )
                        
                        # 替换模型中的层
                        self._replace_module_in_model(model, name, new_linear)
                        logger.info(f"✅ Successfully updated downstream Linear {name}")
                        
                    except Exception as e:
                        logger.error(f"Failed to update downstream Linear {name}: {e}")
    
    def _sync_residual_shortcut_channels(self, model: nn.Module, conv_layer_name: str,
                                       old_out_channels: int, new_out_channels: int,
                                       split_indices: List[int]) -> None:
        """
        🔗 残差连接修复：更新ResidualBlock的shortcut层
        
        当main_path中的Conv层通道发生变化时，对应的shortcut层也需要相应更新
        以确保残差相加时通道数匹配
        """
        logger.debug(f"🔍 Checking residual shortcut for {conv_layer_name}")
        
        # 解析层名以找到对应的ResidualBlock
        parts = conv_layer_name.split('.')
        
        # 检查是否是ResidualBlock内的main_path层
        if len(parts) >= 3 and parts[-2] == 'main_path':
            # 构造对应的shortcut层名
            block_name = '.'.join(parts[:-2])  # 例如：block1
            shortcut_layer_name = f"{block_name}.shortcut.0"
            
            try:
                shortcut_conv = self._get_module_by_name(model, shortcut_layer_name)
                
                # 如果shortcut是Conv层且输出通道匹配，需要更新
                if isinstance(shortcut_conv, nn.Conv2d) and shortcut_conv.out_channels == old_out_channels:
                    logger.info(f"🔄 Updating residual shortcut {shortcut_layer_name}: out_channels {old_out_channels} -> {new_out_channels}")
                    
                    # 创建新的shortcut Conv层
                    new_shortcut_conv = self._expand_conv_output_channels(
                        shortcut_conv, old_out_channels, new_out_channels, split_indices
                    )
                    
                    # 替换模型中的层
                    self._replace_module_in_model(model, shortcut_layer_name, new_shortcut_conv)
                    
                    # 同步对应的BatchNorm
                    shortcut_bn_name = f"{block_name}.shortcut.1"
                    self._sync_batchnorm_after_conv_split(model, shortcut_layer_name, old_out_channels, new_out_channels, split_indices)
                    
                    logger.info(f"✅ Successfully updated residual shortcut {shortcut_layer_name}")
                    
            except Exception as e:
                logger.error(f"Failed to update residual shortcut for {conv_layer_name}: {e}")
    
    def _expand_conv_output_channels(self, conv_layer: nn.Conv2d, old_out_channels: int,
                                   new_out_channels: int, split_indices: List[int]) -> nn.Conv2d:
        """扩展Conv层的输出通道（用于shortcut层更新）"""
        # 创建新的Conv层
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
        
        # 权重继承策略
        with torch.no_grad():
            # 复制原始权重 [out_channels, in_channels, kernel_h, kernel_w]
            new_conv.weight[:old_out_channels, :, :, :] = conv_layer.weight.data
            
            # 为新的输出通道初始化权重（继承自分裂的父通道）
            for i, split_idx in enumerate(split_indices):
                new_out_idx = old_out_channels + i
                # 继承父通道的权重并添加少量噪声
                parent_weight = conv_layer.weight.data[split_idx, :, :, :]
                noise_scale = 0.01 * torch.std(parent_weight)
                noise = torch.randn_like(parent_weight) * noise_scale
                new_conv.weight[new_out_idx, :, :, :] = parent_weight + noise
            
            # 复制bias
            if conv_layer.bias is not None:
                new_conv.bias[:old_out_channels] = conv_layer.bias.data
                # 为新通道初始化bias
                for i, split_idx in enumerate(split_indices):
                    new_out_idx = old_out_channels + i
                    new_conv.bias[new_out_idx] = conv_layer.bias.data[split_idx]
        
        return new_conv
    
    def _expand_linear_input_features(self, linear_layer: nn.Linear, old_in_features: int,
                                    new_in_features: int, split_indices: List[int]) -> nn.Linear:
        """扩展Linear层的输入特征数"""
        # 创建新的Linear层
        new_linear = nn.Linear(
            in_features=new_in_features,
            out_features=linear_layer.out_features,
            bias=linear_layer.bias is not None
        ).to(linear_layer.weight.device)
        
        # 权重继承策略
        with torch.no_grad():
            # 复制原始权重 [out_features, in_features]
            new_linear.weight[:, :old_in_features] = linear_layer.weight.data
            
            # 为新的输入特征初始化权重（继承自分裂的父特征）
            for i, split_idx in enumerate(split_indices):
                new_in_idx = old_in_features + i
                # 继承父特征的权重
                new_linear.weight[:, new_in_idx] = linear_layer.weight.data[:, split_idx]
            
            # 复制bias
            if linear_layer.bias is not None:
                new_linear.bias.data = linear_layer.bias.data
        
        return new_linear
    
    def _find_corresponding_batchnorm(self, model: nn.Module, conv_layer_name: str) -> Optional[str]:
        """查找Conv层对应的BatchNorm层 - 增强版本支持ResNet架构"""
        
        # 首先尝试直接匹配的模式
        direct_patterns = [
            # 标准模式: conv1 -> bn1
            conv_layer_name.replace('conv', 'bn'),
            # norm变体: conv1 -> norm1  
            conv_layer_name.replace('conv', 'norm'),
            # 后缀模式
            conv_layer_name + '.bn',
            conv_layer_name + '.norm',
        ]
        
        # 收集所有BatchNorm层用于调试
        all_bn_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                all_bn_layers.append(name)
        
        logger.debug(f"Looking for BatchNorm for Conv layer: {conv_layer_name}")
        logger.debug(f"Available BatchNorm layers: {all_bn_layers}")
        
        # 1. 检查直接模式匹配
        for pattern in direct_patterns:
            if pattern in all_bn_layers:
                logger.info(f"✅ Found BatchNorm by direct pattern: {conv_layer_name} -> {pattern}")
                return pattern
        
        # 2. 解析层级结构进行智能匹配
        conv_parts = conv_layer_name.split('.')
        
        for bn_name in all_bn_layers:
            bn_parts = bn_name.split('.')
            
            # ResNet模式匹配
            if self._is_resnet_bn_match(conv_parts, bn_parts):
                logger.info(f"✅ Found BatchNorm by ResNet pattern: {conv_layer_name} -> {bn_name}")
                return bn_name
            
            # 序列模式匹配 (用于shortcut等序列)
            if self._is_sequential_bn_match(conv_parts, bn_parts):
                logger.info(f"✅ Found BatchNorm by sequential pattern: {conv_layer_name} -> {bn_name}")
                return bn_name
        
        # 3. 按距离查找最近的BatchNorm
        nearest_bn = self._find_nearest_batchnorm(model, conv_layer_name)
        if nearest_bn:
            logger.info(f"✅ Found BatchNorm by proximity: {conv_layer_name} -> {nearest_bn}")
            return nearest_bn
        
        logger.warning(f"❌ No corresponding BatchNorm found for {conv_layer_name}")
        logger.warning(f"Available BatchNorm layers: {all_bn_layers}")
        return None
    
    def _is_resnet_bn_match(self, conv_parts: List[str], bn_parts: List[str]) -> bool:
        """检查是否为ResNet风格的BatchNorm匹配"""
        if len(conv_parts) != len(bn_parts):
            return False
        
        # 检查所有部分except最后一个是否相同
        if conv_parts[:-1] != bn_parts[:-1]:
            return False
        
        conv_final = conv_parts[-1]
        bn_final = bn_parts[-1]
        
        # 标准匹配: conv1 -> bn1, conv2 -> bn2
        if conv_final.replace('conv', 'bn') == bn_final:
            return True
        
        return False
    
    def _is_sequential_bn_match(self, conv_parts: List[str], bn_parts: List[str]) -> bool:
        """检查是否为Sequential序列中的BatchNorm匹配"""
        # 用于处理序列: stem.0 (Conv) -> stem.1 (BN), layer1.0.shortcut.0 (Conv) -> layer1.0.shortcut.1 (BN)
        if len(conv_parts) != len(bn_parts):
            return False
        
        # 检查前面的路径是否相同
        if conv_parts[:-1] != bn_parts[:-1]:
            return False
        
        try:
            conv_idx = int(conv_parts[-1])
            bn_idx = int(bn_parts[-1])
            # BatchNorm通常紧跟在Conv后面
            if bn_idx == conv_idx + 1:
                return True
        except ValueError:
            # 处理非数字的情况，如果最后一部分相似
            conv_final = conv_parts[-1].lower()
            bn_final = bn_parts[-1].lower()
            
            # 检查是否是conv->bn的变体
            if ('conv' in conv_final and 'bn' in bn_final) or ('conv' in conv_final and 'norm' in bn_final):
                return True
        
        return False
    
    def _find_nearest_batchnorm(self, model: nn.Module, conv_layer_name: str) -> Optional[str]:
        """按模块遍历顺序查找最近的BatchNorm层"""
        modules_list = list(model.named_modules())
        conv_index = None
        
        # 找到Conv层的位置
        for i, (name, module) in enumerate(modules_list):
            if name == conv_layer_name:
                conv_index = i
                break
        
        if conv_index is None:
            return None
        
        # 在Conv层后面查找最近的BatchNorm
        for i in range(conv_index + 1, min(conv_index + 5, len(modules_list))):
            name, module = modules_list[i]
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                return name
        
        return None

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