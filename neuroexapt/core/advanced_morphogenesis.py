#!/usr/bin/env python3
"""
Advanced Morphogenesis Module - 高级形态发生模块

🧬 实现更复杂的结构变异策略：
1. 串行分裂 (Serial Division) - 增加网络深度
2. 并行分裂 (Parallel Division) - 创建多分支结构  
3. 混合分裂 (Hybrid Division) - 组合不同类型的层
4. 跳跃连接 (Skip Connections) - 增强信息流
5. 注意力机制 (Attention Mechanisms) - 提升特征选择能力

🎯 目标：突破传统架构限制，探索更高性能的网络拓扑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import copy
import math
import time
import traceback
from collections import defaultdict

# 配置详细调试日志
logger = logging.getLogger(__name__)

class DebugPrinter:
    """调试输出管理器 - 高级形态发生模块专用"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.indent_level = 0
        
    def print_debug(self, message: str, level: str = "INFO"):
        """打印调试信息"""
        if not self.enabled:
            return
            
        indent = "  " * self.indent_level
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        
        # 颜色编码
        colors = {
            "INFO": "\033[36m",      # 青色
            "SUCCESS": "\033[32m",   # 绿色 
            "WARNING": "\033[33m",   # 黄色
            "ERROR": "\033[31m",     # 红色
            "DEBUG": "\033[35m",     # 紫色
        }
        color = colors.get(level, "\033[0m")
        reset = "\033[0m"
        
        print(f"{color}[{timestamp}] {indent}{level}: {message}{reset}")
        
    def enter_section(self, section_name: str):
        """进入新的调试区域"""
        self.print_debug(f"🔍 进入 {section_name}", "DEBUG")
        self.indent_level += 1
        
    def exit_section(self, section_name: str):
        """退出调试区域"""
        self.indent_level = max(0, self.indent_level - 1)
        self.print_debug(f"✅ 完成 {section_name}", "DEBUG")

# 全局调试器
morpho_debug = DebugPrinter(enabled=True)

class MorphogenesisType(Enum):
    """形态发生类型枚举"""
    WIDTH_EXPANSION = "width_expansion"      # 宽度扩展
    SERIAL_DIVISION = "serial_division"      # 串行分裂
    PARALLEL_DIVISION = "parallel_division"  # 并行分裂
    HYBRID_DIVISION = "hybrid_division"      # 混合分裂
    SKIP_CONNECTION = "skip_connection"      # 跳跃连接
    ATTENTION_INJECTION = "attention_injection"  # 注意力注入

@dataclass
class MorphogenesisDecision:
    """形态发生决策"""
    morphogenesis_type: MorphogenesisType
    target_location: str
    confidence: float
    expected_improvement: float
    complexity_cost: float
    parameters_added: int
    reasoning: str

class AdvancedBottleneckAnalyzer:
    """高级瓶颈分析器"""
    
    def __init__(self):
        self.analysis_history = []
        
    def analyze_network_bottlenecks(self, model: nn.Module, activations: Dict[str, torch.Tensor], 
                                  gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """深度分析网络瓶颈"""
        morpho_debug.enter_section("网络瓶颈分析")
        morpho_debug.print_debug(f"分析输入: 模型层数={len(list(model.named_modules()))}, 激活值层数={len(activations)}, 梯度层数={len(gradients)}", "INFO")
        
        analysis = {}
        
        # 分别分析各类瓶颈
        morpho_debug.print_debug("1/5 分析深度瓶颈", "DEBUG")
        analysis['depth_bottlenecks'] = self._analyze_depth_bottlenecks(activations, gradients)
        
        morpho_debug.print_debug("2/5 分析宽度瓶颈", "DEBUG")
        analysis['width_bottlenecks'] = self._analyze_width_bottlenecks(activations, gradients)
        
        morpho_debug.print_debug("3/5 分析信息流瓶颈", "DEBUG")
        analysis['information_flow_bottlenecks'] = self._analyze_information_flow(activations)
        
        morpho_debug.print_debug("4/5 分析梯度流瓶颈", "DEBUG")
        analysis['gradient_flow_bottlenecks'] = self._analyze_gradient_flow(gradients)
        
        morpho_debug.print_debug("5/5 分析容量瓶颈", "DEBUG")
        analysis['capacity_bottlenecks'] = self._analyze_capacity_bottlenecks(model, activations)
        
        # 输出瓶颈汇总
        morpho_debug.print_debug("瓶颈分析汇总:", "INFO")
        for bottleneck_type, results in analysis.items():
            if isinstance(results, dict):
                count = len([k for k, v in results.items() if v > 0.5])  # 假设0.5为高瓶颈阈值
                morpho_debug.print_debug(f"  {bottleneck_type}: {count}个高瓶颈位置", "DEBUG")
        
        self.analysis_history.append(analysis)
        morpho_debug.exit_section("网络瓶颈分析")
        return analysis
    
    def _analyze_depth_bottlenecks(self, 
                                   activations: Dict[str, torch.Tensor], 
                                   gradients: Dict[str, torch.Tensor],
                                   perform_gc: bool = False,
                                   memory_threshold_mb: Optional[int] = None) -> Dict[str, float]:
        """分析深度瓶颈 - 需要增加层数的位置
        
        Args:
            activations: 层名到激活的映射
            gradients: 层名到梯度的映射
            perform_gc: 是否在每层后执行垃圾回收和CUDA缓存清理
            memory_threshold_mb: 仅当内存使用超过此阈值（MB）时才执行清理
        """
        import gc
        
        depth_scores = {}
        
        layer_names = list(activations.keys())
        for i, layer_name in enumerate(layer_names):
            if layer_name not in gradients:
                continue
                
            activation = activations[layer_name]
            gradient = gradients[layer_name]
            
            # 1. 激活饱和度分析
            saturation_score = self._compute_activation_saturation(activation)
            
            # 2. 梯度消失/爆炸分析
            gradient_health = self._compute_gradient_health(gradient)
            
            # 3. 层间信息损失分析
            if i > 0:
                prev_layer = layer_names[i-1]
                if prev_layer in activations:
                    info_loss = self._compute_information_loss(
                        activations[prev_layer], activation
                    )
                else:
                    info_loss = 0.0
            else:
                info_loss = 0.0
            
            # 4. 计算深度瓶颈分数
            depth_score = (
                0.4 * saturation_score +
                0.3 * (1.0 - gradient_health) +
                0.3 * info_loss
            )
            
            depth_scores[layer_name] = depth_score
            
            # 可选的垃圾回收和CUDA缓存清理
            do_cleanup = False
            if perform_gc:
                if memory_threshold_mb is not None:
                    # 仅当内存使用超过阈值时才清理
                    if torch.cuda.is_available():
                        mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
                        if mem_mb > memory_threshold_mb:
                            do_cleanup = True
                    else:
                        try:
                            import psutil
                            mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
                            if mem_mb > memory_threshold_mb:
                                do_cleanup = True
                        except ImportError:
                            # psutil not available, fallback to always clean if requested
                            do_cleanup = True
                else:
                    do_cleanup = True
                    
            if do_cleanup:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        return depth_scores
    
    def _analyze_width_bottlenecks(self, activations: Dict[str, torch.Tensor], 
                                 gradients: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """分析宽度瓶颈 - 需要增加神经元数量的位置"""
        width_scores = {}
        
        for layer_name, activation in activations.items():
            if layer_name not in gradients:
                continue
                
            gradient = gradients[layer_name]
            
            # 1. 神经元利用率分析
            utilization = self._compute_neuron_utilization(activation)
            
            # 2. 梯度方差分析
            gradient_variance = self._compute_gradient_variance(gradient)
            
            # 3. 激活模式多样性
            activation_diversity = self._compute_activation_diversity(activation)
            
            # 4. 计算宽度瓶颈分数
            width_score = (
                0.4 * (1.0 - utilization) +
                0.3 * gradient_variance +
                0.3 * (1.0 - activation_diversity)
            )
            
            width_scores[layer_name] = width_score
            
        return width_scores
    
    def _analyze_information_flow(self, activations: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """分析信息流瓶颈 - 需要并行分支的位置"""
        morpho_debug.enter_section("信息流瓶颈分析")
        flow_scores = {}
        layer_names = list(activations.keys())
        
        morpho_debug.print_debug(f"分析{len(layer_names)}层的信息流", "INFO")
        
        for i, layer_name in enumerate(layer_names):
            morpho_debug.print_debug(f"分析层 {i+1}/{len(layer_names)}: {layer_name}", "DEBUG")
            activation = activations[layer_name]
            
            # 内存检查
            if activation.numel() > 10**7:  # 超过1000万元素
                morpho_debug.print_debug(f"⚠️ 大张量检测: {activation.shape}, 元素数={activation.numel():,}", "WARNING")
            
            try:
                # 1. 信息瓶颈分析
                morpho_debug.print_debug(f"计算熵值...", "DEBUG")
                entropy = self._compute_entropy(activation)
                
                # 2. 特征相关性分析
                morpho_debug.print_debug(f"计算特征相关性...", "DEBUG")
                feature_correlation = self._compute_feature_correlation(activation)
                
                # 3. 信息冗余分析
                morpho_debug.print_debug(f"计算信息冗余度...", "DEBUG")
                redundancy = self._compute_information_redundancy(activation)
                
                # 4. 计算信息流瓶颈分数
                flow_score = (
                    0.3 * (1.0 - entropy) +
                    0.4 * feature_correlation +
                    0.3 * redundancy
                )
                
                flow_scores[layer_name] = flow_score
                morpho_debug.print_debug(f"层{layer_name}: 熵={entropy:.3f}, 相关性={feature_correlation:.3f}, 冗余={redundancy:.3f}, 分数={flow_score:.3f}", "DEBUG")
                
            except Exception as e:
                morpho_debug.print_debug(f"❌ 层{layer_name}分析失败: {e}", "ERROR")
                flow_scores[layer_name] = 0.0
                
            # 可配置的垃圾回收，仅在需要时执行以避免性能损失
            # 注意：频繁的垃圾回收可能影响性能，建议设置memory_threshold_mb参数
            if i % 5 == 0 and getattr(self, 'enable_gc', False):  # 每5层清理一次，默认关闭
                import gc
                memory_threshold = getattr(self, 'gc_memory_threshold_mb', 1024)  # 默认1GB阈值
                
                do_cleanup = False
                if torch.cuda.is_available():
                    mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    if mem_mb > memory_threshold:
                        do_cleanup = True
                else:
                    try:
                        import psutil
                        mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
                        if mem_mb > memory_threshold:
                            do_cleanup = True
                    except ImportError:
                        do_cleanup = True  # fallback if psutil unavailable
                
                if do_cleanup:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
        morpho_debug.print_debug(f"信息流分析完成，共{len(flow_scores)}层", "SUCCESS")
        morpho_debug.exit_section("信息流瓶颈分析")
        return flow_scores
    
    def _compute_activation_saturation(self, activation: torch.Tensor) -> float:
        """计算激活饱和度"""
        if activation.numel() == 0:
            return 0.0
            
        # 计算激活值接近极值的比例
        activation_flat = activation.flatten()
        
        # 对于不同激活函数的饱和度计算
        if torch.all(activation_flat >= 0):  # ReLU类激活
            saturated = torch.sum(activation_flat == 0).float()
        else:  # Tanh类激活
            saturated = torch.sum(torch.abs(activation_flat) > 0.9).float()
            
        saturation_ratio = saturated / activation_flat.numel()
        return saturation_ratio.item()
    
    def _compute_gradient_health(self, gradient: torch.Tensor) -> float:
        """计算梯度健康度"""
        if gradient is None or gradient.numel() == 0:
            return 0.0
            
        grad_norm = torch.norm(gradient).item()
        grad_mean = torch.mean(torch.abs(gradient)).item()
        
        # 梯度过小或过大都不健康
        if grad_norm < 1e-7:  # 梯度消失
            return 0.1
        elif grad_norm > 10.0:  # 梯度爆炸
            return 0.2
        else:
            # 理想的梯度范围
            health = 1.0 / (1.0 + abs(math.log10(grad_mean + 1e-8)))
            return min(health, 1.0)
    
    def _compute_information_loss(self, prev_activation: torch.Tensor, 
                                curr_activation: torch.Tensor) -> float:
        """计算层间信息损失"""
        try:
            # 简化的信息损失计算
            prev_entropy = self._compute_entropy(prev_activation)
            curr_entropy = self._compute_entropy(curr_activation)
            
            # 信息损失 = (前一层熵 - 当前层熵) / 前一层熵
            if prev_entropy > 1e-8:
                loss = max(0, (prev_entropy - curr_entropy) / prev_entropy)
                return min(loss, 1.0)
            else:
                return 0.0
        except:
            return 0.0
    
    def _compute_entropy(self, activation: torch.Tensor) -> float:
        """计算激活值熵"""
        if activation.numel() == 0:
            return 0.0
            
        # 将激活值转换为概率分布
        activation_flat = activation.flatten()
        activation_abs = torch.abs(activation_flat) + 1e-8
        probs = activation_abs / torch.sum(activation_abs)
        
        # 计算熵
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return entropy.item()
    
    def _compute_neuron_utilization(self, activation: torch.Tensor) -> float:
        """计算神经元利用率"""
        if activation.numel() == 0:
            return 0.0
            
        # 计算激活的神经元比例
        if len(activation.shape) >= 2:
            # 对于每个样本，计算激活的神经元
            batch_size = activation.shape[0]
            activation_reshaped = activation.view(batch_size, -1)
            active_neurons = torch.sum(activation_reshaped > 1e-6, dim=0)
            utilization = torch.mean((active_neurons > 0).float())
            return utilization.item()
        else:
            return 1.0
    
    def _compute_gradient_variance(self, gradient: torch.Tensor) -> float:
        """计算梯度方差"""
        if gradient is None or gradient.numel() == 0:
            return 0.0
            
        grad_flat = gradient.flatten()
        variance = torch.var(grad_flat)
        
        # 归一化方差
        mean_abs = torch.mean(torch.abs(grad_flat))
        if mean_abs > 1e-8:
            normalized_variance = variance / (mean_abs ** 2)
            return min(normalized_variance.item(), 1.0)
        else:
            return 0.0
    
    def _compute_activation_diversity(self, activation: torch.Tensor) -> float:
        """计算激活模式多样性"""
        if activation.numel() == 0 or len(activation.shape) < 2:
            return 0.0
            
        batch_size = activation.shape[0]
        if batch_size < 2:
            return 0.0
            
        # 计算批次内激活模式的相似性
        activation_flat = activation.view(batch_size, -1)
        
        # 计算样本间的余弦相似度
        similarities = []
        for i in range(min(batch_size, 10)):  # 限制计算量
            for j in range(i+1, min(batch_size, 10)):
                sim = F.cosine_similarity(
                    activation_flat[i:i+1], 
                    activation_flat[j:j+1], 
                    dim=1
                )
                similarities.append(sim.item())
        
        if similarities:
            avg_similarity = np.mean(similarities)
            diversity = 1.0 - avg_similarity  # 相似度越低，多样性越高
            return max(0.0, diversity)
        else:
            return 0.0
    
    def _compute_feature_correlation(self, activation: torch.Tensor) -> float:
        """计算特征相关性 - 内存优化版本"""
        if activation.numel() == 0 or len(activation.shape) < 2:
            return 0.0
            
        try:
            activation_flat = activation.view(activation.shape[0], -1)
            if activation_flat.shape[1] < 2:
                return 0.0
            
            # 内存优化：限制特征数量，使用采样
            max_features = 512  # 最大特征数限制
            if activation_flat.shape[1] > max_features:
                # 随机采样特征
                indices = torch.randperm(activation_flat.shape[1])[:max_features]
                activation_flat = activation_flat[:, indices]
            
            # 进一步限制：如果还是太大，使用更小的样本
            if activation_flat.shape[0] > 64:
                indices = torch.randperm(activation_flat.shape[0])[:64]
                activation_flat = activation_flat[indices]
                
            # 计算特征间的相关系数 - 仅在可管理的大小时
            if activation_flat.shape[1] > 1024:
                # 对于非常大的特征，使用近似方法
                # 随机选择特征对计算相关性
                num_pairs = min(100, activation_flat.shape[1] // 2)
                correlations = []
                
                for _ in range(num_pairs):
                    i = torch.randint(0, activation_flat.shape[1], (1,)).item()
                    j = torch.randint(0, activation_flat.shape[1], (1,)).item()
                    if i != j:
                        corr = torch.corrcoef(torch.stack([activation_flat[:, i], activation_flat[:, j]]))[0, 1]
                        if not torch.isnan(corr):
                            correlations.append(torch.abs(corr).item())
                
                return np.mean(correlations) if correlations else 0.0
            else:
                # 标准相关性计算
                correlation_matrix = torch.corrcoef(activation_flat.T)
                
                # 检查矩阵是否有效
                if torch.isnan(correlation_matrix).any():
                    return 0.0
                
                # 计算平均绝对相关系数
                mask = torch.eye(correlation_matrix.shape[0], dtype=torch.bool)
                off_diagonal = correlation_matrix[~mask]
                
                if len(off_diagonal) > 0:
                    avg_correlation = torch.mean(torch.abs(off_diagonal))
                    return avg_correlation.item()
                else:
                    return 0.0
        except Exception as e:
            morpho_debug.print_debug(f"特征相关性计算失败: {e}", "WARNING")
            return 0.0
    
    def _compute_information_redundancy(self, activation: torch.Tensor) -> float:
        """计算信息冗余度 - 内存优化版本"""
        if activation.numel() == 0:
            return 0.0
            
        try:
            # 内存优化：对于大张量使用采样
            activation_flat = activation.flatten()
            
            # 限制分析的元素数量
            max_elements = 100000  # 最大分析10万个元素
            if len(activation_flat) > max_elements:
                # 随机采样
                indices = torch.randperm(len(activation_flat))[:max_elements]
                activation_flat = activation_flat[indices]
            
            # 计算重复值的比例
            unique_values = torch.unique(activation_flat)
            redundancy = 1.0 - (len(unique_values) / len(activation_flat))
            
            return redundancy
        except Exception as e:
            morpho_debug.print_debug(f"信息冗余度计算失败: {e}", "WARNING")
            return 0.0
    
    def _analyze_gradient_flow(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """分析梯度流瓶颈"""
        flow_scores = {}
        
        layer_names = list(gradients.keys())
        grad_norms = []
        
        # 计算每层的梯度范数
        for layer_name in layer_names:
            if gradients[layer_name] is not None:
                norm = torch.norm(gradients[layer_name]).item()
                grad_norms.append(norm)
            else:
                grad_norms.append(0.0)
        
        if not grad_norms:
            return flow_scores
            
        # 计算梯度范数的变化率
        for i, layer_name in enumerate(layer_names):
            if i == 0:
                flow_scores[layer_name] = 0.0
                continue
                
            prev_norm = grad_norms[i-1]
            curr_norm = grad_norms[i]
            
            # 梯度衰减率
            if prev_norm > 1e-8:
                decay_rate = (prev_norm - curr_norm) / prev_norm
                flow_scores[layer_name] = max(0.0, decay_rate)
            else:
                flow_scores[layer_name] = 0.0
                
        return flow_scores
    
    def _analyze_capacity_bottlenecks(self, model: nn.Module, 
                                    activations: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """分析容量瓶颈"""
        capacity_scores = {}
        
        for name, module in model.named_modules():
            if name in activations and isinstance(module, (nn.Linear, nn.Conv2d)):
                activation = activations[name]
                
                # 计算层的理论容量
                if isinstance(module, nn.Linear):
                    theoretical_capacity = module.in_features * module.out_features
                    actual_capacity = self._compute_actual_capacity(activation)
                elif isinstance(module, nn.Conv2d):
                    theoretical_capacity = (module.in_channels * module.out_channels * 
                                          module.kernel_size[0] * module.kernel_size[1])
                    actual_capacity = self._compute_actual_capacity(activation)
                else:
                    continue
                
                # 容量利用率
                if theoretical_capacity > 0:
                    utilization = actual_capacity / theoretical_capacity
                    capacity_scores[name] = 1.0 - utilization  # 利用率越低，瓶颈分数越高
                else:
                    capacity_scores[name] = 0.0
        
        return capacity_scores
    
    def _compute_actual_capacity(self, activation: torch.Tensor) -> float:
        """计算实际使用的容量"""
        if activation.numel() == 0:
            return 0.0
            
        # 计算有效激活的数量
        effective_activations = torch.sum(torch.abs(activation) > 1e-6).item()
        return effective_activations

class AdvancedMorphogenesisExecutor:
    """高级形态发生执行器"""
    
    def __init__(self):
        self.execution_history = []
        
    def execute_morphogenesis(self, model: nn.Module, decision: MorphogenesisDecision) -> Tuple[nn.Module, int]:
        """执行形态发生"""
        try:
            # 获取模型设备
            device = next(model.parameters()).device
            
            if decision.morphogenesis_type == MorphogenesisType.SERIAL_DIVISION:
                new_model, params_added = self._execute_serial_division(model, decision.target_location)
            elif decision.morphogenesis_type == MorphogenesisType.PARALLEL_DIVISION:
                new_model, params_added = self._execute_parallel_division(model, decision.target_location)
            elif decision.morphogenesis_type == MorphogenesisType.HYBRID_DIVISION:
                new_model, params_added = self._execute_hybrid_division(model, decision.target_location)
            elif decision.morphogenesis_type == MorphogenesisType.SKIP_CONNECTION:
                new_model, params_added = self._execute_skip_connection(model, decision.target_location)
            elif decision.morphogenesis_type == MorphogenesisType.ATTENTION_INJECTION:
                new_model, params_added = self._execute_attention_injection(model, decision.target_location)
            else:
                # 默认使用宽度扩展
                new_model, params_added = self._execute_width_expansion(model, decision.target_location)
            
            # 确保新模型在正确的设备上
            new_model = new_model.to(device)
            
            return new_model, params_added
                
        except Exception as e:
            logger.error(f"形态发生执行失败: {e}")
            import traceback
            traceback.print_exc()
            return model, 0
    
    def _execute_serial_division(self, model: nn.Module, target_location: str) -> Tuple[nn.Module, int]:
        """执行串行分裂 - 在目标层后插入新层"""
        logger.info(f"执行串行分裂: {target_location}")
        
        # 创建新模型
        new_model = copy.deepcopy(model)
        parameters_added = 0
        
        # 查找目标层
        target_module = None
        target_parent = None
        target_attr = None
        
        for name, module in new_model.named_modules():
            if name == target_location:
                target_module = module
                # 找到父模块
                if '.' in name:
                    parent_name = '.'.join(name.split('.')[:-1])
                    target_attr = name.split('.')[-1]
                    for pname, pmodule in new_model.named_modules():
                        if pname == parent_name:
                            target_parent = pmodule
                            break
                break
        
        if target_module is None:
            logger.warning(f"未找到目标层: {target_location}")
            return model, 0
        
        if isinstance(target_module, nn.Linear):
            # 获取设备信息
            device = target_module.weight.device
            
            # 在全连接层后插入新的全连接层
            hidden_size = max(target_module.out_features // 2, 64)
            
            # 创建新的中间层
            intermediate_layer = nn.Linear(target_module.out_features, hidden_size).to(device)
            output_layer = nn.Linear(hidden_size, target_module.out_features).to(device)
            
            # 初始化权重使新层组合接近原层
            with torch.no_grad():
                # 中间层：压缩表示
                nn.init.kaiming_normal_(intermediate_layer.weight)
                nn.init.zeros_(intermediate_layer.bias)
                
                # 输出层：重构到原始维度
                nn.init.kaiming_normal_(output_layer.weight)
                nn.init.zeros_(output_layer.bias)
            
            # 创建新的序列模块
            new_sequence = nn.Sequential(
                target_module,
                nn.ReLU(),
                intermediate_layer,
                nn.ReLU(),
                output_layer
            )
            
            # 替换原模块
            if target_parent is not None and target_attr is not None:
                setattr(target_parent, target_attr, new_sequence)
                parameters_added = (hidden_size * target_module.out_features + hidden_size +
                                  target_module.out_features * hidden_size + target_module.out_features)
                
        elif isinstance(target_module, nn.Conv2d):
            # 获取设备信息
            device = target_module.weight.device
            
            # 在卷积层后插入新的卷积层
            intermediate_channels = max(target_module.out_channels // 2, 32)
            
            # 创建新的卷积层序列
            intermediate_conv = nn.Conv2d(
                target_module.out_channels, 
                intermediate_channels,
                kernel_size=3, 
                padding=1
            ).to(device)
            output_conv = nn.Conv2d(
                intermediate_channels,
                target_module.out_channels,
                kernel_size=3,
                padding=1
            ).to(device)
            
            # 初始化权重
            with torch.no_grad():
                nn.init.kaiming_normal_(intermediate_conv.weight)
                nn.init.zeros_(intermediate_conv.bias)
                nn.init.kaiming_normal_(output_conv.weight)
                nn.init.zeros_(output_conv.bias)
            
            # 创建新的序列
            new_sequence = nn.Sequential(
                target_module,
                nn.ReLU(),
                intermediate_conv,
                nn.ReLU(),
                output_conv
            )
            
            # 替换原模块
            if target_parent is not None and target_attr is not None:
                setattr(target_parent, target_attr, new_sequence)
                parameters_added = (intermediate_channels * target_module.out_channels * 9 + intermediate_channels +
                                  target_module.out_channels * intermediate_channels * 9 + target_module.out_channels)
        
        self.execution_history.append({
            'type': 'serial_division',
            'location': target_location,
            'parameters_added': parameters_added
        })
        
        logger.info(f"串行分裂完成，新增参数: {parameters_added}")
        return new_model, parameters_added
    
    def _execute_parallel_division(self, model: nn.Module, target_location: str) -> Tuple[nn.Module, int]:
        """执行并行分裂 - 创建多分支结构"""
        logger.info(f"执行并行分裂: {target_location}")
        
        new_model = copy.deepcopy(model)
        parameters_added = 0
        
        # 查找目标层
        target_module = None
        target_parent = None
        target_attr = None
        
        for name, module in new_model.named_modules():
            if name == target_location:
                target_module = module
                if '.' in name:
                    parent_name = '.'.join(name.split('.')[:-1])
                    target_attr = name.split('.')[-1]
                    for pname, pmodule in new_model.named_modules():
                        if pname == parent_name:
                            target_parent = pmodule
                            break
                break
        
        if target_module is None:
            logger.warning(f"未找到目标层: {target_location}")
            return model, 0
        
        if isinstance(target_module, nn.Linear):
            # 获取设备信息
            device = target_module.weight.device
            
            # 创建并行分支
            branch_size = target_module.out_features // 3
            
            # 三个并行分支：不同的特征提取策略
            branch1 = nn.Linear(target_module.in_features, branch_size).to(device)  # 标准线性变换
            branch2 = nn.Sequential(  # 深度分支
                nn.Linear(target_module.in_features, branch_size),
                nn.ReLU(),
                nn.Linear(branch_size, branch_size)
            ).to(device)
            branch3 = nn.Sequential(  # 残差分支
                nn.Linear(target_module.in_features, branch_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(branch_size, branch_size)
            ).to(device)
            
            # 融合层
            fusion_layer = nn.Linear(branch_size * 3, target_module.out_features).to(device)
            
            # 初始化权重
            for branch in [branch1, branch2, branch3]:
                if isinstance(branch, nn.Linear):
                    nn.init.kaiming_normal_(branch.weight)
                    nn.init.zeros_(branch.bias)
                else:
                    for layer in branch:
                        if isinstance(layer, nn.Linear):
                            nn.init.kaiming_normal_(layer.weight)
                            nn.init.zeros_(layer.bias)
            
            nn.init.kaiming_normal_(fusion_layer.weight)
            nn.init.zeros_(fusion_layer.bias)
            
            # 创建并行模块
            class ParallelBranches(nn.Module):
                def __init__(self, branch1, branch2, branch3, fusion):
                    super().__init__()
                    self.branch1 = branch1
                    self.branch2 = branch2
                    self.branch3 = branch3
                    self.fusion = fusion
                
                def forward(self, x):
                    out1 = self.branch1(x)
                    out2 = self.branch2(x)
                    out3 = self.branch3(x)
                    combined = torch.cat([out1, out2, out3], dim=-1)
                    return self.fusion(combined)
            
            parallel_module = ParallelBranches(branch1, branch2, branch3, fusion_layer)
            
            # 替换原模块
            if target_parent is not None and target_attr is not None:
                setattr(target_parent, target_attr, parallel_module)
                
                # 计算新增参数
                params1 = branch_size * target_module.in_features + branch_size
                params2 = (branch_size * target_module.in_features + branch_size + 
                          branch_size * branch_size + branch_size)
                params3 = (branch_size * target_module.in_features + branch_size + 
                          branch_size * branch_size + branch_size)
                params_fusion = target_module.out_features * (branch_size * 3) + target_module.out_features
                parameters_added = params1 + params2 + params3 + params_fusion
                
        elif isinstance(target_module, nn.Conv2d):
            # 获取设备信息
            device = target_module.weight.device
            
            # 创建卷积并行分支
            branch_channels = target_module.out_channels // 3
            
            # 三个不同尺度的卷积分支
            branch1 = nn.Conv2d(target_module.in_channels, branch_channels, 
                              kernel_size=1, padding=0).to(device)  # 1x1卷积
            branch2 = nn.Conv2d(target_module.in_channels, branch_channels, 
                              kernel_size=3, padding=1).to(device)  # 3x3卷积
            branch3 = nn.Conv2d(target_module.in_channels, branch_channels, 
                              kernel_size=5, padding=2).to(device)  # 5x5卷积
            
            # 融合卷积
            fusion_conv = nn.Conv2d(branch_channels * 3, target_module.out_channels, 
                                  kernel_size=1, padding=0).to(device)
            
            # 初始化权重
            for branch in [branch1, branch2, branch3, fusion_conv]:
                nn.init.kaiming_normal_(branch.weight)
                nn.init.zeros_(branch.bias)
            
            # 创建并行卷积模块
            class ParallelConv(nn.Module):
                def __init__(self, branch1, branch2, branch3, fusion):
                    super().__init__()
                    self.branch1 = branch1
                    self.branch2 = branch2
                    self.branch3 = branch3
                    self.fusion = fusion
                
                def forward(self, x):
                    out1 = self.branch1(x)
                    out2 = self.branch2(x)
                    out3 = self.branch3(x)
                    combined = torch.cat([out1, out2, out3], dim=1)
                    return self.fusion(combined)
            
            parallel_module = ParallelConv(branch1, branch2, branch3, fusion_conv)
            
            # 替换原模块
            if target_parent is not None and target_attr is not None:
                setattr(target_parent, target_attr, parallel_module)
                
                # 计算新增参数
                params1 = branch_channels * target_module.in_channels * 1 + branch_channels
                params2 = branch_channels * target_module.in_channels * 9 + branch_channels
                params3 = branch_channels * target_module.in_channels * 25 + branch_channels
                params_fusion = target_module.out_channels * (branch_channels * 3) + target_module.out_channels
                parameters_added = params1 + params2 + params3 + params_fusion
        
        self.execution_history.append({
            'type': 'parallel_division',
            'location': target_location,
            'parameters_added': parameters_added
        })
        
        logger.info(f"并行分裂完成，新增参数: {parameters_added}")
        return new_model, parameters_added
    
    def _execute_hybrid_division(self, model: nn.Module, target_location: str) -> Tuple[nn.Module, int]:
        """执行混合分裂 - 组合不同类型的层"""
        logger.info(f"执行混合分裂: {target_location}")
        
        new_model = copy.deepcopy(model)
        parameters_added = 0
        
        # 查找目标层
        target_module = None
        target_parent = None
        target_attr = None
        
        for name, module in new_model.named_modules():
            if name == target_location:
                target_module = module
                if '.' in name:
                    parent_name = '.'.join(name.split('.')[:-1])
                    target_attr = name.split('.')[-1]
                    for pname, pmodule in new_model.named_modules():
                        if pname == parent_name:
                            target_parent = pmodule
                            break
                break
        
        if target_module is None:
            logger.warning(f"未找到目标层: {target_location}")
            return model, 0
        
        if isinstance(target_module, nn.Linear):
            # 获取设备信息
            device = target_module.weight.device
            
            # 创建混合结构：线性层 + 注意力机制 + 残差连接
            hidden_size = target_module.out_features
            
            # 主要变换
            main_transform = nn.Linear(target_module.in_features, hidden_size).to(device)
            
            # 注意力机制
            attention = nn.MultiheadAttention(
                embed_dim=target_module.in_features,
                num_heads=max(1, target_module.in_features // 64),
                batch_first=True
            ).to(device)
            attention_projection = nn.Linear(target_module.in_features, hidden_size).to(device)
            
            # 输出融合
            output_layer = nn.Linear(hidden_size * 2, target_module.out_features).to(device)
            
            # 初始化
            nn.init.kaiming_normal_(main_transform.weight)
            nn.init.zeros_(main_transform.bias)
            nn.init.kaiming_normal_(attention_projection.weight)
            nn.init.zeros_(attention_projection.bias)
            nn.init.kaiming_normal_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
            
            class HybridLinear(nn.Module):
                def __init__(self, main_transform, attention, attention_projection, output_layer):
                    super().__init__()
                    self.main_transform = main_transform
                    self.attention = attention
                    self.attention_projection = attention_projection
                    self.output_layer = output_layer
                
                def forward(self, x):
                    # 主要变换
                    main_out = self.main_transform(x)
                    
                    # 注意力分支
                    if len(x.shape) == 2:
                        x_unsqueezed = x.unsqueeze(1)  # 添加序列维度
                        attn_out, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
                        attn_out = attn_out.squeeze(1)  # 移除序列维度
                    else:
                        attn_out, _ = self.attention(x, x, x)
                    
                    attn_out = self.attention_projection(attn_out)
                    
                    # 融合
                    combined = torch.cat([main_out, attn_out], dim=-1)
                    return self.output_layer(combined)
            
            hybrid_module = HybridLinear(main_transform, attention, attention_projection, output_layer)
            
            # 替换原模块
            if target_parent is not None and target_attr is not None:
                setattr(target_parent, target_attr, hybrid_module)
                
                # 计算参数
                main_params = hidden_size * target_module.in_features + hidden_size
                attn_params = attention.in_proj_weight.numel() + attention.out_proj.weight.numel()
                proj_params = hidden_size * target_module.in_features + hidden_size
                output_params = target_module.out_features * (hidden_size * 2) + target_module.out_features
                parameters_added = main_params + attn_params + proj_params + output_params
        
        self.execution_history.append({
            'type': 'hybrid_division',
            'location': target_location,
            'parameters_added': parameters_added
        })
        
        logger.info(f"混合分裂完成，新增参数: {parameters_added}")
        return new_model, parameters_added
    
    def _execute_skip_connection(self, model: nn.Module, target_location: str) -> Tuple[nn.Module, int]:
        """执行跳跃连接添加"""
        logger.warning(f"跳跃连接功能尚未实现: {target_location}")
        # 跳跃连接实现较为复杂，需要修改模型的forward方法
        # 当前版本暂不支持此功能
        raise NotImplementedError("Skip connection morphogenesis is not yet implemented. "
                                "This requires complex model architecture modification.")
    
    def _execute_attention_injection(self, model: nn.Module, target_location: str) -> Tuple[nn.Module, int]:
        """执行注意力机制注入"""
        logger.warning(f"注意力注入功能尚未实现: {target_location}")
        # 注意力机制注入需要仔细的架构设计
        # 当前版本暂不支持此功能
        raise NotImplementedError("Attention injection morphogenesis is not yet implemented. "
                                "This requires careful attention mechanism design and integration.")
    
    def _execute_width_expansion(self, model: nn.Module, target_location: str) -> Tuple[nn.Module, int]:
        """执行宽度扩展（兜底策略）"""
        logger.info(f"执行宽度扩展: {target_location}")
        
        new_model = copy.deepcopy(model)
        parameters_added = 0
        
        # 找到目标层并扩展
        target_module = None
        for name, module in new_model.named_modules():
            if name == target_location:
                target_module = module
                break
        
        if target_module is None:
            logger.warning(f"未找到目标层: {target_location}")
            return model, 0
        
        if isinstance(target_module, nn.Linear):
            # 获取设备信息
            device = target_module.weight.device
            
            old_out = target_module.out_features
            new_out = int(old_out * 1.2)
            expansion = new_out - old_out
            
            # 扩展当前层的权重
            new_weight = torch.zeros(new_out, target_module.in_features, device=device)
            new_bias = torch.zeros(new_out, device=device) if target_module.bias is not None else None
            
            # 复制原权重
            new_weight[:old_out] = target_module.weight.data
            if new_bias is not None:
                new_bias[:old_out] = target_module.bias.data
                
            # 初始化新权重
            nn.init.normal_(new_weight[old_out:], std=0.01)
            if new_bias is not None:
                nn.init.zeros_(new_bias[old_out:])
            
            # 更新当前层
            target_module.out_features = new_out
            target_module.weight = nn.Parameter(new_weight)
            if target_module.bias is not None:
                target_module.bias = nn.Parameter(new_bias)
            
            parameters_added += expansion * (target_module.in_features + 1)
            
            # 找到并更新下一个线性层的输入维度
            all_modules = list(new_model.named_modules())
            target_index = None
            
            for i, (name, module) in enumerate(all_modules):
                if name == target_location:
                    target_index = i
                    break
            
            if target_index is not None:
                # 寻找下一个线性层
                for i in range(target_index + 1, len(all_modules)):
                    next_name, next_module = all_modules[i]
                    if isinstance(next_module, nn.Linear):
                        # 更新下一层的输入维度
                        old_in = next_module.in_features
                        new_in = new_out
                        
                        # 扩展下一层的权重
                        next_weight = torch.zeros(next_module.out_features, new_in, device=device)
                        
                        # 复制原有权重到对应位置
                        next_weight[:, :old_in] = next_module.weight.data
                        
                        # 为新增的输入维度初始化权重
                        nn.init.normal_(next_weight[:, old_in:], std=0.01)
                        
                        # 更新下一层
                        next_module.in_features = new_in
                        next_module.weight = nn.Parameter(next_weight)
                        
                        parameters_added += next_module.out_features * expansion
                        break
        
        return new_model, parameters_added

class IntelligentMorphogenesisDecisionMaker:
    """智能形态发生决策制定器"""
    
    def __init__(self):
        self.decision_history = []
        self.performance_tracker = {}
        
    def make_decision(self, bottleneck_analysis: Dict[str, Any], 
                     performance_history: List[float]) -> Optional[MorphogenesisDecision]:
        """制定形态发生决策"""
        
        # 分析不同类型的瓶颈
        depth_bottlenecks = bottleneck_analysis['depth_bottlenecks']
        width_bottlenecks = bottleneck_analysis['width_bottlenecks']
        flow_bottlenecks = bottleneck_analysis['information_flow_bottlenecks']
        
        # 找到最严重的瓶颈
        all_bottlenecks = []
        
        # 深度瓶颈 -> 串行分裂
        for layer, score in depth_bottlenecks.items():
            if score > 0.6:  # 深度瓶颈阈值
                all_bottlenecks.append({
                    'location': layer,
                    'type': MorphogenesisType.SERIAL_DIVISION,
                    'score': score,
                    'reasoning': f"深度瓶颈检测，分数: {score:.3f}"
                })
        
        # 信息流瓶颈 -> 并行分裂
        for layer, score in flow_bottlenecks.items():
            if score > 0.5:  # 信息流瓶颈阈值
                all_bottlenecks.append({
                    'location': layer,
                    'type': MorphogenesisType.PARALLEL_DIVISION,
                    'score': score,
                    'reasoning': f"信息流瓶颈检测，分数: {score:.3f}"
                })
        
        # 复杂瓶颈 -> 混合分裂
        for layer in set(depth_bottlenecks.keys()) & set(flow_bottlenecks.keys()):
            combined_score = (depth_bottlenecks[layer] + flow_bottlenecks[layer]) / 2
            if combined_score > 0.55:
                all_bottlenecks.append({
                    'location': layer,
                    'type': MorphogenesisType.HYBRID_DIVISION,
                    'score': combined_score,
                    'reasoning': f"复合瓶颈检测，深度:{depth_bottlenecks[layer]:.3f}, 流动:{flow_bottlenecks[layer]:.3f}"
                })
        
        # 宽度瓶颈 -> 宽度扩展
        for layer, score in width_bottlenecks.items():
            if score > 0.4:  # 宽度瓶颈阈值（较低，作为备选）
                all_bottlenecks.append({
                    'location': layer,
                    'type': MorphogenesisType.WIDTH_EXPANSION,
                    'score': score,
                    'reasoning': f"宽度瓶颈检测，分数: {score:.3f}"
                })
        
        if not all_bottlenecks:
            return None
        
        # 选择分数最高的瓶颈
        best_bottleneck = max(all_bottlenecks, key=lambda x: x['score'])
        
        # 估算性能改进和成本
        expected_improvement = self._estimate_improvement(best_bottleneck)
        complexity_cost = self._estimate_complexity_cost(best_bottleneck)
        parameters_added = self._estimate_parameters(best_bottleneck)
        
        decision = MorphogenesisDecision(
            morphogenesis_type=best_bottleneck['type'],
            target_location=best_bottleneck['location'],
            confidence=best_bottleneck['score'],
            expected_improvement=expected_improvement,
            complexity_cost=complexity_cost,
            parameters_added=parameters_added,
            reasoning=best_bottleneck['reasoning']
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _estimate_improvement(self, bottleneck: Dict) -> float:
        """估算性能改进"""
        score = bottleneck['score']
        morph_type = bottleneck['type']
        
        # 基于瓶颈类型和严重程度估算改进
        type_multipliers = {
            MorphogenesisType.SERIAL_DIVISION: 1.5,      # 串行分裂通常带来更大改进
            MorphogenesisType.PARALLEL_DIVISION: 1.3,    # 并行分裂提供多样性
            MorphogenesisType.HYBRID_DIVISION: 1.4,      # 混合分裂综合效果
            MorphogenesisType.WIDTH_EXPANSION: 1.0       # 宽度扩展效果较小
        }
        
        base_improvement = score * 0.05  # 基础改进：5%
        type_bonus = type_multipliers.get(morph_type, 1.0)
        
        return base_improvement * type_bonus
    
    def _estimate_complexity_cost(self, bottleneck: Dict) -> float:
        """估算复杂度成本"""
        morph_type = bottleneck['type']
        
        complexity_costs = {
            MorphogenesisType.SERIAL_DIVISION: 0.3,      # 增加深度，中等成本
            MorphogenesisType.PARALLEL_DIVISION: 0.5,    # 并行结构，较高成本
            MorphogenesisType.HYBRID_DIVISION: 0.6,      # 混合结构，最高成本
            MorphogenesisType.WIDTH_EXPANSION: 0.2       # 宽度扩展，低成本
        }
        
        return complexity_costs.get(morph_type, 0.3)
    
    def _estimate_parameters(self, bottleneck: Dict) -> int:
        """估算新增参数数量"""
        morph_type = bottleneck['type']
        
        # 基于类型的参数估算（粗略）
        base_params = {
            MorphogenesisType.SERIAL_DIVISION: 5000,
            MorphogenesisType.PARALLEL_DIVISION: 8000,
            MorphogenesisType.HYBRID_DIVISION: 10000,
            MorphogenesisType.WIDTH_EXPANSION: 3000
        }
        
        return base_params.get(morph_type, 3000)