#!/usr/bin/env python3
"""
Dynamic Neural Morphogenesis (DNM) Framework - 重构版本

🧬 核心理论支撑：
1. 信息论 (Information Theory) - 信息瓶颈和熵分析
2. 生物学原理 (Biological Principles) - 神经发育和突触可塑性
3. 动力学系统 (Dynamical Systems) - 梯度流和收敛性分析  
4. 认知科学 (Cognitive Science) - 学习曲线和记忆巩固
5. 网络科学 (Network Science) - 连接模式和拓扑分析

🎯 目标：突破传统神经网络的性能瓶颈，实现真正的智能形态发生
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
from collections import defaultdict, deque
import copy

# 配置日志
logger = logging.getLogger(__name__)

@dataclass
class MorphogenesisEvent:
    """形态发生事件记录"""
    epoch: int
    event_type: str  # 'neuron_division', 'connection_growth', 'pruning', 'topology_change'
    location: str    # 层名称或位置
    trigger_reason: str
    performance_before: float
    performance_after: Optional[float] = None
    parameters_added: int = 0
    complexity_change: float = 0.0
    
class MorphogenesisTrigger(ABC):
    """形态发生触发器抽象基类"""
    
    @abstractmethod
    def should_trigger(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """判断是否应该触发形态发生"""
        pass
    
    @abstractmethod
    def get_priority(self) -> float:
        """获取触发器优先级"""
        pass

class InformationTheoryTrigger(MorphogenesisTrigger):
    """基于信息论的触发器"""
    
    def __init__(self, entropy_threshold: float = 0.1, mi_threshold: float = 0.05):
        self.entropy_threshold = entropy_threshold
        self.mi_threshold = mi_threshold
        self.history = deque(maxlen=10)
        
    def should_trigger(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        activations = context.get('activations', {})
        gradients = context.get('gradients', {})
        
        if not activations or not gradients:
            return False, "缺少激活值或梯度信息"
            
        # 计算信息熵变化
        entropy_changes = self._compute_entropy_changes(activations)
        
        # 计算互信息
        mutual_info = self._compute_mutual_information(activations)
        
        # 梯度方差分析
        gradient_variance = self._analyze_gradient_variance(gradients)
        
        self.history.append({
            'entropy_changes': entropy_changes,
            'mutual_info': mutual_info,
            'gradient_variance': gradient_variance
        })
        
        # 信息瓶颈检测
        if self._detect_information_bottleneck():
            return True, f"信息瓶颈检测：熵变化={entropy_changes:.4f}, 互信息={mutual_info:.4f}"
            
        return False, "信息论指标未达到触发条件"
    
    def _compute_entropy_changes(self, activations: Dict[str, torch.Tensor]) -> float:
        """计算激活值熵的变化"""
        total_entropy_change = 0.0
        count = 0
        
        for name, activation in activations.items():
            if len(activation.shape) >= 2:
                # 计算每个神经元的熵
                activation_flat = activation.view(activation.size(0), -1)
                probs = F.softmax(activation_flat, dim=-1) + 1e-8
                entropy = -torch.sum(probs * torch.log(probs), dim=-1).mean()
                
                if len(self.history) > 0:
                    prev_entropy = self.history[-1].get('entropy_changes', 0)
                    entropy_change = abs(entropy.item() - prev_entropy)
                    total_entropy_change += entropy_change
                    count += 1
                    
        return total_entropy_change / max(count, 1)
    
    def _compute_mutual_information(self, activations: Dict[str, torch.Tensor]) -> float:
        """计算层间互信息"""
        layer_names = list(activations.keys())
        if len(layer_names) < 2:
            return 0.0
            
        # 简化的互信息估计
        mi_sum = 0.0
        pairs = 0
        
        for i in range(len(layer_names) - 1):
            for j in range(i + 1, min(i + 3, len(layer_names))):  # 只考虑相邻层
                act1 = activations[layer_names[i]].flatten()
                act2 = activations[layer_names[j]].flatten()
                
                # 使用相关系数近似互信息
                if len(act1) == len(act2):
                    correlation = torch.corrcoef(torch.stack([act1, act2]))[0, 1]
                    mi = -0.5 * torch.log(1 - correlation**2 + 1e-8)
                    mi_sum += mi.item()
                    pairs += 1
                    
        return mi_sum / max(pairs, 1)
    
    def _analyze_gradient_variance(self, gradients: Dict[str, torch.Tensor]) -> float:
        """分析梯度方差"""
        total_variance = 0.0
        count = 0
        
        for name, grad in gradients.items():
            if grad is not None:
                variance = torch.var(grad).item()
                total_variance += variance
                count += 1
                
        return total_variance / max(count, 1)
    
    def _detect_information_bottleneck(self) -> bool:
        """检测信息瓶颈"""
        if len(self.history) < 5:
            return False
            
        recent_entropies = [h['entropy_changes'] for h in list(self.history)[-5:]]
        recent_mis = [h['mutual_info'] for h in list(self.history)[-5:]]
        
        # 熵变化趋于稳定且互信息较低
        entropy_stability = np.std(recent_entropies) < self.entropy_threshold
        low_mi = np.mean(recent_mis) < self.mi_threshold
        
        return entropy_stability and low_mi
    
    def get_priority(self) -> float:
        return 0.8

class BiologicalPrinciplesTrigger(MorphogenesisTrigger):
    """基于生物学原理的触发器"""
    
    def __init__(self, learning_rate_threshold: float = 1e-4, saturation_threshold: float = 0.95):
        self.learning_rate_threshold = learning_rate_threshold
        self.saturation_threshold = saturation_threshold
        self.performance_history = deque(maxlen=20)
        
    def should_trigger(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        current_performance = context.get('current_performance', 0.0)
        learning_rate = context.get('learning_rate', 1e-3)
        activations = context.get('activations', {})
        
        self.performance_history.append(current_performance)
        
        # 模拟神经可塑性 - Hebbian学习原理
        if self._detect_hebbian_potential(activations):
            return True, f"Hebbian可塑性检测：性能={current_performance:.4f}"
            
        # 模拟突触稳态 - 性能平台期检测
        if self._detect_homeostatic_imbalance():
            return True, f"稳态失衡检测：性能停滞，建议结构调整"
            
        # 模拟神经发育的关键期
        if self._detect_critical_period():
            return True, f"关键发育期检测：适合结构重组"
            
        return False, "生物学指标未达到触发条件"
    
    def _detect_hebbian_potential(self, activations: Dict[str, torch.Tensor]) -> bool:
        """检测Hebbian可塑性潜力"""
        if not activations:
            return False
            
        # 分析激活模式的相关性
        correlation_strengths = []
        
        for name, activation in activations.items():
            if len(activation.shape) >= 2:
                # 计算神经元间的相关性
                act_flat = activation.view(activation.size(0), -1)
                if act_flat.size(1) > 1:
                    corr_matrix = torch.corrcoef(act_flat.T)
                    # 移除对角线元素
                    mask = ~torch.eye(corr_matrix.size(0), dtype=bool)
                    corr_values = corr_matrix[mask]
                    avg_correlation = torch.mean(torch.abs(corr_values)).item()
                    correlation_strengths.append(avg_correlation)
        
        if correlation_strengths:
            mean_correlation = np.mean(correlation_strengths)
            return mean_correlation > 0.7  # 高相关性表明可以分裂
            
        return False
    
    def _detect_homeostatic_imbalance(self) -> bool:
        """检测稳态失衡"""
        if len(self.performance_history) < 10:
            return False
            
        recent_performance = list(self.performance_history)[-10:]
        performance_std = np.std(recent_performance)
        performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        
        # 性能停滞且无明显上升趋势
        return performance_std < 0.01 and abs(performance_trend) < 0.001
    
    def _detect_critical_period(self) -> bool:
        """检测关键发育期"""
        if len(self.performance_history) < 15:
            return False
            
        # 模拟生物神经网络的关键期
        recent_performance = list(self.performance_history)[-15:]
        
        # 查找性能快速上升后的平台期
        for i in range(5, len(recent_performance)):
            early_avg = np.mean(recent_performance[:i-5])
            recent_avg = np.mean(recent_performance[i-5:i])
            latest_avg = np.mean(recent_performance[i:])
            
            # 快速上升后停滞
            if (recent_avg - early_avg > 0.05) and (abs(latest_avg - recent_avg) < 0.01):
                return True
                
        return False
    
    def get_priority(self) -> float:
        return 0.9

class DynamicalSystemsTrigger(MorphogenesisTrigger):
    """基于动力学系统的触发器"""
    
    def __init__(self):
        self.gradient_history = deque(maxlen=15)
        self.loss_history = deque(maxlen=20)
        
    def should_trigger(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        gradients = context.get('gradients', {})
        current_loss = context.get('current_loss', float('inf'))
        
        self.loss_history.append(current_loss)
        
        if gradients:
            gradient_norm = self._compute_gradient_norm(gradients)
            self.gradient_history.append(gradient_norm)
        
        # 检测梯度消失/爆炸
        if self._detect_gradient_pathology():
            return True, "梯度病理检测：需要结构调整改善梯度流"
            
        # 检测损失函数的动力学特性
        if self._detect_loss_dynamics_anomaly():
            return True, "损失动力学异常：建议增加模型容量"
            
        # 检测收敛性问题
        if self._detect_convergence_issues():
            return True, "收敛性问题检测：模型可能欠拟合"
            
        return False, "动力学系统指标正常"
    
    def _compute_gradient_norm(self, gradients: Dict[str, torch.Tensor]) -> float:
        """计算梯度范数"""
        total_norm = 0.0
        for grad in gradients.values():
            if grad is not None:
                total_norm += torch.norm(grad).item() ** 2
        return math.sqrt(total_norm)
    
    def _detect_gradient_pathology(self) -> bool:
        """检测梯度病理"""
        if len(self.gradient_history) < 10:
            return False
            
        recent_grads = list(self.gradient_history)[-10:]
        
        # 梯度消失
        if np.mean(recent_grads) < 1e-6:
            return True
            
        # 梯度爆炸
        if np.max(recent_grads) > 100:
            return True
            
        # 梯度振荡
        grad_diff = np.diff(recent_grads)
        if len(grad_diff) > 5:
            oscillation = np.sum(np.diff(np.sign(grad_diff)) != 0) / len(grad_diff)
            if oscillation > 0.7:
                return True
                
        return False
    
    def _detect_loss_dynamics_anomaly(self) -> bool:
        """检测损失动力学异常"""
        if len(self.loss_history) < 15:
            return False
            
        recent_losses = list(self.loss_history)[-15:]
        
        # 损失停滞
        loss_std = np.std(recent_losses)
        if loss_std < 0.001:
            return True
            
        # 损失振荡而不收敛
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        if abs(loss_trend) < 0.001 and loss_std > 0.01:
            return True
            
        return False
    
    def _detect_convergence_issues(self) -> bool:
        """检测收敛性问题"""
        if len(self.loss_history) < 15 or len(self.gradient_history) < 10:
            return False
            
        # 损失下降缓慢且梯度很小
        recent_losses = list(self.loss_history)[-10:]
        recent_grads = list(self.gradient_history)[-5:]
        
        loss_improvement = recent_losses[0] - recent_losses[-1]
        avg_grad = np.mean(recent_grads)
        
        # 损失改善很小且梯度很小，但不是过拟合
        if loss_improvement < 0.01 and avg_grad < 0.01 and recent_losses[-1] > 0.5:
            return True
            
        return False
    
    def get_priority(self) -> float:
        return 0.85

class CognitiveScienceTrigger(MorphogenesisTrigger):
    """基于认知科学的触发器"""
    
    def __init__(self):
        self.learning_curve = deque(maxlen=50)
        self.forgetting_events = []
        
    def should_trigger(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        train_acc = context.get('train_accuracy', 0.0)
        val_acc = context.get('val_accuracy', 0.0)
        epoch = context.get('epoch', 0)
        
        self.learning_curve.append({
            'epoch': epoch,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'generalization_gap': train_acc - val_acc
        })
        
        # 检测学习高原期
        if self._detect_learning_plateau():
            return True, "学习高原期检测：需要增加认知复杂性"
            
        # 检测遗忘现象
        if self._detect_catastrophic_forgetting():
            return True, "灾难性遗忘检测：需要分化专门化神经元"
            
        # 检测认知负荷过载
        if self._detect_cognitive_overload():
            return True, "认知负荷过载：建议分解任务复杂性"
            
        return False, "认知科学指标正常"
    
    def _detect_learning_plateau(self) -> bool:
        """检测学习高原期"""
        if len(self.learning_curve) < 20:
            return False
            
        recent_curves = list(self.learning_curve)[-20:]
        train_accs = [c['train_acc'] for c in recent_curves]
        val_accs = [c['val_acc'] for c in recent_curves]
        
        # 训练和验证准确率都停滞
        train_improvement = max(train_accs) - min(train_accs)
        val_improvement = max(val_accs) - min(val_accs)
        
        return train_improvement < 0.02 and val_improvement < 0.02
    
    def _detect_catastrophic_forgetting(self) -> bool:
        """检测灾难性遗忘"""
        if len(self.learning_curve) < 10:
            return False
            
        recent_curves = list(self.learning_curve)[-10:]
        
        # 检测验证准确率大幅下降
        for i in range(1, len(recent_curves)):
            val_drop = recent_curves[i-1]['val_acc'] - recent_curves[i]['val_acc']
            if val_drop > 0.05:  # 准确率下降超过5%
                self.forgetting_events.append(recent_curves[i]['epoch'])
                return True
                
        return False
    
    def _detect_cognitive_overload(self) -> bool:
        """检测认知负荷过载"""
        if len(self.learning_curve) < 15:
            return False
            
        recent_curves = list(self.learning_curve)[-15:]
        gaps = [c['generalization_gap'] for c in recent_curves]
        
        # 泛化差距持续增大
        gap_trend = np.polyfit(range(len(gaps)), gaps, 1)[0]
        avg_gap = np.mean(gaps)
        
        return gap_trend > 0.002 and avg_gap > 0.15
    
    def get_priority(self) -> float:
        return 0.75

class NetworkScienceTrigger(MorphogenesisTrigger):
    """基于网络科学的触发器"""
    
    def __init__(self):
        self.connectivity_history = deque(maxlen=10)
        
    def should_trigger(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        model = context.get('model')
        activations = context.get('activations', {})
        
        if model is None:
            return False, "缺少模型信息"
            
        # 分析网络拓扑特性
        connectivity_metrics = self._analyze_network_topology(model, activations)
        self.connectivity_history.append(connectivity_metrics)
        
        # 检测网络瓶颈
        if self._detect_network_bottleneck(connectivity_metrics):
            return True, f"网络瓶颈检测：中心性过高={connectivity_metrics.get('centrality', 0):.3f}"
            
        # 检测连接不平衡
        if self._detect_connectivity_imbalance(connectivity_metrics):
            return True, "连接不平衡检测：需要重新分布网络连接"
            
        return False, "网络科学指标正常"
    
    def _analyze_network_topology(self, model: nn.Module, activations: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """分析网络拓扑特性"""
        metrics = {}
        
        # 计算层间连接强度
        layer_connections = self._compute_layer_connections(model)
        metrics['avg_connection_strength'] = np.mean(list(layer_connections.values()))
        
        # 计算网络中心性
        centrality = self._compute_network_centrality(activations)
        metrics['centrality'] = centrality
        
        # 计算聚类系数
        clustering = self._compute_clustering_coefficient(activations)
        metrics['clustering'] = clustering
        
        return metrics
    
    def _compute_layer_connections(self, model: nn.Module) -> Dict[str, float]:
        """计算层间连接强度"""
        connections = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    weight_norm = torch.norm(module.weight).item()
                    connections[name] = weight_norm
                    
        return connections
    
    def _compute_network_centrality(self, activations: Dict[str, torch.Tensor]) -> float:
        """计算网络中心性"""
        if len(activations) < 2:
            return 0.0
            
        # 简化的中心性计算
        activation_norms = {}
        for name, activation in activations.items():
            activation_norms[name] = torch.norm(activation).item()
            
        norm_values = list(activation_norms.values())
        if not norm_values:
            return 0.0
            
        # 计算标准化的中心性
        max_norm = max(norm_values)
        avg_norm = np.mean(norm_values)
        
        return max_norm / (avg_norm + 1e-8)
    
    def _compute_clustering_coefficient(self, activations: Dict[str, torch.Tensor]) -> float:
        """计算聚类系数"""
        if len(activations) < 3:
            return 0.0
            
        # 简化的聚类系数计算
        layer_names = list(activations.keys())
        correlations = []
        
        for i in range(len(layer_names)):
            for j in range(i+1, len(layer_names)):
                act1 = activations[layer_names[i]].flatten()
                act2 = activations[layer_names[j]].flatten()
                
                if len(act1) == len(act2) and len(act1) > 1:
                    corr = torch.corrcoef(torch.stack([act1, act2]))[0, 1]
                    correlations.append(abs(corr.item()))
                    
        return np.mean(correlations) if correlations else 0.0
    
    def _detect_network_bottleneck(self, metrics: Dict[str, float]) -> bool:
        """检测网络瓶颈"""
        centrality = metrics.get('centrality', 0)
        return centrality > 3.0  # 中心性过高表明存在瓶颈
    
    def _detect_connectivity_imbalance(self, metrics: Dict[str, float]) -> bool:
        """检测连接不平衡"""
        if len(self.connectivity_history) < 5:
            return False
            
        recent_metrics = list(self.connectivity_history)[-5:]
        connection_strengths = [m.get('avg_connection_strength', 0) for m in recent_metrics]
        
        # 连接强度方差过大
        strength_std = np.std(connection_strengths)
        return strength_std > 0.5
    
    def get_priority(self) -> float:
        return 0.7

class NeuronDivisionExecutor:
    """神经元分裂执行器"""
    
    def __init__(self):
        self.division_history = []
        
    def execute_division(self, model: nn.Module, layer_name: str, division_type: str = 'width_expansion') -> Tuple[nn.Module, int]:
        """执行神经元分裂"""
        try:
            if division_type == 'width_expansion':
                return self._expand_layer_width(model, layer_name)
            elif division_type == 'depth_expansion':
                return self._expand_network_depth(model, layer_name)
            elif division_type == 'branch_creation':
                return self._create_branch(model, layer_name)
            else:
                logger.warning(f"未知的分裂类型: {division_type}")
                return model, 0
                
        except Exception as e:
            logger.error(f"神经元分裂执行失败: {e}")
            return model, 0
    
    def _expand_layer_width(self, model: nn.Module, layer_name: str) -> Tuple[nn.Module, int]:
        """扩展层宽度（增加神经元数量）"""
        new_model = copy.deepcopy(model)
        parameters_added = 0
        
        # 找到目标层
        target_module = None
        for name, module in new_model.named_modules():
            if name == layer_name:
                target_module = module
                break
                
        if target_module is None:
            logger.warning(f"未找到目标层: {layer_name}")
            return model, 0
            
        if isinstance(target_module, nn.Linear):
            # 扩展全连接层
            old_out_features = target_module.out_features
            new_out_features = int(old_out_features * 1.2)  # 增加20%
            expansion_size = new_out_features - old_out_features
            
            # 创建新的权重和偏置
            new_weight = torch.zeros(new_out_features, target_module.in_features)
            new_bias = torch.zeros(new_out_features) if target_module.bias is not None else None
            
            # 复制原有权重
            new_weight[:old_out_features] = target_module.weight.data
            if new_bias is not None:
                new_bias[:old_out_features] = target_module.bias.data
                
            # 初始化新增的神经元
            with torch.no_grad():
                # 使用小的随机值初始化新神经元
                nn.init.normal_(new_weight[old_out_features:], mean=0, std=0.01)
                if new_bias is not None:
                    nn.init.zeros_(new_bias[old_out_features:])
                    
            # 更新模块
            target_module.out_features = new_out_features
            target_module.weight = nn.Parameter(new_weight)
            if target_module.bias is not None:
                target_module.bias = nn.Parameter(new_bias)
                
            parameters_added = expansion_size * (target_module.in_features + 1)
            
        elif isinstance(target_module, nn.Conv2d):
            # 扩展卷积层
            old_out_channels = target_module.out_channels
            new_out_channels = int(old_out_channels * 1.15)  # 增加15%
            expansion_size = new_out_channels - old_out_channels
            
            # 创建新的卷积层
            new_conv = nn.Conv2d(
                target_module.in_channels,
                new_out_channels,
                target_module.kernel_size,
                target_module.stride,
                target_module.padding,
                target_module.dilation,
                target_module.groups,
                target_module.bias is not None
            )
            
            # 复制原有权重
            with torch.no_grad():
                new_conv.weight.data[:old_out_channels] = target_module.weight.data
                if target_module.bias is not None:
                    new_conv.bias.data[:old_out_channels] = target_module.bias.data
                    
                # 初始化新增的通道
                nn.init.kaiming_normal_(new_conv.weight.data[old_out_channels:])
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias.data[old_out_channels:])
                    
            # 替换模块
            parent_name = '.'.join(layer_name.split('.')[:-1])
            child_name = layer_name.split('.')[-1]
            
            if parent_name:
                parent_module = new_model
                for part in parent_name.split('.'):
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, child_name, new_conv)
            else:
                setattr(new_model, child_name, new_conv)
                
            parameters_added = expansion_size * target_module.in_channels * \
                             target_module.kernel_size[0] * target_module.kernel_size[1]
            
        self.division_history.append({
            'layer': layer_name,
            'type': 'width_expansion',
            'parameters_added': parameters_added
        })
        
        logger.info(f"执行宽度扩展: {layer_name}, 新增参数: {parameters_added}")
        return new_model, parameters_added
    
    def _expand_network_depth(self, model: nn.Module, layer_name: str) -> Tuple[nn.Module, int]:
        """扩展网络深度（添加新层）"""
        # 深度扩展的实现较为复杂，这里提供基础框架
        logger.info(f"深度扩展功能待实现: {layer_name}")
        return model, 0
    
    def _create_branch(self, model: nn.Module, layer_name: str) -> Tuple[nn.Module, int]:
        """创建分支结构"""
        # 分支创建的实现较为复杂，这里提供基础框架
        logger.info(f"分支创建功能待实现: {layer_name}")
        return model, 0

class DNMFramework:
    """Dynamic Neural Morphogenesis Framework - 主框架"""
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = config or {}
        
        # 初始化触发器
        self.triggers = [
            InformationTheoryTrigger(),
            BiologicalPrinciplesTrigger(), 
            DynamicalSystemsTrigger(),
            CognitiveScienceTrigger(),
            NetworkScienceTrigger()
        ]
        
        # 执行器
        self.executor = NeuronDivisionExecutor()
        
        # 状态追踪
        self.morphogenesis_events = []
        self.performance_history = deque(maxlen=100)
        self.activation_cache = {}
        self.gradient_cache = {}
        
        # 配置参数
        self.morphogenesis_interval = self.config.get('morphogenesis_interval', 4)
        self.max_morphogenesis_per_epoch = self.config.get('max_morphogenesis_per_epoch', 2)
        self.performance_improvement_threshold = self.config.get('performance_improvement_threshold', 0.02)
        
    def should_trigger_morphogenesis(self, epoch: int, train_metrics: Dict[str, float], 
                                   val_metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """判断是否应该触发形态发生"""
        
        # 检查触发间隔
        if epoch % self.morphogenesis_interval != 0:
            return False, []
            
        # 准备上下文信息
        context = {
            'epoch': epoch,
            'train_accuracy': train_metrics.get('accuracy', 0.0),
            'val_accuracy': val_metrics.get('accuracy', 0.0),
            'current_loss': train_metrics.get('loss', float('inf')),
            'current_performance': val_metrics.get('accuracy', 0.0),
            'learning_rate': train_metrics.get('learning_rate', 1e-3),
            'model': self.model,
            'activations': self.activation_cache,
            'gradients': self.gradient_cache
        }
        
        # 检查所有触发器
        trigger_results = []
        triggered_reasons = []
        
        for trigger in self.triggers:
            should_trigger, reason = trigger.should_trigger(context)
            if should_trigger:
                trigger_results.append((trigger, reason))
                triggered_reasons.append(f"{trigger.__class__.__name__}: {reason}")
                
        # 根据优先级排序
        trigger_results.sort(key=lambda x: x[0].get_priority(), reverse=True)
        
        # 至少有一个高优先级触发器激活
        if trigger_results and trigger_results[0][0].get_priority() >= 0.8:
            return True, triggered_reasons
            
        # 或者有多个中等优先级触发器激活
        if len(trigger_results) >= 2 and all(t[0].get_priority() >= 0.7 for t in trigger_results[:2]):
            return True, triggered_reasons
            
        return False, []
    
    def execute_morphogenesis(self, epoch: int) -> Dict[str, Any]:
        """执行形态发生"""
        logger.info(f"🔄 Triggering morphogenesis analysis...")
        
        results = {
            'neuron_divisions': 0,
            'connection_growths': 0,
            'optimizations': 0,
            'parameters_added': 0,
            'events': []
        }
        
        # 分析最佳分裂位置
        best_layers = self._identify_optimal_division_layers()
        
        divisions_executed = 0
        for layer_name, score in best_layers[:self.max_morphogenesis_per_epoch]:
            # 执行神经元分裂
            new_model, params_added = self.executor.execute_division(
                self.model, layer_name, 'width_expansion'
            )
            
            if params_added > 0:
                self.model = new_model
                divisions_executed += 1
                results['parameters_added'] += params_added
                
                # 记录事件
                event = MorphogenesisEvent(
                    epoch=epoch,
                    event_type='neuron_division',
                    location=layer_name,
                    trigger_reason=f"优化分数: {score:.4f}",
                    performance_before=self.performance_history[-1] if self.performance_history else 0.0,
                    parameters_added=params_added
                )
                
                self.morphogenesis_events.append(event)
                results['events'].append(event)
                
        results['neuron_divisions'] = divisions_executed
        
        logger.info(f"DNM Neuron Division completed: {divisions_executed} splits executed")
        
        return results
    
    def _identify_optimal_division_layers(self) -> List[Tuple[str, float]]:
        """识别最佳分裂层"""
        layer_scores = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                score = self._compute_division_score(name, module)
                layer_scores.append((name, score))
                
        # 按分数排序
        layer_scores.sort(key=lambda x: x[1], reverse=True)
        return layer_scores
    
    def _compute_division_score(self, layer_name: str, module: nn.Module) -> float:
        """计算层的分裂分数"""
        score = 0.0
        
        # 基于权重分析
        if hasattr(module, 'weight') and module.weight is not None:
            weight = module.weight.data
            
            # 权重方差（高方差表明神经元分化程度高）
            weight_var = torch.var(weight).item()
            score += weight_var * 0.3
            
            # 权重范数（适中的范数最佳）
            weight_norm = torch.norm(weight).item()
            normalized_norm = weight_norm / weight.numel()
            score += (1.0 - abs(normalized_norm - 0.1)) * 0.2
            
        # 基于激活值分析
        if layer_name in self.activation_cache:
            activation = self.activation_cache[layer_name]
            
            # 激活值多样性
            act_std = torch.std(activation).item()
            score += act_std * 0.3
            
            # 激活值饱和度
            saturation = torch.mean((activation > 0.9).float()).item()
            score += (1.0 - saturation) * 0.2
            
        return score
    
    def update_caches(self, activations: Dict[str, torch.Tensor], 
                      gradients: Dict[str, torch.Tensor]):
        """更新激活值和梯度缓存"""
        self.activation_cache = {k: v.detach().clone() for k, v in activations.items()}
        self.gradient_cache = {k: v.detach().clone() if v is not None else None 
                              for k, v in gradients.items()}
    
    def record_performance(self, performance: float):
        """记录性能"""
        self.performance_history.append(performance)
    
    def get_morphogenesis_summary(self) -> Dict[str, Any]:
        """获取形态发生摘要"""
        if not self.morphogenesis_events:
            return {
                'total_events': 0,
                'total_neuron_divisions': 0,
                'total_parameters_added': 0,
                'performance_improvement': 0.0
            }
            
        total_events = len(self.morphogenesis_events)
        neuron_divisions = sum(1 for e in self.morphogenesis_events 
                              if e.event_type == 'neuron_division')
        total_params = sum(e.parameters_added for e in self.morphogenesis_events)
        
        # 计算性能改善
        if len(self.performance_history) >= 2:
            initial_perf = self.performance_history[0]
            final_perf = self.performance_history[-1]
            performance_improvement = final_perf - initial_perf
        else:
            performance_improvement = 0.0
            
        return {
            'total_events': total_events,
            'total_neuron_divisions': neuron_divisions,
            'total_parameters_added': total_params,
            'performance_improvement': performance_improvement,
            'events_detail': [
                {
                    'epoch': e.epoch,
                    'type': e.event_type,
                    'location': e.location,
                    'params_added': e.parameters_added,
                    'reason': e.trigger_reason
                }
                for e in self.morphogenesis_events
            ]
        }