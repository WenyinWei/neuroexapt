"""
信息流分析器

专门负责神经网络中信息流模式的分析
"""

from typing import Dict, Any
import torch
import logging

logger = logging.getLogger(__name__)


class InformationFlowAnalyzer:
    """信息流分析器"""
    
    def __init__(self):
        self.flow_patterns = {}
        
    def analyze_information_flow(self, model: torch.nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析模型的信息流模式
        
        Args:
            model: 要分析的模型
            context: 分析上下文，包含激活值等信息
            
        Returns:
            信息流分析结果
        """
        logger.debug("开始信息流分析")
        
        try:
            # 如果有激活值直接分析
            if 'activations' in context:
                return self.analyze_flow_patterns(context['activations'])
            
            # 否则基于模型结构进行分析
            return self._analyze_model_structure_flow(model, context)
            
        except Exception as e:
            logger.error(f"信息流分析失败: {e}")
            return {'layer_flow_metrics': {}, 'global_bottleneck_score': 0.5}
    
    def _analyze_model_structure_flow(self, model: torch.nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        """基于模型结构分析信息流"""
        flow_metrics = {}
        
        # 分析每一层的信息传递能力
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
                # 计算信息容量
                info_capacity = self._estimate_layer_information_capacity(module)
                
                # 计算传递效率（基于参数配置）
                transfer_efficiency = self._estimate_transfer_efficiency(module)
                
                # 计算瓶颈分数
                bottleneck_score = self._calculate_bottleneck_score(info_capacity, transfer_efficiency)
                
                flow_metrics[name] = {
                    'information_density': info_capacity,
                    'transfer_efficiency': transfer_efficiency,
                    'bottleneck_score': bottleneck_score
                }
        
        return {
            'layer_flow_metrics': flow_metrics,
            'global_bottleneck_score': self._calculate_global_bottleneck_score(flow_metrics)
        }
    
    def _estimate_layer_information_capacity(self, module: torch.nn.Module) -> float:
        """估计层的信息容量"""
        if isinstance(module, torch.nn.Conv2d):
            # 卷积层的信息容量与输出通道数和核大小相关
            capacity = module.out_channels * module.kernel_size[0] * module.kernel_size[1]
            return min(1.0, capacity / 10000.0)  # 标准化
        elif isinstance(module, torch.nn.Linear):
            # 线性层的信息容量与输出特征数相关
            capacity = module.out_features
            return min(1.0, capacity / 5000.0)  # 标准化
        else:
            return 0.5  # 默认值
    
    def _estimate_transfer_efficiency(self, module: torch.nn.Module) -> float:
        """估计传递效率"""
        if isinstance(module, torch.nn.Conv2d):
            # 基于步长、填充等参数估计效率
            stride_penalty = 1.0 / (module.stride[0] * module.stride[1])
            return min(1.0, stride_penalty)
        elif isinstance(module, torch.nn.Linear):
            # 线性层通常有较高的传递效率
            return 0.8
        else:
            return 0.6  # 默认值
        
    def analyze_flow_patterns(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """分析信息流模式"""
        logger.debug("信息流模式分析")
        
        try:
            flow_metrics = {}
            layer_names = list(activations.keys())
            
            for i, layer_name in enumerate(layer_names):
                activation = activations[layer_name]
                
                # 计算信息密度
                info_density = self._calculate_information_density(activation)
                
                # 计算传递效率
                transfer_efficiency = 0.0
                if i < len(layer_names) - 1:
                    next_layer = layer_names[i + 1]
                    if next_layer in activations:
                        transfer_efficiency = self._calculate_transfer_efficiency(
                            activation, activations[next_layer]
                        )
                
                flow_metrics[layer_name] = {
                    'information_density': info_density,
                    'transfer_efficiency': transfer_efficiency,
                    'bottleneck_score': self._calculate_bottleneck_score(
                        info_density, transfer_efficiency
                    )
                }
            
            return {
                'layer_flow_metrics': flow_metrics,
                'global_bottleneck_score': self._calculate_global_bottleneck_score(flow_metrics)
            }
            
        except Exception as e:
            logger.error(f"信息流分析失败: {e}")
            return {'layer_flow_metrics': {}, 'global_bottleneck_score': 0.5}

    def _calculate_information_density(self, activation: torch.Tensor) -> float:
        """计算信息密度"""
        try:
            # 使用激活值的熵作为信息密度指标
            flat_activation = activation.flatten()
            
            # 计算近似熵
            hist = torch.histc(flat_activation, bins=50, min=float(flat_activation.min()), max=float(flat_activation.max()))
            hist = hist / hist.sum()  # 归一化
            
            # 避免log(0)
            hist = hist + 1e-10
            entropy = -torch.sum(hist * torch.log(hist))
            
            # 归一化到[0, 1]
            return float(torch.clamp(entropy / 10.0, 0, 1))
            
        except Exception:
            return 0.5  # 默认值

    def _calculate_transfer_efficiency(self, current_activation: torch.Tensor, 
                                     next_activation: torch.Tensor) -> float:
        """计算信息传递效率"""
        try:
            # 简化的相关性计算
            curr_flat = current_activation.flatten()[:1000]  # 限制大小
            next_flat = next_activation.flatten()[:1000]
            
            min_size = min(len(curr_flat), len(next_flat))
            curr_flat = curr_flat[:min_size]
            next_flat = next_flat[:min_size]
            
            # 计算相关系数
            correlation = torch.corrcoef(torch.stack([curr_flat, next_flat]))[0, 1]
            
            if torch.isnan(correlation):
                return 0.0
                
            return float(torch.abs(correlation))
            
        except Exception:
            return 0.5  # 默认值

    def _calculate_bottleneck_score(self, info_density: float, transfer_efficiency: float) -> float:
        """计算瓶颈分数"""
        # 瓶颈分数 = 信息密度低 + 传递效率低
        bottleneck_score = (1.0 - info_density) * 0.6 + (1.0 - transfer_efficiency) * 0.4
        return float(bottleneck_score)

    def _calculate_global_bottleneck_score(self, flow_metrics: Dict[str, Dict[str, float]]) -> float:
        """计算全局瓶颈分数"""
        if not flow_metrics:
            return 0.5
            
        bottleneck_scores = [metrics['bottleneck_score'] for metrics in flow_metrics.values()]
        return float(sum(bottleneck_scores) / len(bottleneck_scores))