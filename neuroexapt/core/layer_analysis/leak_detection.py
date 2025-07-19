"""
信息泄漏检测器

专门负责检测神经网络中的信息泄漏点
"""

from typing import Dict, Any, List
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class InformationLeakDetector:
    """信息泄漏检测器"""
    
    def __init__(self):
        self.leak_thresholds = {
            'entropy_drop': 0.5,
            'gradient_variance': 0.1,
            'correlation_loss': 0.3
        }
    
    def detect_leaks(self, activations: Dict[str, torch.Tensor],
                    gradients: Dict[str, torch.Tensor],
                    targets: torch.Tensor) -> List[Dict[str, Any]]:
        """检测信息泄露点"""
        logger.debug("信息泄漏检测")
        
        try:
            leak_points = []
            layer_names = list(activations.keys())
            
            for i, layer_name in enumerate(layer_names[1:], 1):  # 跳过第一层
                if layer_name not in gradients:
                    continue
                    
                current_activation = activations[layer_name]
                current_gradient = gradients[layer_name]
                prev_layer = layer_names[i-1]
                
                if prev_layer not in activations:
                    continue
                    
                prev_activation = activations[prev_layer]
                
                # 检测信息密度下降
                leak_severity = self._detect_information_drop(
                    prev_activation, current_activation
                )
                
                # 检测梯度学习困难
                learning_difficulty = self._detect_learning_difficulty(current_gradient)
                
                # 检测与目标的相关性损失
                correlation_loss = self._detect_correlation_loss(current_activation, targets)
                
                # 综合评估
                overall_severity = (leak_severity + learning_difficulty + correlation_loss) / 3.0
                
                if overall_severity > 0.5:  # 阈值可调
                    leak_points.append({
                        'layer_name': layer_name,
                        'severity': overall_severity,
                        'leak_type': self._classify_leak_type(
                            leak_severity, learning_difficulty, correlation_loss
                        ),
                        'details': {
                            'information_drop': leak_severity,
                            'learning_difficulty': learning_difficulty,
                            'correlation_loss': correlation_loss
                        }
                    })
            
            # 按严重程度排序
            leak_points.sort(key=lambda x: x['severity'], reverse=True)
            
            return leak_points
            
        except Exception as e:
            logger.error(f"信息泄漏检测失败: {e}")
            return []

    def _detect_information_drop(self, prev_activation: torch.Tensor, 
                               current_activation: torch.Tensor) -> float:
        """检测信息密度下降"""
        try:
            prev_entropy = self._calculate_entropy(prev_activation)
            current_entropy = self._calculate_entropy(current_activation)
            
            info_drop = max(0.0, prev_entropy - current_entropy)
            
            # 归一化
            return min(info_drop / 5.0, 1.0)
            
        except Exception:
            return 0.0

    def _detect_learning_difficulty(self, gradient: torch.Tensor) -> float:
        """检测梯度学习困难"""
        try:
            gradient_variance = torch.var(gradient).item()
            
            # 方差越小，学习越困难
            difficulty = 1.0 / (1.0 + gradient_variance * 100)
            
            return float(difficulty)
            
        except Exception:
            return 0.0

    def _detect_correlation_loss(self, activation: torch.Tensor, targets: torch.Tensor) -> float:
        """检测与目标的相关性损失"""
        try:
            # 简化的相关性计算
            activation_mean = torch.mean(activation, dim=tuple(range(1, activation.dim())))
            
            if len(activation_mean) != len(targets):
                return 0.5  # 维度不匹配时返回中等损失
                
            correlation = torch.corrcoef(torch.stack([
                activation_mean.float(),
                targets.float()
            ]))[0, 1]
            
            if torch.isnan(correlation):
                return 0.5
                
            # 相关性越低，损失越大
            correlation_loss = 1.0 - torch.abs(correlation).item()
            
            return correlation_loss
            
        except Exception:
            return 0.5

    def _calculate_entropy(self, tensor: torch.Tensor) -> float:
        """计算张量的近似熵"""
        try:
            flat_tensor = tensor.flatten()
            
            # 计算直方图
            hist = torch.histc(flat_tensor, bins=50, 
                             min=float(flat_tensor.min()), 
                             max=float(flat_tensor.max()))
            hist = hist / hist.sum()
            
            # 避免log(0)
            hist = hist + 1e-10
            entropy = -torch.sum(hist * torch.log(hist))
            
            return float(entropy)
            
        except Exception:
            return 0.0

    def _classify_leak_type(self, info_drop: float, learning_difficulty: float, 
                          correlation_loss: float) -> str:
        """分类漏点类型"""
        if info_drop > 0.6:
            return "information_compression_bottleneck"
        elif learning_difficulty > 0.7:
            return "gradient_learning_bottleneck"
        elif correlation_loss > 0.6:
            return "representational_bottleneck"
        else:
            return "general_bottleneck"