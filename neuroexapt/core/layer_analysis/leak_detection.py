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
    
    def detect_information_leaks(self, model: torch.nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        检测模型中的信息泄漏点
        
        Args:
            model: 要分析的模型
            context: 分析上下文，包含激活值、梯度等信息
            
        Returns:
            信息泄漏检测结果
        """
        logger.debug("开始信息泄漏检测")
        
        try:
            # 如果有激活值和梯度，直接使用
            if 'activations' in context and 'gradients' in context:
                leak_points = self.detect_leaks(
                    context['activations'], 
                    context['gradients'],
                    context.get('targets', torch.tensor([]))
                )
            else:
                # 基于模型结构进行泄漏风险评估
                leak_points = self._assess_structural_leak_risks(model, context)
            
            # 处理检测结果
            result = self._process_leak_analysis(leak_points, model, context)
            
            logger.debug(f"检测到 {len(leak_points)} 个潜在泄漏点")
            return result
            
        except Exception as e:
            logger.error(f"信息泄漏检测失败: {e}")
            return self._fallback_leak_analysis()
    
    def _assess_structural_leak_risks(self, model: torch.nn.Module, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于模型结构评估泄漏风险"""
        leak_risks = []
        
        # 分析每一层的潜在泄漏风险
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
                risk_score = self._calculate_structural_leak_risk(name, module, context)
                
                if risk_score > 0.5:  # 风险阈值
                    leak_risks.append({
                        'layer_name': name,
                        'severity': risk_score,
                        'leak_type': self._classify_structural_leak_type(module, risk_score),
                        'details': {
                            'structural_risk': risk_score,
                            'layer_type': type(module).__name__,
                            'parameter_count': sum(p.numel() for p in module.parameters())
                        }
                    })
        
        # 按风险程度排序
        leak_risks.sort(key=lambda x: x['severity'], reverse=True)
        return leak_risks
    
    def _calculate_structural_leak_risk(self, layer_name: str, module: torch.nn.Module, context: Dict[str, Any]) -> float:
        """计算结构性泄漏风险"""
        risk_factors = []
        
        # 基于层类型的风险评估
        if isinstance(module, torch.nn.Conv2d):
            # 卷积层风险：通道数过少可能导致信息压缩
            channel_risk = max(0.0, (64 - module.out_channels) / 64.0)
            risk_factors.append(channel_risk)
            
            # 核大小风险：核太大可能导致信息过度聚合
            kernel_size_risk = max(0.0, (module.kernel_size[0] - 3) / 7.0)
            risk_factors.append(kernel_size_risk)
            
        elif isinstance(module, torch.nn.Linear):
            # 线性层风险：输出特征过少可能导致信息丢失
            feature_risk = max(0.0, (512 - module.out_features) / 512.0)
            risk_factors.append(feature_risk)
            
        elif isinstance(module, torch.nn.BatchNorm2d):
            # BatchNorm风险相对较低
            risk_factors.append(0.2)
        
        # 参数数量风险：参数过少可能无法充分表达信息
        param_count = sum(p.numel() for p in module.parameters())
        param_risk = max(0.0, (10000 - param_count) / 10000.0)
        risk_factors.append(param_risk)
        
        # 综合风险评分
        if risk_factors:
            return float(np.mean(risk_factors))
        else:
            return 0.3  # 默认中等风险
    
    def _classify_structural_leak_type(self, module: torch.nn.Module, risk_score: float) -> str:
        """基于结构分类泄漏类型"""
        if isinstance(module, torch.nn.Conv2d):
            if module.out_channels < 32:
                return "information_compression_bottleneck"
            else:
                return "representational_bottleneck"
        elif isinstance(module, torch.nn.Linear):
            if module.out_features < 128:
                return "information_compression_bottleneck"
            else:
                return "gradient_learning_bottleneck"
        else:
            return "general_bottleneck"
    
    def _process_leak_analysis(self, leak_points: List[Dict[str, Any]], 
                             model: torch.nn.Module, 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """处理泄漏分析结果"""
        
        # 统计不同类型的泄漏
        leak_types = {}
        for leak in leak_points:
            leak_type = leak['leak_type']
            if leak_type not in leak_types:
                leak_types[leak_type] = []
            leak_types[leak_type].append(leak)
        
        # 计算总体泄漏风险
        overall_risk = 0.0
        if leak_points:
            overall_risk = np.mean([leak['severity'] for leak in leak_points])
        
        # 生成修复建议
        repair_suggestions = self._generate_repair_suggestions(leak_points)
        
        return {
            'leak_points': leak_points,
            'leak_types_distribution': {
                leak_type: len(leaks) for leak_type, leaks in leak_types.items()
            },
            'overall_leak_risk': float(overall_risk),
            'total_leak_count': len(leak_points),
            'high_risk_leaks': [leak for leak in leak_points if leak['severity'] > 0.7],
            'repair_suggestions': repair_suggestions,
            'leak_analysis_metadata': {
                'detection_method': 'activations' if 'activations' in context else 'structural',
                'model_layers_analyzed': len(list(model.named_modules())),
                'analysis_timestamp': context.get('timestamp', 'unknown')
            }
        }
    
    def _generate_repair_suggestions(self, leak_points: List[Dict[str, Any]]) -> List[str]:
        """生成泄漏修复建议"""
        suggestions = []
        
        if not leak_points:
            return ["未检测到明显的信息泄漏，模型结构良好"]
        
        # 基于泄漏类型生成建议
        leak_types = set(leak['leak_type'] for leak in leak_points)
        
        if 'information_compression_bottleneck' in leak_types:
            suggestions.append("建议增加瓶颈层的宽度以减少信息压缩")
            suggestions.append("考虑使用Net2Wider技术扩展压缩层")
        
        if 'gradient_learning_bottleneck' in leak_types:
            suggestions.append("建议添加残差连接改善梯度流")
            suggestions.append("考虑使用BatchNorm或LayerNorm提高训练稳定性")
        
        if 'representational_bottleneck' in leak_types:
            suggestions.append("建议增加层的表达能力或添加更多层")
            suggestions.append("考虑使用注意力机制增强表征能力")
        
        # 高风险泄漏的特殊建议
        high_risk_count = len([leak for leak in leak_points if leak['severity'] > 0.7])
        if high_risk_count > 0:
            suggestions.append(f"发现 {high_risk_count} 个高风险泄漏点，建议优先处理")
            suggestions.append("建议进行架构搜索或使用进化算法优化结构")
        
        return suggestions
    
    def _fallback_leak_analysis(self) -> Dict[str, Any]:
        """fallback泄漏分析"""
        return {
            'leak_points': [],
            'leak_types_distribution': {},
            'overall_leak_risk': 0.5,
            'total_leak_count': 0,
            'high_risk_leaks': [],
            'repair_suggestions': ["泄漏检测失败，建议人工检查模型结构"],
            'leak_analysis_metadata': {
                'detection_method': 'fallback',
                'analysis_status': 'failed'
            }
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