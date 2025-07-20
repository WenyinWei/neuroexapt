"""
贝叶斯形态发生特征提取器

从模型和上下文中提取用于贝叶斯分析的特征
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ArchitectureFeatureExtractor:
    """架构特征提取器"""
    
    def __init__(self):
        pass
    
    def extract_features(self, model: torch.nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        """提取架构特征"""
        
        # 1. 基础架构信息
        architecture_info = self._extract_architecture_info(model)
        
        # 2. 激活统计特征
        activation_features = self._extract_activation_features(context.get('activations', {}))
        
        # 3. 梯度统计特征
        gradient_features = self._extract_gradient_features(context.get('gradients', {}))
        
        # 4. 性能历史特征
        performance_features = self._extract_performance_features(context.get('performance_history', []))
        
        # 5. 层级关系特征
        layer_relationship_features = self._extract_layer_relationship_features(model)
        
        return {
            'architecture_info': architecture_info,
            'activation_features': activation_features,
            'gradient_features': gradient_features,
            'performance_features': performance_features,
            'layer_relationship_features': layer_relationship_features,
            'feature_summary': self._build_feature_summary(
                architecture_info, activation_features, gradient_features, performance_features
            )
        }
    
    def _extract_architecture_info(self, model: torch.nn.Module) -> Dict[str, Any]:
        """提取基础架构信息"""
        
        layer_info = []
        total_params = 0
        layer_types = {}
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                param_count = sum(p.numel() for p in module.parameters())
                total_params += param_count
                
                module_type = type(module).__name__
                layer_types[module_type] = layer_types.get(module_type, 0) + 1
                
                layer_info.append({
                    'name': name,
                    'type': module_type,
                    'param_count': param_count,
                    'depth': len(name.split('.'))
                })
        
        return {
            'total_layers': len(layer_info),
            'total_params': total_params,
            'layer_types': layer_types,
            'layer_info': layer_info,
            'max_depth': max(info['depth'] for info in layer_info) if layer_info else 0,
            'avg_params_per_layer': total_params / len(layer_info) if layer_info else 0
        }
    
    def _extract_activation_features(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """提取激活特征"""
        
        if not activations:
            return {'available': False}
        
        features = {'available': True, 'layer_features': {}, 'global_features': {}}
        
        all_means = []
        all_stds = []
        all_zeros_ratios = []
        
        for layer_name, activation in activations.items():
            if isinstance(activation, torch.Tensor):
                # 转换为numpy进行计算
                act_data = activation.detach().cpu().numpy()
                
                # 基础统计
                mean_val = np.mean(act_data)
                std_val = np.std(act_data)
                zeros_ratio = np.mean(act_data == 0)
                
                # 分布特征
                percentiles = np.percentile(act_data.flatten(), [25, 50, 75, 95])
                
                layer_features = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'zeros_ratio': float(zeros_ratio),
                    'percentiles': percentiles.tolist(),
                    'shape': list(act_data.shape)
                }
                
                features['layer_features'][layer_name] = layer_features
                
                all_means.append(mean_val)
                all_stds.append(std_val)
                all_zeros_ratios.append(zeros_ratio)
        
        # 全局特征
        if all_means:
            features['global_features'] = {
                'avg_activation': float(np.mean(all_means)),
                'activation_diversity': float(np.std(all_means)),
                'avg_activation_std': float(np.mean(all_stds)),
                'avg_sparsity': float(np.mean(all_zeros_ratios)),
                'activation_layers': len(all_means)
            }
        
        return features
    
    def _extract_gradient_features(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """提取梯度特征"""
        
        if not gradients:
            return {'available': False}
        
        features = {'available': True, 'layer_features': {}, 'global_features': {}}
        
        all_norms = []
        all_means = []
        all_stds = []
        
        for layer_name, gradient in gradients.items():
            if isinstance(gradient, torch.Tensor):
                # 转换为numpy进行计算
                grad_data = gradient.detach().cpu().numpy()
                
                # 基础统计
                grad_norm = np.linalg.norm(grad_data)
                mean_val = np.mean(grad_data)
                std_val = np.std(grad_data)
                
                # 梯度分布
                abs_grad = np.abs(grad_data)
                percentiles = np.percentile(abs_grad.flatten(), [50, 90, 95, 99])
                
                layer_features = {
                    'norm': float(grad_norm),
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'abs_percentiles': percentiles.tolist(),
                    'shape': list(grad_data.shape)
                }
                
                features['layer_features'][layer_name] = layer_features
                
                all_norms.append(grad_norm)
                all_means.append(abs(mean_val))
                all_stds.append(std_val)
        
        # 全局梯度特征
        if all_norms:
            features['global_features'] = {
                'total_grad_norm': float(np.sqrt(np.sum(np.array(all_norms) ** 2))),
                'avg_grad_norm': float(np.mean(all_norms)),
                'grad_norm_std': float(np.std(all_norms)),
                'avg_grad_magnitude': float(np.mean(all_means)),
                'gradient_layers': len(all_norms)
            }
        
        return features
    
    def _extract_performance_features(self, performance_history: List[float]) -> Dict[str, Any]:
        """提取性能历史特征"""
        
        if not performance_history:
            return {'available': False}
        
        history = np.array(performance_history)
        
        # 基础统计
        features = {
            'available': True,
            'length': len(history),
            'current': float(history[-1]),
            'best': float(np.max(history)),
            'worst': float(np.min(history)),
            'mean': float(np.mean(history)),
            'std': float(np.std(history))
        }
        
        # 趋势分析
        if len(history) >= 3:
            # 短期趋势（最近3个点）
            recent = history[-3:]
            short_trend = np.polyfit(range(len(recent)), recent, 1)[0]
            
            # 长期趋势（全部数据）
            long_trend = np.polyfit(range(len(history)), history, 1)[0]
            
            features.update({
                'short_term_trend': float(short_trend),
                'long_term_trend': float(long_trend),
                'trend_acceleration': float(short_trend - long_trend)
            })
        
        # 改进分析
        if len(history) >= 2:
            improvements = np.diff(history)
            features.update({
                'total_improvement': float(history[-1] - history[0]),
                'avg_improvement': float(np.mean(improvements)),
                'improvement_volatility': float(np.std(improvements)),
                'positive_improvements': int(np.sum(improvements > 0)),
                'improvement_ratio': float(np.mean(improvements > 0))
            })
        
        return features
    
    def _extract_layer_relationship_features(self, model: torch.nn.Module) -> Dict[str, Any]:
        """提取层级关系特征"""
        
        relationships = {
            'sequential_layers': 0,
            'parallel_branches': 0,
            'skip_connections': 0,
            'layer_groups': {},
            'connection_complexity': 0
        }
        
        # 分析模块结构
        for name, module in model.named_modules():
            module_type = type(module).__name__
            
            # 计算连接复杂度
            if hasattr(module, 'weight') and hasattr(module, 'bias'):
                relationships['sequential_layers'] += 1
            
            # 检测跳跃连接和并行结构（简化版）
            if 'shortcut' in name.lower() or 'skip' in name.lower():
                relationships['skip_connections'] += 1
            
            # 层组分析
            parts = name.split('.')
            if len(parts) >= 2:
                group_name = parts[0]
                relationships['layer_groups'][group_name] = relationships['layer_groups'].get(group_name, 0) + 1
        
        # 计算连接复杂度
        relationships['connection_complexity'] = (
            relationships['sequential_layers'] + 
            relationships['skip_connections'] * 2 +  # 跳跃连接复杂度更高
            len(relationships['layer_groups'])
        )
        
        return relationships
    
    def _build_feature_summary(self, 
                             architecture_info: Dict[str, Any], 
                             activation_features: Dict[str, Any], 
                             gradient_features: Dict[str, Any], 
                             performance_features: Dict[str, Any]) -> Dict[str, Any]:
        """构建特征摘要"""
        
        summary = {
            'total_layers': architecture_info.get('total_layers', 0),
            'total_params': architecture_info.get('total_params', 0),
            'has_activations': activation_features.get('available', False),
            'has_gradients': gradient_features.get('available', False),
            'has_performance_history': performance_features.get('available', False)
        }
        
        # 添加关键指标
        if activation_features.get('available'):
            global_act = activation_features.get('global_features', {})
            summary.update({
                'avg_activation': global_act.get('avg_activation', 0),
                'activation_sparsity': global_act.get('avg_sparsity', 0)
            })
        
        if gradient_features.get('available'):
            global_grad = gradient_features.get('global_features', {})
            summary.update({
                'total_grad_norm': global_grad.get('total_grad_norm', 0),
                'avg_grad_norm': global_grad.get('avg_grad_norm', 0)
            })
        
        if performance_features.get('available'):
            summary.update({
                'current_performance': performance_features.get('current', 0),
                'performance_trend': performance_features.get('short_term_trend', 0)
            })
        
        return summary