#!/usr/bin/env python3
"""
@defgroup group_net2net_subnetwork_analyzer Net2Net Subnetwork Analyzer
@ingroup core
Net2Net Subnetwork Analyzer module for NeuroExapt framework.

Net2Net子网络分析器 - Net2Net Subnetwork Analyzer

🎯 核心功能：
1. 从指定层到输出层提取子网络
2. 评估子网络的变异潜力
3. 预测变异后的准确率提升空间
4. 分析可行参数空间在总参数空间中的占比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import OrderedDict, defaultdict
import copy
import logging
import time

from .logging_utils import logger

class SubnetworkExtractor:
    """子网络提取器"""
    
    def __init__(self):
        self.extracted_subnetworks = {}
        self.layer_dependencies = {}
        
    def extract_subnetwork_from_layer(self, 
                                    model: nn.Module, 
                                    start_layer_name: str,
                                    include_start_layer: bool = True) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        从指定层开始提取到输出层的子网络
        
        Args:
            model: 原始模型
            start_layer_name: 起始层名称
            include_start_layer: 是否包含起始层
            
        Returns:
            子网络模型和提取信息
        """
        logger.enter_section(f"提取子网络: {start_layer_name}")
        
        # 1. 分析模型结构，找到所有层的依赖关系
        layer_graph = self._build_layer_dependency_graph(model)
        
        # 2. 从起始层开始，找到所有后续层
        target_layers = self._find_downstream_layers(layer_graph, start_layer_name, include_start_layer)
        
        logger.info(f"识别出{len(target_layers)}个下游层")
        
        # 3. 构建子网络
        subnetwork = self._build_subnetwork(model, target_layers, start_layer_name)
        
        # 4. 分析子网络信息
        subnetwork_info = self._analyze_subnetwork(subnetwork, target_layers, start_layer_name)
        
        logger.info(f"子网络构建完成: {subnetwork_info['total_params']:,}参数")
        logger.exit_section(f"提取子网络: {start_layer_name}")
        
        return subnetwork, subnetwork_info
    
    def _build_layer_dependency_graph(self, model: nn.Module) -> Dict[str, List[str]]:
        """构建层依赖关系图"""
        layer_graph = defaultdict(list)
        named_modules = dict(model.named_modules())
        
        # 分析ResNet风格的前向传播依赖
        for name, module in named_modules.items():
            if name == '':  # 跳过根模块
                continue
                
            # 解析层的逻辑位置
            parts = name.split('.')
            
            if len(parts) >= 2:
                # 对于层级结构，添加依赖关系
                parent_parts = parts[:-1]
                parent_name = '.'.join(parent_parts)
                
                if parent_name in named_modules:
                    layer_graph[parent_name].append(name)
        
        # 添加特殊的依赖关系（基于ResNet架构）
        self._add_resnet_dependencies(layer_graph, named_modules)
        
        return dict(layer_graph)
    
    def _add_resnet_dependencies(self, layer_graph: Dict[str, List[str]], named_modules: Dict[str, nn.Module]):
        """添加ResNet特定的依赖关系"""
        
        # 主干依赖：conv1 -> feature_blocks -> classifier
        main_sequence = []
        
        # 查找主要组件
        if 'conv1' in named_modules:
            main_sequence.append('conv1')
        
        # 查找feature blocks
        feature_blocks = []
        for name in named_modules.keys():
            if name.startswith('feature_block'):
                if '.' not in name[len('feature_block'):]:  # 只要顶级feature_block
                    feature_blocks.append(name)
        
        feature_blocks.sort()  # 按名称排序
        main_sequence.extend(feature_blocks)
        
        # 查找分类器
        if 'classifier' in named_modules:
            main_sequence.append('classifier')
        
        # 建立主干依赖
        for i in range(len(main_sequence) - 1):
            current = main_sequence[i]
            next_layer = main_sequence[i + 1]
            if current not in layer_graph:
                layer_graph[current] = []
            layer_graph[current].append(next_layer)
    
    def _find_downstream_layers(self, 
                               layer_graph: Dict[str, List[str]], 
                               start_layer: str, 
                               include_start: bool = True) -> Set[str]:
        """找到指定层之后的所有下游层"""
        
        downstream_layers = set()
        
        if include_start:
            downstream_layers.add(start_layer)
        
        # BFS遍历找到所有下游层
        queue = [start_layer]
        visited = set([start_layer])
        
        while queue:
            current_layer = queue.pop(0)
            
            # 添加当前层的所有下游层
            if current_layer in layer_graph:
                for next_layer in layer_graph[current_layer]:
                    if next_layer not in visited:
                        downstream_layers.add(next_layer)
                        queue.append(next_layer)
                        visited.add(next_layer)
        
        return downstream_layers
    
    def _build_subnetwork(self, 
                         model: nn.Module, 
                         target_layers: Set[str], 
                         start_layer_name: str) -> nn.Module:
        """构建包含目标层的子网络"""
        
        # 获取模型的所有命名模块
        named_modules = dict(model.named_modules())
        
        # 创建子网络模块字典
        subnetwork_modules = OrderedDict()
        
        # 添加目标层到子网络
        for layer_name in sorted(target_layers):
            if layer_name in named_modules:
                subnetwork_modules[layer_name] = copy.deepcopy(named_modules[layer_name])
        
        # 创建动态子网络类
        class DynamicSubnetwork(nn.Module):
            def __init__(self, modules_dict, start_layer):
                super().__init__()
                self.start_layer = start_layer
                self.modules_dict = nn.ModuleDict(modules_dict)
                
                # 分析输入输出维度
                self._analyze_io_dims()
            
            def _analyze_io_dims(self):
                """分析输入输出维度"""
                # 这里简化处理，实际应该根据模型结构动态分析
                self.input_dim = None
                self.output_dim = None
                
                # 查找输入和输出层
                layer_names = list(self.modules_dict.keys())
                if layer_names:
                    first_layer = self.modules_dict[layer_names[0]]
                    last_layer = self.modules_dict[layer_names[-1]]
                    
                    # 尝试获取输入维度
                    if hasattr(first_layer, 'in_features'):
                        self.input_dim = first_layer.in_features
                    elif hasattr(first_layer, 'in_channels'):
                        self.input_dim = first_layer.in_channels
                    
                    # 尝试获取输出维度
                    if hasattr(last_layer, 'out_features'):
                        self.output_dim = last_layer.out_features
                    elif hasattr(last_layer, 'out_channels'):
                        self.output_dim = last_layer.out_channels
            
            def forward(self, x):
                """简化的前向传播"""
                # 这是一个简化版本，实际需要根据具体架构实现
                current = x
                
                for name, module in self.modules_dict.items():
                    try:
                        if isinstance(module, (nn.Linear, nn.Conv2d)):
                            current = module(current)
                        elif isinstance(module, nn.Sequential):
                            current = module(current)
                        else:
                            # 对于其他类型的模块，尝试直接调用
                            current = module(current)
                    except Exception as e:
                        logger.warning(f"子网络前向传播在层{name}失败: {e}")
                        break
                
                return current
        
        subnetwork = DynamicSubnetwork(subnetwork_modules, start_layer_name)
        return subnetwork
    
    def _analyze_subnetwork(self, 
                           subnetwork: nn.Module, 
                           target_layers: Set[str], 
                           start_layer_name: str) -> Dict[str, Any]:
        """分析子网络的特性"""
        
        total_params = sum(p.numel() for p in subnetwork.parameters())
        trainable_params = sum(p.numel() for p in subnetwork.parameters() if p.requires_grad)
        
        # 分析层类型分布
        layer_types = defaultdict(int)
        for name, module in subnetwork.named_modules():
            if name:  # 跳过根模块
                layer_types[type(module).__name__] += 1
        
        return {
            'start_layer': start_layer_name,
            'layer_count': len(target_layers),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'layer_types': dict(layer_types),
            'input_dim': getattr(subnetwork, 'input_dim', None),
            'output_dim': getattr(subnetwork, 'output_dim', None)
        }

class ParameterSpaceAnalyzer:
    """参数空间分析器"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_parameter_space_efficiency(self, 
                                         subnetwork: nn.Module,
                                         activations: torch.Tensor,
                                         gradients: torch.Tensor,
                                         targets: torch.Tensor) -> Dict[str, float]:
        """
        分析参数空间效率
        
        核心思想：评估当前参数在解决分类任务时的效率
        """
        logger.enter_section("参数空间效率分析")
        
        try:
            # 1. 计算参数利用率
            param_utilization = self._compute_parameter_utilization(subnetwork, gradients)
            
            # 2. 分析表示能力
            representation_capacity = self._compute_representation_capacity(activations, targets)
            
            # 3. 评估冗余度
            redundancy_ratio = self._compute_parameter_redundancy(subnetwork, activations)
            
            # 4. 计算可行参数空间占比
            feasible_space_ratio = self._estimate_feasible_parameter_space(
                subnetwork, activations, targets
            )
            
            # 5. 综合效率评分
            overall_efficiency = (
                0.3 * param_utilization +
                0.3 * representation_capacity +
                0.2 * (1.0 - redundancy_ratio) +  # 冗余度越低越好
                0.2 * feasible_space_ratio
            )
            
            analysis_result = {
                'parameter_utilization': param_utilization,
                'representation_capacity': representation_capacity,
                'redundancy_ratio': redundancy_ratio,
                'feasible_space_ratio': feasible_space_ratio,
                'overall_efficiency': overall_efficiency
            }
            
            logger.info(f"参数空间分析完成: 整体效率={overall_efficiency:.3f}")
            logger.exit_section("参数空间效率分析")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"参数空间分析失败: {e}")
            logger.exit_section("参数空间效率分析")
            return {
                'parameter_utilization': 0.0,
                'representation_capacity': 0.0,
                'redundancy_ratio': 1.0,
                'feasible_space_ratio': 0.0,
                'overall_efficiency': 0.0
            }
    
    def _compute_parameter_utilization(self, subnetwork: nn.Module, gradients: torch.Tensor) -> float:
        """计算参数利用率 - 有多少参数在积极参与学习"""
        
        total_params = sum(p.numel() for p in subnetwork.parameters())
        if total_params == 0:
            return 0.0
        
        # 计算有效梯度的参数数量
        active_params = 0
        for param in subnetwork.parameters():
            if param.grad is not None:
                # 梯度显著非零的参数被认为是活跃的
                significant_grads = torch.abs(param.grad) > 1e-6
                active_params += significant_grads.sum().item()
        
        utilization = active_params / total_params
        return min(utilization, 1.0)
    
    def _compute_representation_capacity(self, activations: torch.Tensor, targets: torch.Tensor) -> float:
        """计算表示能力 - 网络能多好地区分不同类别"""
        
        try:
            # 计算类间分离度
            if len(activations.shape) > 2:
                # 对于卷积层输出，取平均池化
                activations_flat = F.adaptive_avg_pool2d(activations, (1, 1)).flatten(1)
            else:
                activations_flat = activations
            
            # 计算不同类别的激活分布
            unique_targets = torch.unique(targets)
            if len(unique_targets) < 2:
                return 0.0
            
            class_centers = []
            for target_class in unique_targets:
                mask = targets == target_class
                if mask.sum() > 0:
                    class_center = activations_flat[mask].mean(dim=0)
                    class_centers.append(class_center)
            
            if len(class_centers) < 2:
                return 0.0
            
            # 计算类间距离
            class_centers = torch.stack(class_centers)
            distances = torch.pdist(class_centers)
            avg_inter_class_distance = distances.mean().item()
            
            # 计算类内方差
            intra_class_variance = 0.0
            for target_class in unique_targets:
                mask = targets == target_class
                if mask.sum() > 1:
                    class_activations = activations_flat[mask]
                    class_center = class_activations.mean(dim=0)
                    variance = ((class_activations - class_center) ** 2).mean().item()
                    intra_class_variance += variance
            
            intra_class_variance /= len(unique_targets)
            
            # 表示能力 = 类间距离 / 类内方差
            if intra_class_variance > 0:
                representation_capacity = avg_inter_class_distance / (intra_class_variance + 1e-8)
            else:
                representation_capacity = avg_inter_class_distance
            
            # 归一化到[0, 1]
            return min(representation_capacity / 10.0, 1.0)
            
        except Exception as e:
            logger.warning(f"表示能力计算失败: {e}")
            return 0.0
    
    def _compute_parameter_redundancy(self, subnetwork: nn.Module, activations: torch.Tensor) -> float:
        """计算参数冗余度"""
        
        try:
            redundancy_scores = []
            
            for name, module in subnetwork.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    weight = module.weight.data
                    
                    # 计算权重矩阵的有效秩
                    if len(weight.shape) == 2:  # Linear layer
                        rank = torch.matrix_rank(weight).item()
                        max_rank = min(weight.shape[0], weight.shape[1])
                    else:  # Conv2d layer
                        # 将卷积权重重塑为2D矩阵
                        weight_2d = weight.view(weight.shape[0], -1)
                        rank = torch.matrix_rank(weight_2d).item()
                        max_rank = min(weight_2d.shape[0], weight_2d.shape[1])
                    
                    if max_rank > 0:
                        rank_ratio = rank / max_rank
                        redundancy = 1.0 - rank_ratio  # 秩越低，冗余度越高
                        redundancy_scores.append(redundancy)
            
            if redundancy_scores:
                return np.mean(redundancy_scores)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"参数冗余度计算失败: {e}")
            return 0.0
    
    def _estimate_feasible_parameter_space(self, 
                                         subnetwork: nn.Module, 
                                         activations: torch.Tensor, 
                                         targets: torch.Tensor) -> float:
        """
        估计可行参数空间占比
        
        核心思想：在当前参数附近，有多大比例的参数变化能够维持或改善性能
        """
        
        try:
            # 使用采样方法估计可行参数空间
            original_params = [p.data.clone() for p in subnetwork.parameters()]
            
            # 计算原始性能
            with torch.no_grad():
                original_output = subnetwork(activations)
                if len(original_output.shape) > 1 and original_output.shape[1] > 1:
                    original_loss = F.cross_entropy(original_output, targets, reduction='mean')
                else:
                    original_loss = F.mse_loss(original_output.squeeze(), targets.float())
            
            # 采样测试
            feasible_count = 0
            total_samples = 50  # 减少采样数量以提高速度
            noise_scale = 0.01  # 小扰动
            
            for _ in range(total_samples):
                # 添加随机扰动
                for param in subnetwork.parameters():
                    noise = torch.randn_like(param.data) * noise_scale
                    param.data.add_(noise)
                
                # 测试扰动后的性能
                try:
                    with torch.no_grad():
                        perturbed_output = subnetwork(activations)
                        if len(perturbed_output.shape) > 1 and perturbed_output.shape[1] > 1:
                            perturbed_loss = F.cross_entropy(perturbed_output, targets, reduction='mean')
                        else:
                            perturbed_loss = F.mse_loss(perturbed_output.squeeze(), targets.float())
                    
                    # 如果损失没有显著增加，认为是可行的
                    if perturbed_loss <= original_loss * 1.1:  # 允许10%的性能下降
                        feasible_count += 1
                        
                except Exception:
                    pass  # 扰动导致的错误视为不可行
                
                # 恢复原始参数
                for param, original_param in zip(subnetwork.parameters(), original_params):
                    param.data.copy_(original_param)
            
            feasible_ratio = feasible_count / total_samples
            return feasible_ratio
            
        except Exception as e:
            logger.warning(f"可行参数空间估计失败: {e}")
            return 0.0

class MutationPotentialPredictor:
    """变异潜力预测器"""
    
    def __init__(self):
        self.predictor_cache = {}
    
    def predict_mutation_potential(self, 
                                 subnetwork: nn.Module,
                                 subnetwork_info: Dict[str, Any],
                                 parameter_space_analysis: Dict[str, float],
                                 current_accuracy: float) -> Dict[str, Any]:
        """
        预测变异潜力和可能的准确率提升
        
        Args:
            subnetwork: 提取的子网络
            subnetwork_info: 子网络分析信息
            parameter_space_analysis: 参数空间分析结果
            current_accuracy: 当前准确率
            
        Returns:
            变异潜力预测结果
        """
        logger.enter_section("变异潜力预测")
        
        try:
            # 1. 基于参数空间效率预测提升空间
            efficiency = parameter_space_analysis['overall_efficiency']
            improvement_potential = self._compute_improvement_potential(efficiency, current_accuracy)
            
            # 2. 预测不同变异策略的效果
            strategy_predictions = self._predict_strategy_effects(
                subnetwork_info, parameter_space_analysis, current_accuracy
            )
            
            # 3. 估计最优变异强度
            optimal_mutation_strength = self._estimate_optimal_mutation_strength(
                parameter_space_analysis, improvement_potential
            )
            
            # 4. 风险评估
            risk_assessment = self._assess_mutation_risks(
                subnetwork_info, parameter_space_analysis
            )
            
            prediction_result = {
                'improvement_potential': improvement_potential,
                'strategy_predictions': strategy_predictions,
                'optimal_mutation_strength': optimal_mutation_strength,
                'risk_assessment': risk_assessment,
                'confidence': self._compute_prediction_confidence(parameter_space_analysis)
            }
            
            logger.info(f"变异潜力预测完成: 改进潜力={improvement_potential:.3f}")
            logger.exit_section("变异潜力预测")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"变异潜力预测失败: {e}")
            logger.exit_section("变异潜力预测")
            return {
                'improvement_potential': 0.0,
                'strategy_predictions': {},
                'optimal_mutation_strength': 0.1,
                'risk_assessment': {'overall_risk': 1.0},
                'confidence': 0.0
            }
    
    def _compute_improvement_potential(self, efficiency: float, current_accuracy: float) -> float:
        """基于效率和当前准确率计算改进潜力"""
        
        # 效率越低，改进空间越大
        efficiency_factor = 1.0 - efficiency
        
        # 准确率越高，改进越困难
        if current_accuracy > 0.95:
            accuracy_factor = 0.1
        elif current_accuracy > 0.90:
            accuracy_factor = 0.3
        elif current_accuracy > 0.80:
            accuracy_factor = 0.6
        else:
            accuracy_factor = 1.0
        
        # 综合改进潜力
        improvement_potential = efficiency_factor * accuracy_factor
        
        return min(improvement_potential, 1.0)
    
    def _predict_strategy_effects(self, 
                                subnetwork_info: Dict[str, Any],
                                parameter_space_analysis: Dict[str, float],
                                current_accuracy: float) -> Dict[str, Dict[str, float]]:
        """预测不同变异策略的效果"""
        
        strategies = {
            'width_expansion': self._predict_width_expansion_effect(
                subnetwork_info, parameter_space_analysis
            ),
            'depth_increase': self._predict_depth_increase_effect(
                subnetwork_info, parameter_space_analysis
            ),
            'parallel_division': self._predict_parallel_division_effect(
                subnetwork_info, parameter_space_analysis
            ),
            'hybrid_mutation': self._predict_hybrid_mutation_effect(
                subnetwork_info, parameter_space_analysis
            )
        }
        
        return strategies
    
    def _predict_width_expansion_effect(self, 
                                      subnetwork_info: Dict[str, Any],
                                      parameter_space_analysis: Dict[str, float]) -> Dict[str, float]:
        """预测宽度扩展效果"""
        
        redundancy = parameter_space_analysis['redundancy_ratio']
        utilization = parameter_space_analysis['parameter_utilization']
        
        # 冗余度低且利用率高的层适合宽度扩展
        expansion_benefit = (1.0 - redundancy) * utilization
        
        return {
            'expected_accuracy_gain': expansion_benefit * 0.02,  # 最多2%的提升
            'parameter_cost': 0.3,  # 相对参数增长
            'implementation_difficulty': 0.2,  # 实现难度
            'stability_risk': 0.1  # 稳定性风险
        }
    
    def _predict_depth_increase_effect(self, 
                                     subnetwork_info: Dict[str, Any],
                                     parameter_space_analysis: Dict[str, float]) -> Dict[str, float]:
        """预测深度增加效果"""
        
        representation_capacity = parameter_space_analysis['representation_capacity']
        
        # 表示能力不足的子网络适合增加深度
        depth_benefit = 1.0 - representation_capacity
        
        return {
            'expected_accuracy_gain': depth_benefit * 0.015,  # 最多1.5%的提升
            'parameter_cost': 0.5,  # 较高的参数增长
            'implementation_difficulty': 0.4,  # 中等实现难度
            'stability_risk': 0.3  # 中等稳定性风险
        }
    
    def _predict_parallel_division_effect(self, 
                                        subnetwork_info: Dict[str, Any],
                                        parameter_space_analysis: Dict[str, float]) -> Dict[str, float]:
        """预测并行分裂效果"""
        
        feasible_space = parameter_space_analysis['feasible_space_ratio']
        
        # 可行空间大的子网络适合并行分裂
        parallel_benefit = feasible_space
        
        return {
            'expected_accuracy_gain': parallel_benefit * 0.025,  # 最多2.5%的提升
            'parameter_cost': 0.4,  # 中等参数增长
            'implementation_difficulty': 0.3,  # 中等实现难度
            'stability_risk': 0.2  # 较低稳定性风险
        }
    
    def _predict_hybrid_mutation_effect(self, 
                                      subnetwork_info: Dict[str, Any],
                                      parameter_space_analysis: Dict[str, float]) -> Dict[str, float]:
        """预测混合变异效果"""
        
        overall_efficiency = parameter_space_analysis['overall_efficiency']
        
        # 效率低的子网络适合混合变异
        hybrid_benefit = (1.0 - overall_efficiency) * 1.2  # 混合策略的加成
        
        return {
            'expected_accuracy_gain': min(hybrid_benefit * 0.03, 0.04),  # 最多4%的提升
            'parameter_cost': 0.6,  # 较高参数增长
            'implementation_difficulty': 0.6,  # 较高实现难度
            'stability_risk': 0.4  # 中等稳定性风险
        }
    
    def _estimate_optimal_mutation_strength(self, 
                                          parameter_space_analysis: Dict[str, float],
                                          improvement_potential: float) -> float:
        """估计最优变异强度"""
        
        efficiency = parameter_space_analysis['overall_efficiency']
        feasible_space = parameter_space_analysis['feasible_space_ratio']
        
        # 基于效率和可行空间确定变异强度
        base_strength = 0.1  # 基础变异强度
        
        # 效率低的子网络可以承受更强的变异
        efficiency_factor = (1.0 - efficiency) * 2.0
        
        # 可行空间大的子网络可以承受更强的变异
        feasible_factor = feasible_space * 1.5
        
        optimal_strength = base_strength + min(efficiency_factor + feasible_factor, 0.8)
        
        return min(optimal_strength, 1.0)
    
    def _assess_mutation_risks(self, 
                             subnetwork_info: Dict[str, Any],
                             parameter_space_analysis: Dict[str, float]) -> Dict[str, float]:
        """评估变异风险"""
        
        # 参数数量风险
        param_count = subnetwork_info['total_params']
        param_risk = min(param_count / 1e6, 1.0)  # 参数越多风险越高
        
        # 效率风险
        efficiency = parameter_space_analysis['overall_efficiency']
        efficiency_risk = efficiency  # 效率高的系统变异风险高
        
        # 可行空间风险
        feasible_space = parameter_space_analysis['feasible_space_ratio']
        space_risk = 1.0 - feasible_space  # 可行空间小风险高
        
        # 综合风险
        overall_risk = (param_risk + efficiency_risk + space_risk) / 3.0
        
        return {
            'parameter_risk': param_risk,
            'efficiency_risk': efficiency_risk,
            'space_risk': space_risk,
            'overall_risk': overall_risk
        }
    
    def _compute_prediction_confidence(self, parameter_space_analysis: Dict[str, float]) -> float:
        """计算预测置信度"""
        
        # 基于分析结果的完整性和一致性计算置信度
        analysis_completeness = len([v for v in parameter_space_analysis.values() if v > 0]) / len(parameter_space_analysis)
        
        # 结果的一致性检查
        efficiency = parameter_space_analysis['overall_efficiency']
        redundancy = parameter_space_analysis['redundancy_ratio']
        
        # 效率和冗余度应该负相关
        consistency = 1.0 - abs(efficiency + redundancy - 1.0)
        
        confidence = (analysis_completeness + consistency) / 2.0
        
        return min(confidence, 1.0)

class Net2NetSubnetworkAnalyzer:
    """Net2Net子网络分析器主类"""
    
    def __init__(self):
        self.extractor = SubnetworkExtractor()
        self.param_analyzer = ParameterSpaceAnalyzer()
        self.predictor = MutationPotentialPredictor()
        
        # 新增：信息流分析器
        self.info_flow_analyzer = InformationFlowAnalyzer()
        self.leak_detector = InformationLeakDetector()
        
        # 新增：贝叶斯变异收益预测器
        self.bayesian_predictor = BayesianMutationBenefitPredictor()
    
    def analyze_all_layers(self, model: nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析所有层的变异潜力和信息流瓶颈
        
        这是实现神经网络最优变异理论的核心方法：
        1. 检测信息流漏点 - 某层成为信息提取瓶颈，导致后续层无法提升准确率
        2. 分析参数空间密度 - 漏点层的参数空间中高准确率区域占比较小
        3. 预测变异收益 - 变异后参数空间中高准确率区域占比提升
        4. 指导架构变异 - 让漏点层变得更复杂，提取更多信息
        
        Args:
            model: 神经网络模型
            context: 分析上下文，包含激活值、梯度、目标等
            
        Returns:
            包含所有层分析结果和变异建议的字典
        """
        logger.enter_section("Net2Net全层分析")
        
        try:
            activations = context.get('activations', {})
            gradients = context.get('gradients', {})
            targets = context.get('targets')
            current_accuracy = context.get('current_accuracy', 0.0)
            
            # 1. 信息流全局分析
            logger.info("🔍 执行信息流全局分析...")
            flow_analysis = self._analyze_global_information_flow(
                model, activations, gradients, targets
            )
            
            # 2. 检测信息泄露漏点
            logger.info("🕳️ 检测信息泄露漏点...")
            leak_points = self._detect_information_leak_points(
                model, activations, gradients, targets, current_accuracy
            )
            
            # 3. 分析每层的变异潜力
            logger.info("📊 分析各层变异潜力...")
            layer_analyses = {}
            
            for layer_name in activations.keys():
                if self._is_analyzable_layer(model, layer_name):
                    layer_analysis = self.analyze_layer_mutation_potential(
                        model, layer_name, activations, gradients, 
                        targets, current_accuracy
                    )
                    
                    # 增强分析：添加信息流漏点评估
                    layer_analysis['leak_assessment'] = self._assess_layer_leak_potential(
                        layer_name, activations, gradients, leak_points
                    )
                    
                    layer_analyses[layer_name] = layer_analysis
            
            # 4. 贝叶斯收益预测
            logger.info("🧠 执行贝叶斯变异收益预测...")
            bayesian_predictions = self.predict_mutation_benefits_with_bayesian(
                layer_analyses, current_accuracy, model
            )
            
            # 5. 综合变异策略预测（Serial/Parallel + 层类型组合）
            logger.info("🎭 预测综合变异策略...")
            comprehensive_strategies = self.predict_comprehensive_strategies_for_top_candidates(
                layer_analyses, current_accuracy, model, top_n=3
            )
            
            # 6. 生成全局变异策略（结合贝叶斯预测和综合策略）
            logger.info("🎯 生成全局变异策略...")
            global_strategy = self._generate_global_mutation_strategy(
                layer_analyses, leak_points, flow_analysis, current_accuracy, 
                bayesian_predictions, comprehensive_strategies
            )
            
            # 7. 组装完整分析结果
            complete_analysis = {
                'global_flow_analysis': flow_analysis,
                'detected_leak_points': leak_points,
                'layer_analyses': layer_analyses,
                'bayesian_benefit_predictions': bayesian_predictions,
                'comprehensive_mutation_strategies': comprehensive_strategies,
                'global_mutation_strategy': global_strategy,
                'analysis_metadata': {
                    'total_layers_analyzed': len(layer_analyses),
                    'critical_leak_points': len([lp for lp in leak_points if lp['severity'] > 0.7]),
                    'high_potential_layers': len([la for la in layer_analyses.values() 
                                                 if la.get('mutation_prediction', {}).get('improvement_potential', 0) > 0.5]),
                    'high_confidence_predictions': len([bp for bp in bayesian_predictions.values() 
                                                       if bp.get('bayesian_prediction', {}).get('uncertainty_metrics', {}).get('prediction_confidence', 0) > 0.7]),
                    'strong_recommendations': len([bp for bp in bayesian_predictions.values() 
                                                  if bp.get('bayesian_prediction', {}).get('recommendation_strength', '') == 'strong_recommend']),
                    'comprehensive_strategies_count': len(comprehensive_strategies),
                    'analysis_timestamp': time.time()
                }
            }
            
            logger.success(f"Net2Net全层分析完成，发现{len(leak_points)}个潜在漏点")
            logger.exit_section("Net2Net全层分析")
            
            return complete_analysis
            
        except Exception as e:
            logger.error(f"Net2Net全层分析失败: {e}")
            logger.exit_section("Net2Net全层分析")
            return {
                'error': str(e),
                'global_mutation_strategy': {'action': 'skip', 'reason': f'分析失败: {e}'}
            }
    
    def _analyze_global_information_flow(self, model: nn.Module, 
                                       activations: Dict[str, torch.Tensor],
                                       gradients: Dict[str, torch.Tensor],
                                       targets: torch.Tensor) -> Dict[str, Any]:
        """分析全局信息流模式"""
        
        flow_metrics = {}
        layer_names = list(activations.keys())
        
        for i, layer_name in enumerate(layer_names):
            if layer_name not in gradients:
                continue
                
            activation = activations[layer_name]
            gradient = gradients[layer_name]
            
            # 计算信息密度指标
            info_density = self._calculate_information_density(activation, gradient)
            
            # 计算信息传递效率（与下一层的相关性）
            transfer_efficiency = 0.0
            if i < len(layer_names) - 1:
                next_layer = layer_names[i + 1]
                if next_layer in activations:
                    transfer_efficiency = self._calculate_transfer_efficiency(
                        activation, activations[next_layer]
                    )
            
            # 计算信息保留率（与目标的相关性）
            target_correlation = self._calculate_target_correlation(activation, targets)
            
            flow_metrics[layer_name] = {
                'information_density': info_density,
                'transfer_efficiency': transfer_efficiency,
                'target_correlation': target_correlation,
                'flow_bottleneck_score': self._calculate_bottleneck_score(
                    info_density, transfer_efficiency, target_correlation
                )
            }
        
        return {
            'layer_flow_metrics': flow_metrics,
            'global_bottleneck_score': np.mean([m['flow_bottleneck_score'] 
                                              for m in flow_metrics.values()]),
            'critical_bottlenecks': [name for name, metrics in flow_metrics.items() 
                                   if metrics['flow_bottleneck_score'] > 0.7]
        }
    
    def _detect_information_leak_points(self, model: nn.Module,
                                      activations: Dict[str, torch.Tensor],
                                      gradients: Dict[str, torch.Tensor],
                                      targets: torch.Tensor,
                                      current_accuracy: float) -> List[Dict[str, Any]]:
        """
        检测信息泄露漏点
        
        漏点的特征：
        1. 该层的信息密度显著低于前层
        2. 该层的梯度方差很小（学习困难）
        3. 后续子网络的参数空间中高准确率区域占比小
        4. 变异该层后，后续子网络性能提升明显
        """
        
        leak_points = []
        layer_names = list(activations.keys())
        
        for i, layer_name in enumerate(layer_names[1:], 1):  # 跳过第一层
            if layer_name not in gradients:
                continue
                
            # 获取当前层和前一层的数据
            current_activation = activations[layer_name]
            current_gradient = gradients[layer_name]
            prev_layer = layer_names[i-1]
            
            if prev_layer not in activations:
                continue
                
            prev_activation = activations[prev_layer]
            
            # 1. 信息密度下降检测
            current_info_density = self._calculate_information_density(
                current_activation, current_gradient
            )
            prev_info_density = self._calculate_information_density(
                prev_activation, gradients.get(prev_layer, torch.zeros_like(prev_activation))
            )
            
            info_drop = prev_info_density - current_info_density
            
            # 2. 梯度学习困难检测
            gradient_variance = torch.var(current_gradient).item()
            learning_difficulty = 1.0 / (1.0 + gradient_variance)  # 方差越小，学习越困难
            
            # 3. 后续子网络效率评估
            posterior_efficiency = self._evaluate_posterior_subnetwork_efficiency(
                model, layer_name, activations, targets
            )
            
            # 4. 变异潜力评估
            mutation_potential = self._estimate_mutation_improvement_potential(
                current_activation, current_gradient, targets, current_accuracy
            )
            
            # 综合评估漏点严重程度
            leak_severity = (
                info_drop * 0.3 +
                learning_difficulty * 0.2 +
                (1.0 - posterior_efficiency) * 0.3 +
                mutation_potential * 0.2
            )
            
            if leak_severity > 0.5:  # 阈值可调
                leak_points.append({
                    'layer_name': layer_name,
                    'severity': leak_severity,
                    'info_density_drop': info_drop,
                    'learning_difficulty': learning_difficulty,
                    'posterior_efficiency': posterior_efficiency,
                    'mutation_potential': mutation_potential,
                    'leak_type': self._classify_leak_type(
                        info_drop, learning_difficulty, posterior_efficiency
                    )
                })
        
        # 按严重程度排序
        leak_points.sort(key=lambda x: x['severity'], reverse=True)
        
        return leak_points
    
    def _assess_layer_leak_potential(self, layer_name: str,
                                   activations: Dict[str, torch.Tensor],
                                   gradients: Dict[str, torch.Tensor],
                                   leak_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估特定层的漏点潜力"""
        
        # 检查该层是否被识别为漏点
        is_leak_point = any(lp['layer_name'] == layer_name for lp in leak_points)
        
        if is_leak_point:
            leak_info = next(lp for lp in leak_points if lp['layer_name'] == layer_name)
            
            return {
                'is_leak_point': True,
                'leak_severity': leak_info['severity'],
                'leak_type': leak_info['leak_type'],
                'recommended_mutation_priority': 'high' if leak_info['severity'] > 0.7 else 'medium',
                'expected_improvement': leak_info['mutation_potential']
            }
        else:
            return {
                'is_leak_point': False,
                'leak_severity': 0.0,
                'recommended_mutation_priority': 'low',
                'expected_improvement': 0.0
            }
    
    def _generate_global_mutation_strategy(self, layer_analyses: Dict[str, Any],
                                         leak_points: List[Dict[str, Any]],
                                         flow_analysis: Dict[str, Any],
                                         current_accuracy: float,
                                         bayesian_predictions: Dict[str, Dict[str, Any]] = None,
                                         comprehensive_strategies: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成全局变异策略"""
        
        # 1. 优先处理严重漏点
        priority_targets = []
        for leak_point in leak_points:
            if leak_point['severity'] > 0.7:
                priority_targets.append({
                    'layer_name': leak_point['layer_name'],
                    'priority': 'critical',
                    'expected_improvement': leak_point['mutation_potential'],
                    'strategy': self._select_optimal_mutation_strategy(leak_point)
                })
        
        # 2. 考虑高潜力非漏点层（结合贝叶斯预测）
        if bayesian_predictions:
            # 使用贝叶斯预测结果重新排序
            bayesian_sorted = sorted(
                bayesian_predictions.items(),
                key=lambda x: x[1].get('combined_score', 0),
                reverse=True
            )
            
            for layer_name, bayesian_result in bayesian_sorted:
                if layer_name in layer_analyses:
                    combined_score = bayesian_result.get('combined_score', 0)
                    bayesian_gain = bayesian_result.get('bayesian_prediction', {}).get('expected_accuracy_gain', 0)
                    confidence = bayesian_result.get('bayesian_prediction', {}).get('uncertainty_metrics', {}).get('prediction_confidence', 0)
                    
                    # 贝叶斯驱动的选择标准
                    if (combined_score > 0.02 and confidence > 0.6 and bayesian_gain > 0.005 and 
                        not any(t['layer_name'] == layer_name for t in priority_targets)):
                        
                        priority_targets.append({
                            'layer_name': layer_name,
                            'priority': 'high' if combined_score > 0.05 else 'medium',
                            'expected_improvement': bayesian_gain,
                            'strategy': bayesian_result.get('mutation_strategy', 'widening'),
                            'bayesian_confidence': confidence,
                            'combined_score': combined_score,
                            'recommendation_strength': bayesian_result.get('bayesian_prediction', {}).get('recommendation_strength', 'neutral')
                        })
        else:
            # fallback到原来的逻辑
            for layer_name, analysis in layer_analyses.items():
                mutation_potential = analysis.get('mutation_prediction', {}).get('improvement_potential', 0)
                if mutation_potential > 0.6 and not any(t['layer_name'] == layer_name for t in priority_targets):
                    priority_targets.append({
                        'layer_name': layer_name,
                        'priority': 'high',
                        'expected_improvement': mutation_potential,
                        'strategy': analysis.get('recommendation', {}).get('strategy', 'widening')
                    })
        
        # 3. 生成执行计划
        execution_plan = self._create_mutation_execution_plan(
            priority_targets, current_accuracy, flow_analysis
        )
        
        # 集成综合策略信息
        enhanced_targets = []
        for target in priority_targets:
            layer_name = target['layer_name']
            enhanced_target = target.copy()
            
            # 添加综合策略信息
            if comprehensive_strategies and layer_name in comprehensive_strategies:
                comp_strategy = comprehensive_strategies[layer_name]['comprehensive_strategy']
                enhanced_target.update({
                    'detailed_mutation_mode': comp_strategy.get('mutation_mode', 'unknown'),
                    'layer_combination_strategy': comp_strategy.get('layer_combination', {}),
                    'implementation_timeline': comp_strategy.get('implementation_details', {}).get('expected_timeline', 'unknown'),
                    'comprehensive_confidence': comp_strategy.get('confidence', 0.5),
                    'total_expected_gain': comp_strategy.get('expected_total_gain', 0.0)
                })
            
            enhanced_targets.append(enhanced_target)
        
        return {
            'priority_targets': enhanced_targets,
            'execution_plan': execution_plan,
            'comprehensive_strategies_summary': self._summarize_comprehensive_strategies(comprehensive_strategies),
            'global_improvement_estimate': sum(t.get('total_expected_gain', t.get('expected_improvement', 0)) for t in enhanced_targets),
            'recommended_sequence': [t['layer_name'] for t in 
                                   sorted(enhanced_targets, key=lambda x: x.get('total_expected_gain', x.get('expected_improvement', 0)), reverse=True)]
        }
    
    def _calculate_information_density(self, activation: torch.Tensor, gradient: torch.Tensor) -> float:
        """计算信息密度"""
        # 使用激活值的熵和梯度的方差作为信息密度指标
        activation_entropy = self._calculate_entropy(activation)
        gradient_variance = torch.var(gradient).item()
        
        # 归一化并组合
        info_density = (activation_entropy + np.log(1 + gradient_variance)) / 2
        return float(info_density)
    
    def _calculate_entropy(self, tensor: torch.Tensor) -> float:
        """计算张量的近似熵"""
        # 将张量展平并计算直方图
        flat_tensor = tensor.flatten()
        hist, _ = np.histogram(flat_tensor.cpu().numpy(), bins=50, density=True)
        
        # 避免log(0)
        hist = hist + 1e-10
        entropy = -np.sum(hist * np.log(hist))
        
        return float(entropy)
    
    def _calculate_transfer_efficiency(self, current_activation: torch.Tensor, 
                                     next_activation: torch.Tensor) -> float:
        """计算信息传递效率"""
        # 计算激活值之间的相关性
        curr_flat = current_activation.flatten()
        next_flat = next_activation.flatten()
        
        # 调整尺寸以匹配
        min_size = min(len(curr_flat), len(next_flat))
        curr_flat = curr_flat[:min_size]
        next_flat = next_flat[:min_size]
        
        correlation = torch.corrcoef(torch.stack([curr_flat, next_flat]))[0, 1]
        
        # 处理NaN情况
        if torch.isnan(correlation):
            return 0.0
            
        return float(torch.abs(correlation))
    
    def _calculate_target_correlation(self, activation: torch.Tensor, targets: torch.Tensor) -> float:
        """计算与目标的相关性"""
        # 简化的相关性计算
        activation_mean = torch.mean(activation, dim=tuple(range(1, activation.dim())))
        
        if len(activation_mean) != len(targets):
            return 0.0
            
        # 计算与目标的相关性
        try:
            correlation = torch.corrcoef(torch.stack([
                activation_mean.float(),
                targets.float()
            ]))[0, 1]
            
            if torch.isnan(correlation):
                return 0.0
                
            return float(torch.abs(correlation))
        except:
            return 0.0
    
    def _calculate_bottleneck_score(self, info_density: float, transfer_efficiency: float, 
                                  target_correlation: float) -> float:
        """计算瓶颈分数"""
        # 瓶颈分数 = 信息密度低 + 传递效率低 + 目标相关性低
        bottleneck_score = (
            (1.0 - min(info_density / 10.0, 1.0)) * 0.4 +
            (1.0 - transfer_efficiency) * 0.3 +
            (1.0 - target_correlation) * 0.3
        )
        
        return float(bottleneck_score)
    
    def _evaluate_posterior_subnetwork_efficiency(self, model: nn.Module, layer_name: str,
                                                activations: Dict[str, torch.Tensor],
                                                targets: torch.Tensor) -> float:
        """评估后续子网络效率"""
        # 获取该层之后的所有层
        layer_names = list(activations.keys())
        try:
            layer_idx = layer_names.index(layer_name)
            posterior_layers = layer_names[layer_idx + 1:]
        except ValueError:
            return 0.5  # 默认中等效率
        
        if not posterior_layers:
            return 1.0  # 最后一层，效率为1
        
        # 计算后续层的平均信息处理效率
        efficiency_scores = []
        
        for post_layer in posterior_layers:
            if post_layer in activations:
                post_activation = activations[post_layer]
                target_corr = self._calculate_target_correlation(post_activation, targets)
                efficiency_scores.append(target_corr)
        
        if not efficiency_scores:
            return 0.5
            
        return float(np.mean(efficiency_scores))
    
    def _estimate_mutation_improvement_potential(self, activation: torch.Tensor,
                                               gradient: torch.Tensor,
                                               targets: torch.Tensor,
                                               current_accuracy: float) -> float:
        """估算变异改进潜力"""
        # 基于梯度和激活模式估算变异后的改进潜力
        
        # 1. 梯度多样性（高多样性 = 高改进潜力）
        gradient_diversity = torch.std(gradient).item()
        
        # 2. 激活饱和度（低饱和度 = 高改进潜力）
        activation_saturation = torch.mean(torch.sigmoid(activation)).item()
        saturation_score = 1.0 - abs(activation_saturation - 0.5) * 2  # 0.5为最佳
        
        # 3. 当前准确率距离上限的空间
        accuracy_headroom = (0.95 - current_accuracy) / 0.95
        
        # 综合评估
        improvement_potential = (
            gradient_diversity * 0.3 +
            saturation_score * 0.3 +
            accuracy_headroom * 0.4
        )
        
        return float(np.clip(improvement_potential, 0.0, 1.0))
    
    def _classify_leak_type(self, info_drop: float, learning_difficulty: float, 
                          posterior_efficiency: float) -> str:
        """分类漏点类型"""
        if info_drop > 0.5:
            return "information_compression_bottleneck"
        elif learning_difficulty > 0.7:
            return "gradient_learning_bottleneck"
        elif posterior_efficiency < 0.3:
            return "representational_bottleneck"
        else:
            return "general_bottleneck"
    
    def _select_optimal_mutation_strategy(self, leak_point: Dict[str, Any]) -> str:
        """为漏点选择最优变异策略"""
        leak_type = leak_point['leak_type']
        severity = leak_point['severity']
        
        if leak_type == "information_compression_bottleneck":
            return "widening"  # 增加通道数
        elif leak_type == "gradient_learning_bottleneck":
            return "deepening"  # 增加层数
        elif leak_type == "representational_bottleneck":
            return "hybrid_expansion"  # 混合扩展
        else:
            # 根据严重程度选择
            if severity > 0.8:
                return "aggressive_widening"
            else:
                return "conservative_widening"
    
    def predict_mutation_benefits_with_bayesian(self, 
                                              layer_analyses: Dict[str, Any],
                                              current_accuracy: float,
                                              model: nn.Module) -> Dict[str, Dict[str, Any]]:
        """
        使用贝叶斯推断为所有候选层预测变异收益
        
        Args:
            layer_analyses: 所有层的分析结果
            current_accuracy: 当前准确率
            model: 神经网络模型
            
        Returns:
            每层的贝叶斯收益预测结果
        """
        logger.enter_section("贝叶斯变异收益批量预测")
        
        bayesian_predictions = {}
        
        # 计算模型复杂度指标
        model_complexity = self._calculate_model_complexity(model)
        
        for layer_name, layer_analysis in layer_analyses.items():
            try:
                # 获取推荐的变异策略
                recommendation = layer_analysis.get('recommendation', {})
                mutation_strategy = recommendation.get('recommended_strategy', 'widening')
                
                # 如果是漏点，使用漏点特定的策略
                leak_assessment = layer_analysis.get('leak_assessment', {})
                if leak_assessment.get('is_leak_point', False):
                    leak_type = leak_assessment.get('leak_type', 'general_bottleneck')
                    mutation_strategy = self._get_strategy_for_leak_type(leak_type)
                
                # 贝叶斯收益预测
                bayesian_result = self.bayesian_predictor.predict_mutation_benefit(
                    layer_analysis=layer_analysis,
                    mutation_strategy=mutation_strategy,
                    current_accuracy=current_accuracy,
                    model_complexity=model_complexity
                )
                
                # 增强分析结果
                bayesian_predictions[layer_name] = {
                    'mutation_strategy': mutation_strategy,
                    'bayesian_prediction': bayesian_result,
                    'is_leak_point': leak_assessment.get('is_leak_point', False),
                    'leak_severity': leak_assessment.get('leak_severity', 0.0),
                    'combined_score': self._calculate_combined_benefit_score(
                        layer_analysis, bayesian_result
                    )
                }
                
                logger.info(f"🎯 {layer_name}: 策略={mutation_strategy}, "
                          f"期望收益={bayesian_result['expected_accuracy_gain']:.4f}, "
                          f"置信度={bayesian_result['uncertainty_metrics']['prediction_confidence']:.3f}")
                
            except Exception as e:
                logger.error(f"❌ 贝叶斯预测失败 {layer_name}: {e}")
                bayesian_predictions[layer_name] = {
                    'mutation_strategy': 'widening',
                    'bayesian_prediction': self.bayesian_predictor._fallback_prediction('widening', current_accuracy),
                    'error': str(e)
                }
        
        logger.success(f"完成{len(bayesian_predictions)}个层的贝叶斯收益预测")
        logger.exit_section("贝叶斯变异收益批量预测")
        
        return bayesian_predictions
    
    def predict_comprehensive_strategies_for_top_candidates(self,
                                                          layer_analyses: Dict[str, Any],
                                                          current_accuracy: float,
                                                          model: nn.Module,
                                                          top_n: int = 3) -> Dict[str, Dict[str, Any]]:
        """
        为前N个候选层预测综合变异策略
        包括变异模式选择和层类型组合预测
        """
        logger.enter_section("综合策略预测")
        
        try:
            comprehensive_strategies = {}
            
            # 选择top N候选层
            candidates = []
            for layer_name, analysis in layer_analyses.items():
                improvement_potential = analysis.get('mutation_prediction', {}).get('improvement_potential', 0)
                leak_severity = analysis.get('leak_assessment', {}).get('leak_severity', 0)
                combined_score = improvement_potential + leak_severity * 0.5
                candidates.append((layer_name, combined_score, analysis))
            
            # 按评分排序并选择前N个
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidates[:top_n]
            
            for layer_name, score, layer_analysis in top_candidates:
                logger.info(f"🎯 预测 {layer_name} 的综合变异策略...")
                
                # 预测综合策略
                comprehensive_strategy = self.bayesian_predictor.predict_comprehensive_mutation_strategy(
                    layer_analysis=layer_analysis,
                    current_accuracy=current_accuracy,
                    model=model,
                    target_layer_name=layer_name
                )
                
                comprehensive_strategies[layer_name] = {
                    'layer_score': score,
                    'comprehensive_strategy': comprehensive_strategy,
                    'detailed_breakdown': {
                        'mode_analysis': self._extract_mode_analysis(comprehensive_strategy),
                        'combination_analysis': self._extract_combination_analysis(comprehensive_strategy),
                        'implementation_plan': comprehensive_strategy.get('implementation_details', {})
                    }
                }
                
                # 详细日志输出
                mode = comprehensive_strategy['mutation_mode']
                combo = comprehensive_strategy['layer_combination']['combination']
                total_gain = comprehensive_strategy['expected_total_gain']
                confidence = comprehensive_strategy['confidence']
                
                logger.info(f"  📋 {layer_name}: {mode} + {combo}")
                logger.info(f"    💡 总期望收益: {total_gain:.4f}")
                logger.info(f"    🎯 置信度: {confidence:.3f}")
            
            logger.success(f"完成{len(comprehensive_strategies)}个层的综合策略预测")
            logger.exit_section("综合策略预测")
            
            return comprehensive_strategies
            
        except Exception as e:
            logger.error(f"综合策略预测失败: {e}")
            logger.exit_section("综合策略预测")
            return {}

    def _extract_mode_analysis(self, comprehensive_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """提取变异模式分析"""
        return {
            'recommended_mode': comprehensive_strategy.get('mutation_mode', 'unknown'),
            'mode_reasoning': "基于瓶颈类型和准确率阶段的最优选择",
            'alternatives': ['serial_division', 'parallel_division', 'hybrid_division']
        }

    def _extract_combination_analysis(self, comprehensive_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """提取层组合分析"""
        layer_combo = comprehensive_strategy.get('layer_combination', {})
        return {
            'recommended_combination': layer_combo.get('combination', 'unknown'),
            'combination_type': layer_combo.get('type', 'unknown'),
            'synergy_score': layer_combo.get('synergy', 0.5),
            'implementation_cost': layer_combo.get('implementation_cost', 1.0)
        }
    
    def _calculate_model_complexity(self, model: nn.Module) -> Dict[str, float]:
        """计算模型复杂度指标"""
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # 计算层深度和平均宽度
        layer_count = 0
        total_width = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layer_count += 1
                
                if isinstance(module, nn.Linear):
                    total_width += module.out_features
                elif isinstance(module, nn.Conv2d):
                    total_width += module.out_channels
        
        avg_width = total_width / max(layer_count, 1)
        
        return {
            'total_parameters': float(total_params),
            'layer_depth': float(layer_count),
            'layer_width': float(avg_width)
        }
    
    def _get_strategy_for_leak_type(self, leak_type: str) -> str:
        """根据漏点类型获取最优策略"""
        
        strategy_mapping = {
            'information_compression_bottleneck': 'widening',
            'gradient_learning_bottleneck': 'deepening',
            'representational_bottleneck': 'hybrid_expansion',
            'general_bottleneck': 'widening'
        }
        
        return strategy_mapping.get(leak_type, 'widening')
    
    def _calculate_combined_benefit_score(self, 
                                        layer_analysis: Dict[str, Any],
                                        bayesian_result: Dict[str, Any]) -> float:
        """计算综合收益评分"""
        
        # 原始变异潜力
        original_potential = layer_analysis.get('mutation_prediction', {}).get('improvement_potential', 0.0)
        
        # 贝叶斯期望收益
        bayesian_gain = bayesian_result.get('expected_accuracy_gain', 0.0)
        
        # 贝叶斯置信度
        confidence = bayesian_result.get('uncertainty_metrics', {}).get('prediction_confidence', 0.5)
        
        # 风险调整收益
        risk_adjusted = bayesian_result.get('risk_adjusted_benefit', {}).get('risk_adjusted_gain', 0.0)
        
        # 综合评分
        combined_score = (
            original_potential * 0.3 +
            bayesian_gain * 0.4 +
            risk_adjusted * 0.3
        ) * confidence
        
        return float(combined_score)
    
    def _create_mutation_execution_plan(self, priority_targets: List[Dict[str, Any]],
                                      current_accuracy: float,
                                      flow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """创建变异执行计划"""
        
        # 根据当前准确率和全局流分析确定执行策略
        if current_accuracy < 0.85:
            execution_mode = "conservative"
            max_concurrent = 1
        elif current_accuracy < 0.92:
            execution_mode = "moderate"
            max_concurrent = 2
        else:
            execution_mode = "aggressive"
            max_concurrent = 3
        
        return {
            'execution_mode': execution_mode,
            'max_concurrent_mutations': max_concurrent,
            'total_expected_improvement': sum(t['expected_improvement'] for t in priority_targets),
            'estimated_parameter_cost': len(priority_targets) * 5000,  # 估算
            'execution_phases': self._plan_execution_phases(priority_targets, max_concurrent)
        }
    
    def _plan_execution_phases(self, targets: List[Dict[str, Any]], max_concurrent: int) -> List[List[str]]:
        """规划执行阶段"""
        phases = []
        
        # 按优先级分组
        critical = [t for t in targets if t['priority'] == 'critical']
        high = [t for t in targets if t['priority'] == 'high']
        
        # 第一阶段：关键漏点
        if critical:
            phases.append([t['layer_name'] for t in critical[:max_concurrent]])
        
        # 第二阶段：高潜力层
        if high:
            phases.append([t['layer_name'] for t in high[:max_concurrent]])
        
        return phases
    
    def _summarize_comprehensive_strategies(self, comprehensive_strategies: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """总结综合策略"""
        if not comprehensive_strategies:
            return {}
        
        # 统计变异模式偏好
        mode_counts = {}
        combination_types = {}
        total_expected_gain = 0.0
        avg_confidence = 0.0
        
        for layer_name, strategy_data in comprehensive_strategies.items():
            comp_strategy = strategy_data['comprehensive_strategy']
            
            # 统计变异模式
            mode = comp_strategy.get('mutation_mode', 'unknown')
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            # 统计层组合类型
            combo_type = comp_strategy.get('layer_combination', {}).get('type', 'unknown')
            combination_types[combo_type] = combination_types.get(combo_type, 0) + 1
            
            # 累加指标
            total_expected_gain += comp_strategy.get('expected_total_gain', 0.0)
            avg_confidence += comp_strategy.get('confidence', 0.0)
        
        n_strategies = len(comprehensive_strategies)
        avg_confidence /= max(n_strategies, 1)
        
        # 找出最受推荐的模式和组合
        preferred_mode = max(mode_counts.items(), key=lambda x: x[1])[0] if mode_counts else 'serial_division'
        preferred_combination = max(combination_types.items(), key=lambda x: x[1])[0] if combination_types else 'heterogeneous'
        
        return {
            'total_strategies_analyzed': n_strategies,
            'preferred_mutation_mode': preferred_mode,
            'preferred_combination_type': preferred_combination,
            'mode_distribution': mode_counts,
            'combination_distribution': combination_types,
            'total_expected_improvement': total_expected_gain,
            'average_confidence': avg_confidence,
            'strategy_recommendations': [
                f"主要推荐: {preferred_mode} 变异模式",
                f"首选组合: {preferred_combination} 层组合",
                f"总期望收益: {total_expected_gain:.4f}",
                f"平均置信度: {avg_confidence:.3f}"
            ]
        }
    
    def _is_analyzable_layer(self, model: nn.Module, layer_name: str) -> bool:
        """判断层是否可分析"""
        try:
            module = dict(model.named_modules())[layer_name]
            return isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))
        except:
            return False

    def analyze_layer_mutation_potential(self, 
                                       model: nn.Module,
                                       layer_name: str,
                                       activations: Dict[str, torch.Tensor],
                                       gradients: Dict[str, torch.Tensor],
                                       targets: torch.Tensor,
                                       current_accuracy: float) -> Dict[str, Any]:
        """
        分析指定层的变异潜力
        
        Args:
            model: 完整模型
            layer_name: 目标层名称
            activations: 激活值字典
            gradients: 梯度字典
            targets: 真实标签
            current_accuracy: 当前准确率
            
        Returns:
            完整的变异潜力分析结果
        """
        logger.enter_section(f"Net2Net分析: {layer_name}")
        
        try:
            # 1. 提取子网络
            subnetwork, subnetwork_info = self.extractor.extract_subnetwork_from_layer(
                model, layer_name
            )
            
            # 2. 获取该层的激活和梯度
            if layer_name in activations and layer_name in gradients:
                layer_activation = activations[layer_name]
                layer_gradient = gradients[layer_name]
            else:
                logger.warning(f"层{layer_name}缺少激活值或梯度信息")
                layer_activation = torch.randn(32, 64)  # 默认值
                layer_gradient = torch.randn(32, 64)
            
            # 3. 分析参数空间效率
            param_space_analysis = self.param_analyzer.analyze_parameter_space_efficiency(
                subnetwork, layer_activation, layer_gradient, targets
            )
            
            # 4. 预测变异潜力
            mutation_prediction = self.predictor.predict_mutation_potential(
                subnetwork, subnetwork_info, param_space_analysis, current_accuracy
            )
            
            # 5. 生成综合分析报告
            analysis_result = {
                'layer_name': layer_name,
                'subnetwork_info': subnetwork_info,
                'parameter_space_analysis': param_space_analysis,
                'mutation_prediction': mutation_prediction,
                'recommendation': self._generate_recommendation(
                    layer_name, param_space_analysis, mutation_prediction
                )
            }
            
            logger.success(f"Net2Net分析完成: {layer_name}")
            logger.exit_section(f"Net2Net分析: {layer_name}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Net2Net分析失败: {layer_name} - {e}")
            logger.exit_section(f"Net2Net分析: {layer_name}")
            return {
                'layer_name': layer_name,
                'error': str(e),
                'recommendation': {'action': 'skip', 'reason': f'分析失败: {e}'}
            }

    def _generate_recommendation(self, 
                               layer_name: str,
                               param_space_analysis: Dict[str, float],
                               mutation_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """生成变异建议"""
        
        improvement_potential = mutation_prediction['improvement_potential']
        risk_assessment = mutation_prediction['risk_assessment']
        strategy_predictions = mutation_prediction['strategy_predictions']
        
        # 选择最佳策略
        best_strategy = None
        best_score = -1.0
        
        for strategy_name, strategy_info in strategy_predictions.items():
            # 综合评分：期望收益 - 风险 - 成本
            score = (
                strategy_info['expected_accuracy_gain'] * 2.0 -
                strategy_info['stability_risk'] -
                strategy_info['parameter_cost'] * 0.5
            )
            
            if score > best_score:
                best_score = score
                best_strategy = strategy_name
        
        # 生成建议
        if improvement_potential > 0.3 and risk_assessment['overall_risk'] < 0.6:
            action = 'mutate'
            priority = 'high' if improvement_potential > 0.6 else 'medium'
        elif improvement_potential > 0.1 and risk_assessment['overall_risk'] < 0.8:
            action = 'consider'
            priority = 'low'
        else:
            action = 'skip'
            priority = 'none'
        
        return {
            'action': action,
            'priority': priority,
            'recommended_strategy': best_strategy,
            'expected_gain': strategy_predictions.get(best_strategy, {}).get('expected_accuracy_gain', 0.0),
            'risk_level': risk_assessment['overall_risk'],
            'reason': f"改进潜力={improvement_potential:.3f}, 风险={risk_assessment['overall_risk']:.3f}"
        }

# 新增：信息流分析器类
class InformationFlowAnalyzer:
    """信息流分析器"""
    
    def __init__(self):
        self.flow_patterns = {}
        
    def analyze_flow_patterns(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """分析信息流模式"""
        # 实现信息流分析逻辑
        return {}

class InformationLeakDetector:
    """信息泄露检测器"""
    
    def __init__(self):
        self.leak_thresholds = {
            'entropy_drop': 0.5,
            'gradient_variance': 0.1,
            'correlation_loss': 0.3
        }
    
    def detect_leaks(self, layer_data: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """检测信息泄露点"""
        # 实现泄露检测逻辑
        return []

# 新增：基于贝叶斯推断的变异收益预测器
class BayesianMutationBenefitPredictor:
    """
    基于贝叶斯推断的变异收益预测器
    
    使用贝叶斯统计、高斯过程回归和蒙特卡罗采样来预测架构变异的期望收益
    """
    
    def __init__(self):
        self.prior_knowledge = self._initialize_prior_knowledge()
        self.gp_hyperparams = {
            'length_scale': 1.0,
            'variance': 1.0,
            'noise_variance': 0.01
        }
        self.mc_samples = 1000  # 蒙特卡罗采样数
        
        # 历史变异数据（用于更新先验）
        self.mutation_history = []
        
    def _initialize_prior_knowledge(self) -> Dict[str, Any]:
        """初始化先验知识"""
        return {
            # 不同变异类型的历史成功率先验
            'mutation_success_priors': {
                'widening': {'alpha': 3, 'beta': 2},  # Beta分布参数，倾向于成功
                'deepening': {'alpha': 2, 'beta': 3},  # 相对保守
                'hybrid_expansion': {'alpha': 4, 'beta': 2},  # 较为激进
                'aggressive_widening': {'alpha': 2, 'beta': 1}  # 高风险高收益
            },
            
            # Serial vs Parallel mutation 先验知识
            'mutation_mode_priors': {
                'serial_division': {
                    'success_rate': {'alpha': 5, 'beta': 3},  # 相对稳定
                    'best_for': ['gradient_learning_bottleneck', 'representational_bottleneck'],
                    'accuracy_preference': {'low': 0.7, 'medium': 0.8, 'high': 0.6}
                },
                'parallel_division': {
                    'success_rate': {'alpha': 4, 'beta': 4},  # 中等风险
                    'best_for': ['information_compression_bottleneck'],
                    'accuracy_preference': {'low': 0.6, 'medium': 0.7, 'high': 0.8}
                },
                'hybrid_division': {
                    'success_rate': {'alpha': 6, 'beta': 2},  # 激进但高收益
                    'best_for': ['general_bottleneck'],
                    'accuracy_preference': {'low': 0.8, 'medium': 0.9, 'high': 0.7}
                }
            },
            
            # 层类型组合策略先验 (同种 vs 异种)
            'layer_combination_priors': {
                'homogeneous': {  # 同种层
                    'conv2d_conv2d': {'effectiveness': 0.7, 'stability': 0.9},
                    'linear_linear': {'effectiveness': 0.6, 'stability': 0.8},
                    'batch_norm_batch_norm': {'effectiveness': 0.5, 'stability': 0.9}
                },
                'heterogeneous': {  # 异种层组合
                    'conv2d_depthwise_conv': {'effectiveness': 0.8, 'stability': 0.7},
                    'conv2d_batch_norm': {'effectiveness': 0.9, 'stability': 0.8},
                    'conv2d_dropout': {'effectiveness': 0.6, 'stability': 0.7},
                    'conv2d_attention': {'effectiveness': 0.85, 'stability': 0.6},
                    'linear_dropout': {'effectiveness': 0.7, 'stability': 0.8},
                    'linear_batch_norm': {'effectiveness': 0.8, 'stability': 0.9},
                    'conv2d_pool': {'effectiveness': 0.5, 'stability': 0.9},
                    'conv2d_residual_block': {'effectiveness': 0.9, 'stability': 0.8}
                }
            },
            
            # 不同网络层操作的适用性先验
            'layer_operation_priors': {
                'conv2d': {
                    'feature_extraction_boost': 0.9,
                    'spatial_processing': 0.95,
                    'parameter_efficiency': 0.7,
                    'computation_cost': 0.6
                },
                'depthwise_conv': {
                    'feature_extraction_boost': 0.7,
                    'spatial_processing': 0.8,
                    'parameter_efficiency': 0.9,
                    'computation_cost': 0.8
                },
                'batch_norm': {
                    'feature_extraction_boost': 0.4,
                    'spatial_processing': 0.3,
                    'parameter_efficiency': 0.9,
                    'computation_cost': 0.9,
                    'stability_boost': 0.9
                },
                'dropout': {
                    'feature_extraction_boost': 0.2,
                    'spatial_processing': 0.1,
                    'parameter_efficiency': 1.0,
                    'computation_cost': 0.95,
                    'overfitting_prevention': 0.8
                },
                'attention': {
                    'feature_extraction_boost': 0.85,
                    'spatial_processing': 0.7,
                    'parameter_efficiency': 0.5,
                    'computation_cost': 0.3,
                    'long_range_dependency': 0.95
                },
                'pool': {
                    'feature_extraction_boost': 0.3,
                    'spatial_processing': 0.6,
                    'parameter_efficiency': 1.0,
                    'computation_cost': 0.9,
                    'dimensionality_reduction': 0.9
                },
                'residual_connection': {
                    'feature_extraction_boost': 0.6,
                    'spatial_processing': 0.5,
                    'parameter_efficiency': 0.8,
                    'computation_cost': 0.7,
                    'gradient_flow': 0.95
                }
            },
            
            # 不同瓶颈类型对变异的响应性先验
            'bottleneck_response_priors': {
                'information_compression_bottleneck': {
                    'widening_response': 0.8,
                    'deepening_response': 0.3,
                    'hybrid_response': 0.6,
                    'preferred_operations': ['conv2d', 'attention', 'residual_connection']
                },
                'gradient_learning_bottleneck': {
                    'widening_response': 0.4,
                    'deepening_response': 0.7,
                    'hybrid_response': 0.5,
                    'preferred_operations': ['batch_norm', 'residual_connection', 'dropout']
                },
                'representational_bottleneck': {
                    'widening_response': 0.6,
                    'deepening_response': 0.5,
                    'hybrid_response': 0.9,
                    'preferred_operations': ['attention', 'conv2d', 'depthwise_conv']
                }
            },
            
            # 准确率阶段对变异收益的影响
            'accuracy_stage_priors': {
                'low': (0.0, 0.85),    # 低准确率阶段，变异收益较大
                'medium': (0.85, 0.92), # 中等准确率，收益递减
                'high': (0.92, 1.0)     # 高准确率，收益微小但关键
            }
        }
    
    def predict_mutation_benefit(self, 
                               layer_analysis: Dict[str, Any],
                               mutation_strategy: str,
                               current_accuracy: float,
                               model_complexity: Dict[str, float]) -> Dict[str, Any]:
        """
        使用贝叶斯推断预测变异收益
        
        Args:
            layer_analysis: 层分析结果
            mutation_strategy: 变异策略类型
            current_accuracy: 当前准确率
            model_complexity: 模型复杂度指标
            
        Returns:
            包含期望收益、置信区间、风险评估的预测结果
        """
        logger.enter_section(f"贝叶斯变异收益预测: {mutation_strategy}")
        
        try:
            # 1. 构建特征向量
            feature_vector = self._extract_feature_vector(
                layer_analysis, current_accuracy, model_complexity
            )
            
            # 2. 贝叶斯后验推断
            posterior_params = self._bayesian_posterior_inference(
                feature_vector, mutation_strategy, layer_analysis
            )
            
            # 3. 高斯过程回归预测
            gp_prediction = self._gaussian_process_prediction(
                feature_vector, posterior_params
            )
            
            # 4. 蒙特卡罗期望估计
            mc_estimate = self._monte_carlo_benefit_estimation(
                gp_prediction, mutation_strategy, current_accuracy
            )
            
            # 5. 不确定性量化
            uncertainty_metrics = self._quantify_prediction_uncertainty(
                gp_prediction, mc_estimate, feature_vector
            )
            
            # 6. 风险调整收益
            risk_adjusted_benefit = self._calculate_risk_adjusted_benefit(
                mc_estimate, uncertainty_metrics, mutation_strategy
            )
            
            prediction_result = {
                'expected_accuracy_gain': mc_estimate['expected_gain'],
                'confidence_interval': mc_estimate['confidence_interval'],
                'success_probability': posterior_params['success_probability'],
                'risk_adjusted_benefit': risk_adjusted_benefit,
                'uncertainty_metrics': uncertainty_metrics,
                'bayesian_evidence': posterior_params['evidence'],
                'recommendation_strength': self._calculate_recommendation_strength(
                    risk_adjusted_benefit, uncertainty_metrics
                )
            }
            
            logger.success(f"贝叶斯预测完成: 期望收益={mc_estimate['expected_gain']:.4f}")
            logger.exit_section(f"贝叶斯变异收益预测: {mutation_strategy}")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"贝叶斯预测失败: {e}")
            logger.exit_section(f"贝叶斯变异收益预测: {mutation_strategy}")
            return self._fallback_prediction(mutation_strategy, current_accuracy)
    
    def _extract_feature_vector(self, layer_analysis: Dict[str, Any], 
                              current_accuracy: float,
                              model_complexity: Dict[str, float]) -> np.ndarray:
        """提取用于预测的特征向量"""
        
        # 从层分析中提取关键特征
        mutation_prediction = layer_analysis.get('mutation_prediction', {})
        param_analysis = layer_analysis.get('parameter_space_analysis', {})
        leak_assessment = layer_analysis.get('leak_assessment', {})
        
        features = [
            # 基础准确率和改进空间
            current_accuracy,
            1.0 - current_accuracy,  # 改进空间
            
            # 层特性
            mutation_prediction.get('improvement_potential', 0.0),
            param_analysis.get('efficiency_score', 0.0),
            param_analysis.get('utilization_rate', 0.0),
            
            # 漏点特征
            leak_assessment.get('leak_severity', 0.0),
            1.0 if leak_assessment.get('is_leak_point', False) else 0.0,
            
            # 模型复杂度
            model_complexity.get('total_parameters', 0.0) / 1e6,  # 百万参数为单位
            model_complexity.get('layer_depth', 0.0) / 50.0,      # 归一化深度
            model_complexity.get('layer_width', 0.0) / 1000.0,    # 归一化宽度
            
            # 梯度和激活统计
            mutation_prediction.get('gradient_diversity', 0.0),
            mutation_prediction.get('activation_saturation', 0.5),
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _bayesian_posterior_inference(self, feature_vector: np.ndarray,
                                    mutation_strategy: str,
                                    layer_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """贝叶斯后验推断"""
        
        # 获取先验参数
        mutation_prior = self.prior_knowledge['mutation_success_priors'].get(
            mutation_strategy, {'alpha': 2, 'beta': 2}
        )
        
        # 获取瓶颈类型相关的先验
        leak_assessment = layer_analysis.get('leak_assessment', {})
        leak_type = leak_assessment.get('leak_type', 'general_bottleneck')
        
        bottleneck_prior = self.prior_knowledge['bottleneck_response_priors'].get(
            leak_type, {}
        )
        
        strategy_response = bottleneck_prior.get(f"{mutation_strategy}_response", 0.5)
        
        # 贝叶斯更新：根据观测特征更新先验
        # 使用共轭先验-后验更新
        observed_evidence = self._calculate_evidence_strength(feature_vector)
        
        # Beta分布的共轭更新
        alpha_posterior = mutation_prior['alpha'] + observed_evidence['positive_evidence']
        beta_posterior = mutation_prior['beta'] + observed_evidence['negative_evidence']
        
        # 计算后验成功概率
        success_probability = alpha_posterior / (alpha_posterior + beta_posterior)
        
        # 贝叶斯证据（边际似然）
        evidence = self._calculate_marginal_likelihood(
            feature_vector, mutation_strategy, strategy_response
        )
        
        return {
            'success_probability': success_probability,
            'alpha_posterior': alpha_posterior,
            'beta_posterior': beta_posterior,
            'evidence': evidence,
            'strategy_response': strategy_response
        }
    
    def _calculate_evidence_strength(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """计算观测证据强度"""
        
        # 基于特征向量计算支持和反对变异的证据
        current_acc = feature_vector[0]
        improvement_space = feature_vector[1]
        improvement_potential = feature_vector[2]
        leak_severity = feature_vector[5]
        is_leak_point = feature_vector[6]
        
        # 正面证据：支持变异的因素
        positive_evidence = (
            improvement_space * 2.0 +           # 改进空间大
            improvement_potential * 1.5 +       # 改进潜力高
            leak_severity * 2.0 +               # 漏点严重
            is_leak_point * 1.0                 # 确实是漏点
        )
        
        # 负面证据：反对变异的因素
        negative_evidence = (
            current_acc * 1.0 +                 # 当前准确率已经很高
            (1.0 - improvement_potential) * 1.0 # 改进潜力低
        )
        
        return {
            'positive_evidence': max(0.1, positive_evidence),
            'negative_evidence': max(0.1, negative_evidence)
        }
    
    def _calculate_marginal_likelihood(self, feature_vector: np.ndarray,
                                     mutation_strategy: str,
                                     strategy_response: float) -> float:
        """计算边际似然（贝叶斯证据）"""
        
        # 使用高斯似然函数
        likelihood = 0.0
        
        # 基于特征相似性计算似然
        for historical_mutation in self.mutation_history:
            if historical_mutation['strategy'] == mutation_strategy:
                feature_similarity = self._calculate_feature_similarity(
                    feature_vector, historical_mutation['features']
                )
                
                success = historical_mutation['success']
                likelihood += feature_similarity * (success if success else (1 - success))
        
        # 如果没有历史数据，使用先验响应性
        if likelihood == 0.0:
            likelihood = strategy_response
        
        return float(np.clip(likelihood, 0.01, 0.99))
    
    def _gaussian_process_prediction(self, feature_vector: np.ndarray,
                                   posterior_params: Dict[str, Any]) -> Dict[str, Any]:
        """高斯过程回归预测"""
        
        # 构建核函数（RBF核）
        def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
            return variance * np.exp(-0.5 * np.sum((x1 - x2)**2) / length_scale**2)
        
        # 如果有历史数据，使用GP回归
        if self.mutation_history:
            # 构建训练数据
            X_train = np.array([m['features'] for m in self.mutation_history])
            y_train = np.array([m['actual_gain'] for m in self.mutation_history])
            
            # 计算核矩阵
            n_train = len(X_train)
            K = np.zeros((n_train, n_train))
            
            for i in range(n_train):
                for j in range(n_train):
                    K[i, j] = rbf_kernel(
                        X_train[i], X_train[j],
                        self.gp_hyperparams['length_scale'],
                        self.gp_hyperparams['variance']
                    )
            
            # 添加噪声项
            K += np.eye(n_train) * self.gp_hyperparams['noise_variance']
            
            # 计算预测
            k_star = np.array([
                rbf_kernel(feature_vector, X_train[i],
                          self.gp_hyperparams['length_scale'],
                          self.gp_hyperparams['variance'])
                for i in range(n_train)
            ])
            
            try:
                K_inv = np.linalg.inv(K)
                mean_pred = k_star.T @ K_inv @ y_train
                
                k_star_star = rbf_kernel(
                    feature_vector, feature_vector,
                    self.gp_hyperparams['length_scale'],
                    self.gp_hyperparams['variance']
                )
                
                var_pred = k_star_star - k_star.T @ K_inv @ k_star
                
            except np.linalg.LinAlgError:
                # 如果矩阵奇异，使用伪逆
                K_pinv = np.linalg.pinv(K)
                mean_pred = k_star.T @ K_pinv @ y_train
                var_pred = self.gp_hyperparams['variance']
        
        else:
            # 没有历史数据时，使用先验均值和方差
            mean_pred = posterior_params['success_probability'] * 0.05  # 假设最大收益5%
            var_pred = self.gp_hyperparams['variance']
        
        return {
            'mean_prediction': float(mean_pred),
            'variance_prediction': float(var_pred),
            'std_prediction': float(np.sqrt(var_pred))
        }
    
    def _monte_carlo_benefit_estimation(self, gp_prediction: Dict[str, Any],
                                      mutation_strategy: str,
                                      current_accuracy: float) -> Dict[str, Any]:
        """蒙特卡罗期望收益估计"""
        
        mean = gp_prediction['mean_prediction']
        std = gp_prediction['std_prediction']
        
        # 从预测分布中采样
        samples = np.random.normal(mean, std, self.mc_samples)
        
        # 考虑变异策略的风险特性
        strategy_risk_factor = {
            'widening': 0.9,
            'deepening': 0.8,
            'hybrid_expansion': 0.7,
            'aggressive_widening': 0.6
        }.get(mutation_strategy, 0.8)
        
        # 应用风险调整
        risk_adjusted_samples = samples * strategy_risk_factor
        
        # 确保收益不超过理论上限
        max_possible_gain = min(0.95 - current_accuracy, 0.1)  # 最大收益限制
        risk_adjusted_samples = np.clip(risk_adjusted_samples, -0.02, max_possible_gain)
        
        # 计算统计量
        expected_gain = np.mean(risk_adjusted_samples)
        gain_std = np.std(risk_adjusted_samples)
        
        # 置信区间
        confidence_interval = {
            '95%': (
                np.percentile(risk_adjusted_samples, 2.5),
                np.percentile(risk_adjusted_samples, 97.5)
            ),
            '90%': (
                np.percentile(risk_adjusted_samples, 5),
                np.percentile(risk_adjusted_samples, 95)
            ),
            '68%': (
                np.percentile(risk_adjusted_samples, 16),
                np.percentile(risk_adjusted_samples, 84)
            )
        }
        
        # 成功概率（收益为正的概率）
        success_probability = np.mean(risk_adjusted_samples > 0)
        
        return {
            'expected_gain': float(expected_gain),
            'gain_std': float(gain_std),
            'confidence_interval': confidence_interval,
            'success_probability': float(success_probability),
            'samples': risk_adjusted_samples
        }
    
    def _quantify_prediction_uncertainty(self, gp_prediction: Dict[str, Any],
                                       mc_estimate: Dict[str, Any],
                                       feature_vector: np.ndarray) -> Dict[str, Any]:
        """量化预测不确定性"""
        
        # 认知不确定性（模型不确定性）
        epistemic_uncertainty = gp_prediction['variance_prediction']
        
        # 偶然不确定性（数据噪声）
        aleatoric_uncertainty = mc_estimate['gain_std']**2
        
        # 总不确定性
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # 基于特征的不确定性评估
        feature_uncertainty = self._assess_feature_uncertainty(feature_vector)
        
        # 预测置信度
        prediction_confidence = 1.0 / (1.0 + total_uncertainty)
        
        return {
            'epistemic_uncertainty': float(epistemic_uncertainty),
            'aleatoric_uncertainty': float(aleatoric_uncertainty),
            'total_uncertainty': float(total_uncertainty),
            'feature_uncertainty': feature_uncertainty,
            'prediction_confidence': float(prediction_confidence)
        }
    
    def _calculate_risk_adjusted_benefit(self, mc_estimate: Dict[str, Any],
                                       uncertainty_metrics: Dict[str, Any],
                                       mutation_strategy: str) -> Dict[str, Any]:
        """计算风险调整后的收益"""
        
        expected_gain = mc_estimate['expected_gain']
        total_uncertainty = uncertainty_metrics['total_uncertainty']
        success_prob = mc_estimate['success_probability']
        
        # 风险调整系数
        risk_aversion_factor = 0.5  # 可调参数
        uncertainty_penalty = risk_aversion_factor * total_uncertainty
        
        # 风险调整收益 = 期望收益 - 不确定性惩罚
        risk_adjusted_gain = expected_gain - uncertainty_penalty
        
        # 夏普比率（收益风险比）
        sharpe_ratio = expected_gain / (mc_estimate['gain_std'] + 1e-8)
        
        # 价值风险（VaR）
        var_95 = mc_estimate['confidence_interval']['95%'][0]  # 5%分位数
        
        # 条件价值风险（CVaR）
        samples = mc_estimate['samples']
        cvar_95 = np.mean(samples[samples <= var_95])
        
        return {
            'risk_adjusted_gain': float(risk_adjusted_gain),
            'sharpe_ratio': float(sharpe_ratio),
            'value_at_risk_95': float(var_95),
            'conditional_var_95': float(cvar_95),
            'risk_reward_score': float(expected_gain / (total_uncertainty + 1e-8))
        }
    
    def _calculate_recommendation_strength(self, risk_adjusted_benefit: Dict[str, Any],
                                         uncertainty_metrics: Dict[str, Any]) -> str:
        """计算推荐强度"""
        
        gain = risk_adjusted_benefit['risk_adjusted_gain']
        confidence = uncertainty_metrics['prediction_confidence']
        sharpe_ratio = risk_adjusted_benefit['sharpe_ratio']
        
        # 综合评分
        score = gain * confidence * (1 + sharpe_ratio)
        
        if score > 0.02 and confidence > 0.7:
            return "strong_recommend"
        elif score > 0.01 and confidence > 0.5:
            return "recommend"
        elif score > 0.005:
            return "weak_recommend"
        elif score > -0.005:
            return "neutral"
        else:
            return "not_recommend"
    
    def _assess_feature_uncertainty(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """评估特征不确定性"""
        
        return {
            'accuracy_uncertainty': 0.01,  # 准确率测量误差
            'layer_analysis_uncertainty': 0.1,  # 层分析的不确定性
            'model_complexity_uncertainty': 0.05  # 复杂度估计误差
        }
    
    def _calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """计算特征相似性"""
        
        # 使用余弦相似性
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return float(np.clip(similarity, 0.0, 1.0))
    
    def _fallback_prediction(self, mutation_strategy: str, current_accuracy: float) -> Dict[str, Any]:
        """fallback预测（当贝叶斯预测失败时）"""
        
        # 简单的启发式预测
        base_gain = max(0.01, (0.95 - current_accuracy) * 0.1)
        
        strategy_multipliers = {
            'widening': 0.8,
            'deepening': 0.6,
            'hybrid_expansion': 1.0,
            'aggressive_widening': 1.2
        }
        
        expected_gain = base_gain * strategy_multipliers.get(mutation_strategy, 0.8)
        
        return {
            'expected_accuracy_gain': expected_gain,
            'confidence_interval': {'95%': (0.0, expected_gain * 2)},
            'success_probability': 0.5,
            'risk_adjusted_benefit': {'risk_adjusted_gain': expected_gain * 0.5},
            'uncertainty_metrics': {'prediction_confidence': 0.3},
            'recommendation_strength': "weak_recommend" if expected_gain > 0.005 else "neutral"
        }
    
    def update_with_mutation_result(self, 
                                  feature_vector: np.ndarray,
                                  mutation_strategy: str,
                                  actual_gain: float,
                                  success: bool):
        """用实际变异结果更新模型"""
        
        self.mutation_history.append({
            'features': feature_vector,
            'strategy': mutation_strategy,
            'actual_gain': actual_gain,
            'success': success,
            'timestamp': time.time()
        })
        
        # 保持历史记录大小
        if len(self.mutation_history) > 100:
            self.mutation_history = self.mutation_history[-100:]
        
        logger.info(f"更新贝叶斯模型: {mutation_strategy}, 实际收益={actual_gain:.4f}")

    def predict_optimal_mutation_mode(self, 
                                    layer_analysis: Dict[str, Any],
                                    current_accuracy: float,
                                    model_complexity: Dict[str, float]) -> Dict[str, Any]:
        """
        预测最优变异模式 (Serial vs Parallel vs Hybrid Division)
        
        Args:
            layer_analysis: 层分析结果
            current_accuracy: 当前准确率
            model_complexity: 模型复杂度
            
        Returns:
            各种变异模式的收益预测和推荐
        """
        logger.enter_section("变异模式预测分析")
        
        try:
            leak_assessment = layer_analysis.get('leak_assessment', {})
            leak_type = leak_assessment.get('leak_type', 'general_bottleneck')
            leak_severity = leak_assessment.get('leak_severity', 0.0)
            
            # 确定准确率阶段
            accuracy_stage = self._get_accuracy_stage(current_accuracy)
            
            mode_predictions = {}
            
            # 预测每种变异模式的收益
            for mode_name, mode_config in self.prior_knowledge['mutation_mode_priors'].items():
                # 计算该模式对当前瓶颈类型的适配度
                bottleneck_fit = 1.0 if leak_type in mode_config['best_for'] else 0.6
                
                # 计算该模式对当前准确率阶段的适配度
                accuracy_fit = mode_config['accuracy_preference'][accuracy_stage]
                
                # 计算复杂度适配度
                complexity_fit = self._calculate_complexity_fit(mode_name, model_complexity)
                
                # 贝叶斯后验概率
                alpha = mode_config['success_rate']['alpha']
                beta = mode_config['success_rate']['beta']
                
                # 观测证据调整
                evidence_adjustment = leak_severity * bottleneck_fit * accuracy_fit
                alpha_posterior = alpha + evidence_adjustment
                beta_posterior = beta + (1.0 - evidence_adjustment)
                
                success_probability = alpha_posterior / (alpha_posterior + beta_posterior)
                
                # 期望收益计算
                base_gain = self._calculate_base_mutation_gain(current_accuracy, leak_severity)
                mode_multiplier = self._get_mode_multiplier(mode_name, leak_type, accuracy_stage)
                expected_gain = base_gain * mode_multiplier * success_probability
                
                # 风险评估
                risk_score = self._calculate_mode_risk(mode_name, model_complexity, current_accuracy)
                
                mode_predictions[mode_name] = {
                    'expected_accuracy_gain': float(expected_gain),
                    'success_probability': float(success_probability),
                    'bottleneck_fit': float(bottleneck_fit),
                    'accuracy_stage_fit': float(accuracy_fit),
                    'complexity_fit': float(complexity_fit),
                    'risk_score': float(risk_score),
                    'recommendation_score': float(expected_gain * success_probability / (risk_score + 0.1)),
                    'optimal_for': mode_config['best_for']
                }
            
            # 选择最优模式
            best_mode = max(mode_predictions.items(), 
                          key=lambda x: x[1]['recommendation_score'])
            
            prediction_result = {
                'recommended_mode': best_mode[0],
                'mode_predictions': mode_predictions,
                'confidence': best_mode[1]['success_probability'],
                'expected_improvement': best_mode[1]['expected_accuracy_gain'],
                'reasoning': self._generate_mode_reasoning(best_mode, leak_type, accuracy_stage)
            }
            
            logger.success(f"最优变异模式: {best_mode[0]} (收益={best_mode[1]['expected_accuracy_gain']:.4f})")
            logger.exit_section("变异模式预测分析")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"变异模式预测失败: {e}")
            logger.exit_section("变异模式预测分析")
            return self._fallback_mode_prediction(current_accuracy)

    def predict_optimal_layer_combinations(self, 
                                         layer_analysis: Dict[str, Any],
                                         target_layer_type: str,
                                         mutation_mode: str,
                                         current_accuracy: float) -> Dict[str, Any]:
        """
        预测最优层类型组合 (同种 vs 异种层)
        
        Args:
            layer_analysis: 层分析结果
            target_layer_type: 目标层类型 (conv2d, linear等)
            mutation_mode: 变异模式 (serial_division, parallel_division等)
            current_accuracy: 当前准确率
            
        Returns:
            层类型组合的收益预测和推荐
        """
        logger.enter_section(f"层组合预测: {target_layer_type}")
        
        try:
            leak_assessment = layer_analysis.get('leak_assessment', {})
            leak_type = leak_assessment.get('leak_type', 'general_bottleneck')
            
            # 获取瓶颈类型的首选操作
            preferred_ops = self.prior_knowledge['bottleneck_response_priors'].get(
                leak_type, {}
            ).get('preferred_operations', ['conv2d', 'batch_norm'])
            
            combination_predictions = {}
            
            # 1. 同种层组合预测
            homo_key = f"{target_layer_type}_{target_layer_type}"
            if homo_key in self.prior_knowledge['layer_combination_priors']['homogeneous']:
                homo_config = self.prior_knowledge['layer_combination_priors']['homogeneous'][homo_key]
                homo_prediction = self._predict_combination_benefit(
                    homo_config, target_layer_type, target_layer_type, 
                    leak_type, mutation_mode, current_accuracy, 'homogeneous'
                )
                combination_predictions['homogeneous'] = homo_prediction
            
            # 2. 异种层组合预测
            hetero_predictions = {}
            for operation in preferred_ops:
                if operation != target_layer_type:  # 避免重复
                    hetero_key = f"{target_layer_type}_{operation}"
                    reverse_key = f"{operation}_{target_layer_type}"
                    
                    # 查找配置
                    hetero_config = None
                    final_key = None
                    if hetero_key in self.prior_knowledge['layer_combination_priors']['heterogeneous']:
                        hetero_config = self.prior_knowledge['layer_combination_priors']['heterogeneous'][hetero_key]
                        final_key = hetero_key
                    elif reverse_key in self.prior_knowledge['layer_combination_priors']['heterogeneous']:
                        hetero_config = self.prior_knowledge['layer_combination_priors']['heterogeneous'][reverse_key]
                        final_key = reverse_key
                    
                    if hetero_config:
                        hetero_prediction = self._predict_combination_benefit(
                            hetero_config, target_layer_type, operation,
                            leak_type, mutation_mode, current_accuracy, 'heterogeneous'
                        )
                        hetero_predictions[final_key] = hetero_prediction
            
            combination_predictions['heterogeneous'] = hetero_predictions
            
            # 选择最优组合
            best_combination = self._select_best_combination(combination_predictions)
            
            prediction_result = {
                'recommended_combination': best_combination,
                'combination_predictions': combination_predictions,
                'target_layer_type': target_layer_type,
                'mutation_mode': mutation_mode,
                'detailed_analysis': self._generate_combination_analysis(
                    best_combination, combination_predictions, leak_type
                )
            }
            
            logger.success(f"最优层组合: {best_combination['type']} - {best_combination['combination']}")
            logger.exit_section(f"层组合预测: {target_layer_type}")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"层组合预测失败: {e}")
            logger.exit_section(f"层组合预测: {target_layer_type}")
            return self._fallback_combination_prediction(target_layer_type)

    def predict_comprehensive_mutation_strategy(self,
                                               layer_analysis: Dict[str, Any],
                                               current_accuracy: float,
                                               model: nn.Module,
                                               target_layer_name: str) -> Dict[str, Any]:
        """
        综合预测完整的变异策略
        包括: 变异模式 + 层类型组合 + 具体参数
        """
        logger.enter_section(f"综合变异策略预测: {target_layer_name}")
        
        try:
            model_complexity = self._calculate_model_complexity(model)
            target_layer_type = self._get_layer_type(model, target_layer_name)
            
            # 1. 预测最优变异模式
            mode_prediction = self.predict_optimal_mutation_mode(
                layer_analysis, current_accuracy, model_complexity
            )
            
            # 2. 预测最优层组合
            combination_prediction = self.predict_optimal_layer_combinations(
                layer_analysis, target_layer_type, 
                mode_prediction['recommended_mode'], current_accuracy
            )
            
            # 3. 预测具体参数配置
            parameter_prediction = self._predict_optimal_parameters(
                layer_analysis, mode_prediction['recommended_mode'],
                combination_prediction['recommended_combination'], 
                current_accuracy, model_complexity
            )
            
            # 4. 综合评分和最终推荐
            comprehensive_score = self._calculate_comprehensive_score(
                mode_prediction, combination_prediction, parameter_prediction
            )
            
            final_strategy = {
                'mutation_mode': mode_prediction['recommended_mode'],
                'layer_combination': combination_prediction['recommended_combination'],
                'parameters': parameter_prediction,
                'comprehensive_score': comprehensive_score,
                'expected_total_gain': (
                    mode_prediction['expected_improvement'] *
                    combination_prediction['recommended_combination']['expected_gain']
                ),
                'confidence': min(
                    mode_prediction['confidence'],
                    combination_prediction['recommended_combination']['confidence']
                ),
                'implementation_details': self._generate_implementation_details(
                    mode_prediction, combination_prediction, parameter_prediction
                )
            }
            
            logger.success(f"综合策略: {final_strategy['mutation_mode']} + "
                         f"{final_strategy['layer_combination']['combination']} "
                         f"(总收益={final_strategy['expected_total_gain']:.4f})")
            logger.exit_section(f"综合变异策略预测: {target_layer_name}")
            
            return final_strategy
            
        except Exception as e:
            logger.error(f"综合预测失败: {e}")
            logger.exit_section(f"综合变异策略预测: {target_layer_name}")
            return self._fallback_comprehensive_prediction(target_layer_name)

    def _get_accuracy_stage(self, accuracy: float) -> str:
        """确定准确率阶段"""
        for stage, (low, high) in self.prior_knowledge['accuracy_stage_priors'].items():
            if low <= accuracy < high:
                return stage
        return 'high'

    def _calculate_complexity_fit(self, mode_name: str, model_complexity: Dict[str, float]) -> float:
        """计算复杂度适配度"""
        total_params = model_complexity.get('total_parameters', 0)
        layer_depth = model_complexity.get('layer_depth', 0)
        
        # 不同模式对复杂度的适配性
        if mode_name == 'serial_division':
            # Serial适合深度增加
            return min(1.0, (50 - layer_depth) / 50.0)  # 层数越少越适合
        elif mode_name == 'parallel_division':
            # Parallel适合宽度增加，但需要足够的参数预算
            return min(1.0, total_params / 1e6)  # 参数越多越适合
        else:  # hybrid_division
            # Hybrid适合中等复杂度
            param_fit = 1.0 - abs(total_params / 1e6 - 0.5) * 2  # 0.5M参数最适合
            depth_fit = 1.0 - abs(layer_depth - 25) / 25.0  # 25层最适合
            return (param_fit + depth_fit) / 2.0

    def _calculate_base_mutation_gain(self, current_accuracy: float, leak_severity: float) -> float:
        """计算基础变异收益"""
        # 基础收益与准确率距离上限和漏点严重程度成正比
        headroom = (0.95 - current_accuracy) / 0.95
        base_gain = headroom * 0.1 * (1 + leak_severity)
        return max(0.005, base_gain)  # 最小收益保障

    def _get_mode_multiplier(self, mode_name: str, leak_type: str, accuracy_stage: str) -> float:
        """获取模式收益倍数"""
        mode_config = self.prior_knowledge['mutation_mode_priors'].get(mode_name, {})
        
        # 基础倍数
        base_multiplier = 1.0
        
        # 瓶颈类型适配加成
        if leak_type in mode_config.get('best_for', []):
            base_multiplier *= 1.3
        
        # 准确率阶段适配加成
        stage_fit = mode_config.get('accuracy_preference', {}).get(accuracy_stage, 0.5)
        base_multiplier *= stage_fit
        
        return base_multiplier

    def _calculate_mode_risk(self, mode_name: str, model_complexity: Dict[str, float], 
                           current_accuracy: float) -> float:
        """计算模式风险"""
        base_risk = {
            'serial_division': 0.3,    # 相对稳定
            'parallel_division': 0.5,  # 中等风险
            'hybrid_division': 0.7     # 高风险高收益
        }.get(mode_name, 0.5)
        
        # 高准确率时风险增加
        if current_accuracy > 0.9:
            base_risk *= 1.5
        
        # 高复杂度时风险增加
        if model_complexity.get('total_parameters', 0) > 5e6:
            base_risk *= 1.2
        
        return base_risk

    def _predict_combination_benefit(self, config: Dict[str, float], 
                                   layer1_type: str, layer2_type: str,
                                   leak_type: str, mutation_mode: str,
                                   current_accuracy: float, combo_type: str) -> Dict[str, Any]:
        """预测特定层组合的收益"""
        
        # 基础效果和稳定性
        effectiveness = config.get('effectiveness', 0.5)
        stability = config.get('stability', 0.5)
        
        # 获取层操作特性
        layer1_props = self.prior_knowledge['layer_operation_priors'].get(layer1_type, {})
        layer2_props = self.prior_knowledge['layer_operation_priors'].get(layer2_type, {})
        
        # 计算协同效应
        synergy = self._calculate_layer_synergy(layer1_props, layer2_props, leak_type)
        
        # 计算期望收益
        base_gain = self._calculate_base_mutation_gain(current_accuracy, 0.5)
        expected_gain = base_gain * effectiveness * synergy
        
        # 计算置信度
        confidence = stability * synergy
        
        # 计算实施成本
        implementation_cost = self._calculate_implementation_cost(
            layer1_type, layer2_type, mutation_mode
        )
        
        return {
            'expected_gain': float(expected_gain),
            'confidence': float(confidence),
            'effectiveness': float(effectiveness),
            'stability': float(stability),
            'synergy': float(synergy),
            'implementation_cost': float(implementation_cost),
            'combination': f"{layer1_type}+{layer2_type}",
            'type': combo_type
        }

    def _calculate_layer_synergy(self, layer1_props: Dict[str, float], 
                               layer2_props: Dict[str, float], leak_type: str) -> float:
        """计算层间协同效应"""
        
        # 基础协同分数
        synergy_factors = []
        
        # 特征提取能力协同
        feat_synergy = (layer1_props.get('feature_extraction_boost', 0.5) + 
                       layer2_props.get('feature_extraction_boost', 0.5)) / 2
        synergy_factors.append(feat_synergy)
        
        # 参数效率协同
        param_synergy = (layer1_props.get('parameter_efficiency', 0.5) + 
                        layer2_props.get('parameter_efficiency', 0.5)) / 2
        synergy_factors.append(param_synergy)
        
        # 计算成本协同
        cost_synergy = 1.0 - abs(layer1_props.get('computation_cost', 0.5) - 
                                layer2_props.get('computation_cost', 0.5))
        synergy_factors.append(cost_synergy)
        
        # 特殊能力互补
        special_abilities = ['stability_boost', 'overfitting_prevention', 
                           'long_range_dependency', 'gradient_flow']
        complementary_bonus = 0.0
        
        for ability in special_abilities:
            if (ability in layer1_props and ability not in layer2_props) or \
               (ability not in layer1_props and ability in layer2_props):
                complementary_bonus += 0.1
        
        base_synergy = np.mean(synergy_factors)
        final_synergy = min(1.0, base_synergy + complementary_bonus)
        
        return final_synergy

    def _calculate_implementation_cost(self, layer1_type: str, layer2_type: str, 
                                     mutation_mode: str) -> float:
        """计算实施成本"""
        
        # 基础成本
        layer_costs = {
            'conv2d': 0.6, 'linear': 0.4, 'batch_norm': 0.2,
            'dropout': 0.1, 'attention': 0.8, 'pool': 0.2,
            'depthwise_conv': 0.5, 'residual_connection': 0.7
        }
        
        cost1 = layer_costs.get(layer1_type, 0.5)
        cost2 = layer_costs.get(layer2_type, 0.5)
        
        # 组合成本
        if layer1_type == layer2_type:
            combo_cost = cost1 * 1.5  # 同种层复制成本较低
        else:
            combo_cost = cost1 + cost2  # 异种层需要更多适配
        
        # 模式成本
        mode_cost_multiplier = {
            'serial_division': 1.0,
            'parallel_division': 1.3,
            'hybrid_division': 1.5
        }.get(mutation_mode, 1.0)
        
        return combo_cost * mode_cost_multiplier

    def _select_best_combination(self, combination_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """选择最佳层组合"""
        
        best_combo = None
        best_score = -1.0
        
        # 评估同种层组合
        if 'homogeneous' in combination_predictions:
            homo = combination_predictions['homogeneous']
            score = (homo['expected_gain'] * homo['confidence']) / (homo['implementation_cost'] + 0.1)
            if score > best_score:
                best_score = score
                best_combo = homo
        
        # 评估异种层组合
        if 'heterogeneous' in combination_predictions:
            for combo_name, hetero in combination_predictions['heterogeneous'].items():
                score = (hetero['expected_gain'] * hetero['confidence']) / (hetero['implementation_cost'] + 0.1)
                if score > best_score:
                    best_score = score
                    best_combo = hetero
        
        return best_combo if best_combo else {'type': 'fallback', 'expected_gain': 0.01}

    def _get_layer_type(self, model: nn.Module, layer_name: str) -> str:
        """获取层类型"""
        try:
            module = dict(model.named_modules())[layer_name]
            if isinstance(module, nn.Conv2d):
                return 'conv2d'
            elif isinstance(module, nn.Linear):
                return 'linear'
            elif isinstance(module, nn.BatchNorm2d):
                return 'batch_norm'
            elif isinstance(module, nn.Dropout):
                return 'dropout'
            else:
                return 'unknown'
        except:
            return 'unknown'

    def _predict_optimal_parameters(self, layer_analysis: Dict[str, Any], 
                                  mutation_mode: str, best_combination: Dict[str, Any],
                                  current_accuracy: float, model_complexity: Dict[str, float]) -> Dict[str, Any]:
        """预测最优参数配置"""
        
        # 基于变异模式和层组合预测参数
        params = {
            'parameter_scaling_factor': 1.5,  # 默认参数扩展因子
            'depth_increase': 1,              # 深度增加
            'width_multiplier': 1.0,          # 宽度倍数
            'learning_rate_adjustment': 1.0    # 学习率调整
        }
        
        # 根据变异模式调整
        if mutation_mode == 'serial_division':
            params['depth_increase'] = 2
            params['parameter_scaling_factor'] = 1.3
        elif mutation_mode == 'parallel_division':
            params['width_multiplier'] = 2.0
            params['parameter_scaling_factor'] = 1.8
        else:  # hybrid_division
            params['depth_increase'] = 1
            params['width_multiplier'] = 1.5
            params['parameter_scaling_factor'] = 2.0
        
        # 根据当前准确率调整
        if current_accuracy > 0.9:
            # 高准确率时更保守
            params['parameter_scaling_factor'] *= 0.8
            params['learning_rate_adjustment'] = 0.5
        
        return params

    def _calculate_comprehensive_score(self, mode_pred: Dict[str, Any], 
                                     combo_pred: Dict[str, Any], 
                                     param_pred: Dict[str, Any]) -> float:
        """计算综合评分"""
        
        mode_score = mode_pred['expected_improvement'] * mode_pred['confidence']
        combo_score = combo_pred['recommended_combination']['expected_gain'] * \
                     combo_pred['recommended_combination']['confidence']
        
        # 参数复杂度惩罚
        param_penalty = param_pred['parameter_scaling_factor'] * 0.1
        
        comprehensive_score = (mode_score + combo_score) / 2.0 - param_penalty
        
        return max(0.0, comprehensive_score)

    def _generate_implementation_details(self, mode_pred: Dict[str, Any], 
                                       combo_pred: Dict[str, Any], 
                                       param_pred: Dict[str, Any]) -> Dict[str, Any]:
        """生成实施细节"""
        
        return {
            'mutation_sequence': self._plan_mutation_sequence(mode_pred, combo_pred),
            'parameter_adjustments': param_pred,
            'expected_timeline': self._estimate_implementation_time(mode_pred, combo_pred),
            'resource_requirements': self._estimate_resource_needs(param_pred),
            'rollback_strategy': self._plan_rollback_strategy(mode_pred, combo_pred)
        }

    def _plan_mutation_sequence(self, mode_pred: Dict[str, Any], combo_pred: Dict[str, Any]) -> List[str]:
        """规划变异序列"""
        return [
            f"1. 准备{mode_pred['recommended_mode']}变异",
            f"2. 实施{combo_pred['recommended_combination']['combination']}层组合",
            "3. 参数初始化和微调",
            "4. 渐进式训练验证"
        ]

    def _estimate_implementation_time(self, mode_pred: Dict[str, Any], combo_pred: Dict[str, Any]) -> str:
        """估算实施时间"""
        base_time = 10  # 基础10个epoch
        
        if mode_pred['recommended_mode'] == 'hybrid_division':
            base_time *= 1.5
        
        if combo_pred['recommended_combination']['type'] == 'heterogeneous':
            base_time *= 1.2
        
        return f"{int(base_time)} epochs"

    def _estimate_resource_needs(self, param_pred: Dict[str, Any]) -> Dict[str, float]:
        """估算资源需求"""
        scaling = param_pred['parameter_scaling_factor']
        
        return {
            'memory_increase': scaling * 1.2,
            'computation_increase': scaling * 1.5,
            'storage_increase': scaling * 1.1
        }

    def _plan_rollback_strategy(self, mode_pred: Dict[str, Any], combo_pred: Dict[str, Any]) -> List[str]:
        """规划回滚策略"""
        return [
            "1. 保存变异前模型检查点",
            "2. 监控关键性能指标",
            "3. 设置性能下降阈值 (2%)",
            "4. 自动回滚机制"
        ]

    def _fallback_mode_prediction(self, current_accuracy: float) -> Dict[str, Any]:
        """模式预测fallback"""
        return {
            'recommended_mode': 'serial_division',
            'confidence': 0.5,
            'expected_improvement': 0.01,
            'reasoning': 'Fallback to conservative serial division'
        }

    def _fallback_combination_prediction(self, target_layer_type: str) -> Dict[str, Any]:
        """层组合预测fallback"""
        return {
            'recommended_combination': {
                'combination': f"{target_layer_type}+batch_norm",
                'type': 'heterogeneous',
                'expected_gain': 0.005,
                'confidence': 0.4
            }
        }

    def _fallback_comprehensive_prediction(self, target_layer_name: str) -> Dict[str, Any]:
        """综合预测fallback"""
        return {
            'mutation_mode': 'serial_division',
            'layer_combination': {
                'combination': 'conv2d+batch_norm',
                'type': 'heterogeneous'
            },
            'expected_total_gain': 0.005,
            'confidence': 0.3
        }

    def _generate_mode_reasoning(self, best_mode: tuple, leak_type: str, accuracy_stage: str) -> str:
        """生成模式选择推理"""
        mode_name, mode_data = best_mode
        
        return (f"{mode_name}最适合当前情况: "
               f"瓶颈类型={leak_type}, 准确率阶段={accuracy_stage}, "
               f"期望收益={mode_data['expected_accuracy_gain']:.4f}")

    def _generate_combination_analysis(self, best_combo: Dict[str, Any], 
                                     all_predictions: Dict[str, Any], 
                                     leak_type: str) -> Dict[str, Any]:
        """生成组合分析"""
        return {
            'selected_combination': best_combo['combination'],
            'selection_reason': f"最高综合评分，适合{leak_type}瓶颈",
            'alternative_options': list(all_predictions.get('heterogeneous', {}).keys())[:3],
            'synergy_analysis': f"协同效应评分: {best_combo.get('synergy', 0.5):.3f}"
        }