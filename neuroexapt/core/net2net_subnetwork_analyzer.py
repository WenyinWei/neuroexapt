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