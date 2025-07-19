#!/usr/bin/env python3
"""
defgroup group_aggressive_morphogenesis Aggressive Morphogenesis
ingroup core
Aggressive Morphogenesis module for NeuroExapt framework.
"""

激进多点形态发生系统 - Aggressive Multi-Point Morphogenesis

🎯 专门针对准确率饱和状态的激进架构变异策略
- 多点同步变异
- 反向梯度投影分析
- 参数空间扩展优化
- 动态瓶颈识别与突破
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import logging
import copy

from .logging_utils import logger
from .advanced_morphogenesis import MorphogenesisType, MorphogenesisDecision

@dataclass
class MultiPointMutation:
    """多点变异决策"""
    target_locations: List[str]  # 多个变异位置
    mutation_types: List[MorphogenesisType]  # 对应的变异类型
    coordination_strategy: str  # 协调策略: 'parallel', 'cascade', 'hybrid'
    expected_improvement: float
    risk_assessment: float
    parameter_budget: int

@dataclass
class BottleneckSignature:
    """瓶颈特征签名"""
    layer_name: str
    bottleneck_type: str  # 'gradient_vanishing', 'activation_saturation', 'capacity_limit', 'information_loss'
    severity: float
    upstream_impact: float  # 对上游的影响
    downstream_impact: float  # 对下游的影响
    parameter_efficiency: float  # 参数效率

class AggressiveMorphogenesisAnalyzer:
    """激进形态发生分析器"""
    
    def __init__(self, 
                 accuracy_plateau_threshold: float = 0.1,  # 准确率停滞阈值
                 plateau_window: int = 5,  # 停滞检测窗口
                 aggressive_trigger_threshold: float = 0.05):  # 激进触发阈值
        self.accuracy_plateau_threshold = accuracy_plateau_threshold
        self.plateau_window = plateau_window
        self.aggressive_trigger_threshold = aggressive_trigger_threshold
        self.accuracy_history = []
        self.bottleneck_history = []
        
    def detect_accuracy_plateau(self, performance_history: List[float]) -> Tuple[bool, float]:
        """检测准确率停滞状态"""
        if len(performance_history) < self.plateau_window:
            return False, 0.0
            
        recent_performance = performance_history[-self.plateau_window:]
        improvement = max(recent_performance) - min(recent_performance)
        
        is_plateau = improvement < self.accuracy_plateau_threshold
        stagnation_severity = 1.0 - (improvement / self.accuracy_plateau_threshold)
        
        logger.info(f"准确率停滞检测: 改进={improvement:.4f}, 阈值={self.accuracy_plateau_threshold:.4f}, 严重程度={stagnation_severity:.4f}")
        
        return is_plateau, stagnation_severity
    
    def analyze_reverse_gradient_projection(self, 
                                          activations: Dict[str, torch.Tensor],
                                          gradients: Dict[str, torch.Tensor],
                                          output_targets: torch.Tensor) -> Dict[str, BottleneckSignature]:
        """反向梯度投影分析 - 从输出反推关键瓶颈层"""
        logger.enter_section("反向梯度投影分析")
        
        bottleneck_signatures = {}
        layer_names = list(activations.keys())
        
        # 计算输出层的梯度强度作为基准
        output_layer = layer_names[-1] if layer_names else None
        if not output_layer or output_layer not in gradients:
            logger.warning("无法找到输出层梯度，跳过反向投影分析")
            logger.exit_section("反向梯度投影分析")
            return bottleneck_signatures
            
        output_grad_intensity = torch.norm(gradients[output_layer]).item()
        logger.info(f"输出层梯度强度基准: {output_grad_intensity:.6f}")
        
        # 从后向前分析每一层
        for i, layer_name in enumerate(reversed(layer_names)):
            if layer_name not in gradients or layer_name not in activations:
                continue
                
            gradient = gradients[layer_name]
            activation = activations[layer_name]
            
            # 1. 梯度消失/爆炸检测
            grad_norm = torch.norm(gradient).item()
            grad_ratio = grad_norm / (output_grad_intensity + 1e-8)
            
            # 2. 激活饱和度分析
            activation_flat = activation.flatten()
            saturation_ratio = self._compute_saturation_ratio(activation_flat)
            
            # 3. 信息传递效率
            info_efficiency = self._compute_information_efficiency(activation, gradient)
            
            # 4. 参数空间利用率
            param_efficiency = self._compute_parameter_efficiency(layer_name, activation, gradient)
            
            # 综合瓶颈严重程度评估
            bottleneck_severity = self._assess_bottleneck_severity(
                grad_ratio, saturation_ratio, info_efficiency, param_efficiency
            )
            
            # 影响范围评估
            layer_index = len(layer_names) - 1 - i
            upstream_impact = layer_index / len(layer_names)  # 越靠前影响越大
            downstream_impact = 1.0 - upstream_impact
            
            signature = BottleneckSignature(
                layer_name=layer_name,
                bottleneck_type=self._classify_bottleneck_type(grad_ratio, saturation_ratio, info_efficiency),
                severity=bottleneck_severity,
                upstream_impact=upstream_impact,
                downstream_impact=downstream_impact,
                parameter_efficiency=param_efficiency
            )
            
            bottleneck_signatures[layer_name] = signature
            
            logger.debug(f"层{layer_name}: 严重程度={bottleneck_severity:.3f}, "
                        f"梯度比={grad_ratio:.6f}, 饱和度={saturation_ratio:.3f}, "
                        f"信息效率={info_efficiency:.3f}")
        
        logger.info(f"识别出{len(bottleneck_signatures)}个潜在瓶颈层")
        logger.exit_section("反向梯度投影分析")
        
        return bottleneck_signatures
    
    def _compute_saturation_ratio(self, activation_flat: torch.Tensor) -> float:
        """计算激活饱和比例"""
        if activation_flat.numel() == 0:
            return 0.0
            
        # 检测接近0或接近极值的激活
        near_zero = torch.abs(activation_flat) < 0.01
        near_max = activation_flat > 0.99 * activation_flat.max()
        near_min = activation_flat < 0.99 * activation_flat.min()
        
        saturated = near_zero | near_max | near_min
        return saturated.float().mean().item()
    
    def _compute_information_efficiency(self, activation: torch.Tensor, gradient: torch.Tensor) -> float:
        """计算信息传递效率"""
        try:
            # 计算激活的信息熵
            activation_flat = activation.flatten()
            if activation_flat.numel() < 2:
                return 0.0
                
            # 使用直方图估计信息熵
            hist, _ = torch.histogram(activation_flat, bins=50)
            hist = hist.float()
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # 移除零值
            
            entropy = -(hist * torch.log(hist)).sum().item()
            
            # 梯度信息量
            grad_flat = gradient.flatten()
            grad_var = torch.var(grad_flat).item()
            
            # 信息效率 = 熵 × 梯度方差
            return entropy * grad_var
            
        except Exception:
            return 0.0
    
    def _compute_parameter_efficiency(self, layer_name: str, activation: torch.Tensor, gradient: torch.Tensor) -> float:
        """计算参数效率 - 参数产生的信息量与参数数量的比值"""
        try:
            # 估算该层的参数数量（基于激活形状推断）
            if len(activation.shape) == 4:  # Conv2D
                param_estimate = activation.shape[1] * 9  # 假设3x3卷积核
            elif len(activation.shape) == 2:  # Linear
                param_estimate = activation.shape[1] * 1000  # 粗略估计
            else:
                param_estimate = activation.numel()
            
            # 计算信息产出
            grad_norm = torch.norm(gradient).item()
            activation_norm = torch.norm(activation).item()
            information_output = grad_norm * activation_norm
            
            # 参数效率
            efficiency = information_output / (param_estimate + 1e-8)
            return min(efficiency, 10.0)  # 限制上界
            
        except Exception:
            return 0.0
    
    def _assess_bottleneck_severity(self, grad_ratio: float, saturation_ratio: float, 
                                  info_efficiency: float, param_efficiency: float) -> float:
        """综合评估瓶颈严重程度"""
        # 梯度问题权重
        grad_problem = 1.0 if grad_ratio < 0.01 or grad_ratio > 100 else 0.0
        
        # 饱和问题权重  
        saturation_problem = saturation_ratio
        
        # 效率问题权重
        efficiency_problem = 1.0 - min(info_efficiency / 0.1, 1.0)
        param_problem = 1.0 - min(param_efficiency / 0.1, 1.0)
        
        # 加权综合
        severity = (
            0.3 * grad_problem +
            0.3 * saturation_problem +
            0.2 * efficiency_problem +
            0.2 * param_problem
        )
        
        return min(severity, 1.0)
    
    def _classify_bottleneck_type(self, grad_ratio: float, saturation_ratio: float, info_efficiency: float) -> str:
        """分类瓶颈类型"""
        if grad_ratio < 0.01:
            return 'gradient_vanishing'
        elif grad_ratio > 100:
            return 'gradient_exploding'
        elif saturation_ratio > 0.7:
            return 'activation_saturation'
        elif info_efficiency < 0.01:
            return 'information_loss'
        else:
            return 'capacity_limit'

class MultiPointMutationPlanner:
    """多点变异规划器"""
    
    def __init__(self, max_concurrent_mutations: int = 3, parameter_budget: int = 10000):
        self.max_concurrent_mutations = max_concurrent_mutations
        self.parameter_budget = parameter_budget
    
    def plan_aggressive_mutations(self, 
                                bottleneck_signatures: Dict[str, BottleneckSignature],
                                performance_history: List[float],
                                stagnation_severity: float) -> List[MultiPointMutation]:
        """规划激进的多点变异策略"""
        logger.enter_section("多点变异规划")
        
        mutations = []
        
        # 根据停滞严重程度决定激进程度
        max_mutations = min(
            self.max_concurrent_mutations,
            int(stagnation_severity * 5) + 1  # 停滞越严重，变异点越多
        )
        
        logger.info(f"停滞严重程度: {stagnation_severity:.3f}, 计划变异点数: {max_mutations}")
        
        # 按瓶颈严重程度排序
        sorted_bottlenecks = sorted(
            bottleneck_signatures.items(),
            key=lambda x: x[1].severity * (x[1].upstream_impact + x[1].downstream_impact),
            reverse=True
        )
        
        # 策略1: 关键瓶颈层的密集变异
        if max_mutations >= 2:
            mutations.extend(self._plan_dense_mutations(sorted_bottlenecks[:max_mutations]))
        
        # 策略2: 跨层级的协调变异
        if max_mutations >= 3 and len(sorted_bottlenecks) >= 3:
            mutations.extend(self._plan_coordinated_mutations(sorted_bottlenecks))
        
        # 策略3: 激进的架构重构
        if stagnation_severity > 0.8:
            mutations.extend(self._plan_radical_restructuring(sorted_bottlenecks))
        
        logger.info(f"规划了{len(mutations)}个多点变异策略")
        logger.exit_section("多点变异规划")
        
        return mutations
    
    def _plan_dense_mutations(self, bottlenecks: List[Tuple[str, BottleneckSignature]]) -> List[MultiPointMutation]:
        """密集变异策略 - 在关键层进行多种类型的同步变异"""
        mutations = []
        
        for layer_name, signature in bottlenecks[:2]:  # 选择前2个最严重的瓶颈
            target_locations = [layer_name]
            mutation_types = []
            
            # 根据瓶颈类型选择合适的变异
            if signature.bottleneck_type == 'gradient_vanishing':
                mutation_types = [MorphogenesisType.SERIAL_DIVISION, MorphogenesisType.WIDTH_EXPANSION]
            elif signature.bottleneck_type == 'activation_saturation':
                mutation_types = [MorphogenesisType.PARALLEL_DIVISION, MorphogenesisType.HYBRID_DIVISION]
            elif signature.bottleneck_type == 'capacity_limit':
                mutation_types = [MorphogenesisType.WIDTH_EXPANSION, MorphogenesisType.SERIAL_DIVISION]
            else:
                mutation_types = [MorphogenesisType.HYBRID_DIVISION]
            
            mutation = MultiPointMutation(
                target_locations=target_locations,
                mutation_types=mutation_types,
                coordination_strategy='parallel',
                expected_improvement=signature.severity * 0.1,
                risk_assessment=0.3,
                parameter_budget=self.parameter_budget // 2
            )
            mutations.append(mutation)
        
        return mutations
    
    def _plan_coordinated_mutations(self, bottlenecks: List[Tuple[str, BottleneckSignature]]) -> List[MultiPointMutation]:
        """协调变异策略 - 多层同步变异以维持信息流"""
        mutations = []
        
        # 选择分布在不同深度的层
        early_layers = [b for b in bottlenecks if b[1].upstream_impact > 0.7]
        middle_layers = [b for b in bottlenecks if 0.3 <= b[1].upstream_impact <= 0.7]
        late_layers = [b for b in bottlenecks if b[1].upstream_impact < 0.3]
        
        if early_layers and late_layers:
            # 早期层扩展容量，后期层增强表达
            target_locations = [early_layers[0][0], late_layers[0][0]]
            mutation_types = [MorphogenesisType.WIDTH_EXPANSION, MorphogenesisType.SERIAL_DIVISION]
            
            mutation = MultiPointMutation(
                target_locations=target_locations,
                mutation_types=mutation_types,
                coordination_strategy='cascade',
                expected_improvement=0.15,
                risk_assessment=0.4,
                parameter_budget=self.parameter_budget
            )
            mutations.append(mutation)
        
        return mutations
    
    def _plan_radical_restructuring(self, bottlenecks: List[Tuple[str, BottleneckSignature]]) -> List[MultiPointMutation]:
        """激进重构策略 - 大幅度架构变异"""
        mutations = []
        
        # 选择影响最大的前3层进行激进变异
        top_bottlenecks = bottlenecks[:3]
        target_locations = [b[0] for b in top_bottlenecks]
        
        # 混合使用所有变异类型
        mutation_types = [
            MorphogenesisType.HYBRID_DIVISION,
            MorphogenesisType.PARALLEL_DIVISION,
            MorphogenesisType.WIDTH_EXPANSION
        ]
        
        mutation = MultiPointMutation(
            target_locations=target_locations,
            mutation_types=mutation_types,
            coordination_strategy='hybrid',
            expected_improvement=0.3,  # 高期望，但风险也高
            risk_assessment=0.7,
            parameter_budget=self.parameter_budget * 2  # 允许超预算
        )
        mutations.append(mutation)
        
        return mutations

class AggressiveMorphogenesisExecutor:
    """激进形态发生执行器"""
    
    def __init__(self):
        self.execution_history = []
    
    def execute_multi_point_mutation(self, 
                                   model: nn.Module,
                                   mutation: MultiPointMutation) -> Tuple[nn.Module, int, Dict]:
        """执行多点变异"""
        logger.enter_section(f"多点变异执行: {mutation.coordination_strategy}")
        
        try:
            if mutation.coordination_strategy == 'parallel':
                return self._execute_parallel_mutations(model, mutation)
            elif mutation.coordination_strategy == 'cascade':
                return self._execute_cascade_mutations(model, mutation)
            elif mutation.coordination_strategy == 'hybrid':
                return self._execute_hybrid_mutations(model, mutation)
            else:
                logger.error(f"未知的协调策略: {mutation.coordination_strategy}")
                return model, 0, {'error': 'unknown_strategy'}
                
        except Exception as e:
            logger.error(f"多点变异执行失败: {e}")
            return model, 0, {'error': str(e)}
        finally:
            logger.exit_section(f"多点变异执行: {mutation.coordination_strategy}")
    
    def _execute_parallel_mutations(self, model: nn.Module, mutation: MultiPointMutation) -> Tuple[nn.Module, int, Dict]:
        """并行执行多个变异 - 同时在多个位置进行独立变异"""
        logger.info(f"并行变异: {len(mutation.target_locations)}个位置")
        
        new_model = copy.deepcopy(model)
        total_params_added = 0
        execution_details = []
        
        for i, (location, morph_type) in enumerate(zip(mutation.target_locations, mutation.mutation_types)):
            try:
                # 为每个位置执行独立的变异
                from .advanced_morphogenesis import AdvancedMorphogenesisExecutor
                executor = AdvancedMorphogenesisExecutor()
                
                decision = MorphogenesisDecision(
                    morphogenesis_type=morph_type,
                    target_location=location,
                    confidence=0.8,
                    expected_improvement=mutation.expected_improvement / len(mutation.target_locations),
                    complexity_cost=0.3,
                    parameters_added=mutation.parameter_budget // len(mutation.target_locations),
                    reasoning=f"并行变异{i+1}: {morph_type.value}"
                )
                
                new_model, params_added = executor.execute_morphogenesis(new_model, decision)
                total_params_added += params_added
                
                execution_details.append({
                    'location': location,
                    'type': morph_type.value,
                    'params_added': params_added,
                    'success': True
                })
                
                logger.info(f"位置{location}变异成功: +{params_added}参数")
                
            except Exception as e:
                logger.warning(f"位置{location}变异失败: {e}")
                execution_details.append({
                    'location': location,
                    'type': morph_type.value,
                    'error': str(e),
                    'success': False
                })
        
        result = {
            'strategy': 'parallel',
            'total_mutations': len(mutation.target_locations),
            'successful_mutations': sum(1 for d in execution_details if d.get('success', False)),
            'execution_details': execution_details
        }
        
        return new_model, total_params_added, result
    
    def _execute_cascade_mutations(self, model: nn.Module, mutation: MultiPointMutation) -> Tuple[nn.Module, int, Dict]:
        """级联执行变异 - 按深度顺序依次变异，后续变异考虑前面的影响"""
        logger.info(f"级联变异: {len(mutation.target_locations)}个位置")
        
        new_model = copy.deepcopy(model)
        total_params_added = 0
        execution_details = []
        
        # 按层的深度排序（假设层名包含位置信息）
        sorted_mutations = list(zip(mutation.target_locations, mutation.mutation_types))
        
        for i, (location, morph_type) in enumerate(sorted_mutations):
            try:
                from .advanced_morphogenesis import AdvancedMorphogenesisExecutor
                executor = AdvancedMorphogenesisExecutor()
                
                # 级联变异中，后续变异的参数预算会根据前面的结果调整
                remaining_budget = mutation.parameter_budget - total_params_added
                adjusted_budget = max(remaining_budget // (len(sorted_mutations) - i), 1000)
                
                decision = MorphogenesisDecision(
                    morphogenesis_type=morph_type,
                    target_location=location,
                    confidence=0.7,  # 级联变异风险稍高
                    expected_improvement=mutation.expected_improvement * (1.2 ** i),  # 后续变异期望更高
                    complexity_cost=0.4,
                    parameters_added=adjusted_budget,
                    reasoning=f"级联变异{i+1}: {morph_type.value}"
                )
                
                new_model, params_added = executor.execute_morphogenesis(new_model, decision)
                total_params_added += params_added
                
                execution_details.append({
                    'location': location,
                    'type': morph_type.value,
                    'params_added': params_added,
                    'cascade_order': i + 1,
                    'success': True
                })
                
                logger.info(f"级联{i+1}({location})变异成功: +{params_added}参数")
                
            except Exception as e:
                logger.warning(f"级联{i+1}({location})变异失败: {e}")
                execution_details.append({
                    'location': location,
                    'type': morph_type.value,
                    'cascade_order': i + 1,
                    'error': str(e),
                    'success': False
                })
                # 级联变异中，如果某一步失败，继续执行后续步骤
        
        result = {
            'strategy': 'cascade',
            'total_mutations': len(mutation.target_locations),
            'successful_mutations': sum(1 for d in execution_details if d.get('success', False)),
            'execution_details': execution_details
        }
        
        return new_model, total_params_added, result
    
    def _execute_hybrid_mutations(self, model: nn.Module, mutation: MultiPointMutation) -> Tuple[nn.Module, int, Dict]:
        """混合变异策略 - 结合并行和级联的优势"""
        logger.info(f"混合变异: {len(mutation.target_locations)}个位置")
        
        # 将变异分为两组：并行组和级联组
        mid_point = len(mutation.target_locations) // 2
        parallel_group = list(zip(mutation.target_locations[:mid_point], mutation.mutation_types[:mid_point]))
        cascade_group = list(zip(mutation.target_locations[mid_point:], mutation.mutation_types[mid_point:]))
        
        new_model = copy.deepcopy(model)
        total_params_added = 0
        execution_details = []
        
        # 先执行并行组
        if parallel_group:
            parallel_mutation = MultiPointMutation(
                target_locations=[loc for loc, _ in parallel_group],
                mutation_types=[mt for _, mt in parallel_group],
                coordination_strategy='parallel',
                expected_improvement=mutation.expected_improvement * 0.6,
                risk_assessment=mutation.risk_assessment,
                parameter_budget=mutation.parameter_budget // 2
            )
            
            new_model, parallel_params, parallel_result = self._execute_parallel_mutations(new_model, parallel_mutation)
            total_params_added += parallel_params
            execution_details.extend(parallel_result['execution_details'])
        
        # 再执行级联组
        if cascade_group:
            cascade_mutation = MultiPointMutation(
                target_locations=[loc for loc, _ in cascade_group],
                mutation_types=[mt for _, mt in cascade_group],
                coordination_strategy='cascade',
                expected_improvement=mutation.expected_improvement * 0.4,
                risk_assessment=mutation.risk_assessment,
                parameter_budget=mutation.parameter_budget - total_params_added
            )
            
            new_model, cascade_params, cascade_result = self._execute_cascade_mutations(new_model, cascade_mutation)
            total_params_added += cascade_params
            execution_details.extend(cascade_result['execution_details'])
        
        result = {
            'strategy': 'hybrid',
            'total_mutations': len(mutation.target_locations),
            'successful_mutations': sum(1 for d in execution_details if d.get('success', False)),
            'parallel_mutations': len(parallel_group),
            'cascade_mutations': len(cascade_group),
            'execution_details': execution_details
        }
        
        return new_model, total_params_added, result