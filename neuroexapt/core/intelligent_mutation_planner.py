"""
智能变异规划器
基于瓶颈检测结果，结合任务特性和架构特点，制定精确的变异策略
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import copy

from .intelligent_bottleneck_detector import BottleneckReport, BottleneckType

logger = logging.getLogger(__name__)


class MutationType(Enum):
    """变异类型分类"""
    # 容量扩展类
    EXPAND_WIDTH = "expand_width"           # 增加通道数/神经元数
    EXPAND_DEPTH = "expand_depth"           # 增加层数
    EXPAND_CAPACITY = "expand_capacity"     # 综合容量扩展
    
    # 结构优化类  
    ADD_ATTENTION = "add_attention"         # 添加注意力机制
    ADD_RESIDUAL = "add_residual"          # 添加残差连接
    INSERT_BOTTLENECK = "insert_bottleneck" # 插入瓶颈层
    
    # 正则化类
    ADD_NORMALIZATION = "add_normalization" # 添加规范化层
    ADD_DROPOUT = "add_dropout"            # 添加Dropout
    ADD_REGULARIZATION = "add_regularization" # 添加正则化
    
    # 激活函数类
    CHANGE_ACTIVATION = "change_activation" # 变更激活函数
    ADD_GATING = "add_gating"              # 添加门控机制
    
    # 压缩优化类
    FEATURE_SELECTION = "feature_selection" # 特征选择
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction" # 降维
    PRUNING = "pruning"                    # 剪枝
    
    # 连接优化类
    CHANGE_CONNECTIVITY = "change_connectivity" # 改变连接模式
    ADD_SKIP_CONNECTION = "add_skip_connection"  # 添加跳跃连接


@dataclass
class MutationPlan:
    """变异计划"""
    target_layer: str
    mutation_type: MutationType
    parameters: Dict[str, Any]
    priority: float  # 优先级 [0, 1]
    expected_improvement: float  # 预期改进 [0, 1]
    
    # 详细说明
    reasoning: str
    risk_assessment: str
    
    # Net2Net参数迁移相关
    preserve_function: bool = True  # 是否保持功能一致性
    transfer_method: str = "weight_expansion"  # 参数迁移方法


class IntelligentMutationPlanner:
    """
    智能变异规划器
    
    设计理念：
    1. 精确定位：基于瓶颈检测的精确变异位置选择
    2. 策略匹配：根据瓶颈类型选择最适合的变异策略
    3. 参数优化：基于Net2Net思想保证参数平滑迁移
    4. 风险评估：评估变异的潜在风险和收益
    """
    
    def __init__(self):
        # 变异策略映射表
        self.mutation_strategies = self._initialize_mutation_strategies()
        
        # 任务特定的变异权重
        self.task_weights = {
            'vision': {
                'spatial_operations': 1.2,    # 视觉任务偏重空间操作
                'attention_mechanisms': 1.1,
                'channel_operations': 1.0
            },
            'nlp': {
                'sequence_operations': 1.2,   # NLP任务偏重序列操作
                'attention_mechanisms': 1.3,
                'embedding_operations': 1.1
            },
            'graph': {
                'graph_operations': 1.3,     # 图任务偏重图操作
                'aggregation_mechanisms': 1.2,
                'message_passing': 1.1
            }
        }
        
    def _initialize_mutation_strategies(self) -> Dict[BottleneckType, List[MutationType]]:
        """初始化变异策略映射"""
        return {
            BottleneckType.INFORMATION_LEAKAGE: [
                MutationType.EXPAND_CAPACITY,
                MutationType.ADD_ATTENTION,
                MutationType.ADD_RESIDUAL,
                MutationType.INSERT_BOTTLENECK
            ],
            BottleneckType.HIGH_UNCERTAINTY: [
                MutationType.ADD_NORMALIZATION,
                MutationType.ADD_REGULARIZATION,
                MutationType.ADD_DROPOUT,
                MutationType.CHANGE_ACTIVATION
            ],
            BottleneckType.REDUNDANT_FEATURES: [
                MutationType.FEATURE_SELECTION,
                MutationType.DIMENSIONALITY_REDUCTION,
                MutationType.PRUNING,
                MutationType.INSERT_BOTTLENECK
            ],
            BottleneckType.GRADIENT_BOTTLENECK: [
                MutationType.ADD_RESIDUAL,
                MutationType.ADD_SKIP_CONNECTION,
                MutationType.CHANGE_ACTIVATION,
                MutationType.ADD_NORMALIZATION
            ],
            BottleneckType.CAPACITY_BOTTLENECK: [
                MutationType.EXPAND_WIDTH,
                MutationType.EXPAND_DEPTH,
                MutationType.ADD_ATTENTION,
                MutationType.CHANGE_ACTIVATION
            ]
        }
    
    def plan_mutations(self,
                      bottleneck_reports: List[BottleneckReport],
                      model: nn.Module,
                      task_type: str = 'vision',
                      max_mutations: int = 3,
                      risk_tolerance: float = 0.7) -> List[MutationPlan]:
        """
        制定变异计划
        
        Args:
            bottleneck_reports: 瓶颈检测报告列表
            model: 待变异的模型
            task_type: 任务类型 ('vision', 'nlp', 'graph')
            max_mutations: 最大变异数量
            risk_tolerance: 风险容忍度 [0, 1]
            
        Returns:
            变异计划列表，按优先级排序
        """
        logger.info(f"📋 制定智能变异计划，任务类型: {task_type}")
        
        mutation_plans = []
        
        # 对每个瓶颈生成变异计划
        for report in bottleneck_reports[:max_mutations * 2]:  # 生成更多候选，后续筛选
            plans = self._generate_mutation_plans_for_bottleneck(
                report, model, task_type, risk_tolerance
            )
            mutation_plans.extend(plans)
        
        # 按优先级排序并筛选
        mutation_plans.sort(key=lambda x: x.priority * x.expected_improvement, reverse=True)
        
        # 避免同一层多次变异
        filtered_plans = self._filter_conflicting_mutations(mutation_plans)
        
        # 限制数量
        final_plans = filtered_plans[:max_mutations]
        
        logger.info(f"生成 {len(final_plans)} 个变异计划")
        return final_plans
    
    def _generate_mutation_plans_for_bottleneck(self,
                                              report: BottleneckReport,
                                              model: nn.Module,
                                              task_type: str,
                                              risk_tolerance: float) -> List[MutationPlan]:
        """为单个瓶颈生成变异计划"""
        plans = []
        
        # 获取该瓶颈类型对应的变异策略
        candidate_mutations = self.mutation_strategies.get(report.bottleneck_type, [])
        
        for mutation_type in candidate_mutations:
            try:
                plan = self._create_specific_mutation_plan(
                    target_layer=report.layer_name,
                    mutation_type=mutation_type,
                    bottleneck_report=report,
                    model=model,
                    task_type=task_type,
                    risk_tolerance=risk_tolerance
                )
                
                if plan:
                    plans.append(plan)
                    
            except Exception as e:
                logger.warning(f"Failed to create mutation plan {mutation_type} for {report.layer_name}: {e}")
        
        return plans
    
    def _create_specific_mutation_plan(self,
                                     target_layer: str,
                                     mutation_type: MutationType,
                                     bottleneck_report: BottleneckReport,
                                     model: nn.Module,
                                     task_type: str,
                                     risk_tolerance: float) -> Optional[MutationPlan]:
        """创建具体的变异计划"""
        
        # 获取目标层的信息
        layer_info = self._analyze_target_layer(model, target_layer)
        if not layer_info:
            return None
        
        # 根据变异类型生成具体参数
        parameters = self._generate_mutation_parameters(
            mutation_type, layer_info, bottleneck_report, task_type
        )
        
        # 计算优先级和预期改进
        priority = self._calculate_priority(
            mutation_type, bottleneck_report, task_type
        )
        
        expected_improvement = self._estimate_improvement(
            mutation_type, bottleneck_report, layer_info
        )
        
        # 风险评估
        risk_score = self._assess_risk(mutation_type, layer_info, bottleneck_report)
        
        # 风险过高则跳过
        if risk_score > risk_tolerance:
            return None
        
        # 生成推理说明
        reasoning = self._generate_reasoning(
            mutation_type, bottleneck_report, expected_improvement
        )
        
        risk_assessment = self._generate_risk_assessment(risk_score, mutation_type)
        
        return MutationPlan(
            target_layer=target_layer,
            mutation_type=mutation_type,
            parameters=parameters,
            priority=priority,
            expected_improvement=expected_improvement,
            preserve_function=risk_score < 0.5,  # 低风险时保持功能一致性
            transfer_method=self._select_transfer_method(mutation_type, layer_info),
            reasoning=reasoning,
            risk_assessment=risk_assessment
        )
    
    def _analyze_target_layer(self, model: nn.Module, target_layer: str) -> Optional[Dict[str, Any]]:
        """分析目标层的信息"""
        try:
            # 通过层名找到对应的模块
            layer_module = None
            for name, module in model.named_modules():
                if name == target_layer:
                    layer_module = module
                    break
            
            if layer_module is None:
                return None
            
            layer_info = {
                'module': layer_module,
                'type': type(layer_module).__name__,
                'parameters': dict(layer_module.named_parameters()),
                'input_dim': None,
                'output_dim': None
            }
            
            # 获取维度信息
            if hasattr(layer_module, 'in_features'):
                layer_info['input_dim'] = layer_module.in_features
                layer_info['output_dim'] = layer_module.out_features
            elif hasattr(layer_module, 'in_channels'):
                layer_info['input_dim'] = layer_module.in_channels
                layer_info['output_dim'] = layer_module.out_channels
            
            return layer_info
            
        except Exception as e:
            logger.warning(f"Failed to analyze layer {target_layer}: {e}")
            return None
    
    def _generate_mutation_parameters(self,
                                    mutation_type: MutationType,
                                    layer_info: Dict[str, Any],
                                    bottleneck_report: BottleneckReport,
                                    task_type: str) -> Dict[str, Any]:
        """生成变异参数"""
        
        if mutation_type == MutationType.EXPAND_WIDTH:
            # 宽度扩展：增加通道数或神经元数
            current_dim = layer_info.get('output_dim', 128)
            expansion_factor = 1.5 if bottleneck_report.severity > 0.7 else 1.25
            new_dim = int(current_dim * expansion_factor)
            
            return {
                'new_output_dim': new_dim,
                'expansion_factor': expansion_factor,
                'initialization': 'kaiming_normal'
            }
        
        elif mutation_type == MutationType.ADD_ATTENTION:
            # 添加注意力机制
            input_dim = layer_info.get('output_dim', 128)
            
            return {
                'attention_type': 'self_attention',
                'num_heads': 8 if input_dim >= 256 else 4,
                'hidden_dim': input_dim,
                'dropout': 0.1
            }
        
        elif mutation_type == MutationType.ADD_RESIDUAL:
            # 添加残差连接
            return {
                'residual_type': 'additive',
                'use_projection': layer_info.get('input_dim') != layer_info.get('output_dim'),
                'activation': 'relu'
            }
        
        elif mutation_type == MutationType.ADD_NORMALIZATION:
            # 添加规范化层
            norm_type = 'layer_norm' if task_type == 'nlp' else 'batch_norm'
            
            return {
                'norm_type': norm_type,
                'momentum': 0.1,
                'eps': 1e-5,
                'affine': True
            }
        
        elif mutation_type == MutationType.FEATURE_SELECTION:
            # 特征选择
            current_dim = layer_info.get('output_dim', 128)
            reduction_factor = 0.7 if bottleneck_report.uncertainty > 1.0 else 0.8
            new_dim = int(current_dim * reduction_factor)
            
            return {
                'new_output_dim': new_dim,
                'selection_method': 'importance_based',
                'reduction_factor': reduction_factor
            }
        
        elif mutation_type == MutationType.CHANGE_ACTIVATION:
            # 变更激活函数
            current_type = layer_info['type']
            if 'relu' in current_type.lower():
                new_activation = 'gelu'
            else:
                new_activation = 'swish'
            
            return {
                'new_activation': new_activation,
                'inplace': True
            }
        
        else:
            # 默认参数
            return {
                'mutation_type': mutation_type.value,
                'severity': bottleneck_report.severity
            }
    
    def _calculate_priority(self,
                          mutation_type: MutationType,
                          bottleneck_report: BottleneckReport,
                          task_type: str) -> float:
        """计算变异优先级"""
        
        # 基础优先级：基于瓶颈严重程度和置信度
        base_priority = bottleneck_report.severity * bottleneck_report.confidence
        
        # 变异类型权重
        type_weights = {
            MutationType.EXPAND_CAPACITY: 0.9,
            MutationType.ADD_ATTENTION: 0.8,
            MutationType.ADD_RESIDUAL: 0.85,
            MutationType.ADD_NORMALIZATION: 0.7,
            MutationType.FEATURE_SELECTION: 0.6,
            MutationType.CHANGE_ACTIVATION: 0.5
        }
        
        type_weight = type_weights.get(mutation_type, 0.5)
        
        # 任务特定权重
        task_weight = self.task_weights.get(task_type, {}).get(
            self._get_operation_category(mutation_type), 1.0
        )
        
        priority = base_priority * type_weight * task_weight
        return min(priority, 1.0)
    
    def _estimate_improvement(self,
                            mutation_type: MutationType,
                            bottleneck_report: BottleneckReport,
                            layer_info: Dict[str, Any]) -> float:
        """估计预期改进"""
        
        # 基于瓶颈类型和变异类型的匹配度
        type_matching = {
            (BottleneckType.INFORMATION_LEAKAGE, MutationType.EXPAND_CAPACITY): 0.8,
            (BottleneckType.HIGH_UNCERTAINTY, MutationType.ADD_NORMALIZATION): 0.7,
            (BottleneckType.REDUNDANT_FEATURES, MutationType.FEATURE_SELECTION): 0.75,
            (BottleneckType.GRADIENT_BOTTLENECK, MutationType.ADD_RESIDUAL): 0.8,
            (BottleneckType.CAPACITY_BOTTLENECK, MutationType.EXPAND_WIDTH): 0.85
        }
        
        matching_score = type_matching.get(
            (bottleneck_report.bottleneck_type, mutation_type), 0.5
        )
        
        # 考虑瓶颈严重程度
        severity_factor = bottleneck_report.severity
        
        # 考虑层的当前状态
        layer_factor = 1.0
        if 'output_dim' in layer_info:
            # 如果层过小，扩展的改进效果更明显
            if layer_info['output_dim'] < 64 and mutation_type in [
                MutationType.EXPAND_WIDTH, MutationType.EXPAND_CAPACITY
            ]:
                layer_factor = 1.2
        
        improvement = matching_score * severity_factor * layer_factor
        return min(improvement, 1.0)
    
    def _assess_risk(self,
                    mutation_type: MutationType,
                    layer_info: Dict[str, Any],
                    bottleneck_report: BottleneckReport) -> float:
        """评估变异风险"""
        
        # 基础风险：不同变异类型的固有风险
        base_risks = {
            MutationType.ADD_NORMALIZATION: 0.2,  # 低风险
            MutationType.ADD_RESIDUAL: 0.3,
            MutationType.CHANGE_ACTIVATION: 0.4,
            MutationType.EXPAND_WIDTH: 0.5,       # 中等风险
            MutationType.ADD_ATTENTION: 0.6,
            MutationType.FEATURE_SELECTION: 0.7,  # 高风险
            MutationType.EXPAND_DEPTH: 0.8
        }
        
        base_risk = base_risks.get(mutation_type, 0.5)
        
        # 层位置风险：越靠近输出层风险越高
        position_risk = 0.0  # 简化处理，实际可以根据层在网络中的位置计算
        
        # 瓶颈严重程度：严重程度高时，大变异的风险相对较低
        severity_factor = 1 - bottleneck_report.severity * 0.3
        
        total_risk = base_risk * severity_factor + position_risk
        return min(total_risk, 1.0)
    
    def _get_operation_category(self, mutation_type: MutationType) -> str:
        """获取操作类别"""
        spatial_ops = [MutationType.EXPAND_WIDTH, MutationType.ADD_ATTENTION]
        if mutation_type in spatial_ops:
            return 'spatial_operations'
        return 'channel_operations'
    
    def _select_transfer_method(self, mutation_type: MutationType, layer_info: Dict[str, Any]) -> str:
        """选择参数迁移方法"""
        if mutation_type in [MutationType.EXPAND_WIDTH, MutationType.EXPAND_CAPACITY]:
            return 'weight_expansion'
        elif mutation_type in [MutationType.ADD_RESIDUAL, MutationType.ADD_ATTENTION]:
            return 'identity_initialization'
        else:
            return 'fine_tuning'
    
    def _generate_reasoning(self,
                          mutation_type: MutationType,
                          bottleneck_report: BottleneckReport,
                          expected_improvement: float) -> str:
        """生成推理说明"""
        return (f"针对{bottleneck_report.layer_name}层的{bottleneck_report.bottleneck_type.value}问题，"
                f"采用{mutation_type.value}策略，预期改进{expected_improvement:.2f}。"
                f"原因：{bottleneck_report.explanation}")
    
    def _generate_risk_assessment(self, risk_score: float, mutation_type: MutationType) -> str:
        """生成风险评估"""
        if risk_score < 0.3:
            risk_level = "低"
        elif risk_score < 0.6:
            risk_level = "中等"
        else:
            risk_level = "高"
        
        return f"风险等级：{risk_level} ({risk_score:.2f})，{mutation_type.value}操作的预期影响可控"
    
    def _filter_conflicting_mutations(self, mutation_plans: List[MutationPlan]) -> List[MutationPlan]:
        """过滤冲突的变异"""
        seen_layers = set()
        filtered_plans = []
        
        for plan in mutation_plans:
            if plan.target_layer not in seen_layers:
                filtered_plans.append(plan)
                seen_layers.add(plan.target_layer)
        
        return filtered_plans
    
    def visualize_mutation_plans(self, plans: List[MutationPlan]) -> str:
        """可视化变异计划"""
        if not plans:
            return "📋 无变异计划"
        
        visualization = "📋 智能变异计划\n" + "="*50 + "\n"
        
        for i, plan in enumerate(plans, 1):
            priority_icon = "🔥" if plan.priority > 0.7 else "⚡" if plan.priority > 0.4 else "💡"
            
            visualization += f"\n{priority_icon} #{i} 目标层: {plan.target_layer}\n"
            visualization += f"   变异类型: {plan.mutation_type.value}\n"
            visualization += f"   优先级: {plan.priority:.3f} | 预期改进: {plan.expected_improvement:.3f}\n"
            visualization += f"   参数迁移: {plan.transfer_method}\n"
            visualization += f"   推理: {plan.reasoning}\n"
            visualization += f"   风险: {plan.risk_assessment}\n"
        
        return visualization