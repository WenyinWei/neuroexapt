#!/usr/bin/env python3
"""
Enhanced Dynamic Neural Morphogenesis (DNM) Framework - 增强版

🧬 核心改进：
1. 多维度瓶颈分析 - 深度、宽度、信息流、梯度流、容量瓶颈
2. 高级形态发生策略 - 串行分裂、并行分裂、混合分裂
3. 智能决策制定 - 基于瓶颈类型的最优策略选择
4. 性能导向 - 追求更高的准确率突破

🎯 目标：实现90%+的准确率，探索网络结构的无限可能
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

# 导入高级形态发生模块
from .advanced_morphogenesis import (
    AdvancedBottleneckAnalyzer,
    AdvancedMorphogenesisExecutor,
    IntelligentMorphogenesisDecisionMaker,
    MorphogenesisType,
    MorphogenesisDecision
)

# 配置日志
logger = logging.getLogger(__name__)

@dataclass
class EnhancedMorphogenesisEvent:
    """增强的形态发生事件记录"""
    epoch: int
    event_type: str  # 'width_expansion', 'serial_division', 'parallel_division', 'hybrid_division'
    location: str
    trigger_reason: str
    performance_before: float
    performance_after: Optional[float] = None
    parameters_added: int = 0
    complexity_change: float = 0.0
    morphogenesis_type: MorphogenesisType = MorphogenesisType.WIDTH_EXPANSION
    confidence: float = 0.0
    expected_improvement: float = 0.0

class EnhancedInformationTheoryTrigger:
    """增强的信息论触发器"""
    
    def __init__(self, entropy_threshold: float = 0.1, complexity_threshold: float = 0.7):
        self.entropy_threshold = entropy_threshold
        self.complexity_threshold = complexity_threshold
        self.history = deque(maxlen=15)
        
    def should_trigger(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        activations = context.get('activations', {})
        gradients = context.get('gradients', {})
        
        if not activations or not gradients:
            return False, "缺少激活值或梯度信息"
            
        # 计算综合复杂度分数
        complexity_score = self._compute_complexity_score(activations, gradients)
        
        self.history.append({
            'complexity_score': complexity_score,
            'epoch': context.get('epoch', 0)
        })
        
        # 检查是否需要更复杂的结构变异
        if complexity_score > self.complexity_threshold:
            return True, f"复杂度瓶颈检测：分数={complexity_score:.4f}"
            
        return False, "复杂度指标未达到触发条件"
    
    def _compute_complexity_score(self, activations: Dict[str, torch.Tensor], 
                                gradients: Dict[str, torch.Tensor]) -> float:
        """计算网络复杂度分数"""
        scores = []
        
        for name, activation in activations.items():
            if name not in gradients or gradients[name] is None:
                continue
                
            gradient = gradients[name]
            
            # 1. 信息熵分析
            entropy = self._compute_entropy(activation)
            
            # 2. 梯度复杂度
            grad_complexity = self._compute_gradient_complexity(gradient)
            
            # 3. 激活模式复杂度
            activation_complexity = self._compute_activation_complexity(activation)
            
            # 综合分数
            layer_score = 0.4 * entropy + 0.3 * grad_complexity + 0.3 * activation_complexity
            scores.append(layer_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _compute_entropy(self, activation: torch.Tensor) -> float:
        """计算激活值熵"""
        if activation.numel() == 0:
            return 0.0
            
        activation_flat = activation.flatten()
        activation_abs = torch.abs(activation_flat) + 1e-8
        probs = activation_abs / torch.sum(activation_abs)
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        # 归一化到 [0, 1]
        max_entropy = math.log(len(probs))
        return min(entropy.item() / max_entropy, 1.0) if max_entropy > 0 else 0.0
    
    def _compute_gradient_complexity(self, gradient: torch.Tensor) -> float:
        """计算梯度复杂度"""
        if gradient is None or gradient.numel() == 0:
            return 0.0
            
        grad_flat = gradient.flatten()
        
        # 梯度的标准差/均值比率
        grad_std = torch.std(grad_flat)
        grad_mean = torch.mean(torch.abs(grad_flat))
        
        if grad_mean > 1e-8:
            complexity = grad_std / grad_mean
            return min(complexity.item(), 1.0)
        else:
            return 0.0
    
    def _compute_activation_complexity(self, activation: torch.Tensor) -> float:
        """计算激活模式复杂度"""
        if activation.numel() == 0 or len(activation.shape) < 2:
            return 0.0
            
        # 计算激活模式的变异系数
        activation_flat = activation.view(activation.shape[0], -1)
        
        # 批次间的变异性
        batch_means = torch.mean(activation_flat, dim=1)
        batch_std = torch.std(batch_means)
        batch_mean_avg = torch.mean(batch_means)
        
        if batch_mean_avg > 1e-8:
            complexity = batch_std / batch_mean_avg
            return min(complexity.item(), 1.0)
        else:
            return 0.0

class EnhancedBiologicalPrinciplesTrigger:
    """增强的生物学原理触发器"""
    
    def __init__(self, maturation_threshold: float = 0.6):
        self.maturation_threshold = maturation_threshold
        self.development_history = deque(maxlen=20)
        
    def should_trigger(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """检测是否处于关键发育期"""
        performance_history = context.get('performance_history', [])
        epoch = context.get('epoch', 0)
        
        if len(performance_history) < 10:
            return False, "性能历史数据不足"
        
        # 检测发育阶段
        maturation_score = self._compute_maturation_score(performance_history)
        
        self.development_history.append({
            'epoch': epoch,
            'maturation_score': maturation_score,
            'performance': performance_history[-1] if performance_history else 0.0
        })
        
        # 检测是否需要结构分化
        if self._detect_structural_differentiation_need(maturation_score):
            return True, f"关键发育期检测：成熟度={maturation_score:.3f}，适合结构重组"
            
        return False, "未处于关键发育期"
    
    def _compute_maturation_score(self, performance_history: List[float]) -> float:
        """计算发育成熟度分数"""
        if len(performance_history) < 5:
            return 0.0
        
        recent_performances = performance_history[-10:]
        
        # 1. 性能稳定性
        stability = 1.0 - np.std(recent_performances[-5:]) / (np.mean(recent_performances[-5:]) + 1e-8)
        
        # 2. 改进速度
        if len(recent_performances) >= 5:
            early_avg = np.mean(recent_performances[:5])
            late_avg = np.mean(recent_performances[-5:])
            improvement_rate = (late_avg - early_avg) / (early_avg + 1e-8)
        else:
            improvement_rate = 0.0
        
        # 3. 性能高度
        performance_level = recent_performances[-1] if recent_performances else 0.0
        
        # 综合成熟度分数
        maturation = (
            0.4 * min(stability, 1.0) +
            0.3 * min(improvement_rate, 1.0) +
            0.3 * performance_level
        )
        
        return max(0.0, min(maturation, 1.0))
    
    def _detect_structural_differentiation_need(self, maturation_score: float) -> bool:
        """检测是否需要结构分化"""
        return maturation_score > self.maturation_threshold

class EnhancedCognitiveScienceTrigger:
    """增强的认知科学触发器"""
    
    def __init__(self, forgetting_threshold: float = 0.05):
        self.forgetting_threshold = forgetting_threshold
        self.learning_patterns = deque(maxlen=25)
        
    def should_trigger(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """检测认知瓶颈和灾难性遗忘"""
        performance_history = context.get('performance_history', [])
        activations = context.get('activations', {})
        
        if len(performance_history) < 8:
            return False, "学习历史数据不足"
        
        # 检测灾难性遗忘
        forgetting_detected = self._detect_catastrophic_forgetting(performance_history)
        
        # 检测学习饱和
        saturation_detected = self._detect_learning_saturation(performance_history)
        
        # 检测特征表示冲突
        conflict_detected = self._detect_representation_conflict(activations)
        
        self.learning_patterns.append({
            'epoch': context.get('epoch', 0),
            'performance': performance_history[-1] if performance_history else 0.0,
            'forgetting_risk': forgetting_detected,
            'saturation_risk': saturation_detected,
            'conflict_risk': conflict_detected
        })
        
        if forgetting_detected or conflict_detected:
            reason = []
            if forgetting_detected:
                reason.append("灾难性遗忘风险")
            if conflict_detected:
                reason.append("特征表示冲突")
            return True, f"认知瓶颈检测：{', '.join(reason)}，需要分化专门化神经元"
            
        return False, "认知指标正常"
    
    def _detect_catastrophic_forgetting(self, performance_history: List[float]) -> bool:
        """检测灾难性遗忘"""
        if len(performance_history) < 8:
            return False
        
        # 检查最近性能是否显著下降
        recent_window = 5
        past_window = 5
        
        recent_avg = np.mean(performance_history[-recent_window:])
        past_avg = np.mean(performance_history[-(recent_window + past_window):-recent_window])
        
        if past_avg > 0:
            decline_ratio = (past_avg - recent_avg) / past_avg
            return decline_ratio > self.forgetting_threshold
        
        return False
    
    def _detect_learning_saturation(self, performance_history: List[float]) -> bool:
        """检测学习饱和"""
        if len(performance_history) < 10:
            return False
        
        # 检查最近10个epoch的改进
        recent_performances = performance_history[-10:]
        improvements = [recent_performances[i] - recent_performances[i-1] 
                       for i in range(1, len(recent_performances))]
        
        avg_improvement = np.mean(improvements)
        return avg_improvement < 0.001  # 改进极小
    
    def _detect_representation_conflict(self, activations: Dict[str, torch.Tensor]) -> bool:
        """检测特征表示冲突"""
        if not activations:
            return False
        
        # 简化的冲突检测：检查激活模式的一致性
        conflict_scores = []
        
        for name, activation in activations.items():
            if len(activation.shape) >= 2 and activation.shape[0] > 1:
                # 计算批次内激活模式的一致性
                activation_flat = activation.view(activation.shape[0], -1)
                
                # 计算样本间的相关性
                if activation_flat.shape[1] > 1:
                    try:
                        correlation_matrix = torch.corrcoef(activation_flat)
                        # 对角线外的相关系数
                        mask = ~torch.eye(correlation_matrix.shape[0], dtype=torch.bool)
                        off_diagonal = correlation_matrix[mask]
                        
                        if len(off_diagonal) > 0:
                            consistency = torch.mean(torch.abs(off_diagonal))
                            conflict_scores.append(1.0 - consistency.item())
                    except:
                        continue
        
        if conflict_scores:
            avg_conflict = np.mean(conflict_scores)
            return avg_conflict > 0.7  # 冲突阈值
        
        return False

class EnhancedDNMFramework:
    """增强的DNM框架"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # 初始化组件
        self.bottleneck_analyzer = AdvancedBottleneckAnalyzer()
        self.morphogenesis_executor = AdvancedMorphogenesisExecutor()
        self.decision_maker = IntelligentMorphogenesisDecisionMaker()
        
        # 初始化触发器
        self.triggers = {
            'information_theory': EnhancedInformationTheoryTrigger(),
            'biological_principles': EnhancedBiologicalPrinciplesTrigger(),
            'cognitive_science': EnhancedCognitiveScienceTrigger()
        }
        
        # 跟踪数据
        self.morphogenesis_events = []
        self.performance_history = []
        self.activation_cache = {}
        self.gradient_cache = {}
        
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'trigger_interval': 3,  # 每3个epoch检查一次
            'max_morphogenesis_per_epoch': 1,
            'performance_patience': 8,
            'min_improvement_threshold': 0.001,
            'max_parameter_growth_ratio': 0.5,  # 最大参数增长50%
            'enable_serial_division': True,
            'enable_parallel_division': True,
            'enable_hybrid_division': True,
            'complexity_threshold': 0.6
        }
    
    def should_trigger_morphogenesis(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """检查是否应该触发形态发生"""
        epoch = context.get('epoch', 0)
        
        # 检查触发间隔
        if epoch % self.config['trigger_interval'] != 0:
            return False, []
        
        # 检查各个触发器
        trigger_results = []
        trigger_reasons = []
        
        for name, trigger in self.triggers.items():
            try:
                should_trigger, reason = trigger.should_trigger(context)
                if should_trigger:
                    trigger_results.append(True)
                    trigger_reasons.append(f"{name}: {reason}")
                else:
                    trigger_results.append(False)
            except Exception as e:
                logger.warning(f"触发器 {name} 执行失败: {e}")
                trigger_results.append(False)
        
        # 至少有一个触发器激活
        should_trigger = any(trigger_results)
        
        return should_trigger, trigger_reasons
    
    def execute_morphogenesis(self, model: nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行形态发生"""
        results = {
            'model_modified': False,
            'new_model': model,
            'parameters_added': 0,
            'morphogenesis_events': 0,
            'morphogenesis_type': 'none',
            'trigger_reasons': []
        }
        
        try:
            # 检查是否应该触发
            should_trigger, trigger_reasons = self.should_trigger_morphogenesis(context)
            
            if not should_trigger:
                return results
            
            logger.info("🔄 Triggering advanced morphogenesis analysis...")
            
            # 输出触发原因
            for reason in trigger_reasons:
                print(f"    - {reason}")
            
            results['trigger_reasons'] = trigger_reasons
            
            # 执行瓶颈分析
            activations = context.get('activations', {})
            gradients = context.get('gradients', {})
            
            if not activations or not gradients:
                logger.warning("缺少激活值或梯度信息，跳过形态发生")
                return results
            
            bottleneck_analysis = self.bottleneck_analyzer.analyze_network_bottlenecks(
                model, activations, gradients
            )
            
            # 制定决策
            performance_history = context.get('performance_history', [])
            decision = self.decision_maker.make_decision(bottleneck_analysis, performance_history)
            
            if decision is None:
                logger.info("未发现需要形态发生的瓶颈")
                return results
            
            # 执行形态发生
            new_model, parameters_added = self.morphogenesis_executor.execute_morphogenesis(
                model, decision
            )
            
            if parameters_added > 0:
                # 记录事件
                event = EnhancedMorphogenesisEvent(
                    epoch=context.get('epoch', 0),
                    event_type=decision.morphogenesis_type.value,
                    location=decision.target_location,
                    trigger_reason=decision.reasoning,
                    performance_before=performance_history[-1] if performance_history else 0.0,
                    parameters_added=parameters_added,
                    morphogenesis_type=decision.morphogenesis_type,
                    confidence=decision.confidence,
                    expected_improvement=decision.expected_improvement
                )
                
                self.morphogenesis_events.append(event)
                
                # 更新结果
                results.update({
                    'model_modified': True,
                    'new_model': new_model,
                    'parameters_added': parameters_added,
                    'morphogenesis_events': 1,
                    'morphogenesis_type': decision.morphogenesis_type.value,
                    'decision_confidence': decision.confidence,
                    'expected_improvement': decision.expected_improvement
                })
                
                logger.info(f"高级形态发生完成: {decision.morphogenesis_type.value}, 新增参数: {parameters_added}")
                
        except Exception as e:
            logger.error(f"形态发生执行失败: {e}")
            
        return results
    
    def update_performance_history(self, performance: float):
        """更新性能历史"""
        self.performance_history.append(performance)
        
        # 保持历史长度
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def cache_activations_and_gradients(self, activations: Dict[str, torch.Tensor], 
                                       gradients: Dict[str, torch.Tensor]):
        """缓存激活值和梯度"""
        self.activation_cache = activations
        self.gradient_cache = gradients
    
    def get_morphogenesis_summary(self) -> Dict[str, Any]:
        """获取形态发生总结"""
        if not self.morphogenesis_events:
            return {
                'total_events': 0,
                'total_parameters_added': 0,
                'morphogenesis_types': {},
                'events': []
            }
        
        # 统计各种类型的形态发生
        type_counts = defaultdict(int)
        for event in self.morphogenesis_events:
            type_counts[event.morphogenesis_type.value] += 1
        
        total_params = sum(event.parameters_added for event in self.morphogenesis_events)
        
        return {
            'total_events': len(self.morphogenesis_events),
            'total_parameters_added': total_params,
            'morphogenesis_types': dict(type_counts),
            'events': [
                {
                    'epoch': event.epoch,
                    'type': event.morphogenesis_type.value,
                    'location': event.location,
                    'parameters_added': event.parameters_added,
                    'confidence': event.confidence,
                    'expected_improvement': event.expected_improvement,
                    'reasoning': event.trigger_reason
                }
                for event in self.morphogenesis_events
            ]
        }