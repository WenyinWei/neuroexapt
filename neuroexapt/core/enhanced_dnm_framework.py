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
import traceback
import time
import os

# 导入高级形态发生模块
from .advanced_morphogenesis import (
    AdvancedBottleneckAnalyzer,
    AdvancedMorphogenesisExecutor,
    IntelligentMorphogenesisDecisionMaker,
    MorphogenesisType,
    MorphogenesisDecision
)

class ConfigurableLogger:
    """可配置的高性能日志系统，替代ANSI彩色打印"""
    
    def __init__(self, name: str = "neuroexapt", level: str = "INFO", enable_console: bool = True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            # 控制台处理器
            if enable_console:
                console_handler = logging.StreamHandler()
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                )
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
            
            # 文件处理器（可选）
            log_file = os.environ.get('NEUROEXAPT_LOG_FILE')
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
        
        self.section_stack = []
        
    def debug(self, message: str, *args, **kwargs):
        """记录调试信息"""
        if self.logger.isEnabledFor(logging.DEBUG):
            indent = "  " * len(self.section_stack)
            self.logger.debug(f"{indent}{message}", *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """记录信息"""
        if self.logger.isEnabledFor(logging.INFO):
            indent = "  " * len(self.section_stack)
            self.logger.info(f"{indent}{message}", *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """记录警告"""
        if self.logger.isEnabledFor(logging.WARNING):
            indent = "  " * len(self.section_stack)
            self.logger.warning(f"{indent}{message}", *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """记录错误"""
        if self.logger.isEnabledFor(logging.ERROR):
            indent = "  " * len(self.section_stack)
            self.logger.error(f"{indent}{message}", *args, **kwargs)
    
    def success(self, message: str, *args, **kwargs):
        """记录成功信息（使用INFO级别）"""
        if self.logger.isEnabledFor(logging.INFO):
            indent = "  " * len(self.section_stack)
            self.logger.info(f"{indent}✅ {message}", *args, **kwargs)
    
    def enter_section(self, section_name: str):
        """进入新的日志区域"""
        if self.logger.isEnabledFor(logging.DEBUG):
            indent = "  " * len(self.section_stack)
            self.logger.debug(f"{indent}🔍 进入 {section_name}")
        self.section_stack.append(section_name)
    
    def exit_section(self, section_name: str):
        """退出日志区域"""
        if self.section_stack and self.section_stack[-1] == section_name:
            self.section_stack.pop()
        if self.logger.isEnabledFor(logging.DEBUG):
            indent = "  " * len(self.section_stack)
            self.logger.debug(f"{indent}✅ 完成 {section_name}")
    
    def log_tensor_info(self, tensor: torch.Tensor, name: str):
        """记录张量信息"""
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
            
        if tensor is None:
            self.warning(f"❌ {name}: None")
            return
        
        device_info = f"({tensor.device})" if hasattr(tensor, 'device') else ""
        self.debug(f"📊 {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}, device={device_info}")
    
    def log_model_info(self, model: nn.Module, name: str = "Model"):
        """记录模型信息"""
        if not self.logger.isEnabledFor(logging.INFO):
            return
            
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        device = next(model.parameters()).device if list(model.parameters()) else "Unknown"
        
        self.info(f"🧠 {name}: 总参数={total_params:,}, 可训练={trainable_params:,}, 设备={device}")

# 全局日志器配置
_log_level = os.environ.get('NEUROEXAPT_LOG_LEVEL', 'INFO')
_enable_console = os.environ.get('NEUROEXAPT_CONSOLE_LOG', 'true').lower() == 'true'

# 创建全局日志器实例
logger = ConfigurableLogger("neuroexapt.dnm", _log_level, _enable_console)

# 保持向后兼容性的DebugPrinter类
class DebugPrinter:
    """向后兼容的调试打印器（已废弃，建议使用logger）"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._logger = logger
        import warnings
        warnings.warn(
            "DebugPrinter is deprecated. Use the global 'logger' instance instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    def print_debug(self, message: str, level: str = "INFO"):
        if not self.enabled:
            return
        
        level_map = {
            "INFO": self._logger.info,
            "SUCCESS": self._logger.success,
            "WARNING": self._logger.warning,
            "ERROR": self._logger.error,
            "DEBUG": self._logger.debug
        }
        level_map.get(level, self._logger.info)(message)
    
    def enter_section(self, section_name: str):
        if self.enabled:
            self._logger.enter_section(section_name)
    
    def exit_section(self, section_name: str):
        if self.enabled:
            self._logger.exit_section(section_name)
    
    def print_tensor_info(self, tensor: torch.Tensor, name: str):
        if self.enabled:
            self._logger.log_tensor_info(tensor, name)
    
    def print_model_info(self, model: nn.Module, name: str = "Model"):
        if self.enabled:
            self._logger.log_model_info(model, name)

# 为了向后兼容保留debug_printer实例
debug_printer = DebugPrinter(enabled=True)

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
        logger.enter_section("信息论触发器检查")
        
        activations = context.get('activations', {})
        gradients = context.get('gradients', {})
        
        logger.debug(f"输入数据: 激活值层数={len(activations)}, 梯度层数={len(gradients)}")
        
        if not activations or not gradients:
            logger.warning("❌ 缺少激活值或梯度信息")
            logger.exit_section("信息论触发器检查")
            return False, "缺少激活值或梯度信息"
            
        # 计算综合复杂度分数
        complexity_score = self._compute_complexity_score(activations, gradients)
        
        logger.info(f"复杂度分数: {complexity_score:.4f} (阈值: {self.complexity_threshold})")
        
        self.history.append({
            'complexity_score': complexity_score,
            'epoch': context.get('epoch', 0)
        })
        
        # 检查是否需要更复杂的结构变异
        if complexity_score > self.complexity_threshold:
            logger.success(f"触发条件满足: {complexity_score:.4f} > {self.complexity_threshold}")
            logger.exit_section("信息论触发器检查")
            return True, f"复杂度瓶颈检测：分数={complexity_score:.4f}"
            
        logger.info(f"❌ 未达到触发条件: {complexity_score:.4f} <= {self.complexity_threshold}")
        logger.exit_section("信息论触发器检查")
        return False, "复杂度指标未达到触发条件"
    
    def _compute_complexity_score(self, activations: Dict[str, torch.Tensor], 
                                gradients: Dict[str, torch.Tensor]) -> float:
        """计算网络复杂度分数"""
        logger.enter_section("复杂度分数计算")
        scores = []
        
        for name, activation in activations.items():
            if name not in gradients or gradients[name] is None:
                logger.warning(f"⚠️ 跳过层 {name}: 缺少梯度信息")
                continue
                
            gradient = gradients[name]
            logger.log_tensor_info(activation, f"激活值[{name}]")
            logger.log_tensor_info(gradient, f"梯度[{name}]")
            
            # 1. 信息熵分析
            entropy = self._compute_entropy(activation)
            logger.debug(f"信息熵[{name}]: {entropy:.4f}")
            
            # 2. 梯度复杂度
            grad_complexity = self._compute_gradient_complexity(gradient)
            logger.debug(f"梯度复杂度[{name}]: {grad_complexity:.4f}")
            
            # 3. 激活模式复杂度
            activation_complexity = self._compute_activation_complexity(activation)
            logger.debug(f"激活复杂度[{name}]: {activation_complexity:.4f}")
            
            # 综合分数
            layer_score = 0.4 * entropy + 0.3 * grad_complexity + 0.3 * activation_complexity
            scores.append(layer_score)
            logger.debug(f"层分数[{name}]: {layer_score:.4f}")
        
        final_score = np.mean(scores) if scores else 0.0
        logger.info(f"最终复杂度分数: {final_score:.4f} (共{len(scores)}层)")
        logger.exit_section("复杂度分数计算")
        return final_score
    
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
        logger.enter_section("生物学原理触发器检查")
        
        performance_history = context.get('performance_history', [])
        epoch = context.get('epoch', 0)
        
        debug_printer.print_debug(f"当前epoch: {epoch}, 性能历史长度: {len(performance_history)}", "DEBUG")
        
        if len(performance_history) < 10:
            logger.warning("❌ 性能历史数据不足 (需要至少10个数据点)")
            logger.exit_section("生物学原理触发器检查")
            return False, "性能历史数据不足"
        
        # 检测发育阶段
        logger.debug("计算发育成熟度分数...")
        maturation_score = self._compute_maturation_score(performance_history)
        logger.info(f"成熟度分数: {maturation_score:.4f} (阈值: {self.maturation_threshold})")
        
        self.development_history.append({
            'epoch': epoch,
            'maturation_score': maturation_score,
            'performance': performance_history[-1] if performance_history else 0.0
        })
        
        # 检测是否需要结构分化
        differentiation_needed = self._detect_structural_differentiation_need(maturation_score)
        debug_printer.print_debug(f"结构分化需求: {'✅需要' if differentiation_needed else '❌不需要'}", 
                               "SUCCESS" if differentiation_needed else "DEBUG")
        
        if differentiation_needed:
            logger.success(f"✅ 触发条件满足: 成熟度={maturation_score:.3f}")
            logger.exit_section("生物学原理触发器检查")
            return True, f"关键发育期检测：成熟度={maturation_score:.3f}，适合结构重组"
            
        logger.info("❌ 未达到触发条件: 未处于关键发育期")
        logger.exit_section("生物学原理触发器检查")
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
        logger.enter_section("认知科学触发器检查")
        
        performance_history = context.get('performance_history', [])
        activations = context.get('activations', {})
        
        debug_printer.print_debug(f"性能历史长度: {len(performance_history)}, 激活值层数: {len(activations)}", "DEBUG")
        
        if len(performance_history) < 8:
            logger.warning("❌ 学习历史数据不足 (需要至少8个数据点)")
            logger.exit_section("认知科学触发器检查")
            return False, "学习历史数据不足"
        
        # 检测灾难性遗忘
        logger.debug("检测灾难性遗忘...")
        forgetting_detected = self._detect_catastrophic_forgetting(performance_history)
        debug_printer.print_debug(f"灾难性遗忘检测: {'✅发现' if forgetting_detected else '❌未发现'}", 
                               "WARNING" if forgetting_detected else "DEBUG")
        
        # 检测学习饱和
        logger.debug("检测学习饱和...")
        saturation_detected = self._detect_learning_saturation(performance_history)
        debug_printer.print_debug(f"学习饱和检测: {'✅发现' if saturation_detected else '❌未发现'}", 
                               "WARNING" if saturation_detected else "DEBUG")
        
        # 检测特征表示冲突
        logger.debug("检测特征表示冲突...")
        conflict_detected = self._detect_representation_conflict(activations)
        debug_printer.print_debug(f"特征表示冲突检测: {'✅发现' if conflict_detected else '❌未发现'}", 
                               "WARNING" if conflict_detected else "DEBUG")
        
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
            debug_printer.print_debug(f"✅ 触发条件满足: {', '.join(reason)}", "SUCCESS")
            logger.exit_section("认知科学触发器检查")
            return True, f"认知瓶颈检测：{', '.join(reason)}，需要分化专门化神经元"
            
        logger.info("❌ 未达到触发条件: 认知指标正常")
        logger.exit_section("认知科学触发器检查")
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
    """增强的动态神经形态发生框架
    
    🧬 支持传统形态发生和激进多点变异
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # 现有初始化代码
        default_config = {
            'trigger_interval': 8,
            'performance_monitoring_window': 10,
            'morphogenesis_budget': 5000,
            'enable_aggressive_mode': True,  # 新增：激进模式开关
            'accuracy_plateau_threshold': 0.1,  # 新增：准确率停滞阈值
            'plateau_detection_window': 5,  # 新增：停滞检测窗口
            'aggressive_trigger_accuracy': 0.92,  # 新增：激进模式触发准确率
            'max_concurrent_mutations': 3,  # 新增：最大并发变异数
        }
        
        self.config = {**default_config, **(config or {})}
        
        # 原有组件
        self.triggers = {
            'information_theory': EnhancedInformationTheoryTrigger(),
            'biological_principle': EnhancedBiologicalPrinciplesTrigger(),
            'cognitive_science': EnhancedCognitiveScienceTrigger()
        }
        
        self.bottleneck_analyzer = AdvancedBottleneckAnalyzer()
        self.decision_maker = IntelligentMorphogenesisDecisionMaker()
        self.executor = AdvancedMorphogenesisExecutor()
        
        # 新增激进形态发生组件
        if self.config['enable_aggressive_mode']:
            from .aggressive_morphogenesis import (
                AggressiveMorphogenesisAnalyzer,
                MultiPointMutationPlanner,
                AggressiveMorphogenesisExecutor
            )
            self.aggressive_analyzer = AggressiveMorphogenesisAnalyzer(
                accuracy_plateau_threshold=self.config['accuracy_plateau_threshold'],
                plateau_window=self.config['plateau_detection_window']
            )
            self.mutation_planner = MultiPointMutationPlanner(
                max_concurrent_mutations=self.config['max_concurrent_mutations'],
                parameter_budget=self.config['morphogenesis_budget']
            )
            self.aggressive_executor = AggressiveMorphogenesisExecutor()
        
        # 记录和监控
        self.morphogenesis_events = []
        self.performance_history = []
        self.aggressive_mode_active = False

    def should_trigger_morphogenesis(self, 
                                   model: nn.Module,
                                   epoch: int,
                                   activations: Dict[str, torch.Tensor],
                                   gradients: Dict[str, torch.Tensor],
                                   performance_history: List[float]) -> Tuple[bool, List[str]]:
        """增强的形态发生触发检查 - 支持激进模式"""
        logger.enter_section("增强形态发生触发检查")
        
        # 检查当前准确率是否达到激进模式阈值
        current_accuracy = performance_history[-1] if performance_history else 0.0
        
        # 激进模式激活条件
        aggressive_mode_triggered = False
        if (self.config['enable_aggressive_mode'] and 
            current_accuracy >= self.config['aggressive_trigger_accuracy']):
            
            # 检测准确率停滞
            is_plateau, stagnation_severity = self.aggressive_analyzer.detect_accuracy_plateau(performance_history)
            
            if is_plateau and stagnation_severity > 0.5:
                logger.warning(f"🚨 检测到准确率停滞，激活激进模式！停滞严重程度: {stagnation_severity:.3f}")
                aggressive_mode_triggered = True
                self.aggressive_mode_active = True
        
        # 如果激进模式被触发，使用不同的判断逻辑
        if aggressive_mode_triggered:
            logger.info("🚀 使用激进形态发生策略")
            # 激进模式下更频繁地触发，不受传统触发间隔限制
            trigger_reasons = [f"激进模式: 准确率停滞(严重程度={stagnation_severity:.3f})"]
            logger.exit_section("增强形态发生触发检查")
            return True, trigger_reasons
        
        # 传统触发逻辑
        logger.info(f"当前epoch: {epoch}, 触发间隔: {self.config['trigger_interval']}")
        
        if epoch % self.config['trigger_interval'] != 0:
            logger.info(f"❌ 不在触发间隔内 ({epoch} % {self.config['trigger_interval']} != 0)")
            logger.exit_section("增强形态发生触发检查")
            return False, []
        
        logger.info("✅ 在触发间隔内，检查各触发器")
        
        # 构建分析上下文
        context = {
            'epoch': epoch,
            'activations': activations,
            'gradients': gradients,
            'performance_history': performance_history,
            'model': model
        }
        
        # 检查各个触发器
        trigger_results = []
        trigger_reasons = []
        
        for name, trigger in self.triggers.items():
            try:
                logger.debug(f"检查触发器: {name}")
                should_trigger, reason = trigger.should_trigger(context)
                trigger_results.append(should_trigger)
                
                logger.info(f"触发器[{name}]: {'✅激活' if should_trigger else '❌未激活'} - {reason}")
                
                if should_trigger:
                    trigger_reasons.append(f"{name}: {reason}")
                    
            except Exception as e:
                logger.error(f"❌ 触发器 {name} 执行失败: {e}")
                logger.error(f"错误详情: {traceback.format_exc()}")
                trigger_results.append(False)
        
        should_trigger = any(trigger_results)
        
        logger.info(f"触发器汇总: {len([r for r in trigger_results if r])}/{len(trigger_results)} 激活")
        logger.info(f"最终决定: {'✅触发形态发生' if should_trigger else '❌不触发'}")
        
        logger.exit_section("增强形态发生触发检查")
        return should_trigger, trigger_reasons

    def execute_morphogenesis(self,
                            model: nn.Module,
                            activations: Dict[str, torch.Tensor],
                            gradients: Dict[str, torch.Tensor],
                            performance_history: List[float],
                            epoch: int) -> Dict[str, Any]:
        """执行形态发生 - 支持传统和激进模式"""
        logger.enter_section("增强形态发生执行")
        logger.log_model_info(model, "输入模型")
        
        try:
            # 检查是否满足触发条件
            should_trigger, trigger_reasons = self.should_trigger_morphogenesis(
                model, epoch, activations, gradients, performance_history
            )
            
            if not should_trigger:
                logger.info("❌ 未满足触发条件，跳过形态发生")
                logger.exit_section("增强形态发生执行")
                return {
                    'model_modified': False,
                    'new_model': model,
                    'parameters_added': 0,
                    'morphogenesis_events': [],
                    'morphogenesis_type': 'none',
                    'trigger_reasons': []
                }
            
            logger.success(f"满足触发条件，原因: {trigger_reasons}")
            
            # 激进模式路径
            if self.aggressive_mode_active and self.config['enable_aggressive_mode']:
                return self._execute_aggressive_morphogenesis(
                    model, activations, gradients, performance_history, epoch, trigger_reasons
                )
            
            # 传统形态发生路径
            return self._execute_traditional_morphogenesis(
                model, activations, gradients, performance_history, epoch, trigger_reasons
            )
            
        except Exception as e:
            logger.error(f"❌ 形态发生执行失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return {
                'model_modified': False,
                'new_model': model,
                'parameters_added': 0,
                'morphogenesis_events': [],
                'morphogenesis_type': 'error',
                'trigger_reasons': trigger_reasons,
                'error': str(e)
            }
        finally:
            logger.exit_section("增强形态发生执行")

    def _execute_aggressive_morphogenesis(self,
                                        model: nn.Module,
                                        activations: Dict[str, torch.Tensor],
                                        gradients: Dict[str, torch.Tensor],
                                        performance_history: List[float],
                                        epoch: int,
                                        trigger_reasons: List[str]) -> Dict[str, Any]:
        """执行激进多点形态发生"""
        logger.enter_section("激进多点形态发生")
        
        try:
            # 反向梯度投影分析
            output_targets = torch.randint(0, 10, (128,))  # 模拟目标，实际使用时应传入真实targets
            bottleneck_signatures = self.aggressive_analyzer.analyze_reverse_gradient_projection(
                activations, gradients, output_targets
            )
            
            if not bottleneck_signatures:
                logger.warning("❌ 未检测到瓶颈签名，回退到传统形态发生")
                return self._execute_traditional_morphogenesis(
                    model, activations, gradients, performance_history, epoch, trigger_reasons
                )
            
            # 检测停滞严重程度
            _, stagnation_severity = self.aggressive_analyzer.detect_accuracy_plateau(performance_history)
            
            # 规划多点变异
            mutations = self.mutation_planner.plan_aggressive_mutations(
                bottleneck_signatures, performance_history, stagnation_severity
            )
            
            if not mutations:
                logger.warning("❌ 未生成有效的变异计划，回退到传统形态发生")
                return self._execute_traditional_morphogenesis(
                    model, activations, gradients, performance_history, epoch, trigger_reasons
                )
            
            # 执行最佳变异策略
            best_mutation = max(mutations, key=lambda m: m.expected_improvement - m.risk_assessment * 0.5)
            logger.info(f"选择最佳变异策略: {best_mutation.coordination_strategy}, "
                       f"目标位置数: {len(best_mutation.target_locations)}, "
                       f"期望改进: {best_mutation.expected_improvement:.3f}")
            
            new_model, params_added, execution_result = self.aggressive_executor.execute_multi_point_mutation(
                model, best_mutation
            )
            
            # 记录激进形态发生事件
            morphogenesis_event = EnhancedMorphogenesisEvent(
                epoch=epoch,
                event_type='aggressive_multi_point',
                location=f"多点({len(best_mutation.target_locations)}位置)",
                trigger_reason='; '.join(trigger_reasons),
                performance_before=performance_history[-1] if performance_history else 0.0,
                parameters_added=params_added,
                morphogenesis_type=MorphogenesisType.HYBRID_DIVISION,  # 代表多点变异
                confidence=1.0 - best_mutation.risk_assessment,
                expected_improvement=best_mutation.expected_improvement
            )
            
            self.morphogenesis_events.append(morphogenesis_event)
            
            logger.success(f"激进多点形态发生完成: 策略={best_mutation.coordination_strategy}, "
                         f"成功变异={execution_result.get('successful_mutations', 0)}/"
                         f"{execution_result.get('total_mutations', 0)}, "
                         f"新增参数: {params_added:,}")
            
            # 重置激进模式状态（给模型几个epoch适应）
            self.aggressive_mode_active = False
            
            return {
                'model_modified': params_added > 0,
                'new_model': new_model,
                'parameters_added': params_added,
                'morphogenesis_events': [morphogenesis_event],
                'morphogenesis_type': 'aggressive_multi_point',
                'trigger_reasons': trigger_reasons,
                'aggressive_details': {
                    'mutation_strategy': best_mutation.coordination_strategy,
                    'target_locations': best_mutation.target_locations,
                    'bottleneck_count': len(bottleneck_signatures),
                    'stagnation_severity': stagnation_severity,
                    'execution_result': execution_result
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 激进形态发生失败: {e}")
            logger.warning("回退到传统形态发生")
            return self._execute_traditional_morphogenesis(
                model, activations, gradients, performance_history, epoch, trigger_reasons
            )
        finally:
            logger.exit_section("激进多点形态发生")

    def _execute_traditional_morphogenesis(self,
                                         model: nn.Module,
                                         activations: Dict[str, torch.Tensor],
                                         gradients: Dict[str, torch.Tensor],
                                         performance_history: List[float],
                                         epoch: int,
                                         trigger_reasons: List[str]) -> Dict[str, Any]:
        """执行传统单点形态发生"""
        logger.enter_section("传统形态发生")
        
        try:
            # 原有的传统形态发生逻辑
            logger.info("执行传统单点形态发生策略")
            
            # 瓶颈分析
            logger.enter_section("瓶颈分析")
            
            if not activations or not gradients:
                logger.error("❌ 缺少激活值或梯度信息，跳过形态发生")
                logger.exit_section("瓶颈分析")
                logger.exit_section("传统形态发生")
                return {
                    'model_modified': False,
                    'new_model': model,
                    'parameters_added': 0,
                    'morphogenesis_events': [],
                    'morphogenesis_type': 'failed',
                    'trigger_reasons': trigger_reasons,
                    'error': 'missing_analysis_data'
                }
            
            logger.info(f"分析数据: 激活值{len(activations)}层, 梯度{len(gradients)}层")
            bottleneck_analysis = self.bottleneck_analyzer.analyze_network_bottlenecks(activations, gradients)
            
            logger.success(f"瓶颈分析完成: {len(bottleneck_analysis) if bottleneck_analysis else 0}个瓶颈")
            logger.exit_section("瓶颈分析")
            
            # 形态发生决策
            logger.enter_section("形态发生决策")
            logger.info(f"性能历史: {len(performance_history)}个数据点")
            
            decision = self.decision_maker.make_morphogenesis_decision(bottleneck_analysis, performance_history)
            if not decision:
                logger.warning("❌ 未发现需要形态发生的瓶颈")
                logger.exit_section("形态发生决策")
                logger.exit_section("传统形态发生")
                return {
                    'model_modified': False,
                    'new_model': model,
                    'parameters_added': 0,
                    'morphogenesis_events': [],
                    'morphogenesis_type': 'none',
                    'trigger_reasons': trigger_reasons
                }
            
            logger.success(f"决策制定完成: {decision.morphogenesis_type.value} (置信度: {decision.confidence:.3f})")
            logger.exit_section("形态发生决策")
            
            # 形态发生执行
            logger.enter_section("形态发生执行")
            logger.info(f"执行策略: {decision.morphogenesis_type.value} 在 {decision.target_location}")
            
            new_model, parameters_added = self.executor.execute_morphogenesis(model, decision)
            
            logger.info(f"形态发生结果: 新增参数={parameters_added}")
            logger.log_model_info(new_model, "新模型")
            logger.exit_section("形态发生执行")
            
            if parameters_added > 0:
                logger.success("✅ 形态发生成功，记录事件")
                
                # 记录形态发生事件
                morphogenesis_event = EnhancedMorphogenesisEvent(
                    epoch=epoch,
                    event_type=decision.morphogenesis_type.value,
                    location=decision.target_location,
                    trigger_reason='; '.join(trigger_reasons),
                    performance_before=performance_history[-1] if performance_history else 0.0,
                    parameters_added=parameters_added,
                    morphogenesis_type=decision.morphogenesis_type,
                    confidence=decision.confidence,
                    expected_improvement=decision.expected_improvement
                )
                
                self.morphogenesis_events.append(morphogenesis_event)
                
                logger.success(f"传统形态发生完成: {decision.morphogenesis_type.value}, 新增参数: {parameters_added:,}")
                
                return {
                    'model_modified': True,
                    'new_model': new_model,
                    'parameters_added': parameters_added,
                    'morphogenesis_events': [morphogenesis_event],
                    'morphogenesis_type': decision.morphogenesis_type.value,
                    'trigger_reasons': trigger_reasons
                }
            else:
                logger.warning("❌ 形态发生未添加任何参数")
                return {
                    'model_modified': False,
                    'new_model': model,
                    'parameters_added': 0,
                    'morphogenesis_events': [],
                    'morphogenesis_type': 'failed',
                    'trigger_reasons': trigger_reasons
                }
                
        except Exception as e:
            logger.error(f"❌ 传统形态发生失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return {
                'model_modified': False,
                'new_model': model,
                'parameters_added': 0,
                'morphogenesis_events': [],
                'morphogenesis_type': 'error',
                'trigger_reasons': trigger_reasons,
                'error': str(e)
            }
        finally:
            logger.exit_section("传统形态发生")

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