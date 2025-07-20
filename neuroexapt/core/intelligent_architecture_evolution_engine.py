"""
智能架构进化引擎
整合瓶颈检测、变异规划和参数迁移的完整架构进化系统
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import time
import copy

from .intelligent_bottleneck_detector import IntelligentBottleneckDetector, BottleneckReport
from .intelligent_mutation_planner import IntelligentMutationPlanner, MutationPlan
from .advanced_net2net_transfer import AdvancedNet2NetTransfer

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """进化配置"""
    # 检测参数
    confidence_threshold: float = 0.7
    max_mutations_per_iteration: int = 3
    
    # 迁移参数
    risk_tolerance: float = 0.7
    preserve_function: bool = True
    
    # 进化参数
    max_iterations: int = 10
    patience: int = 3  # 连续无改进的容忍轮数
    min_improvement: float = 0.01  # 最小改进阈值
    
    # 任务参数
    task_type: str = 'vision'  # 'vision', 'nlp', 'graph'
    
    # 评估参数
    evaluation_samples: int = 1000
    evaluation_metric: str = 'accuracy'  # 'accuracy', 'loss', 'f1'


@dataclass
class EvolutionIteration:
    """单次进化迭代结果"""
    iteration: int
    bottleneck_reports: List[BottleneckReport]
    mutation_plans: List[MutationPlan]
    transfer_reports: List[Dict[str, Any]]
    
    # 性能指标
    performance_before: float
    performance_after: float
    improvement: float
    
    # 模型复杂度
    parameters_before: int
    parameters_after: int
    parameter_growth: float
    
    # 时间开销
    detection_time: float
    planning_time: float
    transfer_time: float
    total_time: float


class IntelligentArchitectureEvolutionEngine:
    """
    智能架构进化引擎
    
    核心理念：
    1. 科学的进化流程：检测 -> 规划 -> 迁移 -> 评估 -> 迭代
    2. 智能的决策机制：基于互信息和贝叶斯不确定性的精确定位
    3. 稳健的参数迁移：保证功能等价性和训练稳定性
    4. 自适应的进化策略：根据改进效果动态调整策略
    """
    
    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        
        # 核心组件
        self.bottleneck_detector = IntelligentBottleneckDetector()
        self.mutation_planner = IntelligentMutationPlanner()
        self.transfer_engine = AdvancedNet2NetTransfer()
        
        # 进化历史
        self.evolution_history: List[EvolutionIteration] = []
        self.best_model = None
        self.best_performance = float('-inf')
        
        # 性能跟踪
        self.performance_plateau_count = 0
        self.convergence_detected = False
        
    def evolve(self,
              model: nn.Module,
              data_loader,
              evaluation_fn: Callable[[nn.Module], float],
              feature_extractor_fn: Optional[Callable] = None,
              gradient_extractor_fn: Optional[Callable] = None) -> Tuple[nn.Module, List[EvolutionIteration]]:
        """
        执行智能架构进化
        
        Args:
            model: 初始模型
            data_loader: 数据加载器
            evaluation_fn: 评估函数，返回性能分数
            feature_extractor_fn: 特征提取函数，返回各层特征
            gradient_extractor_fn: 梯度提取函数，返回各层梯度
            
        Returns:
            (最佳模型, 进化历史)
        """
        logger.info("🚀 启动智能架构进化")
        
        current_model = copy.deepcopy(model)
        current_performance = evaluation_fn(current_model)
        
        self.best_model = copy.deepcopy(current_model)
        self.best_performance = current_performance
        
        logger.info(f"初始性能: {current_performance:.4f}")
        
        # 迭代进化
        for iteration in range(self.config.max_iterations):
            if self.convergence_detected:
                logger.info(f"在第{iteration}轮检测到收敛，停止进化")
                break
                
            logger.info(f"\n🔄 开始第{iteration + 1}轮进化")
            
            # 执行单次进化迭代
            evolution_result = self._execute_evolution_iteration(
                current_model, data_loader, evaluation_fn,
                feature_extractor_fn, gradient_extractor_fn, iteration
            )
            
            self.evolution_history.append(evolution_result)
            
            # 更新当前模型和性能
            if evolution_result.improvement > self.config.min_improvement:
                current_model = self._load_evolved_model(current_model, evolution_result)
                current_performance = evolution_result.performance_after
                
                # 更新最佳模型
                if current_performance > self.best_performance:
                    self.best_model = copy.deepcopy(current_model)
                    self.best_performance = current_performance
                    self.performance_plateau_count = 0
                    logger.info(f"🎉 发现更佳模型，性能: {current_performance:.4f}")
                else:
                    self.performance_plateau_count += 1
            else:
                self.performance_plateau_count += 1
                logger.info(f"本轮改进不足 ({evolution_result.improvement:.4f})")
            
            # 检查收敛
            if self.performance_plateau_count >= self.config.patience:
                self.convergence_detected = True
                logger.info("达到性能平台期，标记收敛")
        
        logger.info(f"🏁 进化完成，最佳性能: {self.best_performance:.4f}")
        return self.best_model, self.evolution_history
    
    def _execute_evolution_iteration(self,
                                   model: nn.Module,
                                   data_loader,
                                   evaluation_fn: Callable,
                                   feature_extractor_fn: Optional[Callable],
                                   gradient_extractor_fn: Optional[Callable],
                                   iteration: int) -> EvolutionIteration:
        """执行单次进化迭代"""
        
        start_time = time.time()
        
        # 1. 提取特征和梯度
        feature_dict, labels, gradient_dict = self._extract_features_and_gradients(
            model, data_loader, feature_extractor_fn, gradient_extractor_fn
        )
        
        # 2. 瓶颈检测
        detection_start = time.time()
        bottleneck_reports = self.bottleneck_detector.detect_bottlenecks(
            model=model,
            feature_dict=feature_dict,
            labels=labels,
            gradient_dict=gradient_dict,
            num_classes=self._infer_num_classes(labels),
            confidence_threshold=self.config.confidence_threshold
        )
        detection_time = time.time() - detection_start
        
        logger.info(f"检测到 {len(bottleneck_reports)} 个瓶颈，耗时 {detection_time:.2f}s")
        
        # 3. 变异规划
        planning_start = time.time()
        mutation_plans = self.mutation_planner.plan_mutations(
            bottleneck_reports=bottleneck_reports,
            model=model,
            task_type=self.config.task_type,
            max_mutations=self.config.max_mutations_per_iteration,
            risk_tolerance=self.config.risk_tolerance
        )
        planning_time = time.time() - planning_start
        
        logger.info(f"生成 {len(mutation_plans)} 个变异计划，耗时 {planning_time:.2f}s")
        
        # 4. 参数迁移
        transfer_start = time.time()
        evolved_model = copy.deepcopy(model)
        transfer_reports = []
        
        if mutation_plans:
            evolved_model, transfer_reports = self.transfer_engine.batch_transfer(
                evolved_model, mutation_plans
            )
        
        transfer_time = time.time() - transfer_start
        
        logger.info(f"执行参数迁移，耗时 {transfer_time:.2f}s")
        
        # 5. 性能评估
        performance_before = evaluation_fn(model)
        performance_after = evaluation_fn(evolved_model) if mutation_plans else performance_before
        improvement = performance_after - performance_before
        
        # 6. 复杂度分析
        params_before = sum(p.numel() for p in model.parameters())
        params_after = sum(p.numel() for p in evolved_model.parameters())
        param_growth = (params_after - params_before) / params_before if params_before > 0 else 0.0
        
        total_time = time.time() - start_time
        
        logger.info(f"性能变化: {performance_before:.4f} -> {performance_after:.4f} "
                   f"(+{improvement:.4f}), 参数增长: {param_growth:.2%}")
        
        return EvolutionIteration(
            iteration=iteration,
            bottleneck_reports=bottleneck_reports,
            mutation_plans=mutation_plans,
            transfer_reports=transfer_reports,
            performance_before=performance_before,
            performance_after=performance_after,
            improvement=improvement,
            parameters_before=params_before,
            parameters_after=params_after,
            parameter_growth=param_growth,
            detection_time=detection_time,
            planning_time=planning_time,
            transfer_time=transfer_time,
            total_time=total_time
        )
    
    def _extract_features_and_gradients(self,
                                      model: nn.Module,
                                      data_loader,
                                      feature_extractor_fn: Optional[Callable],
                                      gradient_extractor_fn: Optional[Callable]) -> Tuple[Dict, torch.Tensor, Dict]:
        """提取特征和梯度"""
        
        if feature_extractor_fn:
            # 使用自定义特征提取函数
            feature_dict, labels = feature_extractor_fn(model, data_loader)
        else:
            # 使用默认特征提取
            feature_dict, labels = self._default_feature_extraction(model, data_loader)
        
        gradient_dict = {}
        if gradient_extractor_fn:
            gradient_dict = gradient_extractor_fn(model, data_loader)
        
        return feature_dict, labels, gradient_dict
    
    def _default_feature_extraction(self, model: nn.Module, data_loader) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """默认特征提取"""
        feature_dict = {}
        all_labels = []
        
        # 注册hook来收集特征
        def get_hook(name):
            def hook(module, input, output):
                if name not in feature_dict:
                    feature_dict[name] = []
                feature_dict[name].append(output.detach().cpu())
            return hook
        
        # 为主要层注册hook
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                hook = module.register_forward_hook(get_hook(name))
                hooks.append(hook)
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= 3:  # 只处理几个批次
                    break
                    
                data = data.to(next(model.parameters()).device)
                _ = model(data)
                all_labels.append(target)
        
        # 清理hooks
        for hook in hooks:
            hook.remove()
        
        # 合并特征
        for name in feature_dict:
            if feature_dict[name]:
                feature_dict[name] = torch.cat(feature_dict[name], dim=0)
        
        labels = torch.cat(all_labels, dim=0) if all_labels else torch.tensor([])
        
        return feature_dict, labels
    
    def _infer_num_classes(self, labels: torch.Tensor) -> Optional[int]:
        """推断类别数量"""
        if labels.numel() == 0:
            return None
        
        if labels.dtype in [torch.long, torch.int]:
            return int(labels.max().item()) + 1
        
        return None
    
    def _load_evolved_model(self, current_model: nn.Module, evolution_result: EvolutionIteration) -> nn.Module:
        """加载进化后的模型"""
        # 这里应该根据transfer_reports重新构建模型
        # 简化实现：如果有成功的迁移，返回新模型
        if evolution_result.transfer_reports and any(r.get('quality_score', 0) > 0.5 for r in evolution_result.transfer_reports):
            # 重新执行迁移以获得新模型
            evolved_model = copy.deepcopy(current_model)
            evolved_model, _ = self.transfer_engine.batch_transfer(evolved_model, evolution_result.mutation_plans)
            return evolved_model
        
        return current_model
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """获取进化摘要"""
        if not self.evolution_history:
            return {'status': 'no_evolution', 'message': '尚未执行进化'}
        
        total_iterations = len(self.evolution_history)
        successful_iterations = sum(1 for iter_result in self.evolution_history 
                                  if iter_result.improvement > self.config.min_improvement)
        
        total_improvement = self.best_performance - self.evolution_history[0].performance_before
        total_param_growth = (self.evolution_history[-1].parameters_after - 
                            self.evolution_history[0].parameters_before) / self.evolution_history[0].parameters_before
        
        avg_detection_time = np.mean([iter_result.detection_time for iter_result in self.evolution_history])
        avg_planning_time = np.mean([iter_result.planning_time for iter_result in self.evolution_history])
        avg_transfer_time = np.mean([iter_result.transfer_time for iter_result in self.evolution_history])
        
        return {
            'status': 'completed',
            'total_iterations': total_iterations,
            'successful_iterations': successful_iterations,
            'success_rate': successful_iterations / total_iterations,
            'initial_performance': self.evolution_history[0].performance_before,
            'final_performance': self.evolution_history[-1].performance_after,
            'best_performance': self.best_performance,
            'total_improvement': total_improvement,
            'total_parameter_growth': total_param_growth,
            'average_times': {
                'detection': avg_detection_time,
                'planning': avg_planning_time,
                'transfer': avg_transfer_time
            },
            'convergence_detected': self.convergence_detected
        }
    
    def visualize_evolution(self) -> str:
        """可视化进化过程"""
        if not self.evolution_history:
            return "📊 尚未执行进化"
        
        visualization = "📊 智能架构进化报告\n" + "="*60 + "\n"
        
        # 总体摘要
        summary = self.get_evolution_summary()
        visualization += f"\n🎯 进化摘要:\n"
        visualization += f"   总轮数: {summary['total_iterations']}\n"
        visualization += f"   成功轮数: {summary['successful_iterations']} (成功率: {summary['success_rate']:.1%})\n"
        visualization += f"   性能提升: {summary['initial_performance']:.4f} -> {summary['best_performance']:.4f} "
        visualization += f"(+{summary['total_improvement']:.4f})\n"
        visualization += f"   参数增长: {summary['total_parameter_growth']:.1%}\n"
        
        # 详细迭代历史
        visualization += f"\n📈 迭代历史:\n"
        for i, iter_result in enumerate(self.evolution_history, 1):
            status_icon = "✅" if iter_result.improvement > self.config.min_improvement else "⏸️"
            
            visualization += f"\n{status_icon} 第{i}轮:\n"
            visualization += f"   瓶颈: {len(iter_result.bottleneck_reports)}个 | "
            visualization += f"变异: {len(iter_result.mutation_plans)}个\n"
            visualization += f"   性能: {iter_result.performance_before:.4f} -> {iter_result.performance_after:.4f} "
            visualization += f"({iter_result.improvement:+.4f})\n"
            visualization += f"   参数: {iter_result.parameter_growth:+.1%} | "
            visualization += f"耗时: {iter_result.total_time:.1f}s\n"
        
        # 性能趋势
        performances = [iter_result.performance_after for iter_result in self.evolution_history]
        if len(performances) > 1:
            trend = "📈 上升" if performances[-1] > performances[0] else "📉 下降"
            visualization += f"\n{trend} 性能趋势: "
            visualization += " -> ".join([f"{p:.3f}" for p in performances[:5]])
            if len(performances) > 5:
                visualization += " -> ..."
        
        return visualization
    
    def export_best_model(self, save_path: str):
        """导出最佳模型"""
        if self.best_model is not None:
            torch.save({
                'model_state_dict': self.best_model.state_dict(),
                'performance': self.best_performance,
                'evolution_history': self.evolution_history,
                'config': self.config
            }, save_path)
            logger.info(f"最佳模型已保存到: {save_path}")
        else:
            logger.warning("无最佳模型可导出")
    
    def clear_history(self):
        """清理历史记录"""
        self.evolution_history.clear()
        self.best_model = None
        self.best_performance = float('-inf')
        self.performance_plateau_count = 0
        self.convergence_detected = False
        
        # 清理子组件缓存
        self.bottleneck_detector.clear_cache()
        
        logger.info("已清理进化历史")