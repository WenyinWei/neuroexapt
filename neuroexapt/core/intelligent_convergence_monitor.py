"""
智能收敛监控模块
监控网络在变异后的收敛状态，决定何时进行下一次变异
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from collections import deque

logger = logging.getLogger(__name__)


class IntelligentConvergenceMonitor:
    """
    智能收敛监控器
    
    核心思想：
    1. 变异后必须等待网络充分适应新架构
    2. 检测性能饱和和收敛稳定性
    3. 只有在网络稳定且出现瓶颈时才允许下一次变异
    """
    
    def __init__(self):
        # 性能历史追踪
        self.performance_history = deque(maxlen=20)
        self.loss_history = deque(maxlen=20) 
        self.gradient_norm_history = deque(maxlen=15)
        
        # 变异历史追踪
        self.last_morphogenesis_epoch = -1
        self.morphogenesis_history = []
        self.post_morphogenesis_performance = []
        
        # 收敛检测参数
        self.min_epochs_between_morphogenesis = 8  # 变异间最小间隔
        self.convergence_patience = 5              # 收敛检测耐心值
        self.stability_threshold = 0.02            # 稳定性阈值
        
        # 性能饱和检测
        self.saturation_window = 6                 # 饱和检测窗口
        self.improvement_threshold = 0.01          # 改进阈值
        
    def should_allow_morphogenesis(self, 
                                  current_epoch: int,
                                  current_performance: float,
                                  current_loss: float,
                                  gradient_norm: Optional[float] = None) -> Dict[str, Any]:
        """
        判断是否应该允许形态发生
        
        Returns:
            Dict包含是否允许、原因、建议等信息
        """
        
        # 更新历史
        self.performance_history.append(current_performance)
        self.loss_history.append(current_loss)
        if gradient_norm is not None:
            self.gradient_norm_history.append(gradient_norm)
        
        # 1. 检查变异间隔
        epochs_since_last = current_epoch - self.last_morphogenesis_epoch
        if epochs_since_last < self.min_epochs_between_morphogenesis:
            return {
                'allow': False,
                'reason': 'insufficient_time_since_last_morphogenesis',
                'suggestion': f'等待至少{self.min_epochs_between_morphogenesis}个epoch后再变异',
                'epochs_to_wait': self.min_epochs_between_morphogenesis - epochs_since_last,
                'confidence': 0.9
            }
        
        # 2. 检查网络收敛状态
        convergence_analysis = self._analyze_convergence()
        if not convergence_analysis['converged']:
            return {
                'allow': False,
                'reason': 'network_not_converged',
                'suggestion': '网络仍在适应上次变异，需要更多时间收敛',
                'convergence_info': convergence_analysis,
                'confidence': 0.8
            }
        
        # 3. 检查性能饱和状态
        saturation_analysis = self._analyze_performance_saturation()
        if not saturation_analysis['saturated']:
            return {
                'allow': False,
                'reason': 'performance_still_improving',
                'suggestion': '网络性能仍在提升，无需变异',
                'saturation_info': saturation_analysis,
                'confidence': 0.7
            }
        
        # 4. 深度分析变异必要性
        necessity_analysis = self._analyze_morphogenesis_necessity(current_epoch)
        
        if necessity_analysis['urgency_score'] < 0.6:
            return {
                'allow': False,
                'reason': 'low_morphogenesis_urgency',
                'suggestion': '当前网络状态良好，变异不紧急',
                'necessity_info': necessity_analysis,
                'confidence': 0.6
            }
        
        # 5. 所有条件满足，允许变异
        return {
            'allow': True,
            'reason': 'optimal_morphogenesis_timing',
            'confidence': min(0.95, necessity_analysis['urgency_score']),
            'convergence_info': convergence_analysis,
            'saturation_info': saturation_analysis,
            'necessity_info': necessity_analysis,
            'suggestion': '网络已收敛且出现瓶颈，建议进行智能变异'
        }
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """分析网络收敛状态"""
        
        if len(self.performance_history) < self.convergence_patience:
            return {
                'converged': False,
                'reason': 'insufficient_data',
                'stability_score': 0.0
            }
        
        # 计算性能稳定性
        recent_performance = list(self.performance_history)[-self.convergence_patience:]
        performance_std = np.std(recent_performance)
        performance_mean = np.mean(recent_performance)
        
        # 相对标准差作为稳定性指标
        relative_std = performance_std / max(performance_mean, 0.01)
        stability_score = max(0, 1 - relative_std / self.stability_threshold)
        
        # 检查损失收敛
        if len(self.loss_history) >= self.convergence_patience:
            recent_loss = list(self.loss_history)[-self.convergence_patience:]
            loss_trend = np.polyfit(range(len(recent_loss)), recent_loss, 1)[0]
            # 损失下降趋势应该平缓
            loss_converged = abs(loss_trend) < 0.01
        else:
            loss_converged = False
        
        # 检查梯度范数稳定性
        gradient_stable = True
        if len(self.gradient_norm_history) >= 5:
            recent_grads = list(self.gradient_norm_history)[-5:]
            grad_std = np.std(recent_grads)
            grad_mean = np.mean(recent_grads)
            if grad_mean > 0:
                grad_relative_std = grad_std / grad_mean
                gradient_stable = grad_relative_std < 0.3
        
        converged = (stability_score > 0.7 and loss_converged and gradient_stable)
        
        return {
            'converged': converged,
            'stability_score': stability_score,
            'relative_std': relative_std,
            'loss_converged': loss_converged,
            'gradient_stable': gradient_stable,
            'performance_trend': np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        }
    
    def _analyze_performance_saturation(self) -> Dict[str, Any]:
        """分析性能饱和状态"""
        
        if len(self.performance_history) < self.saturation_window:
            return {
                'saturated': False,
                'reason': 'insufficient_data',
                'saturation_score': 0.0
            }
        
        recent_performance = list(self.performance_history)[-self.saturation_window:]
        
        # 计算最近的改进率
        first_half = recent_performance[:len(recent_performance)//2]
        second_half = recent_performance[len(recent_performance)//2:]
        
        improvement = np.mean(second_half) - np.mean(first_half)
        relative_improvement = improvement / max(np.mean(first_half), 0.01)
        
        # 计算饱和分数
        saturation_score = max(0, 1 - relative_improvement / self.improvement_threshold)
        
        # 检查是否出现性能停滞或下降
        performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        saturated = (saturation_score > 0.8 or performance_trend <= 0)
        
        return {
            'saturated': saturated,
            'saturation_score': saturation_score,
            'improvement': improvement,
            'relative_improvement': relative_improvement,
            'performance_trend': performance_trend,
            'recent_performance': recent_performance
        }
    
    def _analyze_morphogenesis_necessity(self, current_epoch: int) -> Dict[str, Any]:
        """分析变异必要性"""
        
        urgency_factors = []
        
        # 1. 性能停滞紧急度
        if len(self.performance_history) >= 10:
            recent_10 = list(self.performance_history)[-10:]
            max_perf = max(recent_10)
            current_perf = recent_10[-1]
            stagnation_urgency = (max_perf - current_perf) / max(max_perf, 0.01)
            urgency_factors.append(('stagnation', min(1.0, stagnation_urgency * 3)))
        
        # 2. 训练进度紧急度
        training_progress = current_epoch / 80.0  # 假设总共80个epoch
        progress_urgency = min(1.0, training_progress * 1.5)  # 后期更紧急
        urgency_factors.append(('training_progress', progress_urgency))
        
        # 3. 历史变异效果
        if self.morphogenesis_history:
            last_morphogenesis = self.morphogenesis_history[-1]
            epochs_since = current_epoch - last_morphogenesis['epoch']
            # 如果上次变异效果不佳，提高紧急度
            if last_morphogenesis.get('performance_improvement', 0) < 0.02:
                history_urgency = min(1.0, epochs_since / 10.0)
                urgency_factors.append(('poor_history', history_urgency))
        
        # 4. 性能水平绝对值
        if self.performance_history:
            current_perf = self.performance_history[-1]
            # 性能越低，变异越紧急
            performance_urgency = max(0, 1 - current_perf / 0.9)  # 90%以上不紧急
            urgency_factors.append(('absolute_performance', performance_urgency))
        
        # 综合紧急度评分
        if urgency_factors:
            weights = [1.0, 0.8, 0.6, 0.7][:len(urgency_factors)]
            urgency_score = sum(w * f[1] for w, f in zip(weights, urgency_factors)) / sum(weights)
        else:
            urgency_score = 0.5
        
        return {
            'urgency_score': urgency_score,
            'urgency_factors': urgency_factors,
            'recommendation': self._generate_urgency_recommendation(urgency_score)
        }
    
    def _generate_urgency_recommendation(self, urgency_score: float) -> str:
        """生成紧急度建议"""
        if urgency_score > 0.8:
            return "强烈建议立即进行激进变异以突破瓶颈"
        elif urgency_score > 0.6:
            return "建议进行适度变异以改善性能"
        elif urgency_score > 0.4:
            return "可考虑保守变异，但不紧急"
        else:
            return "当前状态良好，建议继续训练"
    
    def record_morphogenesis(self, 
                           epoch: int, 
                           morphogenesis_type: str,
                           performance_before: float,
                           performance_after: Optional[float] = None) -> None:
        """记录变异事件"""
        
        self.last_morphogenesis_epoch = epoch
        
        morphogenesis_record = {
            'epoch': epoch,
            'type': morphogenesis_type,
            'performance_before': performance_before,
            'performance_after': performance_after
        }
        
        # 如果有之前的记录，计算改进
        if len(self.morphogenesis_history) > 0 and performance_after is not None:
            prev_record = self.morphogenesis_history[-1]
            improvement = performance_after - prev_record.get('performance_before', 0)
            morphogenesis_record['performance_improvement'] = improvement
        
        self.morphogenesis_history.append(morphogenesis_record)
        logger.info(f"📝 记录变异: Epoch {epoch}, 类型: {morphogenesis_type}")
    
    def get_convergence_report(self) -> Dict[str, Any]:
        """获取收敛状态报告"""
        
        if not self.performance_history:
            return {'status': 'no_data'}
        
        convergence_info = self._analyze_convergence()
        saturation_info = self._analyze_performance_saturation()
        
        return {
            'current_performance': self.performance_history[-1] if self.performance_history else 0,
            'performance_trend': convergence_info.get('performance_trend', 0),
            'stability_score': convergence_info.get('stability_score', 0),
            'saturation_score': saturation_info.get('saturation_score', 0),
            'converged': convergence_info.get('converged', False),
            'saturated': saturation_info.get('saturated', False),
            'epochs_since_last_morphogenesis': len(self.performance_history) - 1 if self.last_morphogenesis_epoch >= 0 else -1,
            'morphogenesis_count': len(self.morphogenesis_history)
        }