"""
增强收敛监控器

解决原始监控器过于保守的问题，提供更智能的收敛判断
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from collections import deque

logger = logging.getLogger(__name__)


class EnhancedConvergenceMonitor:
    """
    增强收敛监控器
    
    核心改进：
    1. 更灵活的收敛检测标准
    2. 基于性能态势的动态调整
    3. 多维度收敛评估
    4. 支持探索性变异
    """
    
    def __init__(self, mode='balanced'):
        # 性能历史追踪
        self.performance_history = deque(maxlen=25)
        self.loss_history = deque(maxlen=25) 
        self.gradient_norm_history = deque(maxlen=20)
        
        # 变异历史追踪
        self.last_morphogenesis_epoch = -1
        self.morphogenesis_history = []
        self.post_morphogenesis_performance = []
        
        # 模式配置
        self.mode = mode
        self.config = self._get_mode_config(mode)
        
        # 收敛状态跟踪
        self.convergence_streak = 0
        self.stagnation_streak = 0
        
    def _get_mode_config(self, mode: str) -> Dict[str, Any]:
        """获取模式配置"""
        
        configs = {
            'aggressive': {
                'min_epochs_between_morphogenesis': 3,  # 最小间隔3epoch
                'convergence_patience': 3,               # 收敛检测耐心值
                'stability_threshold': 0.05,             # 更宽松的稳定性阈值
                'improvement_threshold': 0.005,          # 更低的改进阈值
                'saturation_window': 4,                  # 更短的饱和检测窗口
                'urgency_threshold': 0.3,                # 更低的紧急度阈值
                'exploration_enabled': True,             # 启用探索性变异
                'exploration_interval': 8,               # 探索性变异间隔
            },
            'balanced': {
                'min_epochs_between_morphogenesis': 5,  # 平衡的最小间隔
                'convergence_patience': 4,               # 平衡的收敛检测
                'stability_threshold': 0.03,             # 平衡的稳定性阈值
                'improvement_threshold': 0.008,          # 平衡的改进阈值
                'saturation_window': 5,                  # 平衡的饱和检测窗口
                'urgency_threshold': 0.4,                # 平衡的紧急度阈值
                'exploration_enabled': True,             # 启用探索性变异
                'exploration_interval': 12,              # 探索性变异间隔
            },
            'conservative': {
                'min_epochs_between_morphogenesis': 8,  # 保守的最小间隔
                'convergence_patience': 6,               # 保守的收敛检测
                'stability_threshold': 0.02,             # 严格的稳定性阈值
                'improvement_threshold': 0.01,           # 较高的改进阈值
                'saturation_window': 8,                  # 较长的饱和检测窗口
                'urgency_threshold': 0.6,                # 较高的紧急度阈值
                'exploration_enabled': False,            # 禁用探索性变异
                'exploration_interval': 20,              # 很长的探索间隔
            }
        }
        
        return configs.get(mode, configs['balanced'])
    
    def should_allow_morphogenesis(self, 
                                  current_epoch: int,
                                  current_performance: float,
                                  current_loss: float,
                                  gradient_norm: Optional[float] = None) -> Dict[str, Any]:
        """
        判断是否应该允许形态发生（增强版）
        """
        
        # 更新历史
        self.performance_history.append(current_performance)
        self.loss_history.append(current_loss)
        if gradient_norm is not None:
            self.gradient_norm_history.append(gradient_norm)
        
        # 检查基本间隔要求
        epochs_since_last = current_epoch - self.last_morphogenesis_epoch
        min_interval = self.config['min_epochs_between_morphogenesis']
        
        # 1. 硬性最小间隔检查（较短）
        if epochs_since_last < max(2, min_interval // 2):  # 至少2个epoch
            return {
                'allow': False,
                'reason': 'minimum_interval_not_met',
                'suggestion': f'等待至少{max(2, min_interval // 2)}个epoch的硬性间隔',
                'epochs_to_wait': max(2, min_interval // 2) - epochs_since_last,
                'confidence': 0.95
            }
        
        # 2. 多维度收敛分析
        convergence_analysis = self._enhanced_convergence_analysis()
        
        # 3. 性能态势分析
        performance_situation = self._analyze_performance_situation()
        
        # 4. 变异紧急度评估
        urgency_analysis = self._evaluate_morphogenesis_urgency(current_epoch, performance_situation)
        
        # 5. 智能决策逻辑
        decision = self._make_intelligent_decision(
            epochs_since_last, 
            convergence_analysis, 
            performance_situation, 
            urgency_analysis
        )
        
        # 更新状态（只更新统计，不更新last_morphogenesis_epoch）
        # last_morphogenesis_epoch 应该在实际执行变异后才更新
        if decision['allow']:
            self.convergence_streak = 0
            self.stagnation_streak = 0
        else:
            self.convergence_streak += 1
            if performance_situation['trend'] == 'stagnant':
                self.stagnation_streak += 1
        
        # 添加详细信息
        decision.update({
            'convergence_analysis': convergence_analysis,
            'performance_situation': performance_situation,
            'urgency_analysis': urgency_analysis,
            'epochs_since_last': epochs_since_last,
            'mode': self.mode
        })
        
        return decision
    
    def _enhanced_convergence_analysis(self) -> Dict[str, Any]:
        """增强的收敛分析"""
        
        if len(self.performance_history) < 3:
            return {
                'status': 'insufficient_data',
                'converged': False,
                'stability_score': 0.0,
                'confidence': 0.0
            }
        
        performance_data = list(self.performance_history)
        
        # 1. 性能稳定性分析
        recent_window = min(self.config['convergence_patience'], len(performance_data))
        recent_performance = performance_data[-recent_window:]
        
        performance_std = np.std(recent_performance)
        performance_mean = np.mean(recent_performance)
        relative_std = performance_std / max(performance_mean, 0.01)
        
        stability_score = max(0, 1 - relative_std / self.config['stability_threshold'])
        
        # 2. 趋势分析
        if len(recent_performance) >= 3:
            trend_slope = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
            trend_direction = 'improving' if trend_slope > 0.001 else 'declining' if trend_slope < -0.001 else 'stable'
        else:
            trend_slope = 0
            trend_direction = 'unknown'
        
        # 3. 损失收敛分析
        loss_converged = True
        if len(self.loss_history) >= 3:
            recent_loss = list(self.loss_history)[-3:]
            loss_trend = np.polyfit(range(len(recent_loss)), recent_loss, 1)[0]
            loss_volatility = np.std(recent_loss) / max(np.mean(recent_loss), 0.01)
            loss_converged = abs(loss_trend) < 0.02 and loss_volatility < 0.1
        
        # 4. 梯度稳定性分析
        gradient_stable = True
        if len(self.gradient_norm_history) >= 3:
            recent_grads = list(self.gradient_norm_history)[-3:]
            grad_std = np.std(recent_grads)
            grad_mean = np.mean(recent_grads)
            if grad_mean > 0:
                grad_relative_std = grad_std / grad_mean
                gradient_stable = grad_relative_std < 0.5  # 更宽松
        
        # 5. 综合判断（更宽松的标准）
        converged = (
            stability_score > 0.4 and  # 降低稳定性要求
            loss_converged and 
            gradient_stable
        )
        
        confidence = (stability_score + (1 if loss_converged else 0) + (1 if gradient_stable else 0)) / 3
        
        return {
            'status': 'converged' if converged else 'converging',
            'converged': converged,
            'stability_score': stability_score,
            'relative_std': relative_std,
            'trend_direction': trend_direction,
            'trend_slope': trend_slope,
            'loss_converged': loss_converged,
            'gradient_stable': gradient_stable,
            'confidence': confidence
        }
    
    def _analyze_performance_situation(self) -> Dict[str, Any]:
        """分析性能态势"""
        
        if len(self.performance_history) < 4:
            return {
                'trend': 'unknown',
                'situation': 'insufficient_data',
                'urgency': 0.0
            }
        
        performance_data = list(self.performance_history)
        
        # 短期趋势（最近4个点）
        short_term = performance_data[-4:]
        short_slope = np.polyfit(range(len(short_term)), short_term, 1)[0]
        
        # 中期趋势（最近8个点或全部）
        mid_term_len = min(8, len(performance_data))
        mid_term = performance_data[-mid_term_len:]
        mid_slope = np.polyfit(range(len(mid_term)), mid_term, 1)[0]
        
        # 改进率分析
        window = min(self.config['saturation_window'], len(performance_data))
        recent_perf = performance_data[-window:]
        
        if len(recent_perf) >= 2:
            first_half = recent_perf[:len(recent_perf)//2]
            second_half = recent_perf[len(recent_perf)//2:]
            improvement = np.mean(second_half) - np.mean(first_half)
        else:
            improvement = 0
        
        # 分类性能态势
        if short_slope > 0.002:
            trend = 'improving'
            urgency = 0.2  # 正在改进，不急
        elif short_slope < -0.002:
            trend = 'declining'
            urgency = 0.8  # 性能下降，紧急
        elif abs(improvement) < self.config['improvement_threshold']:
            trend = 'stagnant'
            urgency = 0.6  # 停滞，需要变异
        else:
            trend = 'stable'
            urgency = 0.3  # 稳定，不太急
        
        # 确定整体情况
        if trend == 'declining':
            situation = 'performance_degradation'
        elif trend == 'stagnant' and len(performance_data) >= 6:
            situation = 'performance_plateau' 
        elif trend == 'improving':
            situation = 'performance_growth'
        else:
            situation = 'stable_performance'
        
        return {
            'trend': trend,
            'situation': situation,
            'urgency': urgency,
            'short_slope': short_slope,
            'mid_slope': mid_slope,
            'improvement': improvement,
            'stagnation_streak': self.stagnation_streak
        }
    
    def _evaluate_morphogenesis_urgency(self, current_epoch: int, performance_situation: Dict[str, Any]) -> Dict[str, Any]:
        """评估变异紧急度"""
        
        base_urgency = performance_situation['urgency']
        
        # 调整因子
        adjustment_factors = []
        
        # 1. 停滞时间调整
        if self.stagnation_streak > 3:
            adjustment_factors.append(0.2 * min(self.stagnation_streak / 5, 1.0))
        
        # 2. 长期无变异调整
        epochs_since_last = current_epoch - self.last_morphogenesis_epoch
        if epochs_since_last > self.config['exploration_interval']:
            adjustment_factors.append(0.3)  # 探索性变异
        
        # 3. 模式调整
        if self.mode == 'aggressive':
            adjustment_factors.append(0.2)  # 积极模式更容易变异
        elif self.mode == 'conservative':
            adjustment_factors.append(-0.2)  # 保守模式不易变异
        
        # 4. 性能水平调整
        if self.performance_history:
            current_perf = self.performance_history[-1]
            if current_perf < 0.6:  # 性能较低时更需要变异
                adjustment_factors.append(0.2)
            elif current_perf > 0.85:  # 性能很高时不急于变异
                adjustment_factors.append(-0.1)
        
        total_adjustment = sum(adjustment_factors)
        final_urgency = max(0, min(1, base_urgency + total_adjustment))
        
        return {
            'base_urgency': base_urgency,
            'adjustments': adjustment_factors,
            'total_adjustment': total_adjustment,
            'final_urgency': final_urgency,
            'urgency_level': 'high' if final_urgency > 0.7 else 'medium' if final_urgency > 0.4 else 'low'
        }
    
    def _make_intelligent_decision(self, 
                                 epochs_since_last: int,
                                 convergence_analysis: Dict[str, Any],
                                 performance_situation: Dict[str, Any],
                                 urgency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """智能决策逻辑"""
        
        # 基础条件检查
        min_interval = self.config['min_epochs_between_morphogenesis']
        urgency_threshold = self.config['urgency_threshold']
        
        # 1. 紧急情况：立即允许变异
        if urgency_analysis['final_urgency'] > 0.8:
            return {
                'allow': True,
                'reason': 'high_urgency_morphogenesis',
                'suggestion': '检测到紧急情况（性能下降/严重停滞），立即执行变异',
                'confidence': 0.9,
                'decision_type': 'urgent'
            }
        
        # 2. 探索性变异：定期探索
        if (self.config['exploration_enabled'] and 
            epochs_since_last >= self.config['exploration_interval']):
            return {
                'allow': True,
                'reason': 'exploratory_morphogenesis',
                'suggestion': '执行探索性变异以发现新的架构优化机会',
                'confidence': 0.7,
                'decision_type': 'exploratory'
            }
        
        # 3. 标准决策：基于收敛和紧急度
        if epochs_since_last >= min_interval:
            # 检查收敛状态
            if convergence_analysis['converged']:
                # 已收敛，检查是否需要变异
                if urgency_analysis['final_urgency'] >= urgency_threshold:
                    return {
                        'allow': True,
                        'reason': 'converged_with_sufficient_urgency',
                        'suggestion': '网络已收敛且检测到足够的变异必要性',
                        'confidence': convergence_analysis['confidence'] * 0.8,
                        'decision_type': 'standard'
                    }
                else:
                    return {
                        'allow': False,
                        'reason': 'converged_but_low_urgency',
                        'suggestion': '网络已收敛但变异紧急度不足，继续训练',
                        'confidence': 0.7
                    }
            else:
                # 未完全收敛，检查是否停滞
                if performance_situation['trend'] == 'stagnant' and self.stagnation_streak >= 4:
                    return {
                        'allow': True,
                        'reason': 'stagnation_override',
                        'suggestion': '网络虽未完全收敛但出现明显停滞，执行变异',
                        'confidence': 0.6,
                        'decision_type': 'stagnation_break'
                    }
                else:
                    return {
                        'allow': False,
                        'reason': 'still_converging',
                        'suggestion': '网络仍在收敛过程中，等待稳定',
                        'confidence': 0.8
                    }
        
        # 4. 默认：间隔不足
        return {
            'allow': False,
            'reason': 'interval_insufficient',
            'suggestion': f'距离上次变异仅{epochs_since_last}个epoch，需要更多时间',
            'confidence': 0.9
        }
    
    def set_mode(self, mode: str):
        """设置监控模式"""
        if mode in ['aggressive', 'balanced', 'conservative']:
            self.mode = mode
            self.config = self._get_mode_config(mode)
            logger.info(f"收敛监控模式已设置为: {mode}")
        else:
            logger.warning(f"未知模式: {mode}，保持当前模式: {self.mode}")
    
    def record_morphogenesis_execution(self, current_epoch: int, mutation_info: Dict[str, Any]):
        """记录实际执行的变异"""
        self.last_morphogenesis_epoch = current_epoch
        self.morphogenesis_history.append({
            'epoch': current_epoch,
            'mutation_info': mutation_info,
            'timestamp': len(self.morphogenesis_history)
        })
        
        # 限制历史长度
        if len(self.morphogenesis_history) > 50:
            self.morphogenesis_history = self.morphogenesis_history[-50:]
        
        logger.info(f"📝 记录变异执行: epoch {current_epoch}, 类型 {mutation_info.get('mutation_type', 'unknown')}")
    
    def record_morphogenesis(self, epoch: int, morphogenesis_type: str, performance_before: float = 0.0):
        """兼容性方法：记录变异事件"""
        mutation_info = {
            'mutation_type': morphogenesis_type,
            'performance_before': performance_before
        }
        self.record_morphogenesis_execution(epoch, mutation_info)
    
    def reset_history(self):
        """重置历史记录"""
        self.performance_history.clear()
        self.loss_history.clear() 
        self.gradient_norm_history.clear()
        self.morphogenesis_history.clear()
        self.post_morphogenesis_performance.clear()
        self.convergence_streak = 0
        self.stagnation_streak = 0
        logger.info("收敛监控历史已重置")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            'mode': self.mode,
            'last_morphogenesis_epoch': self.last_morphogenesis_epoch,
            'convergence_streak': self.convergence_streak,
            'stagnation_streak': self.stagnation_streak,
            'performance_history_length': len(self.performance_history),
            'config': self.config
        }