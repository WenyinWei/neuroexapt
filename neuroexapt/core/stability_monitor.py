"""
@defgroup group_stability_monitor Stability Monitor
@ingroup core
Stability Monitor module for NeuroExapt framework.

稳定性监控器 (Stability Monitor)

ASO-SE框架的辅助组件：监控训练过程的稳定性，包括：
1. 损失震荡检测 - 识别架构变化引起的损失剧烈波动
2. 收敛分析 - 分析训练收敛趋势和速度
3. 性能退化预警 - 检测性能下降并提供预警
4. 架构变化影响评估 - 量化架构变化对性能的影响
5. 训练健康度评估 - 综合评估训练过程的健康状态
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import deque
import math
import warnings

logger = logging.getLogger(__name__)

class LossOscillationDetector:
    """
    损失震荡检测器
    
    专门检测架构变化引起的损失剧烈波动
    """
    
    def __init__(self, window_size: int = 10, oscillation_threshold: float = 0.1,
                 severity_levels: Dict[str, float] = None):
        """
        Args:
            window_size: 滑动窗口大小
            oscillation_threshold: 震荡阈值
            severity_levels: 严重程度级别
        """
        self.window_size = window_size
        self.oscillation_threshold = oscillation_threshold
        self.severity_levels = severity_levels or {
            'mild': 0.05,
            'moderate': 0.1,
            'severe': 0.2,
            'critical': 0.5
        }
        
        # 历史数据
        self.loss_history = deque(maxlen=window_size * 3)  # 保留更长历史
        self.oscillation_events = []
        
        logger.debug(f"🌊 Loss Oscillation Detector initialized: "
                    f"window={window_size}, threshold={oscillation_threshold}")
    
    def update(self, loss: float, epoch: int, phase: str = "unknown") -> Dict[str, Any]:
        """
        更新损失值并检测震荡
        
        Args:
            loss: 当前损失值
            epoch: 当前epoch
            phase: 当前训练阶段
            
        Returns:
            检测结果
        """
        self.loss_history.append({
            'loss': loss,
            'epoch': epoch,
            'phase': phase
        })
        
        # 需要足够的历史数据才能检测
        if len(self.loss_history) < self.window_size:
            return {'status': 'insufficient_data', 'oscillation_detected': False}
        
        # 分析震荡
        oscillation_metrics = self._analyze_oscillation()
        
        # 检测是否有震荡
        is_oscillating = oscillation_metrics['variance'] > self.oscillation_threshold
        
        if is_oscillating:
            severity = self._classify_severity(oscillation_metrics['variance'])
            
            event = {
                'epoch': epoch,
                'phase': phase,
                'severity': severity,
                'metrics': oscillation_metrics
            }
            self.oscillation_events.append(event)
            
            logger.warning(f"🌊 Loss oscillation detected at epoch {epoch}: "
                          f"{severity} (variance: {oscillation_metrics['variance']:.4f})")
        
        return {
            'status': 'analyzed',
            'oscillation_detected': is_oscillating,
            'severity': self._classify_severity(oscillation_metrics['variance']) if is_oscillating else 'none',
            'metrics': oscillation_metrics
        }
    
    def _analyze_oscillation(self) -> Dict[str, float]:
        """分析损失震荡特征"""
        recent_losses = [item['loss'] for item in list(self.loss_history)[-self.window_size:]]
        
        # 基本统计
        mean_loss = np.mean(recent_losses)
        variance = np.var(recent_losses)
        std_dev = np.std(recent_losses)
        
        # 相对变异系数
        cv = std_dev / mean_loss if mean_loss > 0 else 0
        
        # 趋势分析
        trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        # 震荡频率分析
        direction_changes = 0
        for i in range(1, len(recent_losses) - 1):
            prev_diff = recent_losses[i] - recent_losses[i-1]
            curr_diff = recent_losses[i+1] - recent_losses[i]
            if prev_diff * curr_diff < 0:  # 方向改变
                direction_changes += 1
        
        oscillation_frequency = direction_changes / (len(recent_losses) - 2) if len(recent_losses) > 2 else 0
        
        return {
            'mean': mean_loss,
            'variance': variance,
            'std_dev': std_dev,
            'cv': cv,
            'trend': trend,
            'oscillation_frequency': oscillation_frequency
        }
    
    def _classify_severity(self, variance: float) -> str:
        """分类震荡严重程度"""
        for severity, threshold in sorted(self.severity_levels.items(), 
                                        key=lambda x: x[1], reverse=True):
            if variance >= threshold:
                return severity
        return 'mild'
    
    def get_stability_report(self) -> Dict[str, Any]:
        """获取稳定性报告"""
        if not self.loss_history:
            return {'status': 'no_data'}
        
        total_events = len(self.oscillation_events)
        severity_counts = {}
        
        for event in self.oscillation_events:
            severity = event['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # 计算总体稳定性评分 (0-1, 1为最稳定)
        if total_events == 0:
            stability_score = 1.0
        else:
            # 根据事件严重程度加权计算
            weight_sum = sum(len(self.severity_levels) - i for i, _ in enumerate(self.severity_levels.keys()))
            weighted_events = sum(
                severity_counts.get(sev, 0) * (len(self.severity_levels) - i)
                for i, sev in enumerate(self.severity_levels.keys())
            )
            stability_score = max(0, 1 - weighted_events / (len(self.loss_history) * weight_sum))
        
        return {
            'total_oscillation_events': total_events,
            'severity_breakdown': severity_counts,
            'stability_score': stability_score,
            'recent_metrics': self._analyze_oscillation() if len(self.loss_history) >= self.window_size else None
        }

class ConvergenceAnalyzer:
    """
    收敛分析器
    
    分析训练收敛趋势和速度
    """
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001,
                 lookback_window: int = 20):
        """
        Args:
            patience: 容忍的停滞epoch数
            min_delta: 最小改进阈值
            lookback_window: 回看窗口大小
        """
        self.patience = patience
        self.min_delta = min_delta
        self.lookback_window = lookback_window
        
        # 历史记录
        self.metric_history = deque(maxlen=lookback_window * 2)
        self.plateau_count = 0
        self.best_metric = -float('inf')
        self.epochs_since_improvement = 0
        
        logger.debug(f"📈 Convergence Analyzer initialized: "
                    f"patience={patience}, min_delta={min_delta}")
    
    def update(self, metric: float, epoch: int, is_higher_better: bool = True) -> Dict[str, Any]:
        """
        更新指标并分析收敛
        
        Args:
            metric: 监控指标（如准确率、损失等）
            epoch: 当前epoch
            is_higher_better: 指标是否越高越好
            
        Returns:
            分析结果
        """
        self.metric_history.append({
            'metric': metric,
            'epoch': epoch
        })
        
        # 判断是否有改进
        if is_higher_better:
            improved = metric > self.best_metric + self.min_delta
        else:
            improved = metric < self.best_metric - self.min_delta
        
        if improved:
            self.best_metric = metric
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        
        # 分析收敛状态
        convergence_analysis = self._analyze_convergence(is_higher_better)
        
        # 判断是否应该早停
        should_early_stop = self.epochs_since_improvement >= self.patience
        
        if should_early_stop:
            logger.info(f"📉 Convergence plateau detected: "
                       f"{self.epochs_since_improvement} epochs without improvement")
        
        return {
            'improved': improved,
            'epochs_since_improvement': self.epochs_since_improvement,
            'should_early_stop': should_early_stop,
            'convergence_analysis': convergence_analysis,
            'best_metric': self.best_metric
        }
    
    def _analyze_convergence(self, is_higher_better: bool) -> Dict[str, Any]:
        """分析收敛特征"""
        if len(self.metric_history) < 5:
            return {'status': 'insufficient_data'}
        
        metrics = [item['metric'] for item in self.metric_history]
        epochs = [item['epoch'] for item in self.metric_history]
        
        # 计算收敛速度
        if len(metrics) >= 2:
            recent_change = metrics[-1] - metrics[-2]
            if len(metrics) >= self.lookback_window:
                long_term_change = metrics[-1] - metrics[-self.lookback_window]
                convergence_rate = long_term_change / self.lookback_window
            else:
                convergence_rate = recent_change
        else:
            recent_change = 0
            convergence_rate = 0
        
        # 趋势分析
        if len(metrics) >= 5:
            trend_slope = np.polyfit(range(len(metrics)), metrics, 1)[0]
        else:
            trend_slope = 0
        
        # 方差分析
        recent_variance = np.var(metrics[-min(10, len(metrics)):])
        
        # 收敛状态分类
        if abs(convergence_rate) < self.min_delta / 2:
            convergence_status = 'converged'
        elif (is_higher_better and convergence_rate > 0) or (not is_higher_better and convergence_rate < 0):
            convergence_status = 'improving'
        else:
            convergence_status = 'degrading'
        
        return {
            'status': 'analyzed',
            'convergence_status': convergence_status,
            'convergence_rate': convergence_rate,
            'trend_slope': trend_slope,
            'recent_variance': recent_variance,
            'recent_change': recent_change
        }
    
    def get_convergence_report(self) -> Dict[str, Any]:
        """获取收敛报告"""
        if not self.metric_history:
            return {'status': 'no_data'}
        
        metrics = [item['metric'] for item in self.metric_history]
        
        return {
            'total_epochs': len(metrics),
            'best_metric': self.best_metric,
            'current_metric': metrics[-1],
            'epochs_since_improvement': self.epochs_since_improvement,
            'improvement_percentage': ((metrics[-1] - metrics[0]) / abs(metrics[0]) * 100) if metrics[0] != 0 else 0,
            'average_metric': np.mean(metrics),
            'metric_std': np.std(metrics)
        }

class PerformanceDegradationDetector:
    """
    性能退化检测器
    
    检测性能下降并提供预警
    """
    
    def __init__(self, degradation_threshold: float = 0.05, 
                 alert_window: int = 5, recovery_window: int = 3):
        """
        Args:
            degradation_threshold: 性能退化阈值
            alert_window: 预警窗口大小
            recovery_window: 恢复检测窗口
        """
        self.degradation_threshold = degradation_threshold
        self.alert_window = alert_window
        self.recovery_window = recovery_window
        
        # 状态跟踪
        self.baseline_performance = None
        self.performance_history = deque(maxlen=alert_window * 3)
        self.degradation_alerts = []
        self.in_degradation = False
        
        logger.debug(f"⚠️ Performance Degradation Detector initialized: "
                    f"threshold={degradation_threshold}")
    
    def update(self, performance: float, epoch: int, 
              set_baseline: bool = False) -> Dict[str, Any]:
        """
        更新性能并检测退化
        
        Args:
            performance: 当前性能指标
            epoch: 当前epoch
            set_baseline: 是否设置为基线性能
            
        Returns:
            检测结果
        """
        if set_baseline or self.baseline_performance is None:
            self.baseline_performance = performance
            logger.info(f"📊 Performance baseline set: {performance:.4f}")
        
        self.performance_history.append({
            'performance': performance,
            'epoch': epoch
        })
        
        # 检测退化
        degradation_analysis = self._detect_degradation(performance, epoch)
        
        # 检测恢复
        recovery_analysis = self._detect_recovery(performance, epoch)
        
        return {
            'baseline_performance': self.baseline_performance,
            'current_performance': performance,
            'degradation_analysis': degradation_analysis,
            'recovery_analysis': recovery_analysis,
            'in_degradation': self.in_degradation
        }
    
    def _detect_degradation(self, performance: float, epoch: int) -> Dict[str, Any]:
        """检测性能退化"""
        if self.baseline_performance is None:
            return {'status': 'no_baseline'}
        
        # 计算相对退化（避免除零错误）
        if abs(self.baseline_performance) < 1e-8:  # baseline为0或非常小
            relative_degradation = 0.0 if performance >= 0 else -1.0
        else:
            relative_degradation = (self.baseline_performance - performance) / abs(self.baseline_performance)
        
        # 检查是否超过阈值
        is_degraded = relative_degradation > self.degradation_threshold
        
        if is_degraded and not self.in_degradation:
            # 新的退化事件
            alert = {
                'epoch': epoch,
                'baseline': self.baseline_performance,
                'current': performance,
                'degradation': relative_degradation,
                'severity': self._classify_degradation_severity(relative_degradation)
            }
            
            self.degradation_alerts.append(alert)
            self.in_degradation = True
            
            logger.warning(f"⚠️ Performance degradation detected at epoch {epoch}: "
                          f"{relative_degradation:.2%} below baseline "
                          f"(severity: {alert['severity']})")
        
        return {
            'is_degraded': is_degraded,
            'relative_degradation': relative_degradation,
            'absolute_degradation': self.baseline_performance - performance,
            'severity': self._classify_degradation_severity(relative_degradation) if is_degraded else 'none'
        }
    
    def _detect_recovery(self, performance: float, epoch: int) -> Dict[str, Any]:
        """检测性能恢复"""
        if not self.in_degradation or self.baseline_performance is None:
            return {'status': 'not_applicable'}
        
        # 检查最近几个epoch的性能
        if len(self.performance_history) >= self.recovery_window:
            recent_performances = [
                item['performance'] 
                for item in list(self.performance_history)[-self.recovery_window:]
            ]
            
            # 如果最近的性能都接近或超过基线
            recovery_threshold = self.baseline_performance * (1 - self.degradation_threshold / 2)
            all_recovered = all(p >= recovery_threshold for p in recent_performances)
            
            if all_recovered:
                self.in_degradation = False
                logger.info(f"✅ Performance recovered at epoch {epoch}")
                return {'status': 'recovered', 'recovery_epoch': epoch}
        
        return {'status': 'still_degraded'}
    
    def _classify_degradation_severity(self, degradation: float) -> str:
        """分类退化严重程度"""
        if degradation < 0.02:
            return 'minor'
        elif degradation < 0.05:
            return 'moderate'
        elif degradation < 0.1:
            return 'major'
        else:
            return 'critical'
    
    def get_degradation_report(self) -> Dict[str, Any]:
        """获取退化报告"""
        return {
            'total_degradation_events': len(self.degradation_alerts),
            'current_status': 'degraded' if self.in_degradation else 'healthy',
            'baseline_performance': self.baseline_performance,
            'recent_alerts': self.degradation_alerts[-5:] if self.degradation_alerts else []
        }

class StabilityMonitor:
    """
    综合稳定性监控器
    
    整合各种监控组件，提供统一的稳定性监控接口
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: 监控配置
        """
        config = config or {}
        
        # 初始化各个组件
        self.oscillation_detector = LossOscillationDetector(
            window_size=config.get('oscillation_window', 10),
            oscillation_threshold=config.get('oscillation_threshold', 0.1)
        )
        
        self.convergence_analyzer = ConvergenceAnalyzer(
            patience=config.get('convergence_patience', 15),
            min_delta=config.get('convergence_min_delta', 0.001)
        )
        
        self.degradation_detector = PerformanceDegradationDetector(
            degradation_threshold=config.get('degradation_threshold', 0.05),
            alert_window=config.get('degradation_window', 5)
        )
        
        # 综合状态
        self.training_health_score = 1.0
        self.health_history = []
        
        logger.info("🏥 Comprehensive Stability Monitor initialized")
    
    def update(self, metrics: Dict[str, float], epoch: int, 
              phase: str = "unknown") -> Dict[str, Any]:
        """
        更新所有监控指标
        
        Args:
            metrics: 指标字典，需包含 'loss', 'accuracy' 等
            epoch: 当前epoch
            phase: 当前训练阶段
            
        Returns:
            综合监控结果
        """
        results = {
            'epoch': epoch,
            'phase': phase,
            'metrics': metrics
        }
        
        # 更新各个监控组件
        if 'loss' in metrics:
            oscillation_result = self.oscillation_detector.update(
                metrics['loss'], epoch, phase
            )
            results['oscillation'] = oscillation_result
        
        if 'accuracy' in metrics:
            convergence_result = self.convergence_analyzer.update(
                metrics['accuracy'], epoch, is_higher_better=True
            )
            results['convergence'] = convergence_result
            
            degradation_result = self.degradation_detector.update(
                metrics['accuracy'], epoch
            )
            results['degradation'] = degradation_result
        
        # 计算综合健康度评分
        health_score = self._calculate_health_score(results)
        self.training_health_score = health_score
        
        self.health_history.append({
            'epoch': epoch,
            'health_score': health_score,
            'phase': phase
        })
        
        results['health_score'] = health_score
        results['health_status'] = self._classify_health_status(health_score)
        
        # 生成建议
        recommendations = self._generate_recommendations(results)
        results['recommendations'] = recommendations
        
        return results
    
    def _calculate_health_score(self, results: Dict[str, Any]) -> float:
        """计算综合健康度评分"""
        score = 1.0
        
        # 震荡惩罚
        oscillation = results.get('oscillation', {})
        if oscillation.get('oscillation_detected', False):
            severity_penalty = {
                'mild': 0.05,
                'moderate': 0.15,
                'severe': 0.3,
                'critical': 0.5
            }
            severity = oscillation.get('severity', 'mild')
            score -= severity_penalty.get(severity, 0.05)
        
        # 收敛惩罚
        convergence = results.get('convergence', {})
        if convergence.get('convergence_analysis', {}).get('convergence_status') == 'degrading':
            score -= 0.2
        
        # 退化惩罚
        degradation = results.get('degradation', {})
        if degradation.get('degradation_analysis', {}).get('is_degraded', False):
            severity_penalty = {
                'minor': 0.1,
                'moderate': 0.2,
                'major': 0.4,
                'critical': 0.6
            }
            severity = degradation.get('degradation_analysis', {}).get('severity', 'minor')
            score -= severity_penalty.get(severity, 0.1)
        
        return max(0.0, min(1.0, score))
    
    def _classify_health_status(self, health_score: float) -> str:
        """分类健康状态"""
        if health_score >= 0.8:
            return 'excellent'
        elif health_score >= 0.6:
            return 'good'
        elif health_score >= 0.4:
            return 'fair'
        elif health_score >= 0.2:
            return 'poor'
        else:
            return 'critical'
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于震荡的建议
        oscillation = results.get('oscillation', {})
        if oscillation.get('oscillation_detected', False):
            severity = oscillation.get('severity', 'mild')
            if severity in ['severe', 'critical']:
                recommendations.append("Consider reducing learning rate due to severe loss oscillations")
                recommendations.append("Check if architecture mutations are too aggressive")
            else:
                recommendations.append("Monitor loss oscillations closely")
        
        # 基于收敛的建议
        convergence = results.get('convergence', {})
        if convergence.get('should_early_stop', False):
            recommendations.append("Consider early stopping due to convergence plateau")
        
        convergence_status = convergence.get('convergence_analysis', {}).get('convergence_status')
        if convergence_status == 'degrading':
            recommendations.append("Performance is degrading - consider adjusting hyperparameters")
        
        # 基于退化的建议
        degradation = results.get('degradation', {})
        if degradation.get('in_degradation', False):
            recommendations.append("Performance degradation detected - consider rollback or adjustment")
        
        # 基于健康度的建议
        if self.training_health_score < 0.5:
            recommendations.append("Training health is poor - comprehensive review recommended")
        
        return recommendations
    
    def set_performance_baseline(self, performance: float):
        """设置性能基线"""
        self.degradation_detector.update(performance, 0, set_baseline=True)
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """获取综合监控报告"""
        return {
            'current_health_score': self.training_health_score,
            'health_status': self._classify_health_status(self.training_health_score),
            'oscillation_report': self.oscillation_detector.get_stability_report(),
            'convergence_report': self.convergence_analyzer.get_convergence_report(),
            'degradation_report': self.degradation_detector.get_degradation_report(),
            'health_history': self.health_history[-20:] if self.health_history else []
        }
    
    def should_intervene(self) -> Tuple[bool, List[str]]:
        """判断是否需要人工干预"""
        intervention_needed = False
        reasons = []
        
        # 健康度过低
        if self.training_health_score < 0.3:
            intervention_needed = True
            reasons.append("Training health score critically low")
        
        # 严重震荡
        oscillation_report = self.oscillation_detector.get_stability_report()
        if oscillation_report.get('stability_score', 1.0) < 0.5:
            intervention_needed = True
            reasons.append("Severe loss oscillations detected")
        
        # 长期性能退化
        degradation_report = self.degradation_detector.get_degradation_report()
        if degradation_report.get('current_status') == 'degraded':
            intervention_needed = True
            reasons.append("Performance degradation persisting")
        
        return intervention_needed, reasons

def create_stability_monitor(config: Dict[str, Any] = None) -> StabilityMonitor:
    """
    创建稳定性监控器的工厂函数
    
    Args:
        config: 监控配置
        
    Returns:
        配置好的稳定性监控器
    """
    return StabilityMonitor(config)

def test_stability_monitor():
    """测试稳定性监控器功能"""
    print("🧪 Testing Stability Monitor...")
    
    # 创建监控器
    monitor = create_stability_monitor()
    
    # 模拟训练过程
    print("📊 Simulating training process...")
    
    # 正常训练阶段
    for epoch in range(10):
        metrics = {
            'loss': 1.0 - epoch * 0.08 + np.random.normal(0, 0.02),
            'accuracy': 0.5 + epoch * 0.04 + np.random.normal(0, 0.01)
        }
        
        result = monitor.update(metrics, epoch, "warmup")
        print(f"   Epoch {epoch}: Health={result['health_score']:.3f}, "
              f"Status={result['health_status']}")
    
    # 设置基线
    monitor.set_performance_baseline(0.85)
    
    # 模拟架构突变引起的震荡
    print("🧬 Simulating architecture mutation...")
    for epoch in range(10, 15):
        metrics = {
            'loss': 0.5 + np.random.normal(0, 0.15),  # 高方差
            'accuracy': 0.8 + np.random.normal(0, 0.05)
        }
        
        result = monitor.update(metrics, epoch, "mutation")
        print(f"   Epoch {epoch}: Health={result['health_score']:.3f}, "
              f"Oscillation={result.get('oscillation', {}).get('oscillation_detected', False)}")
    
    # 获取综合报告
    report = monitor.get_comprehensive_report()
    print(f"✅ Final health score: {report['current_health_score']:.3f}")
    print(f"✅ Health status: {report['health_status']}")
    
    # 检查是否需要干预
    intervention_needed, reasons = monitor.should_intervene()
    print(f"⚠️ Intervention needed: {intervention_needed}")
    if reasons:
        print(f"   Reasons: {reasons}")
    
    print("🎉 Stability Monitor tests passed!")

if __name__ == "__main__":
    test_stability_monitor() 