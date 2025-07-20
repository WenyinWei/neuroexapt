"""
智能瓶颈检测器
基于互信息I(H_k; Y)、条件互信息I(H_k; Y|H_{k+1})和贝叶斯不确定性的综合瓶颈检测
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

from .mutual_information_estimator import MutualInformationEstimator
from .bayesian_uncertainty_estimator import BayesianUncertaintyEstimator

logger = logging.getLogger(__name__)


class BottleneckType(Enum):
    """瓶颈类型"""
    INFORMATION_LEAKAGE = "information_leakage"  # 信息泄露：I(H_k; Y|H_{k+1}) ≈ 0
    HIGH_UNCERTAINTY = "high_uncertainty"       # 高不确定性：U(H_k) >> 阈值
    REDUNDANT_FEATURES = "redundant_features"   # 冗余特征：I(H_k; Y) 低但维度高
    GRADIENT_BOTTLENECK = "gradient_bottleneck" # 梯度瓶颈：梯度流动受阻
    CAPACITY_BOTTLENECK = "capacity_bottleneck" # 容量瓶颈：表征能力不足


@dataclass
class BottleneckReport:
    """瓶颈检测报告"""
    layer_name: str
    bottleneck_type: BottleneckType
    severity: float  # 严重程度 [0, 1]
    confidence: float  # 检测置信度 [0, 1]
    
    # 详细指标
    mutual_info: float
    conditional_mutual_info: float
    uncertainty: float
    
    # 解释和建议
    explanation: str
    suggested_mutations: List[str]
    
    # 原始数据
    raw_metrics: Dict[str, Any]


class IntelligentBottleneckDetector:
    """
    智能瓶颈检测器
    
    核心理念：
    1. 多维度分析：互信息 + 不确定性 + 梯度流 + 结构分析
    2. 自适应阈值：根据网络状态和任务特点动态调整
    3. 分级诊断：从粗粒度到细粒度的递进分析
    4. 可解释性：提供明确的瓶颈原因和修复建议
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 核心分析器
        self.mi_estimator = MutualInformationEstimator(device)
        self.uncertainty_estimator = BayesianUncertaintyEstimator(device)
        
        # 动态阈值
        self.thresholds = {
            'mi_low': 0.01,           # 互信息过低阈值
            'conditional_mi_low': 0.005,  # 条件互信息过低阈值
            'uncertainty_high': 1.0,   # 不确定性过高阈值
            'redundancy_ratio': 0.8,   # 冗余特征比例阈值
            'gradient_flow_low': 0.1   # 梯度流动过低阈值
        }
        
        # 历史记录
        self.detection_history = []
        self.adaptive_thresholds_history = []
        
    def detect_bottlenecks(self,
                          model: nn.Module,
                          feature_dict: Dict[str, torch.Tensor],
                          labels: torch.Tensor,
                          gradient_dict: Dict[str, torch.Tensor] = None,
                          num_classes: int = None,
                          confidence_threshold: float = 0.7) -> List[BottleneckReport]:
        """
        综合瓶颈检测
        
        Args:
            model: 神经网络模型
            feature_dict: 各层特征字典 {layer_name: features}
            labels: 目标标签
            gradient_dict: 各层梯度字典（可选）
            num_classes: 分类任务的类别数
            confidence_threshold: 检测置信度阈值
            
        Returns:
            瓶颈报告列表，按严重程度排序
        """
        logger.info("🔍 开始智能瓶颈检测")
        
        # 1. 批量计算互信息
        logger.info("计算分层互信息...")
        mi_results = self.mi_estimator.batch_estimate_layerwise_mi(
            feature_dict, labels, num_classes
        )
        
        # 2. 批量计算条件互信息
        logger.info("计算条件互信息...")
        conditional_mi_results = self._compute_conditional_mi(
            feature_dict, labels, num_classes
        )
        
        # 3. 批量计算不确定性
        logger.info("计算贝叶斯不确定性...")
        uncertainty_results = self.uncertainty_estimator.estimate_feature_uncertainty(
            feature_dict, labels
        )
        
        # 4. 梯度流分析（如果提供）
        gradient_flow_results = {}
        if gradient_dict:
            logger.info("分析梯度流...")
            gradient_flow_results = self._analyze_gradient_flow(gradient_dict)
        
        # 5. 综合分析生成报告
        bottleneck_reports = []
        layer_names = list(feature_dict.keys())
        
        for layer_name in layer_names:
            report = self._analyze_layer_bottleneck(
                layer_name=layer_name,
                features=feature_dict[layer_name],
                mi_value=mi_results.get(layer_name, 0.0),
                conditional_mi=conditional_mi_results.get(layer_name, 0.0),
                uncertainty=uncertainty_results.get(layer_name, float('inf')),
                gradient_flow=gradient_flow_results.get(layer_name, None),
                num_classes=num_classes
            )
            
            if report and report.confidence >= confidence_threshold:
                bottleneck_reports.append(report)
        
        # 6. 按严重程度排序
        bottleneck_reports.sort(key=lambda x: x.severity, reverse=True)
        
        # 7. 更新自适应阈值
        self._update_adaptive_thresholds(mi_results, uncertainty_results)
        
        # 8. 记录历史
        self.detection_history.append({
            'reports': bottleneck_reports,
            'mi_results': mi_results,
            'uncertainty_results': uncertainty_results,
            'timestamp': torch.tensor(len(self.detection_history))
        })
        
        logger.info(f"检测到 {len(bottleneck_reports)} 个潜在瓶颈")
        return bottleneck_reports
    
    def _compute_conditional_mi(self,
                               feature_dict: Dict[str, torch.Tensor],
                               labels: torch.Tensor,
                               num_classes: int = None) -> Dict[str, float]:
        """计算条件互信息"""
        conditional_mi_results = {}
        layer_names = list(feature_dict.keys())
        
        # 构建相邻层对
        feature_pairs = []
        for i in range(len(layer_names) - 1):
            current_layer = layer_names[i]
            next_layer = layer_names[i + 1]
            feature_pairs.append((
                current_layer,
                feature_dict[current_layer],
                feature_dict[next_layer]
            ))
        
        # 批量计算条件互信息
        if feature_pairs:
            conditional_mi_results = self.mi_estimator.batch_estimate_conditional_mi(
                feature_pairs, labels, num_classes
            )
        
        return conditional_mi_results
    
    def _analyze_gradient_flow(self, gradient_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """分析梯度流动"""
        gradient_flow_results = {}
        
        for layer_name, gradients in gradient_dict.items():
            try:
                # 计算梯度范数
                grad_norm = torch.norm(gradients).item()
                
                # 计算梯度分布的方差（反映梯度分布的均匀性）
                grad_flat = gradients.view(-1)
                grad_variance = torch.var(grad_flat).item()
                
                # 综合梯度流指标
                gradient_flow = grad_norm * (1 + grad_variance)  # 高范数+高方差=好的梯度流
                gradient_flow_results[layer_name] = gradient_flow
                
            except Exception as e:
                logger.warning(f"Failed to analyze gradient flow for {layer_name}: {e}")
                gradient_flow_results[layer_name] = 0.0
                
        return gradient_flow_results
    
    def _analyze_layer_bottleneck(self,
                                 layer_name: str,
                                 features: torch.Tensor,
                                 mi_value: float,
                                 conditional_mi: float,
                                 uncertainty: float,
                                 gradient_flow: Optional[float],
                                 num_classes: int = None) -> Optional[BottleneckReport]:
        """分析单层的瓶颈情况"""
        
        # 收集所有指标
        raw_metrics = {
            'mi_value': mi_value,
            'conditional_mi': conditional_mi,
            'uncertainty': uncertainty,
            'gradient_flow': gradient_flow,
            'feature_shape': features.shape,
            'feature_mean': features.mean().item(),
            'feature_std': features.std().item()
        }
        
        # 多维度瓶颈检测
        bottleneck_detections = []
        
        # 1. 信息泄露检测：I(H_k; Y|H_{k+1}) ≈ 0
        if conditional_mi < self.thresholds['conditional_mi_low']:
            severity = 1.0 - conditional_mi / self.thresholds['conditional_mi_low']
            bottleneck_detections.append({
                'type': BottleneckType.INFORMATION_LEAKAGE,
                'severity': min(severity, 1.0),
                'confidence': 0.9,  # 高置信度
                'explanation': f"条件互信息过低 ({conditional_mi:.4f})，当前层信息被后续层完全包含",
                'mutations': ['expand_capacity', 'add_attention', 'insert_residual']
            })
        
        # 2. 高不确定性检测
        if uncertainty > self.thresholds['uncertainty_high']:
            severity = min(uncertainty / self.thresholds['uncertainty_high'] - 1.0, 1.0)
            bottleneck_detections.append({
                'type': BottleneckType.HIGH_UNCERTAINTY,
                'severity': severity,
                'confidence': 0.8,
                'explanation': f"不确定性过高 ({uncertainty:.4f})，特征表征不稳定",
                'mutations': ['regularization', 'batch_norm', 'layer_norm']
            })
        
        # 3. 互信息过低检测：I(H_k; Y) 低
        if mi_value < self.thresholds['mi_low']:
            severity = 1.0 - mi_value / self.thresholds['mi_low']
            
            # 检查是否是冗余特征（高维度但低信息）
            feature_dim = np.prod(features.shape[1:])
            if feature_dim > 256:  # 高维特征
                bottleneck_detections.append({
                    'type': BottleneckType.REDUNDANT_FEATURES,
                    'severity': severity,
                    'confidence': 0.7,
                    'explanation': f"高维特征 ({feature_dim}) 但互信息低 ({mi_value:.4f})，存在冗余",
                    'mutations': ['feature_selection', 'dimensionality_reduction', 'pruning']
                })
            else:
                bottleneck_detections.append({
                    'type': BottleneckType.CAPACITY_BOTTLENECK,
                    'severity': severity,
                    'confidence': 0.75,
                    'explanation': f"互信息过低 ({mi_value:.4f})，表征能力不足",
                    'mutations': ['expand_width', 'add_depth', 'change_activation']
                })
        
        # 4. 梯度流检测
        if gradient_flow is not None and gradient_flow < self.thresholds['gradient_flow_low']:
            severity = 1.0 - gradient_flow / self.thresholds['gradient_flow_low']
            bottleneck_detections.append({
                'type': BottleneckType.GRADIENT_BOTTLENECK,
                'severity': min(severity, 1.0),
                'confidence': 0.6,
                'explanation': f"梯度流动受阻 ({gradient_flow:.4f})，训练效率低",
                'mutations': ['residual_connection', 'gradient_clipping', 'change_optimizer']
            })
        
        # 选择最严重的瓶颈
        if not bottleneck_detections:
            return None
            
        primary_bottleneck = max(bottleneck_detections, key=lambda x: x['severity'] * x['confidence'])
        
        return BottleneckReport(
            layer_name=layer_name,
            bottleneck_type=primary_bottleneck['type'],
            severity=primary_bottleneck['severity'],
            confidence=primary_bottleneck['confidence'],
            mutual_info=mi_value,
            conditional_mutual_info=conditional_mi,
            uncertainty=uncertainty,
            explanation=primary_bottleneck['explanation'],
            suggested_mutations=primary_bottleneck['mutations'],
            raw_metrics=raw_metrics
        )
    
    def _update_adaptive_thresholds(self,
                                   mi_results: Dict[str, float],
                                   uncertainty_results: Dict[str, float]):
        """更新自适应阈值"""
        # 基于当前网络状态调整阈值
        if mi_results:
            mi_values = list(mi_results.values())
            mi_mean = np.mean(mi_values)
            mi_std = np.std(mi_values)
            
            # 动态调整互信息阈值（均值的10%作为低阈值）
            self.thresholds['mi_low'] = max(0.001, mi_mean * 0.1)
            self.thresholds['conditional_mi_low'] = max(0.0005, mi_mean * 0.05)
        
        if uncertainty_results:
            uncertainty_values = [u for u in uncertainty_results.values() if u != float('inf')]
            if uncertainty_values:
                uncertainty_mean = np.mean(uncertainty_values)
                uncertainty_std = np.std(uncertainty_values)
                
                # 动态调整不确定性阈值（均值+2*标准差作为高阈值）
                self.thresholds['uncertainty_high'] = max(0.5, uncertainty_mean + 2 * uncertainty_std)
        
        # 记录阈值历史
        self.adaptive_thresholds_history.append(dict(self.thresholds))
        
        logger.debug(f"Updated adaptive thresholds: {self.thresholds}")
    
    def get_bottleneck_summary(self, reports: List[BottleneckReport]) -> Dict[str, Any]:
        """生成瓶颈检测摘要"""
        if not reports:
            return {'status': 'no_bottlenecks', 'message': '未检测到显著瓶颈'}
        
        # 按类型统计
        type_counts = {}
        for report in reports:
            type_name = report.bottleneck_type.value
            if type_name not in type_counts:
                type_counts[type_name] = 0
            type_counts[type_name] += 1
        
        # 计算平均严重程度
        avg_severity = np.mean([r.severity for r in reports])
        avg_confidence = np.mean([r.confidence for r in reports])
        
        # 获取最严重的瓶颈
        most_severe = max(reports, key=lambda x: x.severity)
        
        return {
            'status': 'bottlenecks_detected',
            'total_bottlenecks': len(reports),
            'type_distribution': type_counts,
            'average_severity': avg_severity,
            'average_confidence': avg_confidence,
            'most_severe_layer': most_severe.layer_name,
            'most_severe_type': most_severe.bottleneck_type.value,
            'most_severe_severity': most_severe.severity,
            'recommended_priority': [r.layer_name for r in reports[:3]]  # 前3个优先处理
        }
    
    def visualize_bottlenecks(self, reports: List[BottleneckReport]) -> str:
        """生成瓶颈可视化报告"""
        if not reports:
            return "✅ 未检测到显著瓶颈"
        
        visualization = "🔍 瓶颈检测报告\n" + "="*50 + "\n"
        
        for i, report in enumerate(reports[:5], 1):  # 显示前5个
            icon = "🔴" if report.severity > 0.7 else "🟡" if report.severity > 0.4 else "🟢"
            
            visualization += f"\n{icon} #{i} 层: {report.layer_name}\n"
            visualization += f"   类型: {report.bottleneck_type.value}\n"
            visualization += f"   严重程度: {report.severity:.3f} | 置信度: {report.confidence:.3f}\n"
            visualization += f"   互信息: {report.mutual_info:.4f} | 条件互信息: {report.conditional_mutual_info:.4f}\n"
            visualization += f"   不确定性: {report.uncertainty:.4f}\n"
            visualization += f"   原因: {report.explanation}\n"
            visualization += f"   建议: {', '.join(report.suggested_mutations)}\n"
        
        summary = self.get_bottleneck_summary(reports)
        visualization += f"\n📊 总计: {summary['total_bottlenecks']} 个瓶颈"
        visualization += f" | 平均严重程度: {summary['average_severity']:.3f}\n"
        
        return visualization
    
    def clear_cache(self):
        """清理缓存"""
        self.mi_estimator.clear_discriminators()
        self.uncertainty_estimator.clear_probes()
        self.detection_history.clear()
        self.adaptive_thresholds_history.clear()