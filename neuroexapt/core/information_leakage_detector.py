"""
信息泄漏检测器 - 简化版本
使用专注的分析器子模块，减少复杂性
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from .leakage_analyzers import (
    EntropyAnalyzer, DiversityAnalyzer, GradientFlowAnalyzer,
    InformationFlowAnalyzer, RepresentationQualityAnalyzer, LeakagePointIdentifier
)

logger = logging.getLogger(__name__)


class InformationLeakageDetector:
    """
    信息泄漏检测器 - 重构版本
    
    使用专注的分析器子模块，降低类的复杂性
    """
    
    def __init__(self):
        self.activation_cache = {}
        self.gradient_cache = {}
        self.information_metrics = {}
        
        # 初始化专注的分析器组件
        self.entropy_analyzer = EntropyAnalyzer()
        self.diversity_analyzer = DiversityAnalyzer()
        self.gradient_analyzer = GradientFlowAnalyzer()
        self.flow_analyzer = InformationFlowAnalyzer()
        self.quality_analyzer = RepresentationQualityAnalyzer()
        self.leakage_identifier = LeakagePointIdentifier()
        
    def detect_information_leakage(self, 
                                  model: nn.Module,
                                  activations: Dict[str, torch.Tensor],
                                  gradients: Dict[str, torch.Tensor],
                                  targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        检测信息泄漏和特征丢失
        
        Returns:
            Dict包含泄漏点分析和修复建议
        """
        
        logger.info("🔍 开始信息泄漏检测分析...")
        
        # 1. 信息熵分析 - 使用专注的分析器
        entropy_analysis = self.entropy_analyzer.analyze_information_entropy(activations)
        
        # 2. 特征多样性分析 - 使用专注的分析器
        diversity_analysis = self.diversity_analyzer.analyze_feature_diversity(activations)
        
        # 3. 梯度流分析 - 使用专注的分析器
        gradient_flow_analysis = self.gradient_analyzer.analyze_gradient_flow(gradients)
        
        # 4. 层间信息传递分析 - 使用专注的分析器
        information_flow_analysis = self.flow_analyzer.analyze_information_flow(activations)
        
        # 5. 表示质量分析 - 使用专注的分析器
        representation_quality = self.quality_analyzer.analyze_representation_quality(activations, targets)
        
        # 6. 综合分析找出真正的泄漏点 - 使用专注的识别器
        leakage_points = self.leakage_identifier.identify_critical_leakage_points(
            entropy_analysis, diversity_analysis, gradient_flow_analysis, 
            information_flow_analysis, representation_quality
        )
        
        # 7. 生成精准的修复建议
        repair_suggestions = self._generate_targeted_repair_suggestions(leakage_points)
        
        return {
            'leakage_points': leakage_points,
            'repair_suggestions': repair_suggestions,
            'detailed_analysis': {
                'entropy_analysis': entropy_analysis,
                'diversity_analysis': diversity_analysis,
                'gradient_flow_analysis': gradient_flow_analysis,
                'information_flow_analysis': information_flow_analysis,
                'representation_quality': representation_quality
            },
            'summary': self._generate_leakage_summary(leakage_points)
        }
    
    def _generate_targeted_repair_suggestions(self, leakage_points: List[Dict]) -> List[Dict[str, Any]]:
        """生成针对性的修复建议"""
        
        suggestions = []
        
        for point in leakage_points:
            layer_name = point['layer_name']
            problems = point['problems']
            
            # 根据问题类型生成具体建议
            repair_actions = []
            
            for problem in problems:
                if problem['type'] == 'information_bottleneck':
                    repair_actions.extend([
                        'width_expansion',  # 增加通道数
                        'residual_connection',  # 添加跳跃连接
                        'attention_enhancement'  # 增强信息流
                    ])
                elif problem['type'] == 'feature_collapse':
                    repair_actions.extend([
                        'depth_expansion',  # 增加层深度
                        'parallel_division',  # 并行分支
                        'batch_norm_insertion'  # 规范化
                    ])
                elif problem['type'] == 'gradient_blocked':
                    repair_actions.extend([
                        'residual_connection',  # 改善梯度流
                        'batch_norm_insertion',  # 稳定训练
                        'layer_norm'  # 层归一化
                    ])
                elif problem['type'] == 'representation_degradation':
                    repair_actions.extend([
                        'width_expansion',  # 增强表示能力
                        'attention_enhancement',  # 注意力机制
                        'information_enhancement'  # 信息增强
                    ])
            
            # 去重并排序
            unique_actions = list(set(repair_actions))
            
            suggestion = {
                'layer_name': layer_name,
                'priority': point['priority'],
                'recommended_actions': unique_actions,
                'primary_action': unique_actions[0] if unique_actions else 'width_expansion',
                'rationale': f"检测到{len(problems)}个问题: " + 
                           ', '.join(p['description'] for p in problems),
                'expected_improvement': min(1.0, point['total_severity'] * 0.5)
            }
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_leakage_summary(self, leakage_points: List[Dict]) -> Dict[str, Any]:
        """生成泄漏分析摘要"""
        
        if not leakage_points:
            return {
                'total_leakage_points': 0,
                'average_severity': 0.0,
                'most_critical_layer': None,
                'summary': "未检测到明显的信息泄漏问题"
            }
        
        total_points = len(leakage_points)
        avg_severity = np.mean([p['average_severity'] for p in leakage_points])
        most_critical = leakage_points[0] if leakage_points else None
        
        # 问题类型统计
        from collections import defaultdict
        problem_types = defaultdict(int)
        for point in leakage_points:
            for problem in point['problems']:
                problem_types[problem['type']] += 1
        
        summary_text = f"检测到{total_points}个泄漏点，平均严重性{avg_severity:.2f}。"
        if most_critical:
            summary_text += f"最关键层: {most_critical['layer_name']}"
        
        return {
            'total_leakage_points': total_points,
            'average_severity': avg_severity,
            'most_critical_layer': most_critical['layer_name'] if most_critical else None,
            'problem_type_distribution': dict(problem_types),
            'summary': summary_text
        }