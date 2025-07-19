"""
变异模式预测器

专门负责Serial/Parallel/Hybrid Division的智能选择
"""

from typing import Dict, Any, List
import numpy as np
import logging
from ..bayesian_prediction.prior_knowledge import PriorKnowledgeBase

logger = logging.getLogger(__name__)


class MutationModePredictor:
    """变异模式预测器 - Serial vs Parallel vs Hybrid Division"""
    
    def __init__(self, prior_knowledge: PriorKnowledgeBase):
        self.prior_knowledge = prior_knowledge
        
    def predict_optimal_mutation_mode(self, 
                                    layer_analysis: Dict[str, Any],
                                    current_accuracy: float,
                                    model_complexity: Dict[str, float]) -> Dict[str, Any]:
        """
        预测最优变异模式 (Serial vs Parallel vs Hybrid Division)
        
        Args:
            layer_analysis: 层分析结果
            current_accuracy: 当前准确率
            model_complexity: 模型复杂度
            
        Returns:
            各种变异模式的收益预测和推荐
        """
        logger.debug("开始变异模式预测分析")
        
        try:
            leak_assessment = layer_analysis.get('leak_assessment', {})
            leak_type = leak_assessment.get('leak_type', 'general_bottleneck')
            leak_severity = leak_assessment.get('leak_severity', 0.0)
            
            # 确定准确率阶段
            accuracy_stage = self.prior_knowledge.get_accuracy_stage(current_accuracy)
            
            mode_predictions = {}
            
            # 预测每种变异模式的收益
            for mode_name in ['serial_division', 'parallel_division', 'hybrid_division']:
                mode_config = self.prior_knowledge.get_mode_prior(mode_name)
                
                if not mode_config:
                    continue
                    
                # 计算该模式对当前瓶颈类型的适配度
                bottleneck_fit = 1.0 if leak_type in mode_config.get('best_for', []) else 0.6
                
                # 计算该模式对当前准确率阶段的适配度
                accuracy_fit = mode_config.get('accuracy_preference', {}).get(accuracy_stage, 0.5)
                
                # 计算复杂度适配度
                complexity_fit = self._calculate_complexity_fit(mode_name, model_complexity)
                
                # 贝叶斯后验概率
                alpha = mode_config.get('success_rate', {}).get('alpha', 2)
                beta = mode_config.get('success_rate', {}).get('beta', 2)
                
                # 观测证据调整
                evidence_adjustment = leak_severity * bottleneck_fit * accuracy_fit
                alpha_posterior = alpha + evidence_adjustment
                beta_posterior = beta + (1.0 - evidence_adjustment)
                
                success_probability = alpha_posterior / (alpha_posterior + beta_posterior)
                
                # 期望收益计算
                base_gain = self._calculate_base_mutation_gain(current_accuracy, leak_severity)
                mode_multiplier = self._get_mode_multiplier(mode_name, leak_type, accuracy_stage)
                expected_gain = base_gain * mode_multiplier * success_probability
                
                # 风险评估
                risk_score = self._calculate_mode_risk(mode_name, model_complexity, current_accuracy)
                
                mode_predictions[mode_name] = {
                    'expected_accuracy_gain': float(expected_gain),
                    'success_probability': float(success_probability),
                    'bottleneck_fit': float(bottleneck_fit),
                    'accuracy_stage_fit': float(accuracy_fit),
                    'complexity_fit': float(complexity_fit),
                    'risk_score': float(risk_score),
                    'recommendation_score': float(expected_gain * success_probability / (risk_score + 0.1)),
                    'optimal_for': mode_config.get('best_for', [])
                }
            
            # 选择最优模式
            best_mode = max(mode_predictions.items(), 
                          key=lambda x: x[1]['recommendation_score']) if mode_predictions else ('serial_division', {})
            
            prediction_result = {
                'recommended_mode': best_mode[0],
                'mode_predictions': mode_predictions,
                'confidence': best_mode[1].get('success_probability', 0.5),
                'expected_improvement': best_mode[1].get('expected_accuracy_gain', 0.01),
                'reasoning': self._generate_mode_reasoning(best_mode, leak_type, accuracy_stage)
            }
            
            logger.debug(f"最优变异模式: {best_mode[0]} (收益={best_mode[1].get('expected_accuracy_gain', 0):.4f})")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"变异模式预测失败: {e}")
            return self._fallback_mode_prediction(current_accuracy)

    def _calculate_complexity_fit(self, mode_name: str, model_complexity: Dict[str, float]) -> float:
        """计算复杂度适配度"""
        total_params = model_complexity.get('total_parameters', 0)
        layer_depth = model_complexity.get('layer_depth', 0)
        
        # 不同模式对复杂度的适配性
        if mode_name == 'serial_division':
            # Serial适合深度增加
            return min(1.0, (50 - layer_depth) / 50.0)  # 层数越少越适合
        elif mode_name == 'parallel_division':
            # Parallel适合宽度增加，但需要足够的参数预算
            return min(1.0, total_params / 1e6)  # 参数越多越适合
        else:  # hybrid_division
            # Hybrid适合中等复杂度
            param_fit = 1.0 - abs(total_params / 1e6 - 0.5) * 2  # 0.5M参数最适合
            depth_fit = 1.0 - abs(layer_depth - 25) / 25.0  # 25层最适合
            return (param_fit + depth_fit) / 2.0

    def _calculate_base_mutation_gain(self, current_accuracy: float, leak_severity: float) -> float:
        """计算基础变异收益"""
        # 基础收益与准确率距离上限和漏点严重程度成正比
        headroom = (0.95 - current_accuracy) / 0.95
        base_gain = headroom * 0.1 * (1 + leak_severity)
        return max(0.005, base_gain)  # 最小收益保障

    def _get_mode_multiplier(self, mode_name: str, leak_type: str, accuracy_stage: str) -> float:
        """获取模式收益倍数"""
        mode_config = self.prior_knowledge.get_mode_prior(mode_name)
        
        # 基础倍数
        base_multiplier = 1.0
        
        # 瓶颈类型适配加成
        if leak_type in mode_config.get('best_for', []):
            base_multiplier *= 1.3
        
        # 准确率阶段适配加成
        stage_fit = mode_config.get('accuracy_preference', {}).get(accuracy_stage, 0.5)
        base_multiplier *= stage_fit
        
        return base_multiplier

    def _calculate_mode_risk(self, mode_name: str, model_complexity: Dict[str, float], 
                           current_accuracy: float) -> float:
        """计算模式风险"""
        base_risk = {
            'serial_division': 0.3,    # 相对稳定
            'parallel_division': 0.5,  # 中等风险
            'hybrid_division': 0.7     # 高风险高收益
        }.get(mode_name, 0.5)
        
        # 高准确率时风险增加
        if current_accuracy > 0.9:
            base_risk *= 1.5
        
        # 高复杂度时风险增加
        if model_complexity.get('total_parameters', 0) > 5e6:
            base_risk *= 1.2
        
        return base_risk

    def _generate_mode_reasoning(self, best_mode: tuple, leak_type: str, accuracy_stage: str) -> str:
        """生成模式选择推理"""
        if not best_mode or len(best_mode) < 2:
            return "默认使用serial_division策略"
            
        mode_name, mode_data = best_mode
        
        return (f"{mode_name}最适合当前情况: "
               f"瓶颈类型={leak_type}, 准确率阶段={accuracy_stage}, "
               f"期望收益={mode_data.get('expected_accuracy_gain', 0):.4f}")

    def _fallback_mode_prediction(self, current_accuracy: float) -> Dict[str, Any]:
        """模式预测fallback"""
        return {
            'recommended_mode': 'serial_division',
            'confidence': 0.5,
            'expected_improvement': 0.01,
            'reasoning': 'Fallback to conservative serial division'
        }