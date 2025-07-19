"""
综合策略生成器

整合变异模式预测和层组合预测，生成完整的变异策略
"""

from typing import Dict, Any, List
import torch.nn as nn
import logging
from .mode_prediction import MutationModePredictor
from .layer_combination import LayerCombinationPredictor
from ..bayesian_prediction.prior_knowledge import PriorKnowledgeBase

logger = logging.getLogger(__name__)


class ComprehensiveStrategyGenerator:
    """综合策略生成器"""
    
    def __init__(self, prior_knowledge: PriorKnowledgeBase):
        self.prior_knowledge = prior_knowledge
        self.mode_predictor = MutationModePredictor(prior_knowledge)
        self.combination_predictor = LayerCombinationPredictor(prior_knowledge)
        
    def predict_comprehensive_mutation_strategy(self,
                                               layer_analysis: Dict[str, Any],
                                               current_accuracy: float,
                                               model: nn.Module,
                                               target_layer_name: str) -> Dict[str, Any]:
        """
        综合预测完整的变异策略
        包括: 变异模式 + 层类型组合 + 具体参数
        """
        logger.debug(f"开始综合变异策略预测: {target_layer_name}")
        
        try:
            model_complexity = self._calculate_model_complexity(model)
            target_layer_type = self._get_layer_type(model, target_layer_name)
            
            # 1. 预测最优变异模式
            mode_prediction = self.mode_predictor.predict_optimal_mutation_mode(
                layer_analysis, current_accuracy, model_complexity
            )
            
            # 2. 预测最优层组合
            combination_prediction = self.combination_predictor.predict_optimal_layer_combinations(
                layer_analysis, target_layer_type, 
                mode_prediction['recommended_mode'], current_accuracy
            )
            
            # 3. 预测具体参数配置
            parameter_prediction = self._predict_optimal_parameters(
                layer_analysis, mode_prediction['recommended_mode'],
                combination_prediction['recommended_combination'], 
                current_accuracy, model_complexity
            )
            
            # 4. 综合评分和最终推荐
            comprehensive_score = self._calculate_comprehensive_score(
                mode_prediction, combination_prediction, parameter_prediction
            )
            
            final_strategy = {
                'mutation_mode': mode_prediction['recommended_mode'],
                'layer_combination': combination_prediction['recommended_combination'],
                'parameters': parameter_prediction,
                'comprehensive_score': comprehensive_score,
                'expected_total_gain': (
                    mode_prediction['expected_improvement'] *
                    combination_prediction['recommended_combination'].get('expected_gain', 0.01)
                ),
                'confidence': min(
                    mode_prediction['confidence'],
                    combination_prediction['recommended_combination'].get('confidence', 0.5)
                ),
                'implementation_details': self._generate_implementation_details(
                    mode_prediction, combination_prediction, parameter_prediction
                )
            }
            
            logger.debug(f"综合策略: {final_strategy['mutation_mode']} + "
                        f"{final_strategy['layer_combination'].get('combination', 'unknown')} "
                        f"(总收益={final_strategy['expected_total_gain']:.4f})")
            
            return final_strategy
            
        except Exception as e:
            logger.error(f"综合预测失败: {e}")
            return self._fallback_comprehensive_prediction(target_layer_name)

    def _calculate_model_complexity(self, model: nn.Module) -> Dict[str, float]:
        """计算模型复杂度指标"""
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # 计算层深度和平均宽度
        layer_count = 0
        total_width = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layer_count += 1
                
                if isinstance(module, nn.Linear):
                    total_width += module.out_features
                elif isinstance(module, nn.Conv2d):
                    total_width += module.out_channels
        
        avg_width = total_width / max(layer_count, 1)
        
        return {
            'total_parameters': float(total_params),
            'layer_depth': float(layer_count),
            'layer_width': float(avg_width)
        }

    def _get_layer_type(self, model: nn.Module, layer_name: str) -> str:
        """获取层类型"""
        try:
            module = dict(model.named_modules())[layer_name]
            if isinstance(module, nn.Conv2d):
                return 'conv2d'
            elif isinstance(module, nn.Linear):
                return 'linear'
            elif isinstance(module, nn.BatchNorm2d):
                return 'batch_norm'
            elif isinstance(module, nn.Dropout):
                return 'dropout'
            else:
                return 'unknown'
        except:
            return 'unknown'

    def _predict_optimal_parameters(self, layer_analysis: Dict[str, Any], 
                                  mutation_mode: str, best_combination: Dict[str, Any],
                                  current_accuracy: float, model_complexity: Dict[str, float]) -> Dict[str, Any]:
        """预测最优参数配置"""
        
        # 基于变异模式和层组合预测参数
        params = {
            'parameter_scaling_factor': 1.5,  # 默认参数扩展因子
            'depth_increase': 1,              # 深度增加
            'width_multiplier': 1.0,          # 宽度倍数
            'learning_rate_adjustment': 1.0    # 学习率调整
        }
        
        # 根据变异模式调整
        if mutation_mode == 'serial_division':
            params['depth_increase'] = 2
            params['parameter_scaling_factor'] = 1.3
        elif mutation_mode == 'parallel_division':
            params['width_multiplier'] = 2.0
            params['parameter_scaling_factor'] = 1.8
        else:  # hybrid_division
            params['depth_increase'] = 1
            params['width_multiplier'] = 1.5
            params['parameter_scaling_factor'] = 2.0
        
        # 根据当前准确率调整
        if current_accuracy > 0.9:
            # 高准确率时更保守
            params['parameter_scaling_factor'] *= 0.8
            params['learning_rate_adjustment'] = 0.5
        
        return params

    def _calculate_comprehensive_score(self, mode_pred: Dict[str, Any], 
                                     combo_pred: Dict[str, Any], 
                                     param_pred: Dict[str, Any]) -> float:
        """计算综合评分"""
        
        mode_score = mode_pred['expected_improvement'] * mode_pred['confidence']
        combo_score = combo_pred['recommended_combination'].get('expected_gain', 0) * \
                     combo_pred['recommended_combination'].get('confidence', 0.5)
        
        # 参数复杂度惩罚
        param_penalty = param_pred['parameter_scaling_factor'] * 0.1
        
        comprehensive_score = (mode_score + combo_score) / 2.0 - param_penalty
        
        return max(0.0, comprehensive_score)

    def _generate_implementation_details(self, mode_pred: Dict[str, Any], 
                                       combo_pred: Dict[str, Any], 
                                       param_pred: Dict[str, Any]) -> Dict[str, Any]:
        """生成实施细节"""
        
        return {
            'mutation_sequence': self._plan_mutation_sequence(mode_pred, combo_pred),
            'parameter_adjustments': param_pred,
            'expected_timeline': self._estimate_implementation_time(mode_pred, combo_pred),
            'resource_requirements': self._estimate_resource_needs(param_pred),
            'rollback_strategy': self._plan_rollback_strategy(mode_pred, combo_pred)
        }

    def _plan_mutation_sequence(self, mode_pred: Dict[str, Any], combo_pred: Dict[str, Any]) -> List[str]:
        """规划变异序列"""
        recommended_combo = combo_pred.get('recommended_combination', {})
        return [
            f"1. 准备{mode_pred['recommended_mode']}变异",
            f"2. 实施{recommended_combo.get('combination', 'unknown')}层组合",
            "3. 参数初始化和微调",
            "4. 渐进式训练验证"
        ]

    def _estimate_implementation_time(self, mode_pred: Dict[str, Any], combo_pred: Dict[str, Any]) -> str:
        """估算实施时间"""
        base_time = 10  # 基础10个epoch
        
        if mode_pred['recommended_mode'] == 'hybrid_division':
            base_time *= 1.5
        
        recommended_combo = combo_pred.get('recommended_combination', {})
        if recommended_combo.get('type') == 'heterogeneous':
            base_time *= 1.2
        
        return f"{int(base_time)} epochs"

    def _estimate_resource_needs(self, param_pred: Dict[str, Any]) -> Dict[str, float]:
        """估算资源需求"""
        scaling = param_pred['parameter_scaling_factor']
        
        return {
            'memory_increase': scaling * 1.2,
            'computation_increase': scaling * 1.5,
            'storage_increase': scaling * 1.1
        }

    def _plan_rollback_strategy(self, mode_pred: Dict[str, Any], combo_pred: Dict[str, Any]) -> List[str]:
        """规划回滚策略"""
        return [
            "1. 保存变异前模型检查点",
            "2. 监控关键性能指标",
            "3. 设置性能下降阈值 (2%)",
            "4. 自动回滚机制"
        ]

    def _fallback_comprehensive_prediction(self, target_layer_name: str) -> Dict[str, Any]:
        """综合预测fallback"""
        return {
            'mutation_mode': 'serial_division',
            'layer_combination': {
                'combination': 'conv2d+batch_norm',
                'type': 'heterogeneous',
                'expected_gain': 0.005,
                'confidence': 0.4
            },
            'expected_total_gain': 0.005,
            'confidence': 0.3,
            'implementation_details': {
                'expected_timeline': '10 epochs',
                'mutation_sequence': ['1. 默认serial_division变异']
            }
        }