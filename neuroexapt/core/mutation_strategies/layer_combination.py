"""
层组合预测器

专门负责同种/异种层组合的智能选择
包括卷积、池化、注意力等各种操作的最优组合预测
"""

from typing import Dict, Any, List
import numpy as np
import logging
from ..bayesian_prediction.prior_knowledge import PriorKnowledgeBase

logger = logging.getLogger(__name__)


class LayerCombinationPredictor:
    """层组合预测器 - 同种/异种层组合优化"""
    
    def __init__(self, prior_knowledge: PriorKnowledgeBase):
        self.prior_knowledge = prior_knowledge
        
    def predict_optimal_layer_combinations(self, 
                                         layer_analysis: Dict[str, Any],
                                         target_layer_type: str,
                                         mutation_mode: str,
                                         current_accuracy: float) -> Dict[str, Any]:
        """
        预测最优层类型组合 (同种 vs 异种层)
        
        Args:
            layer_analysis: 层分析结果
            target_layer_type: 目标层类型 (conv2d, linear等)
            mutation_mode: 变异模式 (serial_division, parallel_division等)
            current_accuracy: 当前准确率
            
        Returns:
            层类型组合的收益预测和推荐
        """
        logger.debug(f"开始层组合预测: {target_layer_type}")
        
        try:
            leak_assessment = layer_analysis.get('leak_assessment', {})
            leak_type = leak_assessment.get('leak_type', 'general_bottleneck')
            
            # 获取瓶颈类型的首选操作
            bottleneck_prior = self.prior_knowledge.get_bottleneck_response_prior(leak_type)
            preferred_ops = bottleneck_prior.get('preferred_operations', ['conv2d', 'batch_norm'])
            
            combination_predictions = {}
            
            # 1. 同种层组合预测
            homo_prediction = self._predict_homogeneous_combination(
                target_layer_type, leak_type, mutation_mode, current_accuracy
            )
            if homo_prediction:
                combination_predictions['homogeneous'] = homo_prediction
            
            # 2. 异种层组合预测
            hetero_predictions = self._predict_heterogeneous_combinations(
                target_layer_type, preferred_ops, leak_type, mutation_mode, current_accuracy
            )
            if hetero_predictions:
                combination_predictions['heterogeneous'] = hetero_predictions
            
            # 选择最优组合
            best_combination = self._select_best_combination(combination_predictions)
            
            prediction_result = {
                'recommended_combination': best_combination,
                'combination_predictions': combination_predictions,
                'target_layer_type': target_layer_type,
                'mutation_mode': mutation_mode,
                'detailed_analysis': self._generate_combination_analysis(
                    best_combination, combination_predictions, leak_type
                )
            }
            
            logger.debug(f"最优层组合: {best_combination.get('type', 'unknown')} - {best_combination.get('combination', 'unknown')}")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"层组合预测失败: {e}")
            return self._fallback_combination_prediction(target_layer_type)

    def _predict_homogeneous_combination(self, target_layer_type: str, leak_type: str,
                                       mutation_mode: str, current_accuracy: float) -> Dict[str, Any]:
        """预测同种层组合"""
        homo_key = f"{target_layer_type}_{target_layer_type}"
        homo_config = self.prior_knowledge.get_layer_combination_prior('homogeneous', homo_key)
        
        if homo_config.get('effectiveness', 0) > 0:
            return self._predict_combination_benefit(
                homo_config, target_layer_type, target_layer_type, 
                leak_type, mutation_mode, current_accuracy, 'homogeneous'
            )
        return None

    def _predict_heterogeneous_combinations(self, target_layer_type: str, preferred_ops: List[str],
                                          leak_type: str, mutation_mode: str, 
                                          current_accuracy: float) -> Dict[str, Any]:
        """预测异种层组合"""
        hetero_predictions = {}
        
        for operation in preferred_ops:
            if operation != target_layer_type:  # 避免重复
                hetero_key = f"{target_layer_type}_{operation}"
                reverse_key = f"{operation}_{target_layer_type}"
                
                # 查找配置
                hetero_config = self.prior_knowledge.get_layer_combination_prior('heterogeneous', hetero_key)
                final_key = hetero_key
                
                if hetero_config.get('effectiveness', 0) == 0.5:  # 默认值，尝试反向
                    reverse_config = self.prior_knowledge.get_layer_combination_prior('heterogeneous', reverse_key)
                    if reverse_config.get('effectiveness', 0) > 0.5:
                        hetero_config = reverse_config
                        final_key = reverse_key
                
                if hetero_config.get('effectiveness', 0) > 0:
                    hetero_prediction = self._predict_combination_benefit(
                        hetero_config, target_layer_type, operation,
                        leak_type, mutation_mode, current_accuracy, 'heterogeneous'
                    )
                    hetero_predictions[final_key] = hetero_prediction
        
        return hetero_predictions

    def _predict_combination_benefit(self, config: Dict[str, float], 
                                   layer1_type: str, layer2_type: str,
                                   leak_type: str, mutation_mode: str,
                                   current_accuracy: float, combo_type: str) -> Dict[str, Any]:
        """预测特定层组合的收益"""
        
        # 基础效果和稳定性
        effectiveness = config.get('effectiveness', 0.5)
        stability = config.get('stability', 0.5)
        
        # 获取层操作特性
        layer1_props = self.prior_knowledge.get_operation_prior(layer1_type)
        layer2_props = self.prior_knowledge.get_operation_prior(layer2_type)
        
        # 计算协同效应
        synergy = self._calculate_layer_synergy(layer1_props, layer2_props, leak_type)
        
        # 计算期望收益
        base_gain = self._calculate_base_mutation_gain(current_accuracy, 0.5)
        expected_gain = base_gain * effectiveness * synergy
        
        # 计算置信度
        confidence = stability * synergy
        
        # 计算实施成本
        implementation_cost = self._calculate_implementation_cost(
            layer1_type, layer2_type, mutation_mode
        )
        
        return {
            'expected_gain': float(expected_gain),
            'confidence': float(confidence),
            'effectiveness': float(effectiveness),
            'stability': float(stability),
            'synergy': float(synergy),
            'implementation_cost': float(implementation_cost),
            'combination': f"{layer1_type}+{layer2_type}",
            'type': combo_type
        }

    def _calculate_layer_synergy(self, layer1_props: Dict[str, float], 
                               layer2_props: Dict[str, float], leak_type: str) -> float:
        """计算层间协同效应"""
        
        # 基础协同分数
        synergy_factors = []
        
        # 特征提取能力协同
        feat_synergy = (layer1_props.get('feature_extraction_boost', 0.5) + 
                       layer2_props.get('feature_extraction_boost', 0.5)) / 2
        synergy_factors.append(feat_synergy)
        
        # 参数效率协同
        param_synergy = (layer1_props.get('parameter_efficiency', 0.5) + 
                        layer2_props.get('parameter_efficiency', 0.5)) / 2
        synergy_factors.append(param_synergy)
        
        # 计算成本协同
        cost_synergy = 1.0 - abs(layer1_props.get('computation_cost', 0.5) - 
                                layer2_props.get('computation_cost', 0.5))
        synergy_factors.append(cost_synergy)
        
        # 特殊能力互补
        special_abilities = ['stability_boost', 'overfitting_prevention', 
                           'long_range_dependency', 'gradient_flow']
        complementary_bonus = 0.0
        
        for ability in special_abilities:
            if (ability in layer1_props and ability not in layer2_props) or \
               (ability not in layer1_props and ability in layer2_props):
                complementary_bonus += 0.1
        
        base_synergy = np.mean(synergy_factors)
        final_synergy = min(1.0, base_synergy + complementary_bonus)
        
        return final_synergy

    def _calculate_implementation_cost(self, layer1_type: str, layer2_type: str, 
                                     mutation_mode: str) -> float:
        """计算实施成本"""
        
        # 基础成本
        layer_costs = {
            'conv2d': 0.6, 'linear': 0.4, 'batch_norm': 0.2,
            'dropout': 0.1, 'attention': 0.8, 'pool': 0.2,
            'depthwise_conv': 0.5, 'residual_connection': 0.7
        }
        
        cost1 = layer_costs.get(layer1_type, 0.5)
        cost2 = layer_costs.get(layer2_type, 0.5)
        
        # 组合成本
        if layer1_type == layer2_type:
            combo_cost = cost1 * 1.5  # 同种层复制成本较低
        else:
            combo_cost = cost1 + cost2  # 异种层需要更多适配
        
        # 模式成本
        mode_cost_multiplier = {
            'serial_division': 1.0,
            'parallel_division': 1.3,
            'hybrid_division': 1.5
        }.get(mutation_mode, 1.0)
        
        return combo_cost * mode_cost_multiplier

    def _select_best_combination(self, combination_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """选择最佳层组合"""
        
        best_combo = None
        best_score = -1.0
        
        # 评估同种层组合
        if 'homogeneous' in combination_predictions:
            homo = combination_predictions['homogeneous']
            score = (homo['expected_gain'] * homo['confidence']) / (homo['implementation_cost'] + 0.1)
            if score > best_score:
                best_score = score
                best_combo = homo
        
        # 评估异种层组合
        if 'heterogeneous' in combination_predictions:
            for combo_name, hetero in combination_predictions['heterogeneous'].items():
                score = (hetero['expected_gain'] * hetero['confidence']) / (hetero['implementation_cost'] + 0.1)
                if score > best_score:
                    best_score = score
                    best_combo = hetero
        
        return best_combo if best_combo else {'type': 'fallback', 'expected_gain': 0.01, 'combination': 'unknown'}

    def _calculate_base_mutation_gain(self, current_accuracy: float, leak_severity: float) -> float:
        """计算基础变异收益"""
        headroom = (0.95 - current_accuracy) / 0.95
        base_gain = headroom * 0.1 * (1 + leak_severity)
        return max(0.005, base_gain)

    def _generate_combination_analysis(self, best_combo: Dict[str, Any], 
                                     all_predictions: Dict[str, Any], 
                                     leak_type: str) -> Dict[str, Any]:
        """生成组合分析"""
        return {
            'selected_combination': best_combo.get('combination', 'unknown'),
            'selection_reason': f"最高综合评分，适合{leak_type}瓶颈",
            'alternative_options': list(all_predictions.get('heterogeneous', {}).keys())[:3],
            'synergy_analysis': f"协同效应评分: {best_combo.get('synergy', 0.5):.3f}"
        }

    def _fallback_combination_prediction(self, target_layer_type: str) -> Dict[str, Any]:
        """层组合预测fallback"""
        return {
            'recommended_combination': {
                'combination': f"{target_layer_type}+batch_norm",
                'type': 'heterogeneous',
                'expected_gain': 0.005,
                'confidence': 0.4
            },
            'target_layer_type': target_layer_type
        }