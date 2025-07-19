"""
先验知识管理模块

管理贝叶斯推断所需的所有先验知识，包括：
- 变异类型成功率先验
- 层组合效果先验  
- 瓶颈响应性先验
- 操作适用性先验
- Net2Net参数迁移成功率
"""

from typing import Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PriorKnowledgeBase:
    """先验知识库管理器"""
    
    def __init__(self):
        self.knowledge_base = self._initialize_prior_knowledge()
        
    def _initialize_prior_knowledge(self) -> Dict[str, Any]:
        """初始化先验知识"""
        return {
            # 不同变异类型的历史成功率先验
            'mutation_success_priors': {
                'widening': {'alpha': 3, 'beta': 2},  # Beta分布参数，倾向于成功
                'deepening': {'alpha': 2, 'beta': 3},  # 相对保守
                'hybrid_expansion': {'alpha': 4, 'beta': 2},  # 较为激进
                'aggressive_widening': {'alpha': 2, 'beta': 1},  # 高风险高收益
                'moderate_widening': {'alpha': 5, 'beta': 3}  # 稳健策略
            },
            
            # Net2Net参数迁移成功率先验
            'net2net_transfer_priors': {
                'net2wider_conv': {'alpha': 8, 'beta': 2},  # Net2Wider通常很稳定
                'net2deeper_conv': {'alpha': 6, 'beta': 4},  # Net2Deeper有一定风险
                'net2branch': {'alpha': 7, 'beta': 3},  # 分支策略适中
                'smooth_transition': {'alpha': 9, 'beta': 1}  # 平滑过渡极其稳定
            },
            
            # Serial vs Parallel mutation 先验知识
            'mutation_mode_priors': {
                'serial_division': {
                    'success_rate': {'alpha': 5, 'beta': 3},  # 相对稳定
                    'best_for': ['gradient_learning_bottleneck', 'representational_bottleneck'],
                    'accuracy_preference': {'low': 0.7, 'medium': 0.8, 'high': 0.6}
                },
                'parallel_division': {
                    'success_rate': {'alpha': 4, 'beta': 4},  # 中等风险
                    'best_for': ['information_compression_bottleneck'],
                    'accuracy_preference': {'low': 0.6, 'medium': 0.7, 'high': 0.8}
                },
                'hybrid_division': {
                    'success_rate': {'alpha': 6, 'beta': 2},  # 激进但高收益
                    'best_for': ['general_bottleneck'],
                    'accuracy_preference': {'low': 0.8, 'medium': 0.9, 'high': 0.7}
                }
            },
            
            # 层类型组合策略先验 (同种 vs 异种)
            'layer_combination_priors': {
                'homogeneous': {  # 同种层
                    'conv2d_conv2d': {'effectiveness': 0.7, 'stability': 0.9},
                    'linear_linear': {'effectiveness': 0.6, 'stability': 0.8},
                    'batch_norm_batch_norm': {'effectiveness': 0.5, 'stability': 0.9}
                },
                'heterogeneous': {  # 异种层组合
                    'conv2d_depthwise_conv': {'effectiveness': 0.8, 'stability': 0.7},
                    'conv2d_batch_norm': {'effectiveness': 0.9, 'stability': 0.8},
                    'conv2d_dropout': {'effectiveness': 0.6, 'stability': 0.7},
                    'conv2d_attention': {'effectiveness': 0.85, 'stability': 0.6},
                    'linear_dropout': {'effectiveness': 0.7, 'stability': 0.8},
                    'linear_batch_norm': {'effectiveness': 0.8, 'stability': 0.9},
                    'conv2d_pool': {'effectiveness': 0.5, 'stability': 0.9},
                    'conv2d_residual_block': {'effectiveness': 0.9, 'stability': 0.8}
                }
            },
            
            # Net2Net架构变异收益先验
            'net2net_mutation_benefits': {
                'net2wider_expected_gain': {
                    'low_complexity': 0.03,    # 简单模型扩展收益较大
                    'medium_complexity': 0.015, # 中等复杂度适中收益
                    'high_complexity': 0.008   # 复杂模型收益递减
                },
                'net2deeper_expected_gain': {
                    'low_depth': 0.025,        # 浅层网络加深收益明显
                    'medium_depth': 0.012,     # 中等深度收益适中
                    'high_depth': 0.005        # 深层网络收益微小
                },
                'parameter_transfer_confidence': {
                    'function_preserving': 0.95,  # 函数保持性确保高置信度
                    'smooth_transition': 0.90,    # 平滑过渡降低风险
                    'weight_inheritance': 0.85    # 权重继承基本可靠
                }
            },
            
            # 不同网络层操作的适用性先验
            'layer_operation_priors': {
                'conv2d': {
                    'feature_extraction_boost': 0.9,
                    'spatial_processing': 0.95,
                    'parameter_efficiency': 0.7,
                    'computation_cost': 0.6
                },
                'depthwise_conv': {
                    'feature_extraction_boost': 0.7,
                    'spatial_processing': 0.8,
                    'parameter_efficiency': 0.9,
                    'computation_cost': 0.8
                },
                'batch_norm': {
                    'feature_extraction_boost': 0.4,
                    'spatial_processing': 0.3,
                    'parameter_efficiency': 0.9,
                    'computation_cost': 0.9,
                    'stability_boost': 0.9
                },
                'dropout': {
                    'feature_extraction_boost': 0.2,
                    'spatial_processing': 0.1,
                    'parameter_efficiency': 1.0,
                    'computation_cost': 0.95,
                    'overfitting_prevention': 0.8
                },
                'attention': {
                    'feature_extraction_boost': 0.85,
                    'spatial_processing': 0.7,
                    'parameter_efficiency': 0.5,
                    'computation_cost': 0.3,
                    'long_range_dependency': 0.95
                },
                'pool': {
                    'feature_extraction_boost': 0.3,
                    'spatial_processing': 0.6,
                    'parameter_efficiency': 1.0,
                    'computation_cost': 0.9,
                    'dimensionality_reduction': 0.9
                },
                'residual_connection': {
                    'feature_extraction_boost': 0.6,
                    'spatial_processing': 0.5,
                    'parameter_efficiency': 0.8,
                    'computation_cost': 0.7,
                    'gradient_flow': 0.95
                }
            },
            
            # 不同瓶颈类型对变异的响应性先验
            'bottleneck_response_priors': {
                'information_compression_bottleneck': {
                    'widening_response': 0.8,
                    'deepening_response': 0.3,
                    'hybrid_response': 0.6,
                    'net2net_response': 0.9,   # Net2Net对信息压缩瓶颈效果很好
                    'preferred_operations': ['conv2d', 'attention', 'residual_connection']
                },
                'gradient_learning_bottleneck': {
                    'widening_response': 0.4,
                    'deepening_response': 0.7,
                    'hybrid_response': 0.5,
                    'net2net_response': 0.6,   # Net2Net对梯度学习有帮助
                    'preferred_operations': ['batch_norm', 'residual_connection', 'dropout']
                },
                'representational_bottleneck': {
                    'widening_response': 0.6,
                    'deepening_response': 0.5,
                    'hybrid_response': 0.9,
                    'net2net_response': 0.8,   # Net2Net提升表示能力
                    'preferred_operations': ['attention', 'conv2d', 'depthwise_conv']
                }
            },
            
            # 准确率阶段对变异收益的影响
            'accuracy_stage_priors': {
                'low': (0.0, 0.85),    # 低准确率阶段，变异收益较大
                'medium': (0.85, 0.92), # 中等准确率，收益递减
                'high': (0.92, 1.0)     # 高准确率，收益微小但关键
            }
        }
    
    def get_mutation_prior(self, mutation_type: str) -> Dict[str, float]:
        """获取变异类型先验，包含直接计算的成功率"""
        prior_params = self.knowledge_base['mutation_success_priors'].get(
            mutation_type, {'alpha': 2, 'beta': 2}
        )
        
        # 从Beta分布参数计算期望成功率
        alpha, beta = prior_params['alpha'], prior_params['beta']
        success_rate = alpha / (alpha + beta)
        confidence = (alpha + beta) / 10.0  # 样本越多置信度越高
        
        return {
            'alpha': alpha,
            'beta': beta, 
            'success_rate': success_rate,
            'confidence': min(1.0, confidence)
        }
    
    def get_net2net_prior(self, transfer_type: str) -> Dict[str, float]:
        """获取Net2Net参数迁移先验"""
        prior_params = self.knowledge_base['net2net_transfer_priors'].get(
            transfer_type, {'alpha': 5, 'beta': 3}
        )
        
        alpha, beta = prior_params['alpha'], prior_params['beta']
        success_rate = alpha / (alpha + beta)
        
        return {
            'alpha': alpha,
            'beta': beta,
            'success_rate': success_rate,
            'confidence': min(1.0, (alpha + beta) / 15.0)  # Net2Net数据更充分
        }
    
    def get_net2net_benefit_prior(self, complexity_level: str, mutation_type: str) -> float:
        """获取Net2Net变异收益先验"""
        if mutation_type in ['widening', 'net2wider']:
            return self.knowledge_base['net2net_mutation_benefits']['net2wider_expected_gain'].get(
                complexity_level, 0.015
            )
        elif mutation_type in ['deepening', 'net2deeper']:
            return self.knowledge_base['net2net_mutation_benefits']['net2deeper_expected_gain'].get(
                complexity_level, 0.012
            )
        else:
            return 0.01  # 默认期望收益
    
    def get_parameter_transfer_confidence(self, transfer_mechanism: str) -> float:
        """获取参数迁移置信度"""
        return self.knowledge_base['net2net_mutation_benefits']['parameter_transfer_confidence'].get(
            transfer_mechanism, 0.8
        )
    
    def assess_net2net_suitability(self, layer_analysis: Dict[str, Any], mutation_strategy: str) -> Dict[str, float]:
        """评估Net2Net技术的适用性"""
        
        # 基于层类型判断适用性
        param_analysis = layer_analysis.get('parameter_analysis', {})
        mutation_readiness = param_analysis.get('mutation_readiness', 0.5)
        
        # Net2Net适用性评分
        suitability_scores = {
            'function_preserving_suitable': 0.9,  # Net2Net保持函数一致性
            'parameter_efficiency': 0.8,          # 参数利用效率高
            'training_stability': 0.85,           # 训练稳定性好
            'convergence_speed': 0.7              # 收敛速度提升
        }
        
        # 根据变异准备度调整评分
        adjustment_factor = min(1.2, max(0.8, mutation_readiness + 0.2))
        
        for key in suitability_scores:
            suitability_scores[key] *= adjustment_factor
            
        return suitability_scores
    
    def get_mode_prior(self, mode: str) -> Dict[str, Any]:
        """获取变异模式先验"""
        return self.knowledge_base['mutation_mode_priors'].get(mode, {})
    
    def get_layer_combination_prior(self, combination_type: str, combination_key: str) -> Dict[str, float]:
        """获取层组合先验"""
        return self.knowledge_base['layer_combination_priors'].get(
            combination_type, {}
        ).get(combination_key, {'effectiveness': 0.5, 'stability': 0.5})
    
    def get_operation_prior(self, operation: str) -> Dict[str, float]:
        """获取操作先验"""
        return self.knowledge_base['layer_operation_priors'].get(operation, {})
    
    def get_bottleneck_response_prior(self, bottleneck_type: str) -> Dict[str, Any]:
        """获取瓶颈响应先验"""
        return self.knowledge_base['bottleneck_response_priors'].get(
            bottleneck_type, 
            {
                'widening_response': 0.5,
                'deepening_response': 0.5, 
                'hybrid_response': 0.5,
                'net2net_response': 0.7,   # 默认Net2Net响应较好
                'preferred_operations': ['conv2d', 'batch_norm']
            }
        )
    
    def get_accuracy_stage(self, accuracy: float) -> str:
        """获取准确率阶段"""
        for stage, (low, high) in self.knowledge_base['accuracy_stage_priors'].items():
            if low <= accuracy < high:
                return stage
        return 'high'
    
    def update_knowledge(self, category: str, key: str, value: Any):
        """更新先验知识"""
        if category in self.knowledge_base:
            self.knowledge_base[category][key] = value
            logger.info(f"更新先验知识: {category}.{key}")
    
    def get_all_knowledge(self) -> Dict[str, Any]:
        """获取完整知识库"""
        return self.knowledge_base.copy()