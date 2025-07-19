"""
先验知识管理模块

管理贝叶斯预测所需的所有先验知识，包括：
- 变异类型成功率先验
- 层组合效果先验  
- 瓶颈响应性先验
- 操作适用性先验
"""

from typing import Dict, Any
import logging

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
                'aggressive_widening': {'alpha': 2, 'beta': 1}  # 高风险高收益
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
                    'preferred_operations': ['conv2d', 'attention', 'residual_connection']
                },
                'gradient_learning_bottleneck': {
                    'widening_response': 0.4,
                    'deepening_response': 0.7,
                    'hybrid_response': 0.5,
                    'preferred_operations': ['batch_norm', 'residual_connection', 'dropout']
                },
                'representational_bottleneck': {
                    'widening_response': 0.6,
                    'deepening_response': 0.5,
                    'hybrid_response': 0.9,
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
        """获取变异类型先验"""
        return self.knowledge_base['mutation_success_priors'].get(
            mutation_type, {'alpha': 2, 'beta': 2}
        )
    
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