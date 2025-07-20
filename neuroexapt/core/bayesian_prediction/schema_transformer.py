"""
贝叶斯结果模式转换器

提供可复用的转换逻辑，避免重复代码
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class BayesianSchemaTransformer:
    """贝叶斯结果模式转换器"""
    
    def __init__(self):
        # 标准模式映射
        self.standard_schema_mapping = {
            'analysis_summary': {
                'performance_situation': {
                    'situation_type': 'bayesian_optimized',
                    'urgency_level': 'intelligent',
                    'improvement_trend': 'bayesian_predicted'
                }
            },
            'execution_plan': {
                'monitoring_metrics': [
                    'accuracy_improvement',
                    'loss_reduction',
                    'gradient_stability',
                    'computational_efficiency'
                ]
            }
        }
    
    def convert_bayesian_to_standard_format(self, bayesian_result: Dict[str, Any]) -> Dict[str, Any]:
        """将贝叶斯分析结果转换为标准格式"""
        
        optimal_decisions = bayesian_result.get('optimal_decisions', [])
        bayesian_analysis = bayesian_result.get('bayesian_analysis', {})
        execution_plan = bayesian_result.get('execution_plan', {})
        
        # 使用预定义的映射结构
        converted_result = {
            'analysis_summary': self._build_analysis_summary(bayesian_analysis, optimal_decisions),
            'mutation_candidates': self._convert_decisions_to_candidates(optimal_decisions),
            'mutation_strategies': self._convert_decisions_to_strategies(optimal_decisions),
            'final_decisions': optimal_decisions,
            'execution_plan': execution_plan,
            'intelligent_analysis': self._build_intelligent_analysis(bayesian_analysis, optimal_decisions),
            'bayesian_insights': bayesian_result.get('bayesian_insights', {}),
            'source_engine': 'bayesian'
        }
        
        logger.info(f"🔄 贝叶斯结果转换完成: {len(optimal_decisions)}个决策")
        return converted_result
    
    def _build_analysis_summary(self, bayesian_analysis: Dict[str, Any], optimal_decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建分析摘要"""
        
        base_summary = self.standard_schema_mapping['analysis_summary'].copy()
        
        base_summary.update({
            'structural_analysis': {
                'total_layers_analyzed': bayesian_analysis.get('candidates_found', 0),
                'bottlenecks_found': len(optimal_decisions),
                'severity_distribution': {'bayesian_detected': len(optimal_decisions)}
            },
            'information_efficiency': {
                'overall_efficiency': bayesian_analysis.get('decision_confidence', 0.5),
                'enhancement_opportunities': len(optimal_decisions)
            },
            'gradient_quality': {
                'overall_quality': 0.7,  # 贝叶斯分析假设合理的梯度质量
                'enhancement_needed': len(optimal_decisions) > 0
            }
        })
        
        return base_summary
    
    def _convert_decisions_to_candidates(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将贝叶斯决策转换为候选点格式"""
        
        candidates = []
        for decision in decisions:
            candidate = {
                'layer_name': decision.get('layer_name', ''),
                'layer_type': 'bayesian_identified',
                'selection_reasons': ['bayesian_optimization'],
                'bottleneck_types': ['bayesian_detected'],
                'improvement_potential': decision.get('expected_improvement', 0.0),
                'priority_score': decision.get('expected_utility', 0.0),
                'recommended_mutations': [decision.get('mutation_type', '')],
                'bayesian_metrics': self._extract_bayesian_metrics(decision)
            }
            candidates.append(candidate)
        
        return candidates
    
    def _convert_decisions_to_strategies(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将贝叶斯决策转换为策略格式"""
        
        strategies = []
        for decision in decisions:
            strategy = {
                'target_layer': decision.get('layer_name', ''),
                'mutation_type': decision.get('mutation_type', ''),
                'rationale': self._build_strategy_rationale(decision),
                'expected_outcome': self._build_expected_outcome(decision),
                'risk_assessment': self._build_risk_assessment(decision),
                'bayesian_reasoning': decision.get('rationale', 'Bayesian optimization recommended'),
                'implementation_priority': decision.get('expected_utility', 0.0)
            }
            strategies.append(strategy)
        
        return strategies
    
    def _build_intelligent_analysis(self, bayesian_analysis: Dict[str, Any], optimal_decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建智能分析结果"""
        
        return {
            'candidates_discovered': len(optimal_decisions),
            'strategies_evaluated': len(optimal_decisions),
            'final_decisions': len(optimal_decisions),
            'execution_confidence': bayesian_analysis.get('decision_confidence', 0.0),
            'performance_trend': 'bayesian_enhanced',
            'saturation_level': 0.0  # 贝叶斯分析关注改进而非饱和
        }
    
    def _extract_bayesian_metrics(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """提取贝叶斯指标"""
        
        return {
            'success_probability': decision.get('success_probability', 0.5),
            'decision_confidence': decision.get('decision_confidence', 0.5),
            'expected_utility': decision.get('expected_utility', 0.0),
            'risk_metrics': decision.get('risk_metrics', {})
        }
    
    def _build_strategy_rationale(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """构建策略理由"""
        
        return {
            'selection_method': 'bayesian_inference',
            'success_probability': decision.get('success_probability', 0.5),
            'expected_improvement': decision.get('expected_improvement', 0.0),
            'decision_confidence': decision.get('decision_confidence', 0.5)
        }
    
    def _build_expected_outcome(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """构建期望结果"""
        
        return {
            'expected_accuracy_improvement': decision.get('expected_improvement', 0.0),
            'confidence_level': decision.get('decision_confidence', 0.5),
            'success_probability': decision.get('success_probability', 0.5)
        }
    
    def _build_risk_assessment(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """构建风险评估"""
        
        risk_metrics = decision.get('risk_metrics', {})
        
        return {
            'overall_risk': 1.0 - decision.get('success_probability', 0.5),
            'risk_factors': [],
            'value_at_risk': risk_metrics.get('value_at_risk', 0.0),
            'expected_shortfall': risk_metrics.get('expected_shortfall', 0.0)
        }
    
    def convert_standard_to_bayesian_format(self, standard_result: Dict[str, Any]) -> Dict[str, Any]:
        """将标准格式转换为贝叶斯格式（用于混合模式）"""
        
        final_decisions = standard_result.get('final_decisions', [])
        
        # 为标准决策添加贝叶斯指标的默认值
        bayesian_decisions = []
        for decision in final_decisions:
            bayesian_decision = decision.copy()
            if 'success_probability' not in bayesian_decision:
                bayesian_decision['success_probability'] = 0.6  # 默认成功率
            if 'expected_improvement' not in bayesian_decision:
                bayesian_decision['expected_improvement'] = 0.02  # 默认期望改进
            if 'decision_confidence' not in bayesian_decision:
                bayesian_decision['decision_confidence'] = 0.5  # 默认置信度
            
            bayesian_decisions.append(bayesian_decision)
        
        return {
            'optimal_decisions': bayesian_decisions,
            'execution_plan': standard_result.get('execution_plan', {}),
            'bayesian_analysis': {
                'candidates_found': len(bayesian_decisions),
                'decision_confidence': 0.5  # 标准系统的默认置信度
            },
            'bayesian_insights': {
                'most_promising_mutation': bayesian_decisions[0] if bayesian_decisions else None,
                'expected_performance_gain': sum(d.get('expected_improvement', 0) for d in bayesian_decisions),
                'risk_assessment': {'overall_risk': 0.3, 'risk_factors': []}
            }
        }
    
    def merge_bayesian_and_standard_results(self, 
                                          bayesian_result: Dict[str, Any], 
                                          standard_result: Dict[str, Any]) -> Dict[str, Any]:
        """合并贝叶斯和标准分析结果"""
        
        bayesian_decisions = bayesian_result.get('optimal_decisions', [])
        standard_decisions = standard_result.get('final_decisions', [])
        
        # 合并决策，优先贝叶斯结果
        merged_decisions = bayesian_decisions.copy()
        
        # 添加标准结果中不冲突的决策
        for std_decision in standard_decisions:
            layer_name = std_decision.get('target_layer', '')
            mutation_type = std_decision.get('mutation_type', '')
            
            # 检查是否已存在相同的决策
            exists = any(
                d.get('layer_name') == layer_name and d.get('mutation_type') == mutation_type
                for d in merged_decisions
            )
            
            if not exists:
                # 转换为贝叶斯格式并添加
                bayesian_std_decision = {
                    'layer_name': layer_name,
                    'mutation_type': mutation_type,
                    'success_probability': 0.6,  # 默认值
                    'expected_improvement': std_decision.get('expected_outcome', {}).get('expected_accuracy_improvement', 0.02),
                    'decision_confidence': 0.5,
                    'expected_utility': 0.01,
                    'rationale': 'Standard analysis + Bayesian enhancement'
                }
                merged_decisions.append(bayesian_std_decision)
        
        # 构建合并后的结果
        merged_result = self.convert_bayesian_to_standard_format({
            'optimal_decisions': merged_decisions,
            'execution_plan': bayesian_result.get('execution_plan', standard_result.get('execution_plan', {})),
            'bayesian_analysis': bayesian_result.get('bayesian_analysis', {}),
            'bayesian_insights': bayesian_result.get('bayesian_insights', {})
        })
        
        # 添加合并标识
        merged_result['source_engine'] = 'bayesian+standard'
        merged_result['merge_info'] = {
            'bayesian_decisions': len(bayesian_decisions),
            'standard_decisions': len(standard_decisions),
            'total_decisions': len(merged_decisions)
        }
        
        logger.info(f"🔀 合并分析结果: 贝叶斯{len(bayesian_decisions)}个 + 标准{len(standard_decisions)}个 = 总计{len(merged_decisions)}个决策")
        
        return merged_result