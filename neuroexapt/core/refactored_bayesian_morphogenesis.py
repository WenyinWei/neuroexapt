"""
重构后的贝叶斯形态发生引擎

使用组件化架构，提高可维护性和可测试性
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RefactoredBayesianMorphogenesisEngine:
    """
    重构后的贝叶斯形态发生引擎
    
    组件化设计：
    1. 特征提取器 - 提取架构特征
    2. 候选点检测器 - 发现变异候选点
    3. 贝叶斯推理器 - 进行贝叶斯推理
    4. 效用评估器 - 评估变异效用
    5. 决策制定器 - 制定最终决策
    """
    
    def __init__(self, config=None, feature_extractor=None, candidate_detector=None):
        # 配置管理
        from .bayesian_prediction.bayesian_config import BayesianConfigManager
        self.config_manager = BayesianConfigManager()
        self.config = self.config_manager.get_config()
        
        if config:
            self.config_manager.update_config(config)
        
        # 组件注入
        self.feature_extractor = feature_extractor or self._create_feature_extractor()
        self.candidate_detector = candidate_detector or self._create_candidate_detector()
        self.bayesian_inference = self._create_bayesian_inference()
        self.utility_evaluator = self._create_utility_evaluator()
        self.decision_maker = self._create_decision_maker()
        
        # 历史记录
        self.mutation_history = []
        self.performance_history = []
        
    def _create_feature_extractor(self):
        """创建特征提取器"""
        from .bayesian_prediction.feature_extractor import ArchitectureFeatureExtractor
        return ArchitectureFeatureExtractor()
    
    def _create_candidate_detector(self):
        """创建候选点检测器"""
        from .bayesian_prediction.candidate_detector import BayesianCandidateDetector
        return BayesianCandidateDetector(self.config)
    
    def _create_bayesian_inference(self):
        """创建贝叶斯推理器"""
        return BayesianInferenceEngine(self.config)
    
    def _create_utility_evaluator(self):
        """创建效用评估器"""
        return UtilityEvaluator(self.config)
    
    def _create_decision_maker(self):
        """创建决策制定器"""
        return DecisionMaker(self.config)
    
    def bayesian_morphogenesis_analysis(self, model: torch.nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行贝叶斯形态发生分析"""
        
        logger.info("🧠 开始贝叶斯形态发生分析")
        
        try:
            # 1. 特征提取
            features = self.feature_extractor.extract_features(model, context)
            logger.info(f"✅ 特征提取完成: {len(features)}个特征维度")
            
            # 2. 候选点检测
            candidates = self.candidate_detector.detect_candidates(features)
            logger.info(f"🔍 候选点检测完成: 发现{len(candidates)}个候选点")
            
            if not candidates:
                return self._create_no_candidates_result()
            
            # 3. 贝叶斯推理
            inference_results = self.bayesian_inference.analyze_candidates(candidates, features, self.mutation_history)
            logger.info(f"🎯 贝叶斯推理完成: 分析了{len(candidates)}个候选点")
            
            # 4. 效用评估
            utility_results = self.utility_evaluator.evaluate_utilities(candidates, inference_results, features)
            logger.info(f"💰 效用评估完成: 计算了{len(candidates)}个候选点的效用")
            
            # 5. 决策制定
            decisions = self.decision_maker.make_decisions(candidates, inference_results, utility_results)
            logger.info(f"🎲 决策制定完成: 生成了{len(decisions.get('optimal_decisions', []))}个最优决策")
            
            # 6. 构建结果
            result = self._build_analysis_result(features, candidates, inference_results, utility_results, decisions)
            
            logger.info(f"✅ 贝叶斯分析完成: {len(result.get('optimal_decisions', []))}个最优决策")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 贝叶斯分析失败: {e}")
            return self._create_error_result(str(e))
    
    def _create_no_candidates_result(self) -> Dict[str, Any]:
        """创建无候选点结果"""
        return {
            'optimal_decisions': [],
            'execution_plan': {'execute': False, 'reason': 'no_candidates_found'},
            'bayesian_analysis': {
                'candidates_found': 0,
                'decision_confidence': 0.0,
                'analysis_status': 'no_candidates'
            },
            'bayesian_insights': {
                'most_promising_mutation': None,
                'expected_performance_gain': 0.0,
                'risk_assessment': {'overall_risk': 0.0, 'risk_factors': []}
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'optimal_decisions': [],
            'execution_plan': {'execute': False, 'reason': 'analysis_error', 'error': error_message},
            'bayesian_analysis': {
                'candidates_found': 0,
                'decision_confidence': 0.0,
                'analysis_status': 'error'
            },
            'bayesian_insights': {
                'error': error_message,
                'expected_performance_gain': 0.0,
                'risk_assessment': {'overall_risk': 1.0, 'risk_factors': ['analysis_error']}
            }
        }
    
    def _build_analysis_result(self, 
                             features: Dict[str, Any],
                             candidates: List[Dict[str, Any]],
                             inference_results: Dict[str, Any],
                             utility_results: Dict[str, Any],
                             decisions: Dict[str, Any]) -> Dict[str, Any]:
        """构建分析结果"""
        
        optimal_decisions = decisions.get('optimal_decisions', [])
        
        return {
            'optimal_decisions': optimal_decisions,
            'execution_plan': self._build_execution_plan(optimal_decisions, decisions),
            'bayesian_analysis': {
                'candidates_found': len(candidates),
                'decision_confidence': decisions.get('overall_confidence', 0.0),
                'analysis_status': 'success',
                'inference_summary': inference_results.get('summary', {}),
                'utility_summary': utility_results.get('summary', {})
            },
            'bayesian_insights': self._build_insights(optimal_decisions, inference_results, utility_results),
            'detailed_analysis': {
                'features': features,
                'candidates': candidates,
                'inference_results': inference_results,
                'utility_results': utility_results
            }
        }
    
    def _build_execution_plan(self, optimal_decisions: List[Dict[str, Any]], decisions: Dict[str, Any]) -> Dict[str, Any]:
        """构建执行计划"""
        
        overall_confidence = decisions.get('overall_confidence', 0.0)
        confidence_threshold = self.config.dynamic_thresholds['confidence_threshold']
        
        # 修复置信度计算问题 - 如果有决策但置信度为0，使用决策本身的置信度
        if len(optimal_decisions) > 0 and overall_confidence == 0.0:
            decision_confidences = [d.get('decision_confidence', 0.0) for d in optimal_decisions]
            if decision_confidences:
                overall_confidence = max(decision_confidences)  # 使用最高的决策置信度
                logger.info(f"🔧 修正执行置信度: {overall_confidence:.3f} (来自最佳决策)")
        
        should_execute = len(optimal_decisions) > 0 and overall_confidence > confidence_threshold
        
        # 计算总期望改进
        total_expected_improvement = sum(d.get('expected_improvement', 0.0) for d in optimal_decisions)
        
        plan = {
            'execute': should_execute,
            'reason': decisions.get('execution_reason', 'bayesian_analysis'),
            'confidence': overall_confidence,
            'expected_improvements': [],
            'total_expected_improvement': total_expected_improvement,
            'decisions_count': len(optimal_decisions)
        }
        
        if should_execute:
            for decision in optimal_decisions:
                plan['expected_improvements'].append({
                    'layer': decision.get('layer_name', ''),
                    'mutation': decision.get('mutation_type', ''),
                    'expected_gain': decision.get('expected_improvement', 0.0)
                })
        
        return plan
    
    def _build_insights(self, 
                       optimal_decisions: List[Dict[str, Any]],
                       inference_results: Dict[str, Any],
                       utility_results: Dict[str, Any]) -> Dict[str, Any]:
        """构建贝叶斯洞察"""
        
        insights = {
            'most_promising_mutation': None,
            'expected_performance_gain': 0.0,
            'risk_assessment': {'overall_risk': 0.0, 'risk_factors': []},
            'confidence_levels': {},
            'mutation_recommendations': []
        }
        
        if optimal_decisions:
            # 最有前景的变异
            best_decision = max(optimal_decisions, key=lambda x: x.get('expected_utility', 0))
            insights['most_promising_mutation'] = {
                'layer_name': best_decision.get('layer_name', ''),
                'mutation_type': best_decision.get('mutation_type', ''),
                'expected_utility': best_decision.get('expected_utility', 0.0),
                'success_probability': best_decision.get('success_probability', 0.0)
            }
            
            # 期望性能增益
            insights['expected_performance_gain'] = sum(
                d.get('expected_improvement', 0.0) for d in optimal_decisions
            )
            
            # 风险评估
            risks = [d.get('risk_metrics', {}).get('overall_risk', 0.0) for d in optimal_decisions]
            insights['risk_assessment'] = {
                'overall_risk': np.mean(risks) if risks else 0.0,
                'risk_factors': [],
                'risk_distribution': risks
            }
            
            # 置信度水平
            confidences = [d.get('decision_confidence', 0.0) for d in optimal_decisions]
            insights['confidence_levels'] = {
                'average': np.mean(confidences) if confidences else 0.0,
                'minimum': np.min(confidences) if confidences else 0.0,
                'maximum': np.max(confidences) if confidences else 0.0
            }
        
        return insights
    
    def update_mutation_outcome(self, mutation_info: Dict[str, Any], success: bool, improvement: float):
        """更新变异结果（在线学习）"""
        
        outcome = {
            'mutation_info': mutation_info,
            'success': success,
            'improvement': improvement,
            'timestamp': len(self.mutation_history)
        }
        
        self.mutation_history.append(outcome)
        
        # 限制历史长度
        if len(self.mutation_history) > self.config.max_mutation_history:
            self.mutation_history = self.mutation_history[-self.config.max_mutation_history:]
        
        # 更新贝叶斯先验
        self.bayesian_inference.update_priors(mutation_info, success, improvement)
        
        logger.info(f"📈 更新变异结果: 成功={success}, 改进={improvement:.4f}")
    
    def set_aggressive_mode(self):
        """设置积极模式"""
        self.config_manager.reset_to_aggressive_mode()
        self.config = self.config_manager.get_config()
        logger.info("🚀 贝叶斯引擎设置为积极模式")
    
    def set_conservative_mode(self):
        """设置保守模式"""
        self.config_manager.reset_to_conservative_mode()
        self.config = self.config_manager.get_config()
        logger.info("🛡️ 贝叶斯引擎设置为保守模式")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        return {
            'mutation_history_length': len(self.mutation_history),
            'performance_history_length': len(self.performance_history),
            'current_mode': 'aggressive' if self.config.dynamic_thresholds['confidence_threshold'] < 0.3 else 'conservative',
            'config_summary': {
                'confidence_threshold': self.config.dynamic_thresholds['confidence_threshold'],
                'min_expected_improvement': self.config.dynamic_thresholds['min_expected_improvement'],
                'mc_samples': self.config.mc_samples
            }
        }


class BayesianInferenceEngine:
    """贝叶斯推理引擎"""
    
    def __init__(self, config):
        self.config = config
        self.mutation_priors = config.mutation_priors.copy()
    
    def analyze_candidates(self, candidates: List[Dict[str, Any]], features: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析候选点的贝叶斯推理"""
        
        results = {'candidate_analyses': [], 'summary': {}}
        
        for candidate in candidates:
            analysis = self._analyze_single_candidate(candidate, features, history)
            results['candidate_analyses'].append(analysis)
        
        # 构建摘要
        if results['candidate_analyses']:
            success_probs = [a['success_probability'] for a in results['candidate_analyses']]
            results['summary'] = {
                'avg_success_probability': np.mean(success_probs),
                'max_success_probability': np.max(success_probs),
                'min_success_probability': np.min(success_probs),
                'total_candidates_analyzed': len(candidates)
            }
        
        return results
    
    def _analyze_single_candidate(self, candidate: Dict[str, Any], features: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析单个候选点"""
        
        # 获取建议的变异类型
        suggested_mutations = candidate.get('suggested_mutations', ['width_expansion'])
        mutation_type = suggested_mutations[0] if suggested_mutations else 'width_expansion'
        
        # 获取先验分布参数
        prior = self.mutation_priors.get(mutation_type, {'alpha': 10, 'beta': 10})
        
        # 根据历史更新后验
        alpha, beta = self._update_posterior_from_history(mutation_type, history, prior['alpha'], prior['beta'])
        
        # 计算成功概率
        success_probability = alpha / (alpha + beta)
        
        # 估计期望改进
        expected_improvement = self._estimate_expected_improvement(candidate, features, success_probability)
        
        # 更好的置信度计算
        # 基于贝叶斯后验分布的不确定性
        total_observations = alpha + beta
        if total_observations > 0:
            # 使用贝塔分布的方差来计算置信度
            variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            confidence = min(1.0, 1.0 - variance * 10)  # 方差越小，置信度越高
        else:
            confidence = success_probability * 0.5  # 无观测数据时的低置信度
        
        # 确保最小置信度
        if expected_improvement > 0:
            confidence = max(confidence, 0.3)  # 如果有期望改进，最少30%置信度
        
        return {
            'candidate': candidate,
            'mutation_type': mutation_type,
            'prior_alpha': prior['alpha'],
            'prior_beta': prior['beta'],
            'posterior_alpha': alpha,
            'posterior_beta': beta,
            'success_probability': success_probability,
            'expected_improvement': expected_improvement,
            'confidence': confidence
        }
    
    def _update_posterior_from_history(self, mutation_type: str, history: List[Dict[str, Any]], alpha: float, beta: float) -> Tuple[float, float]:
        """根据历史更新后验分布"""
        
        for outcome in history:
            mut_info = outcome.get('mutation_info', {})
            if mut_info.get('mutation_type') == mutation_type:
                if outcome.get('success', False):
                    alpha += 1
                else:
                    beta += 1
        
        return alpha, beta
    
    def _estimate_expected_improvement(self, candidate: Dict[str, Any], features: Dict[str, Any], success_prob: float) -> float:
        """估计期望改进"""
        
        # 基础改进估计
        base_improvement = 0.02  # 2%基础改进
        
        # 根据候选点优先级调整
        priority = candidate.get('priority', 0.5)
        priority_factor = 0.5 + priority
        
        # 根据检测方法调整
        method_factors = {
            'gradient_vanishing': 1.5,
            'gradient_explosion': 1.3,
            'low_activation': 1.2,
            'performance_degradation': 2.0,
            'performance_stagnation': 0.8
        }
        
        detection_method = candidate.get('detection_method', '')
        method_factor = method_factors.get(detection_method, 1.0)
        
        # 计算期望改进
        expected_improvement = base_improvement * priority_factor * method_factor * success_prob
        
        return min(expected_improvement, 0.1)  # 限制最大改进为10%
    
    def update_priors(self, mutation_info: Dict[str, Any], success: bool, improvement: float):
        """更新先验分布"""
        
        mutation_type = mutation_info.get('mutation_type', '')
        if mutation_type in self.mutation_priors:
            if success:
                self.mutation_priors[mutation_type]['alpha'] += 1
            else:
                self.mutation_priors[mutation_type]['beta'] += 1


class UtilityEvaluator:
    """效用评估器"""
    
    def __init__(self, config):
        self.config = config
        self.utility_params = config.utility_params
    
    def evaluate_utilities(self, candidates: List[Dict[str, Any]], inference_results: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """评估候选点效用"""
        
        candidate_analyses = inference_results.get('candidate_analyses', [])
        utilities = []
        
        for analysis in candidate_analyses:
            utility = self._calculate_utility(analysis, features)
            utilities.append(utility)
        
        return {
            'utilities': utilities,
            'summary': {
                'max_utility': max(utilities) if utilities else 0.0,
                'avg_utility': np.mean(utilities) if utilities else 0.0,
                'total_evaluated': len(utilities)
            }
        }
    
    def _calculate_utility(self, analysis: Dict[str, Any], features: Dict[str, Any]) -> float:
        """计算单个候选点的效用"""
        
        # 基础效用组件
        accuracy_gain = analysis.get('expected_improvement', 0.0) * self.utility_params['accuracy_weight']
        success_bonus = analysis.get('success_probability', 0.0) * 0.1
        exploration_bonus = self.utility_params['exploration_bonus']
        
        # 风险惩罚
        risk_penalty = (1 - analysis.get('success_probability', 0.0)) * self.utility_params['risk_aversion']
        
        # 计算总效用
        total_utility = accuracy_gain + success_bonus + exploration_bonus - risk_penalty
        
        # 确保效用值合理，避免全为0的情况
        if total_utility <= 0 and analysis.get('expected_improvement', 0.0) > 0:
            # 如果计算出的效用为0但有期望改进，给一个最小值
            total_utility = analysis.get('expected_improvement', 0.0) * 0.5
        
        return max(0.0, total_utility)


class DecisionMaker:
    """决策制定器"""
    
    def __init__(self, config):
        self.config = config
        self.thresholds = config.dynamic_thresholds
    
    def make_decisions(self, candidates: List[Dict[str, Any]], inference_results: Dict[str, Any], utility_results: Dict[str, Any]) -> Dict[str, Any]:
        """制定最终决策"""
        
        candidate_analyses = inference_results.get('candidate_analyses', [])
        utilities = utility_results.get('utilities', [])
        
        if not candidate_analyses or not utilities:
            return {'optimal_decisions': [], 'overall_confidence': 0.0, 'execution_reason': 'no_viable_candidates'}
        
        # 合并分析和效用
        combined_data = []
        for analysis, utility in zip(candidate_analyses, utilities):
            # 更灵活的阈值检查
            expected_improvement = analysis.get('expected_improvement', 0)
            success_probability = analysis.get('success_probability', 0)
            confidence = analysis.get('confidence', 0)
            
            # 检查是否满足阈值（使用OR逻辑，更宽松）
            meets_improvement = expected_improvement >= self.thresholds['min_expected_improvement']
            meets_probability = success_probability >= self.thresholds['confidence_threshold'] 
            meets_confidence = confidence >= self.thresholds['confidence_threshold']
            meets_utility = utility >= self.thresholds.get('min_utility', 0.01)
            
            # 如果满足任意两个条件就认为通过（更宽松的策略）
            conditions_met = sum([meets_improvement, meets_probability, meets_confidence, meets_utility])
            meets_threshold = conditions_met >= 2
            
            combined_data.append({
                'analysis': analysis,
                'utility': utility,
                'meets_threshold': meets_threshold,
                'conditions_met': conditions_met
            })
        
        # 筛选满足阈值的候选点
        viable_candidates = [data for data in combined_data if data['meets_threshold']]
        
        if not viable_candidates:
            return {'optimal_decisions': [], 'overall_confidence': 0.0, 'execution_reason': 'no_candidates_meet_thresholds'}
        
        # 按效用排序
        viable_candidates.sort(key=lambda x: x['utility'], reverse=True)
        
        # 选择最优决策（最多3个）
        max_decisions = min(3, len(viable_candidates))
        selected_candidates = viable_candidates[:max_decisions]
        
        optimal_decisions = []
        for data in selected_candidates:
            analysis = data['analysis']
            candidate = analysis['candidate']
            
            # 调试日志：检查候选点内容
            layer_name = candidate.get('layer_name', '')
            logger.info(f"🔍 构建决策 - 候选点 layer_name: '{layer_name}', 候选点内容: {candidate}")
            
            decision = {
                'layer_name': layer_name,
                'target_layer': layer_name,  # 添加备用字段保持一致性
                'mutation_type': analysis.get('mutation_type', ''),
                'success_probability': analysis.get('success_probability', 0.0),
                'expected_improvement': analysis.get('expected_improvement', 0.0),
                'expected_utility': data['utility'],
                'decision_confidence': analysis.get('confidence', 0.0),
                'rationale': candidate.get('rationale', ''),
                'risk_metrics': {
                    'overall_risk': 1.0 - analysis.get('success_probability', 0.0),
                    'value_at_risk': analysis.get('expected_improvement', 0.0) * 0.5  # 简化的VaR
                },
                'source': 'bayesian_analysis'
            }
            optimal_decisions.append(decision)
        
        # 计算整体置信度
        confidences = [d['decision_confidence'] for d in optimal_decisions]
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        # 确保置信度不为0（如果有决策的话）
        if overall_confidence == 0.0 and len(optimal_decisions) > 0:
            # 使用期望效用作为备用置信度指标
            utilities = [d.get('expected_utility', 0.0) for d in optimal_decisions]
            if utilities and max(utilities) > 0:
                overall_confidence = min(0.8, max(utilities) * 10)  # 将效用转换为置信度
                logger.info(f"🔧 使用效用计算置信度: {overall_confidence:.3f}")
        
        return {
            'optimal_decisions': optimal_decisions,
            'overall_confidence': overall_confidence,
            'execution_reason': 'bayesian_optimization',
            'candidates_evaluated': len(candidate_analyses),
            'candidates_viable': len(viable_candidates),
            'decisions_selected': len(optimal_decisions)
        }