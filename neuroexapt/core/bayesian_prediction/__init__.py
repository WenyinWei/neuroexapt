"""
贝叶斯预测模块

包含：
- BayesianMutationBenefitPredictor: 基于贝叶斯推断的变异收益预测
- 高斯过程回归、蒙特卡罗采样等核心算法
- 不确定性量化和风险评估
"""

from .bayesian_predictor import BayesianMutationBenefitPredictor
from .prior_knowledge import PriorKnowledgeBase
from .uncertainty_quantification import UncertaintyQuantifier

__all__ = [
    'BayesianMutationBenefitPredictor',
    'PriorKnowledgeBase', 
    'UncertaintyQuantifier'
]