"""
Mathematical models and metrics for Neuro Exapt.

This package contains:
- Information-theoretic metrics
- Entropy calculations
- Structural complexity measures
- Optimization utilities
"""

from .metrics import (
    calculate_entropy,
    calculate_mutual_information,
    calculate_conditional_entropy,
    calculate_kl_divergence,
    calculate_js_divergence,
    calculate_layer_complexity,
    calculate_network_complexity,
    calculate_structural_entropy,
    calculate_redundancy_score,
    calculate_discrete_parameter_gradient,
    calculate_information_gain,
    calculate_task_complexity,
    estimate_flops
)

from .optimization import (
    DiscreteParameterOptimizer,
    InformationBottleneckOptimizer,
    StructuralGradientEstimator,
    AdaptiveLearningRateScheduler,
    create_optimization_schedule,
    calculate_gradient_information
)

__all__ = [
    # Metrics
    "calculate_entropy",
    "calculate_mutual_information",
    "calculate_conditional_entropy",
    "calculate_kl_divergence",
    "calculate_js_divergence",
    "calculate_layer_complexity",
    "calculate_network_complexity",
    "calculate_structural_entropy",
    "calculate_redundancy_score",
    "calculate_discrete_parameter_gradient",
    "calculate_information_gain",
    "calculate_task_complexity",
    "estimate_flops",
    # Optimization
    "DiscreteParameterOptimizer",
    "InformationBottleneckOptimizer",
    "StructuralGradientEstimator",
    "AdaptiveLearningRateScheduler",
    "create_optimization_schedule",
    "calculate_gradient_information",
] 