"""
Core modules for Neuro Exapt.
"""

from .entropy_control import AdaptiveEntropy, EntropyMetrics
from .information_theory import InformationBottleneck, AdaptiveInformationBottleneck
from .operators import (
    StructuralOperator,
    PruneByEntropy,
    ExpandWithMI,
    MutateDiscrete,
    CompoundOperator,
    AdaptiveOperator
)
from .structural_evolution import StructuralEvolution, EvolutionStep
from .dynarch import (
    DynamicArchitecture,
    AttentionPolicyNetwork,
    MultiObjectiveReward,
    Experience,
    ExperienceBuffer
)

# Try to import intelligent operators
try:
    from .intelligent_operators import (
        LayerTypeSelector,
        IntelligentExpansionOperator,
        AdaptiveDataFlowOperator,
        BranchSpecializationOperator,
        ChannelAttention
    )
    INTELLIGENT_OPERATORS_AVAILABLE = True
except ImportError:
    INTELLIGENT_OPERATORS_AVAILABLE = False

__all__ = [
    'AdaptiveEntropy',
    'EntropyMetrics',
    'InformationBottleneck',
    'AdaptiveInformationBottleneck',
    'StructuralOperator',
    'PruneByEntropy',
    'ExpandWithMI',
    'MutateDiscrete',
    'CompoundOperator',
    'AdaptiveOperator',
    'StructuralEvolution',
    'EvolutionStep',
    'DynamicArchitecture',
    'AttentionPolicyNetwork',
    'MultiObjectiveReward',
    'Experience',
    'ExperienceBuffer',
    'INTELLIGENT_OPERATORS_AVAILABLE'
]

# Add intelligent operators to __all__ if available
if INTELLIGENT_OPERATORS_AVAILABLE:
    __all__.extend([
        'LayerTypeSelector',
        'IntelligentExpansionOperator',
        'AdaptiveDataFlowOperator',
        'BranchSpecializationOperator',
        'ChannelAttention'
    ]) 