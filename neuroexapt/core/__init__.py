"""
Core modules for Neuro Exapt framework.

This package contains the fundamental components:
- Information bottleneck engine
- Adaptive entropy controller
- Structural evolution mechanisms
"""

from .information_theory import InformationBottleneck, AdaptiveInformationBottleneck
from .entropy_control import AdaptiveEntropy, EntropyMetrics
from .structural_evolution import StructuralEvolution, EvolutionStep
from .operators import (
    StructuralOperator,
    PruneByEntropy,
    ExpandWithMI,
    MutateDiscrete,
    CompoundOperator,
    AdaptiveOperator
)

__all__ = [
    "InformationBottleneck",
    "AdaptiveInformationBottleneck",
    "AdaptiveEntropy",
    "EntropyMetrics",
    "StructuralEvolution",
    "EvolutionStep",
    "StructuralOperator",
    "PruneByEntropy",
    "ExpandWithMI",
    "MutateDiscrete",
    "CompoundOperator",
    "AdaptiveOperator",
] 