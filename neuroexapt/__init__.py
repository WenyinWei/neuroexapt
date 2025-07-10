"""
Neuro Exapt: A revolutionary neural network framework based on information theory
for dynamic architecture optimization.

This framework integrates information bottleneck principles with adaptive entropy
control to enable dynamic structural evolution of neural networks during training.
"""

__version__ = "0.1.0"
__author__ = "Neuro Exapt Team"
__email__ = "team@neuroexapt.ai"

# Version check
import sys
if sys.version_info < (3, 8):
    raise RuntimeError("Neuro Exapt requires Python 3.8 or later")

# Import main classes when available
try:
    from .neuroexapt import NeuroExapt, NeuroExaptWrapper
    from .trainer import Trainer
    
    # Import core components for advanced usage
    from .core import (
        InformationBottleneck,
        AdaptiveEntropy,
        StructuralEvolution,
        PruneByEntropy,
        ExpandWithMI,
        MutateDiscrete
    )
    
    # Import mathematical utilities
    from .math import (
        calculate_entropy,
        calculate_mutual_information,
        calculate_network_complexity,
        DiscreteParameterOptimizer
    )
    
    __all__ = [
        # Main classes
        "NeuroExapt",
        "NeuroExaptWrapper", 
        "Trainer",
        # Core components
        "InformationBottleneck",
        "AdaptiveEntropy",
        "StructuralEvolution",
        "PruneByEntropy",
        "ExpandWithMI",
        "MutateDiscrete",
        # Math utilities
        "calculate_entropy",
        "calculate_mutual_information",
        "calculate_network_complexity",
        "DiscreteParameterOptimizer",
        # Metadata
        "__version__",
    ]
except ImportError:
    # During initial setup, these imports might fail
    __all__ = ["__version__"] 