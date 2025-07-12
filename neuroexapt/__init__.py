"""
NeuroExapt: Information-Theoretic Dynamic Architecture Optimization

A revolutionary neural network framework that enables adaptive architecture evolution
during training using information theory principles.

V3 Features:
- Every-epoch architecture checking with intelligent evolution
- Subnetwork redundancy analysis (n(n-1)/2 combinations)
- Intelligent threshold learning (no subjective task weights)
- Smart visualization (only shows changes)
- Automatic rollback on performance degradation
"""

__version__ = "3.0.0"
__author__ = "NeuroExapt Development Team"
__email__ = "team@neuroexapt.ai"

# Version check
import sys
if sys.version_info < (3, 8):
    raise RuntimeError("NeuroExapt requires Python 3.8 or later")

# Import V3 components (main interface)
try:
    from .neuroexapt_v3 import NeuroExaptV3
    from .trainer_v3 import TrainerV3, train_with_neuroexapt
    
    # V3 Core components
    from .core.information_theory_v3 import InformationTheoryV3
    from .core.adaptive_evolution_engine import AdaptiveEvolutionEngine
    from .core.intelligent_thresholds import IntelligentThresholdSystem
    from .utils.smart_visualization import SmartVisualizer
    
    # Main aliases (V3 is now default)
    NeuroExapt = NeuroExaptV3
    Trainer = TrainerV3
    
    _v3_available = True
    
except ImportError as e:
    print(f"Warning: V3 modules not available ({e}). Falling back to legacy versions.")
    _v3_available = False

# Import legacy components (V1/V2 compatibility)
try:
    from .neuroexapt import NeuroExapt as NeuroExaptLegacy, NeuroExaptWrapper
    from .trainer import Trainer as TrainerLegacy
    
    # Legacy core components
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
    
    # If V3 not available, use legacy as default
    if not _v3_available:
        NeuroExapt = NeuroExaptLegacy
        Trainer = TrainerLegacy
    
    # Legacy aliases
    NeuroExaptV1 = NeuroExaptLegacy
    TrainerV1 = TrainerLegacy
    
    _legacy_available = True
    
except ImportError:
    print("Warning: Legacy modules also not available. Limited functionality.")
    _legacy_available = False

# Export list
__all__ = [
    # Main classes (V3 preferred)
    "NeuroExapt",
    "Trainer",
    # Metadata
    "__version__",
]

# Add V3 exports if available
if _v3_available:
    __all__.extend([
        "NeuroExaptV3",
        "TrainerV3", 
        "train_with_neuroexapt",
        "InformationTheoryV3",
        "AdaptiveEvolutionEngine",
        "IntelligentThresholdSystem",
        "SmartVisualizer",
    ])

# Add legacy exports if available
if _legacy_available:
    __all__.extend([
        "NeuroExaptV1",
        "TrainerV1",
        "NeuroExaptWrapper",
        "InformationBottleneck",
        "AdaptiveEntropy", 
        "StructuralEvolution",
        "PruneByEntropy",
        "ExpandWithMI",
        "MutateDiscrete",
        "calculate_entropy",
        "calculate_mutual_information",
        "calculate_network_complexity",
        "DiscreteParameterOptimizer",
    ])

# Package info
__description__ = "Information-Theoretic Dynamic Architecture Optimization"
__url__ = "https://github.com/neuroexapt/neuroexapt"
__license__ = "MIT"

# Configuration
import os
import warnings

# Suppress warnings for cleaner output (can be overridden)
if os.environ.get('NEUROEXAPT_VERBOSE', '0') == '0':
    warnings.filterwarnings('ignore', category=UserWarning, module='neuroexapt')

# Welcome message
def _print_welcome():
    """Print welcome message on import."""
    if _v3_available:
        print("🧠 NeuroExapt V3: Information-Theoretic Neural Network Optimization")
        print("   ✨ Every-epoch checking • 🧮 Subnetwork redundancy analysis")
        print("   🎯 Intelligent thresholds • 📊 Smart visualization")
        print("   Ready for training with automatic architecture optimization!")
    else:
        print("🧠 NeuroExapt: Information-Theoretic Neural Network Optimization")
        print("   Using legacy version. Consider upgrading to V3 for latest features.")

# Show welcome message on first import
if not hasattr(_print_welcome, '_shown'):
    _print_welcome._shown = True
    if os.environ.get('NEUROEXAPT_QUIET', '0') == '0':
        _print_welcome() 