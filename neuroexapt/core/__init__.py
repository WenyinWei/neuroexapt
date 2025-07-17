# Core components for NeuroExapt
from .model import Network
from .operations import OPS
from .genotypes import Genotype, PRIMITIVES
from .evolvable_model import EvolvableNetwork, EvolvableCell

# ASO-SE Framework components (备份保留)
# from .aso_se_framework import ASOSEFramework, ASOSEConfig
# from .aso_se_trainer import ASOSETrainer
# from .function_preserving_init import FunctionPreservingInitializer
# from .gumbel_softmax_explorer import GumbelSoftmaxExplorer
# from .architecture_mutator import ArchitectureMutator
# from .stability_monitor import StabilityMonitor

# High-Performance Components - NEW!
from .fast_operations import (
    FastMixedOp, BatchedArchitectureUpdate, MemoryEfficientCell,
    FastDeviceManager, get_fast_device_manager, OperationProfiler
)

# Device management
from .device_manager import DeviceManager, get_device_manager, register_model, auto_device, optimize_memory

# Checkpoint management
from .checkpoint_manager import CheckpointManager, get_checkpoint_manager
from .evolution_checkpoint import EvolutionCheckpointManager

# Legacy components
from .architect import Architect
from .separated_training import SeparatedTrainer, SeparatedOptimizer

__all__ = [
    'Network', 'OPS', 'PRIMITIVES', 'Genotype',
    'EvolvableNetwork', 'EvolvableCell',
    'ASOSEFramework', 'ASOSEConfig', 'ASOSETrainer',
    'FunctionPreservingInitializer', 'GumbelSoftmaxExplorer', 'ArchitectureMutator',
    'StabilityMonitor', 
    # High-Performance Components
    'FastMixedOp', 'BatchedArchitectureUpdate', 'MemoryEfficientCell',
    'FastDeviceManager', 'get_fast_device_manager', 'OperationProfiler',
    # Device & Checkpoint Management
    'DeviceManager', 'get_device_manager', 'register_model',
    'auto_device', 'optimize_memory', 'CheckpointManager', 'get_checkpoint_manager',
    'EvolutionCheckpointManager', 'Architect', 'SeparatedTrainer', 'SeparatedOptimizer'
]
