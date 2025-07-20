# Core components for NeuroExapt
# Core architectural components
from .model import Network
from .operations import OPS
from .genotypes import Genotype, PRIMITIVES
from .evolvable_model import EvolvableNetwork, EvolvableCell

# Logging utilities
from .logging_utils import logger, ConfigurableLogger, DebugPrinter, get_logger

# High-Performance Components
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

# DNM (Dynamic Neural Morphogenesis) Framework - 主要框架
from .dnm_framework import DNMFramework, MorphogenesisEvent
from .dnm_neuron_division import AdaptiveNeuronDivision, NeuronDivisionStrategies
from .dnm_layer_analyzer import LayerPerformanceAnalyzer, SmartLayerSelector
from .dnm_connection_growth import DNMConnectionGrowth
from .dnm_net2net import Net2NetTransformer, DNMArchitectureMutator

# Enhanced DNM Components - 增强组件
from .enhanced_bottleneck_detector import EnhancedBottleneckDetector
from .performance_guided_division import PerformanceGuidedDivision, DivisionStrategy

# Intelligent Morphogenesis Components - 智能形态发生组件 (替代传统方法)
from .intelligent_morphogenesis_engine import IntelligentMorphogenesisEngine
from .intelligent_dnm_integration import IntelligentDNMCore

# 🔬 新一代基于互信息和贝叶斯推断的核心框架
from .mutual_information_estimator import MutualInformationEstimator, MINEDiscriminator
from .bayesian_uncertainty_estimator import BayesianUncertaintyEstimator, BayesianLinear, UncertaintyProbe
from .intelligent_bottleneck_detector import IntelligentBottleneckDetector, BottleneckReport, BottleneckType
from .intelligent_mutation_planner import IntelligentMutationPlanner, MutationPlan, MutationType
from .advanced_net2net_transfer import AdvancedNet2NetTransfer, Net2NetTransferMethod
from .intelligent_architecture_evolution_engine import IntelligentArchitectureEvolutionEngine, EvolutionConfig, EvolutionIteration

__all__ = [
    # Core Components
    'Network', 'OPS', 'PRIMITIVES', 'Genotype',
    'EvolvableNetwork', 'EvolvableCell',
    # Logging utilities
    'logger', 'ConfigurableLogger', 'DebugPrinter', 'get_logger',
    # High-Performance Components
    'FastMixedOp', 'BatchedArchitectureUpdate', 'MemoryEfficientCell',
    'FastDeviceManager', 'get_fast_device_manager', 'OperationProfiler',
    # Device & Checkpoint Management
    'DeviceManager', 'get_device_manager', 'register_model',
    'auto_device', 'optimize_memory', 'CheckpointManager', 'get_checkpoint_manager',
    'EvolutionCheckpointManager', 'Architect', 'SeparatedTrainer', 'SeparatedOptimizer',
    # DNM Framework - 神经网络动态形态发生框架
    'DNMFramework', 'MorphogenesisEvent', 'AdaptiveNeuronDivision', 'NeuronDivisionStrategies',
    'LayerPerformanceAnalyzer', 'SmartLayerSelector', 'DNMConnectionGrowth', 
    'Net2NetTransformer', 'DNMArchitectureMutator',
    # Enhanced DNM Components - 增强的DNM组件
    'EnhancedBottleneckDetector', 'PerformanceGuidedDivision', 'DivisionStrategy',
    # Intelligent Morphogenesis Components - 智能形态发生组件
    'IntelligentMorphogenesisEngine', 'IntelligentDNMCore',
    # 🔬 新一代理论框架组件 - 基于互信息和贝叶斯推断的智能架构进化
    'MutualInformationEstimator', 'MINEDiscriminator',
    'BayesianUncertaintyEstimator', 'BayesianLinear', 'UncertaintyProbe',
    'IntelligentBottleneckDetector', 'BottleneckReport', 'BottleneckType',
    'IntelligentMutationPlanner', 'MutationPlan', 'MutationType',
    'AdvancedNet2NetTransfer', 'Net2NetTransferMethod',
    'IntelligentArchitectureEvolutionEngine', 'EvolutionConfig', 'EvolutionIteration'
]
