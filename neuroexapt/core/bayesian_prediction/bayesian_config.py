"""
贝叶斯配置管理器

将硬编码的参数提取为可配置的设置
"""

from typing import Dict, Any
from dataclasses import dataclass, field
import yaml
import os


@dataclass
class BayesianConfig:
    """贝叶斯系统配置"""
    
    # 贝叶斯先验分布参数
    mutation_priors: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'width_expansion': {'alpha': 15, 'beta': 5},
        'depth_expansion': {'alpha': 12, 'beta': 8},
        'attention_enhancement': {'alpha': 10, 'beta': 10},
        'residual_connection': {'alpha': 18, 'beta': 2},
        'batch_norm_insertion': {'alpha': 20, 'beta': 5},
        'parallel_division': {'alpha': 8, 'beta': 12},
        'serial_division': {'alpha': 12, 'beta': 8},
        'channel_attention': {'alpha': 10, 'beta': 10},
        'layer_norm': {'alpha': 16, 'beta': 4},
        'information_enhancement': {'alpha': 9, 'beta': 11}
    })
    
    # 动态阈值配置
    dynamic_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_expected_improvement': 0.002,
        'max_acceptable_risk': 0.4,
        'confidence_threshold': 0.3,
        'exploration_threshold': 0.25,
        'bottleneck_threshold': 0.3
    })
    
    # 效用函数参数
    utility_params: Dict[str, float] = field(default_factory=lambda: {
        'accuracy_weight': 1.0,
        'efficiency_weight': 0.3,
        'risk_aversion': 0.2,
        'exploration_bonus': 0.1
    })
    
    # 高斯过程超参数
    gp_params: Dict[str, float] = field(default_factory=lambda: {
        'length_scale': 1.0,
        'signal_variance': 1.0,
        'noise_variance': 0.1,
        'mean_function': 0.0
    })
    
    # 蒙特卡罗参数
    mc_samples: int = 500
    confidence_levels: list = field(default_factory=lambda: [0.68, 0.95, 0.99])
    
    # 历史记录参数
    max_mutation_history: int = 100
    max_performance_history: int = 50
    max_architecture_features: int = 100


class BayesianConfigManager:
    """贝叶斯配置管理器"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> BayesianConfig:
        """加载配置"""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
                return BayesianConfig(**config_dict)
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
                print("Using default configuration")
        
        return BayesianConfig()
    
    def save_config(self, path: str = None):
        """保存配置"""
        save_path = path or self.config_path or 'bayesian_config.yaml'
        config_dict = {
            'mutation_priors': self.config.mutation_priors,
            'dynamic_thresholds': self.config.dynamic_thresholds,
            'utility_params': self.config.utility_params,
            'gp_params': self.config.gp_params,
            'mc_samples': self.config.mc_samples,
            'confidence_levels': self.config.confidence_levels,
            'max_mutation_history': self.config.max_mutation_history,
            'max_performance_history': self.config.max_performance_history,
            'max_architecture_features': self.config.max_architecture_features
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        for key, value in updates.items():
            if hasattr(self.config, key):
                if isinstance(getattr(self.config, key), dict):
                    getattr(self.config, key).update(value)
                else:
                    setattr(self.config, key, value)
    
    def get_config(self) -> BayesianConfig:
        """获取配置"""
        return self.config
    
    def reset_to_aggressive_mode(self):
        """重置为积极模式"""
        self.config.dynamic_thresholds.update({
            'min_expected_improvement': 0.001,
            'confidence_threshold': 0.2,
            'exploration_threshold': 0.15,
            'bottleneck_threshold': 0.25
        })
        
        self.config.utility_params.update({
            'risk_aversion': 0.1,
            'exploration_bonus': 0.15
        })
    
    def reset_to_conservative_mode(self):
        """重置为保守模式"""
        self.config.dynamic_thresholds.update({
            'min_expected_improvement': 0.005,
            'confidence_threshold': 0.5,
            'exploration_threshold': 0.4,
            'bottleneck_threshold': 0.5
        })
        
        self.config.utility_params.update({
            'risk_aversion': 0.3,
            'exploration_bonus': 0.05
        })