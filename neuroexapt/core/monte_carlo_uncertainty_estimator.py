"""
Monte Carlo不确定性估计器
基于Monte Carlo Dropout的简单有效不确定性量化方法

核心思路：
1. 启用Dropout在推理时保持激活
2. 多次前向传播获得预测分布
3. 计算预测方差作为不确定性度量
4. 比变分贝叶斯方法更直接、更稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
from dataclasses import dataclass
import weakref

logger = logging.getLogger(__name__)


@dataclass
class MCUncertaintyConfig:
    """Monte Carlo不确定性估计配置"""
    n_samples: int = 50                    # Monte Carlo采样次数
    dropout_rate: float = 0.1              # Dropout概率
    max_batches: int = 5                   # 每次估计使用的最大batch数
    uncertainty_threshold: float = 1e-6    # 最小不确定性阈值
    use_wrapper: bool = True               # 是否使用包装器而非动态属性


class DropoutWrapper(nn.Module):
    """
    Dropout包装器 - 安全的替代动态属性添加方案
    
    避免直接修改原始模块结构，提供更好的隔离性
    """
    
    def __init__(self, module: nn.Module, dropout_rate: float = 0.1):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        return self.dropout(self.module(x))


class MonteCarloUncertaintyEstimator:
    """
    基于Monte Carlo Dropout的不确定性估计器
    
    优势：
    - 简单直接，不需要修改网络架构
    - 计算稳定，不会出现NaN或0值
    - 理论基础扎实（Gal & Ghahramani, 2016）
    - 使用包装器避免动态属性修改
    """
    
    def __init__(self, config: MCUncertaintyConfig = None):
        """
        初始化Monte Carlo不确定性估计器
        
        Args:
            config: 配置对象，如果None则使用默认配置
        """
        self.config = config or MCUncertaintyConfig()
        
        # 用于跟踪已注册的hook和包装器
        self._active_hooks = weakref.WeakSet()
        self._module_wrappers = weakref.WeakKeyDictionary()
        
    def _prepare_model_for_mc_dropout(self, model: nn.Module) -> nn.Module:
        """
        为Monte Carlo Dropout准备模型
        
        Args:
            model: 原始模型
            
        Returns:
            准备好的模型（可能包含包装器）
        """
        if self.config.use_wrapper:
            # 使用包装器方案 - 更安全
            return self._apply_dropout_wrappers(model)
        else:
            # 使用原始方案 - 保持向后兼容
            self._enable_mc_dropout_legacy(model)
            return model
            
    def _apply_dropout_wrappers(self, model: nn.Module) -> nn.Module:
        """应用Dropout包装器"""
        def wrap_modules(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Dropout):
                    # 已有Dropout层，设置为训练模式
                    child.p = self.config.dropout_rate
                    child.train()
                elif isinstance(child, (nn.Conv2d, nn.Linear)):
                    # 为卷积和线性层添加包装器
                    if child not in self._module_wrappers:
                        wrapper = DropoutWrapper(child, self.config.dropout_rate)
                        self._module_wrappers[child] = wrapper
                        setattr(module, name, wrapper)
                else:
                    # 递归处理子模块
                    wrap_modules(child)
                    
        # 创建模型副本以避免修改原始模型
        model_copy = type(model).__new__(type(model))
        model_copy.__dict__.update(model.__dict__)
        wrap_modules(model_copy)
        return model_copy
        
    def _enable_mc_dropout_legacy(self, model: nn.Module):
        """
        为模型启用Monte Carlo Dropout（遗留方法）
        
        Args:
            model: 目标模型
        """
        def apply_mc_dropout(module):
            if isinstance(module, nn.Dropout):
                module.p = self.config.dropout_rate
                module.train()  # 保持训练模式以启用Dropout
            # 移除动态属性添加以避免安全问题
                
        model.apply(apply_mc_dropout)
        
    def estimate_layer_uncertainty(self,
                                 model: nn.Module,
                                 layer_name: str,
                                 data_loader,
                                 device: torch.device) -> float:
        """
        估计指定层的不确定性
        
        Args:
            model: 网络模型
            layer_name: 层名称
            data_loader: 数据加载器
            device: 计算设备
            
        Returns:
            不确定性值（预测方差）
        """
        # 准备模型
        mc_model = self._prepare_model_for_mc_dropout(model)
        mc_model.eval()  # 设置为评估模式，但Dropout仍然激活
        
        # 注册hook来捕获指定层的输出
        layer_outputs = []
        handle = None  # 🔧 修复：初始化handle为None
        
        def hook_fn(module, input, output):
            layer_outputs.append(output.detach().clone())
        
        # 找到目标层并注册hook
        target_layer = None
        for name, module in mc_model.named_modules():
            if name == layer_name:
                target_layer = module
                handle = module.register_forward_hook(hook_fn)
                self._active_hooks.add(handle)  # 跟踪active hook
                break
                
        if target_layer is None:
            logger.warning(f"Layer {layer_name} not found")
            return self.config.uncertainty_threshold
            
        try:
            all_predictions = []
            
            # Monte Carlo采样
            for sample_idx in range(self.config.n_samples):
                layer_outputs.clear()
                
                with torch.no_grad():
                    for batch_idx, (data, targets) in enumerate(data_loader):
                        if batch_idx >= self.config.max_batches:  # 配置化批次限制
                            break
                            
                        data = data.to(device)
                        
                        # 前向传播
                        _ = mc_model(data)
                        
                        if layer_outputs:
                            # 计算层输出的统计信息
                            layer_output = layer_outputs[-1]
                            # 使用平均池化减少维度
                            if len(layer_output.shape) > 2:
                                pooled = F.adaptive_avg_pool2d(layer_output, (1, 1))
                                features = pooled.view(pooled.size(0), -1)
                            else:
                                features = layer_output
                                
                            # 计算特征的L2范数作为代表性统计量
                            feature_norms = torch.norm(features, dim=1)
                            all_predictions.append(feature_norms.cpu().numpy())
            
            # 计算不确定性（预测方差）
            if all_predictions:
                predictions_array = np.concatenate(all_predictions)
                uncertainty = np.var(predictions_array)
                
                # 归一化不确定性值
                uncertainty = float(uncertainty) / (np.mean(predictions_array) + 1e-8)
                
                logger.info(f"Layer {layer_name}: MC Uncertainty = {uncertainty:.6f}")
                return max(uncertainty, self.config.uncertainty_threshold)
            else:
                logger.warning(f"No predictions collected for layer {layer_name}")
                return self.config.uncertainty_threshold * 100  # 返回较大的默认不确定性
                
        except Exception as e:
            logger.error(f"Error estimating uncertainty for layer {layer_name}: {e}")
            return self.config.uncertainty_threshold * 100
        finally:
            # 🔧 修复：确保hook被正确清理
            if handle is not None:
                handle.remove()
                if handle in self._active_hooks:
                    self._active_hooks.discard(handle)
                
    def estimate_model_uncertainty(self,
                                 model: nn.Module,
                                 data_loader,
                                 device: torch.device,
                                 target_layers: List[str] = None) -> Dict[str, float]:
        """
        估计模型多层的不确定性
        
        Args:
            model: 网络模型
            data_loader: 数据加载器  
            device: 计算设备
            target_layers: 目标层列表，如果None则估计所有卷积和线性层
            
        Returns:
            字典，键为层名，值为不确定性估计
        """
        if target_layers is None:
            target_layers = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    target_layers.append(name)
                    
        uncertainties = {}
        
        for layer_name in target_layers:
            uncertainty = self.estimate_layer_uncertainty(
                model, layer_name, data_loader, device
            )
            uncertainties[layer_name] = uncertainty
            
        return uncertainties
        
    def estimate_predictive_uncertainty(self,
                                      model: nn.Module,
                                      data_loader,
                                      device: torch.device) -> Tuple[float, float]:
        """
        估计模型的预测不确定性
        
        Args:
            model: 网络模型
            data_loader: 数据加载器
            device: 计算设备
            
        Returns:
            (认知不确定性, 随机不确定性)
        """
        # 准备模型
        mc_model = self._prepare_model_for_mc_dropout(model)
        mc_model.eval()
        
        all_predictions = []
        all_entropies = []
        
        # Monte Carlo采样
        for sample_idx in range(self.config.n_samples):
            batch_predictions = []
            batch_entropies = []
            
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(data_loader):
                    if batch_idx >= self.config.max_batches * 2:  # 预测不确定性使用更多数据
                        break
                        
                    data = data.to(device)
                    outputs = mc_model(data)
                    
                    # 计算预测概率
                    probs = F.softmax(outputs, dim=1)
                    batch_predictions.append(probs.cpu().numpy())
                    
                    # 计算预测熵
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    batch_entropies.append(entropy.cpu().numpy())
                    
            if batch_predictions:
                all_predictions.append(np.concatenate(batch_predictions))
                all_entropies.append(np.concatenate(batch_entropies))
                
        if not all_predictions:
            return 0.0, 0.0
            
        # 计算认知不确定性（预测分布的方差）
        predictions_array = np.array(all_predictions)  # [n_samples, n_data, n_classes]
        mean_predictions = np.mean(predictions_array, axis=0)
        
        # 认知不确定性：预测均值的熵 - 预测熵的均值
        mean_entropy = np.mean(all_entropies)
        predictive_entropy = -np.sum(mean_predictions * np.log(mean_predictions + 1e-8), axis=1)
        epistemic_uncertainty = np.mean(predictive_entropy) - mean_entropy
        
        # 随机不确定性：预测熵的均值
        aleatoric_uncertainty = mean_entropy
        
        logger.info(f"Epistemic uncertainty: {epistemic_uncertainty:.6f}")
        logger.info(f"Aleatoric uncertainty: {aleatoric_uncertainty:.6f}")
        
        return float(max(epistemic_uncertainty, 0.0)), float(max(aleatoric_uncertainty, 0.0))
        
    def cleanup(self):
        """清理所有注册的hook和包装器"""
        # 清理活跃的hook
        for handle in list(self._active_hooks):
            try:
                handle.remove()
            except:
                pass
        self._active_hooks.clear()
        
        # 清理包装器引用
        self._module_wrappers.clear()
        
    def __del__(self):
        """析构函数中确保清理"""
        try:
            self.cleanup()
        except:
            pass