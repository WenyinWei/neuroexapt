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

logger = logging.getLogger(__name__)


class MonteCarloUncertaintyEstimator:
    """
    基于Monte Carlo Dropout的不确定性估计器
    
    优势：
    - 简单直接，不需要修改网络架构
    - 计算稳定，不会出现NaN或0值
    - 理论基础扎实（Gal & Ghahramani, 2016）
    """
    
    def __init__(self, 
                 n_samples: int = 100,
                 dropout_rate: float = 0.1):
        """
        初始化Monte Carlo不确定性估计器
        
        Args:
            n_samples: Monte Carlo采样次数
            dropout_rate: Dropout概率
        """
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        
    def enable_mc_dropout(self, model: nn.Module, dropout_rate: float = None):
        """
        为模型启用Monte Carlo Dropout
        
        Args:
            model: 目标模型
            dropout_rate: Dropout概率，如果None则使用默认值
        """
        if dropout_rate is None:
            dropout_rate = self.dropout_rate
            
        def apply_mc_dropout(module):
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
                module.train()  # 保持训练模式以启用Dropout
            elif isinstance(module, (nn.Conv2d, nn.Linear)):
                # 为没有Dropout的层添加Dropout
                if not hasattr(module, '_mc_dropout'):
                    module._mc_dropout = nn.Dropout(dropout_rate)
                    
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
        model.eval()  # 设置为评估模式，但Dropout仍然激活
        self.enable_mc_dropout(model)
        
        # 注册hook来捕获指定层的输出
        layer_outputs = []
        
        def hook_fn(module, input, output):
            if hasattr(module, '_mc_dropout'):
                output = module._mc_dropout(output)
            layer_outputs.append(output.detach().clone())
        
        # 找到目标层并注册hook
        target_layer = None
        for name, module in model.named_modules():
            if name == layer_name:
                target_layer = module
                handle = module.register_forward_hook(hook_fn)
                break
                
        if target_layer is None:
            logger.warning(f"Layer {layer_name} not found")
            return 0.0
            
        try:
            all_predictions = []
            
            # Monte Carlo采样
            for _ in range(self.n_samples):
                layer_outputs.clear()
                
                with torch.no_grad():
                    for batch_idx, (data, targets) in enumerate(data_loader):
                        if batch_idx >= 5:  # 只使用前5个batch节省计算
                            break
                            
                        data = data.to(device)
                        
                        # 前向传播
                        _ = model(data)
                        
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
                return max(uncertainty, 1e-6)  # 确保非零
            else:
                logger.warning(f"No predictions collected for layer {layer_name}")
                return 1e-3  # 返回小的默认不确定性
                
        except Exception as e:
            logger.error(f"Error estimating uncertainty for layer {layer_name}: {e}")
            return 1e-3
        finally:
            # 清理hook
            if 'handle' in locals():
                handle.remove()
                
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
        model.eval()
        self.enable_mc_dropout(model)
        
        all_predictions = []
        all_entropies = []
        
        # Monte Carlo采样
        for _ in range(self.n_samples):
            batch_predictions = []
            batch_entropies = []
            
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(data_loader):
                    if batch_idx >= 10:  # 限制计算量
                        break
                        
                    data = data.to(device)
                    outputs = model(data)
                    
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