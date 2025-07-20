"""
贝叶斯不确定性估计器
基于变分推断和随机权重平均(SWA)计算特征层的后验方差和不确定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)


class BayesianLinear(nn.Module):
    """
    贝叶斯线性层，使用变分推断
    """
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # 权重的均值和标准差参数
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 5)
        
        # 偏置的均值和标准差参数
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1 - 5)
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            sample: 是否从后验分布采样，False时使用均值
            
        Returns:
            输出张量
        """
        if sample and self.training:
            # 从后验分布采样权重和偏置
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            # 使用均值
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        计算KL散度：KL(q(w) || p(w))
        """
        # 权重的KL散度
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu ** 2 + weight_var) / (self.prior_std ** 2) 
            - 1 - self.weight_logvar + 2 * np.log(self.prior_std)
        )
        
        # 偏置的KL散度
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu ** 2 + bias_var) / (self.prior_std ** 2)
            - 1 - self.bias_logvar + 2 * np.log(self.prior_std)
        )
        
        return weight_kl + bias_kl


class UncertaintyProbe(nn.Module):
    """
    不确定性探针，用于估计特征层的不确定性
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__()
        
        # 使用贝叶斯网络作为探针
        self.layers = nn.ModuleList([
            BayesianLinear(input_dim, hidden_dim),
            BayesianLinear(hidden_dim, hidden_dim),
            BayesianLinear(hidden_dim, output_dim)
        ])
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """前向传播"""
        x = x.view(x.size(0), -1)  # 展平
        
        for i, layer in enumerate(self.layers):
            x = layer(x, sample)
            if i < len(self.layers) - 1:  # 最后一层不加激活
                x = F.relu(x)
                
        return x
    
    def total_kl_divergence(self) -> torch.Tensor:
        """计算总KL散度"""
        total_kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            total_kl += layer.kl_divergence()
        return total_kl
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测并估计不确定性
        
        Returns:
            (mean, variance) 预测均值和方差
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x, sample=True)
                predictions.append(pred)
                
        predictions = torch.stack(predictions)  # [num_samples, batch_size, output_dim]
        
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        return mean, variance


class BayesianUncertaintyEstimator:
    """
    贝叶斯不确定性估计器
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.uncertainty_probes = {}
        self.swa_models = {}  # 随机权重平均模型
        self.uncertainty_history = defaultdict(list)
        
    def create_uncertainty_probe(self, 
                                layer_name: str,
                                input_dim: int,
                                hidden_dim: int = 128) -> UncertaintyProbe:
        """
        为指定层创建不确定性探针
        """
        probe = UncertaintyProbe(input_dim, hidden_dim).to(self.device)
        self.uncertainty_probes[layer_name] = probe
        return probe
    
    def train_uncertainty_probe(self,
                              layer_name: str,
                              features: torch.Tensor,
                              targets: torch.Tensor,
                              num_epochs: int = 100,
                              learning_rate: float = 1e-3,
                              kl_weight: float = 1e-3) -> float:
        """
        训练不确定性探针
        
        Args:
            layer_name: 层名称
            features: 特征张量
            targets: 目标张量（可以是原始标签或重构目标）
            num_epochs: 训练轮数
            learning_rate: 学习率
            kl_weight: KL散度的权重
            
        Returns:
            最终的不确定性估计值
        """
        # 获取特征维度
        if features.dim() > 2:
            feature_dim = np.prod(features.shape[1:])
        else:
            feature_dim = features.shape[1]
            
        # 创建或获取探针
        if layer_name not in self.uncertainty_probes:
            self.create_uncertainty_probe(layer_name, feature_dim)
            
        probe = self.uncertainty_probes[layer_name]
        optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
        
        probe.train()
        features = features.to(self.device)
        targets = targets.to(self.device)
        
        # 如果targets是分类标签，转换为回归目标（特征重构或置信度预测）
        if targets.dim() == 1 and targets.dtype == torch.long:
            # 使用特征的某种统计量作为回归目标
            targets = torch.norm(features.view(features.size(0), -1), dim=1, keepdim=True)
            
        uncertainty_estimates = []
        
        for epoch in range(num_epochs):
            # 变分推断损失
            pred = probe(features, sample=True)
            
            # 重构损失（或预测损失）
            if targets.dim() == pred.dim():
                reconstruction_loss = F.mse_loss(pred, targets)
            else:
                reconstruction_loss = F.mse_loss(pred.squeeze(), targets)
                
            # KL散度损失
            kl_loss = probe.total_kl_divergence()
            
            # 总损失
            total_loss = reconstruction_loss + kl_weight * kl_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 计算不确定性
            if epoch % 10 == 0:
                with torch.no_grad():
                    _, variance = probe.predict_with_uncertainty(features, num_samples=20)
                    uncertainty = variance.mean().item()
                    uncertainty_estimates.append(uncertainty)
                    
            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}: Reconstruction Loss = {reconstruction_loss.item():.4f}, "
                           f"KL Loss = {kl_loss.item():.4f}, Uncertainty = {uncertainty:.4f}")
                
        # 最终不确定性估计
        with torch.no_grad():
            _, final_variance = probe.predict_with_uncertainty(features, num_samples=100)
            final_uncertainty = final_variance.mean().item()
            
        self.uncertainty_history[layer_name].append(final_uncertainty)
        return final_uncertainty
    
    def estimate_uncertainty(self,
                           features: torch.Tensor,
                           labels: torch.Tensor,
                           layer_name: str,
                           num_classes: int = None,
                           num_samples: int = 50,  # 减少采样数
                           num_epochs: int = 50) -> float:   # 减少训练轮数
        """
        估计层特征的不确定性
        
        Args:
            features: 层特征
            labels: 标签
            layer_name: 层名称
            num_classes: 类别数
            num_samples: 不确定性采样次数
            num_epochs: 训练轮数
            
        Returns:
            不确定性估计值
        """
        try:
            # 清理显存
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 确保张量在正确设备上
            features = features.to(self.device).detach()
            labels = labels.to(self.device).detach()
            
            # 限制batch size以节省内存
            if features.size(0) > 32:
                features = features[:32]
                labels = labels[:32]
            
            # 获取特征维度
            if features.dim() > 2:
                feature_dim = np.prod(features.shape[1:])
                features_flat = features.view(features.size(0), -1)
                # 限制特征维度
                if feature_dim > 512:
                    features_flat = features_flat[:, :512]
                    feature_dim = 512
            else:
                features_flat = features
                feature_dim = features.shape[1]
            
            # 创建简化的不确定性探针
            if layer_name not in self.uncertainty_probes:
                hidden_dim = min(64, feature_dim // 2)  # 减少隐藏层维度
                self.uncertainty_probes[layer_name] = UncertaintyProbe(
                    feature_dim, hidden_dim=hidden_dim
                ).to(self.device)
            
            probe = self.uncertainty_probes[layer_name]
            
            # 训练探针
            uncertainty_value = self.train_uncertainty_probe(
                layer_name, features_flat, labels, 
                num_epochs=num_epochs, kl_weight=1e-4
            )
            
            # 估计预测不确定性
            probe.eval()
            with torch.no_grad():
                try:
                    _, variance = probe.predict_with_uncertainty(
                        features_flat, num_samples=num_samples
                    )
                    uncertainty_estimate = torch.mean(variance).item()
                except Exception as e:
                    logger.warning(f"Failed to estimate prediction uncertainty for {layer_name}: {e}")
                    uncertainty_estimate = uncertainty_value
            
            # 确保返回有效值
            if torch.isnan(torch.tensor(uncertainty_estimate)) or torch.isinf(torch.tensor(uncertainty_estimate)):
                uncertainty_estimate = 0.0
            
            logger.info(f"Layer {layer_name}: Uncertainty = {uncertainty_estimate:.4f}")
            
            # 清理显存
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return max(0.0, uncertainty_estimate)
            
        except Exception as e:
            logger.warning(f"Failed to estimate uncertainty for layer {layer_name}: {e}")
            return 0.0
    
    def estimate_feature_uncertainty(self,
                                   feature_dict: Dict[str, torch.Tensor],
                                   targets: torch.Tensor = None) -> Dict[str, float]:
        """
        批量估计多层的特征不确定性
        
        Args:
            feature_dict: 字典，键为层名，值为特征张量
            targets: 目标张量，如果None则使用特征自身作为重构目标
            
        Returns:
            字典，键为层名，值为不确定性估计值
        """
        uncertainty_results = {}
        
        for layer_name, features in feature_dict.items():
            try:
                # 如果没有提供targets，使用特征重构作为代理任务
                if targets is None:
                    # 使用特征的L2范数作为回归目标
                    layer_targets = torch.norm(features.view(features.size(0), -1), dim=1, keepdim=True)
                else:
                    layer_targets = targets
                    
                uncertainty = self.train_uncertainty_probe(
                    layer_name, features, layer_targets
                )
                uncertainty_results[layer_name] = uncertainty
                logger.info(f"Layer {layer_name}: Uncertainty = {uncertainty:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to estimate uncertainty for layer {layer_name}: {e}")
                uncertainty_results[layer_name] = float('inf')  # 高不确定性表示问题
                
        return uncertainty_results
    
    def estimate_predictive_uncertainty(self,
                                      features: torch.Tensor,
                                      layer_name: str,
                                      num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        估计预测不确定性
        
        Args:
            features: 特征张量
            layer_name: 层名称
            num_samples: 采样次数
            
        Returns:
            (epistemic_uncertainty, aleatoric_uncertainty)
        """
        if layer_name not in self.uncertainty_probes:
            raise ValueError(f"No uncertainty probe found for layer {layer_name}")
            
        probe = self.uncertainty_probes[layer_name]
        probe.eval()
        
        with torch.no_grad():
            mean, variance = probe.predict_with_uncertainty(features.to(self.device), num_samples)
            
            # 认知不确定性（epistemic）：模型参数的不确定性
            epistemic_uncertainty = variance.mean(dim=-1)
            
            # 偶然不确定性（aleatoric）：数据固有的噪声（这里简化为预测方差的一个估计）
            predictions = []
            for _ in range(10):  # 少量采样估计偶然不确定性
                pred = probe(features.to(self.device), sample=False)  # 使用均值参数
                predictions.append(pred)
            pred_variance = torch.stack(predictions).var(dim=0).mean(dim=-1)
            aleatoric_uncertainty = pred_variance
            
        return epistemic_uncertainty, aleatoric_uncertainty
    
    def swa_uncertainty_estimation(self,
                                 model: nn.Module,
                                 dataloader,
                                 layer_names: List[str],
                                 num_models: int = 10) -> Dict[str, float]:
        """
        使用随机权重平均(SWA)估计不确定性
        
        Args:
            model: 要分析的模型
            dataloader: 数据加载器
            layer_names: 要分析的层名称列表
            num_models: SWA模型的数量
            
        Returns:
            各层的不确定性估计
        """
        # 保存多个训练检查点作为SWA集合
        if len(self.swa_models) < num_models:
            # 如果SWA模型不够，先收集
            model_state = copy.deepcopy(model.state_dict())
            self.swa_models[f"model_{len(self.swa_models)}"] = model_state
            logger.info(f"Collected SWA model {len(self.swa_models)}/{num_models}")
            return {}
        
        # 使用SWA模型集合估计不确定性
        uncertainty_results = {}
        
        # 注册hook来收集特征
        features_collection = {name: [] for name in layer_names}
        
        def get_hook(name):
            def hook(module, input, output):
                features_collection[name].append(output.detach().cpu())
            return hook
        
        # 注册hooks
        hooks = []
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(get_hook(name))
                hooks.append(hook)
        
        # 对每个SWA模型收集特征
        model_predictions = {name: [] for name in layer_names}
        
        for model_name, state_dict in self.swa_models.items():
            model.load_state_dict(state_dict)
            model.eval()
            
            # 清空特征收集
            for name in layer_names:
                features_collection[name].clear()
            
            # 前向传播一个批次
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(dataloader):
                    if batch_idx >= 1:  # 只处理一个批次
                        break
                    data = data.to(self.device)
                    _ = model(data)
                    break
            
            # 保存这个模型的特征预测
            for name in layer_names:
                if features_collection[name]:
                    model_predictions[name].append(features_collection[name][0])
        
        # 清理hooks
        for hook in hooks:
            hook.remove()
        
        # 计算每层的不确定性
        for layer_name in layer_names:
            if model_predictions[layer_name]:
                # 将所有模型的预测堆叠
                predictions = torch.stack(model_predictions[layer_name])  # [num_models, batch, ...]
                
                # 计算预测方差作为不确定性度量
                variance = predictions.var(dim=0).mean().item()
                uncertainty_results[layer_name] = variance
                
                logger.info(f"SWA uncertainty for {layer_name}: {variance:.4f}")
            else:
                uncertainty_results[layer_name] = float('inf')
        
        return uncertainty_results
    
    def get_uncertainty_history(self, layer_name: str = None) -> Dict[str, List[float]]:
        """获取不确定性历史"""
        if layer_name:
            return {k: v for k, v in self.uncertainty_history.items() if layer_name in k}
        return dict(self.uncertainty_history)
    
    def clear_probes(self):
        """清理探针缓存"""
        self.uncertainty_probes.clear()
        self.uncertainty_history.clear()
        self.swa_models.clear()