"""
基于MINE的互信息估计器
支持分层互信息I(H_k; Y)和条件互信息I(H_k; Y|H_{k+1})的计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class MINEDiscriminator(nn.Module):
    """
    MINE判别器，支持连续特征和离散标签的互信息估计
    """
    
    def __init__(self, 
                 input_dim_features: int,
                 num_classes: int = None,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.input_dim_features = input_dim_features
        self.num_classes = num_classes
        self.is_discrete_output = num_classes is not None
        
        # 构建判别器网络
        layers = []
        prev_dim = input_dim_features
        
        # 如果有离散标签，添加one-hot编码维度
        if self.is_discrete_output:
            prev_dim += num_classes
            
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
            
        # 输出层
        if self.is_discrete_output:
            layers.append(nn.Linear(prev_dim, num_classes))
        else:
            layers.append(nn.Linear(prev_dim, 1))
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 特征张量 [B, ...] -> 会被展平为 [B, feature_dim]
            labels: 标签张量 [B] (离散) 或 [B, label_dim] (连续)
            
        Returns:
            判别器输出 [B, num_classes] 或 [B, 1]
        """
        # 展平特征
        batch_size = features.size(0)
        features_flat = features.view(batch_size, -1)
        
        if self.is_discrete_output and labels is not None:
            # 将离散标签转换为one-hot编码
            if labels.dim() == 1:  # 类别索引
                labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
            else:  # 已经是one-hot或概率分布
                labels_onehot = labels.float()
                
            # 拼接特征和标签
            joint_input = torch.cat([features_flat, labels_onehot], dim=1)
        else:
            joint_input = features_flat
            
        return self.net(joint_input)


class MutualInformationEstimator:
    """
    基于MINE的互信息估计器
    支持分层互信息和条件互信息的计算
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.discriminators = {}
        self.training_history = defaultdict(list)
        
    def estimate_layerwise_mi(self,
                            features: torch.Tensor,
                            labels: torch.Tensor,
                            layer_name: str,
                            num_classes: int = None,
                            num_epochs: int = 100,
                            learning_rate: float = 1e-3) -> float:
        """
        估计分层互信息 I(H_k; Y)
        
        Args:
            features: 层特征 [B, C, H, W] 或 [B, feature_dim]
            labels: 目标标签 [B] (分类) 或 [B, label_dim] (回归)
            layer_name: 层名称，用于缓存判别器
            num_classes: 分类任务的类别数，None表示回归任务
            num_epochs: 训练轮数
            learning_rate: 学习率
            
        Returns:
            互信息估计值
        """
        try:
            # 确保所有张量在同一设备上，并清理显存
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 统一设备处理
            features = features.to(self.device).detach()
            labels = labels.to(self.device).detach()
            
            # 获取特征维度
            if features.dim() > 2:
                feature_dim = np.prod(features.shape[1:])
                # 对于大特征图，采样减少内存使用
                if feature_dim > 1024:  # 如果特征维度太大
                    batch_size = min(features.size(0), 32)  # 限制batch size
                    features = features[:batch_size]
                    labels = labels[:batch_size]
                    feature_dim = min(feature_dim, 1024)  # 限制特征维度
            else:
                feature_dim = features.shape[1]
                
            # 创建或获取判别器
            discriminator_key = f"{layer_name}_layerwise"
            if discriminator_key not in self.discriminators:
                self.discriminators[discriminator_key] = MINEDiscriminator(
                    input_dim_features=feature_dim,
                    num_classes=num_classes
                ).to(self.device)
                
            discriminator = self.discriminators[discriminator_key]
            optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
            
            # 训练判别器
            mi_estimates = []
            
            # 减少训练轮数以节省计算资源
            effective_epochs = min(num_epochs, 50) if features.size(0) < 100 else num_epochs
            
            for epoch in range(effective_epochs):
                try:
                    # 生成联合样本和边缘样本
                    batch_size = features.size(0)
                    
                    # 联合样本：真实的(features, labels)配对
                    joint_features = features
                    joint_labels = labels
                    
                    # 边缘样本：打乱labels，破坏features和labels的依赖关系
                    marginal_labels = joint_labels[torch.randperm(batch_size)]
                    
                    # 计算判别器输出
                    joint_logits = discriminator(joint_features, joint_labels)
                    marginal_logits = discriminator(joint_features, marginal_labels)
                    
                    # 计算MINE损失
                    if num_classes is not None:  # 分类任务
                        # 对于离散标签，使用交叉熵形式的MINE估计
                        joint_probs = F.softmax(joint_logits, dim=1)
                        marginal_probs = F.softmax(marginal_logits, dim=1)
                        
                        # 计算联合样本的对数似然
                        if joint_labels.dim() == 1:
                            joint_ll = F.cross_entropy(joint_logits, joint_labels, reduction='none')
                        else:
                            joint_ll = -torch.sum(joint_labels * torch.log(joint_probs + 1e-8), dim=1)
                            
                        # 计算边缘样本的对数配分函数
                        marginal_ll = torch.logsumexp(marginal_logits, dim=1) - np.log(num_classes)
                        
                        # MINE估计：E[log p(y|x)] - log E[exp(f(x,y'))]  
                        mi_estimate = torch.mean(-joint_ll) - torch.mean(marginal_ll)
                        
                    else:  # 回归任务
                        # 标准MINE估计
                        mi_estimate = torch.mean(joint_logits) - torch.log(torch.mean(torch.exp(marginal_logits)))
                        
                    # 反向传播
                    loss = -mi_estimate  # 最大化互信息 = 最小化负互信息
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 梯度裁剪防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    mi_estimates.append(mi_estimate.item())
                    
                    # 定期清理显存
                    if epoch % 10 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                except RuntimeError as e:
                    if "out of memory" in str(e) or "CUDA" in str(e):
                        # 内存不足，减少batch size重试
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        logger.warning(f"Memory warning during MI estimation for {layer_name}, retrying with smaller batch")
                        break
                    else:
                        raise e
                        
            # 计算平均互信息估计
            if mi_estimates:
                final_mi = np.mean(mi_estimates[-10:])  # 取最后10次的平均值
                logger.info(f"Layer {layer_name}: I(H; Y) = {final_mi:.4f}")
                return max(0.0, final_mi)  # 确保非负
            else:
                logger.warning(f"Failed to estimate MI for layer {layer_name}")
                return 0.0
                
        except Exception as e:
            logger.warning(f"Failed to estimate MI for layer {layer_name}: {e}")
            return 0.0
    
    def estimate_conditional_mi(self,
                              current_features: torch.Tensor,
                              next_features: torch.Tensor,
                              labels: torch.Tensor,
                              layer_name: str,
                              num_classes: int = None,
                              num_epochs: int = 50,  # 减少训练轮数
                              learning_rate: float = 1e-3) -> float:
        """
        估计条件互信息 I(H_k; Y | H_{k+1})
        
        使用公式：I(H_k; Y | H_{k+1}) = I((H_k, H_{k+1}); Y) - I(H_{k+1}; Y)
        
        Args:
            current_features: 当前层特征 H_k
            next_features: 下一层特征 H_{k+1}
            labels: 目标标签 Y
            layer_name: 层名称
            
        Returns:
            条件互信息估计值
        """
        try:
            # 确保所有张量在同一设备上
            current_features = current_features.to(self.device).detach()
            next_features = next_features.to(self.device).detach()
            labels = labels.to(self.device).detach()
            
            # 限制batch size以节省内存
            batch_size = min(current_features.size(0), 32)
            current_features = current_features[:batch_size]
            next_features = next_features[:batch_size]
            labels = labels[:batch_size]
            
            # 1. 估计 I(H_{k+1}; Y)
            mi_next_y = self.estimate_layerwise_mi(
                next_features, labels, f"{layer_name}_next", num_classes, num_epochs, learning_rate
            )
            
            # 2. 估计 I((H_k, H_{k+1}); Y)
            # 将当前层和下一层特征拼接
            current_flat = current_features.view(current_features.size(0), -1)
            next_flat = next_features.view(next_features.size(0), -1)
            
            # 限制拼接后的特征维度
            max_dim = 1024
            if current_flat.size(1) + next_flat.size(1) > max_dim:
                current_dim = min(current_flat.size(1), max_dim // 2)
                next_dim = min(next_flat.size(1), max_dim // 2)
                current_flat = current_flat[:, :current_dim]
                next_flat = next_flat[:, :next_dim]
            
            joint_features = torch.cat([current_flat, next_flat], dim=1)
            
            mi_joint_y = self.estimate_layerwise_mi(
                joint_features, labels, f"{layer_name}_joint", num_classes, num_epochs, learning_rate
            )
            
            # 3. 计算条件互信息
            conditional_mi = mi_joint_y - mi_next_y
            
            logger.info(f"Conditional MI for {layer_name}: "
                       f"I((H_k,H_{{k+1}}); Y) = {mi_joint_y:.4f}, "
                       f"I(H_{{k+1}}; Y) = {mi_next_y:.4f}, "
                       f"I(H_k; Y | H_{{k+1}}) = {conditional_mi:.4f}")
            
            return conditional_mi
            
        except Exception as e:
            logger.warning(f"Failed to estimate conditional MI for layer {layer_name}: {e}")
            return 0.0
    
    def batch_estimate_layerwise_mi(self,
                                  feature_dict: Dict[str, torch.Tensor],
                                  labels: torch.Tensor,
                                  num_classes: int = None) -> Dict[str, float]:
        """
        批量估计多层的分层互信息
        
        Args:
            feature_dict: 字典，键为层名，值为特征张量
            labels: 目标标签
            num_classes: 分类任务的类别数
            
        Returns:
            字典，键为层名，值为互信息估计值
        """
        mi_results = {}
        
        for layer_name, features in feature_dict.items():
            try:
                mi_value = self.estimate_layerwise_mi(
                    features, labels, layer_name, num_classes
                )
                mi_results[layer_name] = mi_value
                logger.info(f"Layer {layer_name}: I(H; Y) = {mi_value:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to estimate MI for layer {layer_name}: {e}")
                mi_results[layer_name] = 0.0
                
        return mi_results
    
    def batch_estimate_conditional_mi(self,
                                    feature_pairs: List[Tuple[str, torch.Tensor, torch.Tensor]],
                                    labels: torch.Tensor,
                                    num_classes: int = None) -> Dict[str, float]:
        """
        批量估计多层的条件互信息
        
        Args:
            feature_pairs: 列表，每个元素为(层名, 当前层特征, 下一层特征)
            labels: 目标标签
            num_classes: 分类任务的类别数
            
        Returns:
            字典，键为层名，值为条件互信息估计值
        """
        conditional_mi_results = {}
        
        for layer_name, current_features, next_features in feature_pairs:
            try:
                conditional_mi = self.estimate_conditional_mi(
                    current_features, next_features, labels, layer_name, num_classes
                )
                conditional_mi_results[layer_name] = conditional_mi
                logger.info(f"Layer {layer_name}: I(H_k; Y | H_{{k+1}}) = {conditional_mi:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to estimate conditional MI for layer {layer_name}: {e}")
                conditional_mi_results[layer_name] = 0.0
                
        return conditional_mi_results
    
    def get_training_history(self, layer_name: str = None) -> Dict[str, List[float]]:
        """获取训练历史"""
        if layer_name:
            return {k: v for k, v in self.training_history.items() if layer_name in k}
        return dict(self.training_history)
    
    def clear_discriminators(self):
        """清理判别器缓存"""
        self.discriminators.clear()
        self.training_history.clear()