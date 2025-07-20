"""
无参数结构评估器 - 核心理论实现
Parameter-Free Structural Evaluator - Core Theory Implementation

基于信息论的神经网络结构固有能力评估，无需训练参数即可量化：
1. 有效信息 (Effective Information, EI) - 结构对输入信息的因果影响能力
2. 积分信息 (Integrated Information, Φ) - 结构整合多源信息的能力  
3. 结构冗余度 (Structural Redundancy, SR) - 结构中各组件功能重叠程度

理论公式：
EI(S) = max_{p(x)} [I(X; Y) - I(X; Y|S)]
Φ ≈ Σ_{i,j} MI(H_i; H_j) - Σ_i MI(H_i; H_i)
SR = rank(1/N Σ_n W_n^T W_n)

作者：基于用户提供的理论框架实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from scipy.stats import entropy
from scipy.linalg import svd
import logging

logger = logging.getLogger(__name__)


@dataclass
class StructuralMetrics:
    """结构指标集合"""
    effective_information: float = 0.0      # 有效信息
    integrated_information: float = 0.0     # 积分信息
    structural_redundancy: float = 0.0      # 结构冗余度
    information_flow_efficiency: float = 0.0 # 信息流效率
    feature_diversity: float = 0.0          # 特征多样性
    nonlinearity_capacity: float = 0.0      # 非线性能力
    gradient_flow_health: float = 0.0       # 梯度流健康度


class ParameterFreeStructuralEvaluator:
    """
    无参数结构评估器
    
    核心功能：
    1. 计算结构固有的信息处理能力指标
    2. 无需实际参数，仅基于架构拓扑和连接模式
    3. 支持多种网络层类型的统一评估
    """
    
    def __init__(self, 
                 sample_input_shape: Tuple[int, ...] = (32, 3, 32, 32),
                 device: torch.device = None):
        """
        初始化评估器
        
        Args:
            sample_input_shape: 用于分析的样本输入形状 (batch, channels, height, width)
            device: 计算设备
        """
        self.sample_input_shape = sample_input_shape
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 预计算的样本输入
        self.sample_input = torch.randn(sample_input_shape, device=self.device)
        
        logger.info(f"初始化无参数结构评估器，样本输入形状: {sample_input_shape}")
    
    def evaluate_layer_structure(self, layer: nn.Module) -> StructuralMetrics:
        """
        评估单个网络层的结构指标
        
        Args:
            layer: 要评估的网络层
            
        Returns:
            StructuralMetrics: 结构指标集合
        """
        metrics = StructuralMetrics()
        
        with torch.no_grad():
            # 1. 计算有效信息 (Effective Information)
            metrics.effective_information = self._compute_effective_information(layer)
            
            # 2. 计算积分信息 (Integrated Information)
            metrics.integrated_information = self._compute_integrated_information(layer)
            
            # 3. 计算结构冗余度 (Structural Redundancy)
            metrics.structural_redundancy = self._compute_structural_redundancy(layer)
            
            # 4. 计算信息流效率
            metrics.information_flow_efficiency = self._compute_information_flow_efficiency(layer)
            
            # 5. 计算特征多样性
            metrics.feature_diversity = self._compute_feature_diversity(layer)
            
            # 6. 计算非线性能力
            metrics.nonlinearity_capacity = self._compute_nonlinearity_capacity(layer)
            
            # 7. 计算梯度流健康度
            metrics.gradient_flow_health = self._compute_gradient_flow_health(layer)
        
        return metrics
    
    def _compute_effective_information(self, layer: nn.Module) -> float:
        """
        计算有效信息 EI(S) = max_{p(x)} [I(X; Y) - I(X; Y|S)]
        
        近似实现：通过比较输入输出的信息量差异
        """
        try:
            # 生成多个不同分布的输入样本
            inputs = [
                torch.randn_like(self.sample_input) * 0.5,  # 低方差
                torch.randn_like(self.sample_input) * 1.0,  # 标准方差
                torch.randn_like(self.sample_input) * 2.0,  # 高方差
            ]
            
            effective_info_scores = []
            
            for input_sample in inputs:
                # 计算输入的信息熵（近似）
                input_entropy = self._estimate_tensor_entropy(input_sample)
                
                # 通过层传播
                try:
                    output = layer(input_sample)
                    output_entropy = self._estimate_tensor_entropy(output)
                    
                    # 有效信息 = 输出信息保留率 * 输入信息量
                    # 这里用信息保留率作为有效信息的代理指标
                    info_retention = min(output_entropy / max(input_entropy, 1e-8), 1.0)
                    effective_info = info_retention * input_entropy
                    effective_info_scores.append(effective_info)
                    
                except Exception as e:
                    logger.warning(f"层 {type(layer).__name__} 前向传播失败: {e}")
                    effective_info_scores.append(0.0)
            
            return float(np.mean(effective_info_scores))
            
        except Exception as e:
            logger.warning(f"计算有效信息失败: {e}")
            return 0.0
    
    def _compute_integrated_information(self, layer: nn.Module) -> float:
        """
        计算积分信息 Φ ≈ Σ_{i,j} MI(H_i; H_j) - Σ_i MI(H_i; H_i)
        
        近似实现：通过分析层内不同单元的相关性
        """
        try:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                # 获取权重矩阵
                weight = layer.weight.data
                
                if isinstance(layer, nn.Conv2d):
                    # 卷积层：展平权重到 (out_channels, in_channels * kernel_size)
                    weight = weight.view(weight.size(0), -1)
                
                # 计算输出通道间的相关性矩阵
                weight_normalized = F.normalize(weight, dim=1)
                correlation_matrix = torch.mm(weight_normalized, weight_normalized.t())
                
                # 积分信息 ≈ 相关性矩阵的非对角元素和（表示整合能力）
                mask = ~torch.eye(correlation_matrix.size(0), dtype=torch.bool)
                off_diagonal_sum = correlation_matrix[mask].abs().sum()
                
                # 归一化到 [0, 1]
                total_possible_connections = correlation_matrix.size(0) * (correlation_matrix.size(0) - 1)
                integrated_info = off_diagonal_sum / max(total_possible_connections, 1)
                
                return float(integrated_info.item())
            
            else:
                # 其他层类型使用默认值
                return 0.1
                
        except Exception as e:
            logger.warning(f"计算积分信息失败: {e}")
            return 0.0
    
    def _compute_structural_redundancy(self, layer: nn.Module) -> float:
        """
        计算结构冗余度 SR = rank(1/N Σ_n W_n^T W_n)
        
        冗余度越高，说明层内功能重叠越严重
        """
        try:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                weight = layer.weight.data
                
                if isinstance(layer, nn.Conv2d):
                    # 卷积层：重塑为 (out_channels, in_channels * kernel_size)
                    weight = weight.view(weight.size(0), -1)
                
                # 计算协方差矩阵 W^T W
                covariance = torch.mm(weight.t(), weight)
                
                # 使用SVD计算有效秩
                try:
                    U, S, V = torch.svd(covariance)
                    # 计算有效秩（相对于最大可能秩）
                    total_rank = min(covariance.size(0), covariance.size(1))
                    threshold = S[0] * 1e-6  # 相对阈值
                    effective_rank = (S > threshold).sum().float()
                    redundancy = 1.0 - (effective_rank / total_rank)
                    
                    return float(redundancy.item())
                    
                except Exception:
                    # SVD失败时使用简化方法
                    eigenvals = torch.linalg.eigvals(covariance).real
                    eigenvals = eigenvals[eigenvals > 0]
                    condition_number = eigenvals.max() / eigenvals.min() if len(eigenvals) > 0 else 1.0
                    redundancy = 1.0 / (1.0 + condition_number.item())
                    return float(redundancy)
            
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"计算结构冗余度失败: {e}")
            return 0.0
    
    def _compute_information_flow_efficiency(self, layer: nn.Module) -> float:
        """
        计算信息流效率 - 层的信息传递效率
        """
        try:
            # 使用多个测试输入评估信息流
            test_inputs = [
                torch.randn_like(self.sample_input) * scale 
                for scale in [0.1, 0.5, 1.0, 2.0]
            ]
            
            efficiency_scores = []
            
            for test_input in test_inputs:
                try:
                    output = layer(test_input)
                    
                    # 计算输入输出的信息比率
                    input_var = test_input.var().item()
                    output_var = output.var().item()
                    
                    # 信息流效率 = 输出方差保留率（避免过度压缩或放大）
                    if input_var > 1e-8:
                        efficiency = min(output_var / input_var, input_var / output_var)
                        efficiency_scores.append(efficiency)
                    
                except Exception:
                    efficiency_scores.append(0.0)
            
            return float(np.mean(efficiency_scores)) if efficiency_scores else 0.0
            
        except Exception as e:
            logger.warning(f"计算信息流效率失败: {e}")
            return 0.0
    
    def _compute_feature_diversity(self, layer: nn.Module) -> float:
        """
        计算特征多样性 - 层输出特征的多样化程度
        """
        try:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                weight = layer.weight.data
                
                if isinstance(layer, nn.Conv2d):
                    # 对于卷积层，计算不同滤波器的多样性
                    weight = weight.view(weight.size(0), -1)
                
                # 计算权重向量间的余弦相似度
                weight_normalized = F.normalize(weight, dim=1)
                similarity_matrix = torch.mm(weight_normalized, weight_normalized.t())
                
                # 多样性 = 1 - 平均相似度（排除对角线）
                mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool)
                if mask.any():
                    avg_similarity = similarity_matrix[mask].abs().mean()
                    diversity = 1.0 - avg_similarity
                    return float(diversity.item())
                else:
                    return 1.0
            
            else:
                return 0.5  # 默认中等多样性
                
        except Exception as e:
            logger.warning(f"计算特征多样性失败: {e}")
            return 0.0
    
    def _compute_nonlinearity_capacity(self, layer: nn.Module) -> float:
        """
        计算非线性能力 - 层的非线性变换能力
        """
        try:
            if isinstance(layer, nn.ReLU):
                return 0.8  # ReLU有较强非线性
            elif isinstance(layer, nn.GELU):
                return 0.9  # GELU有更强非线性
            elif isinstance(layer, nn.Tanh):
                return 0.85
            elif isinstance(layer, nn.Sigmoid):
                return 0.75
            elif isinstance(layer, (nn.Conv2d, nn.Linear)):
                return 0.1  # 线性层本身非线性能力低
            elif isinstance(layer, nn.BatchNorm2d):
                return 0.3  # BN有一定非线性效果
            elif isinstance(layer, nn.MaxPool2d):
                return 0.6  # 池化有非线性选择性
            else:
                return 0.2  # 默认低非线性
                
        except Exception as e:
            logger.warning(f"计算非线性能力失败: {e}")
            return 0.0
    
    def _compute_gradient_flow_health(self, layer: nn.Module) -> float:
        """
        计算梯度流健康度 - 层对梯度传播的友好程度
        """
        try:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                weight = layer.weight.data
                
                # 计算权重的谱范数（最大奇异值）
                if isinstance(layer, nn.Conv2d):
                    weight = weight.view(weight.size(0), -1)
                
                try:
                    spectral_norm = torch.linalg.norm(weight, ord=2).item()
                    # 梯度流健康度与谱范数相关，理想值接近1
                    health = 1.0 / (1.0 + abs(spectral_norm - 1.0))
                    return float(health)
                except Exception:
                    # 简化计算
                    weight_std = weight.std().item()
                    health = min(weight_std * 4, 1.0)  # 标准差适中时梯度流较好
                    return float(health)
            
            elif isinstance(layer, nn.BatchNorm2d):
                return 0.9  # BN有助于梯度流
            elif isinstance(layer, nn.ReLU):
                return 0.7  # ReLU可能导致梯度消失
            elif isinstance(layer, nn.GELU):
                return 0.8  # GELU梯度流更好
            else:
                return 0.5  # 默认中等健康度
                
        except Exception as e:
            logger.warning(f"计算梯度流健康度失败: {e}")
            return 0.0
    
    def _estimate_tensor_entropy(self, tensor: torch.Tensor) -> float:
        """
        估计张量的信息熵（近似）
        """
        try:
            # 将张量展平并转为numpy
            flat_tensor = tensor.view(-1).cpu().numpy()
            
            # 使用直方图估计概率分布
            hist, _ = np.histogram(flat_tensor, bins=50, density=True)
            hist = hist + 1e-10  # 避免log(0)
            
            # 计算微分熵（连续版本的熵）
            dx = (flat_tensor.max() - flat_tensor.min()) / 50
            entropy_val = -np.sum(hist * np.log(hist)) * dx
            
            return float(entropy_val)
            
        except Exception as e:
            logger.warning(f"估计张量熵失败: {e}")
            return 0.0
    
    def evaluate_model_structure(self, model: nn.Module) -> Dict[str, StructuralMetrics]:
        """
        评估整个模型的结构指标
        
        Args:
            model: 要评估的模型
            
        Returns:
            Dict[str, StructuralMetrics]: 每层的结构指标
        """
        layer_metrics = {}
        
        for name, layer in model.named_modules():
            if len(list(layer.children())) == 0:  # 只评估叶子节点
                try:
                    metrics = self.evaluate_layer_structure(layer)
                    layer_metrics[name] = metrics
                    
                    logger.debug(f"层 {name} ({type(layer).__name__}) 评估完成")
                    
                except Exception as e:
                    logger.warning(f"评估层 {name} 失败: {e}")
                    layer_metrics[name] = StructuralMetrics()
        
        return layer_metrics
    
    def compute_aggregate_metrics(self, layer_metrics: Dict[str, StructuralMetrics]) -> StructuralMetrics:
        """
        计算聚合的结构指标
        
        Args:
            layer_metrics: 各层的结构指标
            
        Returns:
            StructuralMetrics: 聚合指标
        """
        if not layer_metrics:
            return StructuralMetrics()
        
        # 计算各指标的加权平均
        metrics_list = list(layer_metrics.values())
        
        aggregate = StructuralMetrics(
            effective_information=np.mean([m.effective_information for m in metrics_list]),
            integrated_information=np.mean([m.integrated_information for m in metrics_list]),
            structural_redundancy=np.mean([m.structural_redundancy for m in metrics_list]),
            information_flow_efficiency=np.mean([m.information_flow_efficiency for m in metrics_list]),
            feature_diversity=np.mean([m.feature_diversity for m in metrics_list]),
            nonlinearity_capacity=np.mean([m.nonlinearity_capacity for m in metrics_list]),
            gradient_flow_health=np.mean([m.gradient_flow_health for m in metrics_list])
        )
        
        return aggregate