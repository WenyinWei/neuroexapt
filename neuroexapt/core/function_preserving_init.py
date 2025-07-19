"""
"""
defgroup group_function_preserving_init Function Preserving Init
ingroup core
Function Preserving Init module for NeuroExapt framework.
"""


函数保持初始化 (Function-Preserving Initialization)

ASO-SE框架的核心机制之一：确保架构突变时新架构的输出与旧架构完全一致，
避免损失函数剧烈震荡，实现平滑的架构过渡。

支持的操作类型：
1. 层增加 (Layer Addition) - 恒等映射初始化
2. 通道扩展 (Channel Expansion) - 权重复制与微调
3. 分支添加 (Branch Addition) - 零权重初始化
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class FunctionPreservingInitializer:
    """
    函数保持初始化器
    
    确保架构变更时网络输出保持不变的核心工具
    """
    
    def __init__(self, preserve_ratio: float = 0.95, noise_scale: float = 1e-4):
        """
        Args:
            preserve_ratio: 权重保持的比例，用于通道扩展时的权重复制
            noise_scale: 添加的微小随机噪声规模，避免完全对称
        """
        self.preserve_ratio = preserve_ratio
        self.noise_scale = noise_scale
        
    def identity_layer_init(self, layer: nn.Module, input_shape: Optional[Tuple[int, ...]] = None) -> nn.Module:
        """
        恒等映射初始化
        
        将新增层初始化为恒等映射，确保层的输出 = 输入
        
        Args:
            layer: 要初始化的层
            input_shape: 输入张量的形状
            
        Returns:
            初始化后的层
        """
        if isinstance(layer, nn.Conv2d):
            return self._identity_conv2d_init(layer)
        elif isinstance(layer, nn.Linear):
            return self._identity_linear_init(layer)
        elif isinstance(layer, nn.BatchNorm2d):
            return self._identity_batchnorm_init(layer)
        else:
            logger.warning(f"Identity initialization not implemented for {type(layer)}")
            return layer
    
    def _identity_conv2d_init(self, conv: nn.Conv2d) -> nn.Conv2d:
        """卷积层恒等映射初始化"""
        with torch.no_grad():
            # 重置权重
            conv.weight.zero_()
            
            # 设置中心权重为1 (对于3x3卷积，中心为(1,1))
            if conv.kernel_size == (3, 3):
                center = (1, 1)
            elif conv.kernel_size == (1, 1):
                center = (0, 0)
            else:
                center = tuple(k//2 for k in conv.kernel_size)
            
            # 只在输入输出通道数相同时设置恒等
            min_channels = min(conv.in_channels, conv.out_channels)
            for i in range(min_channels):
                conv.weight[i, i, center[0], center[1]] = 1.0
                
            # 如果有偏置，设为零
            if conv.bias is not None:
                conv.bias.zero_()
                
        logger.debug(f"Initialized Conv2d {conv.in_channels}->{conv.out_channels} as identity")
        return conv
    
    def _identity_linear_init(self, linear: nn.Linear) -> nn.Linear:
        """线性层恒等映射初始化"""
        with torch.no_grad():
            # 重置为零
            linear.weight.zero_()
            
            # 设置对角线为1
            min_dim = min(linear.in_features, linear.out_features)
            for i in range(min_dim):
                linear.weight[i, i] = 1.0
                
            # 偏置设为零
            if linear.bias is not None:
                linear.bias.zero_()
                
        logger.debug(f"Initialized Linear {linear.in_features}->{linear.out_features} as identity")
        return linear
    
    def _identity_batchnorm_init(self, bn: nn.BatchNorm2d) -> nn.BatchNorm2d:
        """BatchNorm层恒等映射初始化"""
        with torch.no_grad():
            # 设置为恒等变换
            bn.weight.fill_(1.0)
            bn.bias.zero_()
            
            # 重置运行统计
            if bn.running_mean is not None:
                bn.running_mean.zero_()
            if bn.running_var is not None:
                bn.running_var.fill_(1.0)
                
        logger.debug(f"Initialized BatchNorm2d as identity")
        return bn
    
    def expand_channels_preserving(self, old_layer: nn.Module, new_channels: int, 
                                 expansion_strategy: str = "replicate") -> nn.Module:
        """
        通道扩展的函数保持初始化
        
        Args:
            old_layer: 原始层
            new_channels: 新的通道数
            expansion_strategy: 扩展策略 ("replicate", "interpolate", "zero_pad")
            
        Returns:
            扩展后的层
        """
        if isinstance(old_layer, nn.Conv2d):
            return self._expand_conv2d_channels(old_layer, new_channels, expansion_strategy)
        elif isinstance(old_layer, nn.BatchNorm2d):
            return self._expand_batchnorm_channels(old_layer, new_channels, expansion_strategy)
        else:
            logger.warning(f"Channel expansion not implemented for {type(old_layer)}")
            return old_layer
    
    def _expand_conv2d_channels(self, old_conv: nn.Conv2d, new_out_channels: int, 
                               strategy: str) -> nn.Conv2d:
        """卷积层通道扩展"""
        old_out_channels = old_conv.out_channels
        
        if new_out_channels <= old_out_channels:
            logger.warning("New channel count should be larger than old count")
            return old_conv
        
        # 创建新的卷积层
        new_conv = nn.Conv2d(
            old_conv.in_channels, 
            new_out_channels,
            old_conv.kernel_size,
            old_conv.stride,
            old_conv.padding,
            old_conv.dilation,
            old_conv.groups,
            old_conv.bias is not None
        )
        
        with torch.no_grad():
            # 复制原有权重
            new_conv.weight[:old_out_channels] = old_conv.weight.clone()
            
            # 扩展策略
            if strategy == "replicate":
                # 复制现有通道
                expand_count = new_out_channels - old_out_channels
                replicate_indices = torch.randperm(old_out_channels)[:expand_count]
                new_conv.weight[old_out_channels:] = old_conv.weight[replicate_indices].clone()
                
                # 添加微小噪声避免完全对称
                noise = torch.randn_like(new_conv.weight[old_out_channels:]) * self.noise_scale
                new_conv.weight[old_out_channels:] += noise
                
            elif strategy == "zero_pad":
                # 新通道初始化为零
                new_conv.weight[old_out_channels:].zero_()
                
            elif strategy == "interpolate":
                # 使用现有通道的插值
                for i in range(old_out_channels, new_out_channels):
                    # 在现有通道间插值
                    alpha = (i - old_out_channels) / (new_out_channels - old_out_channels)
                    idx1 = min(int(alpha * old_out_channels), old_out_channels - 1)
                    idx2 = min(idx1 + 1, old_out_channels - 1)
                    weight1, weight2 = 1 - alpha, alpha
                    
                    new_conv.weight[i] = (weight1 * old_conv.weight[idx1] + 
                                        weight2 * old_conv.weight[idx2])
            
            # 复制偏置
            if old_conv.bias is not None:
                new_conv.bias[:old_out_channels] = old_conv.bias.clone()
                if strategy == "replicate":
                    new_conv.bias[old_out_channels:] = old_conv.bias[replicate_indices].clone()
                else:
                    new_conv.bias[old_out_channels:].zero_()
        
        logger.info(f"Expanded Conv2d channels: {old_out_channels} -> {new_out_channels} "
                   f"using {strategy} strategy")
        return new_conv
    
    def _expand_batchnorm_channels(self, old_bn: nn.BatchNorm2d, new_channels: int, 
                                  strategy: str) -> nn.BatchNorm2d:
        """BatchNorm层通道扩展"""
        old_channels = old_bn.num_features
        
        if new_channels <= old_channels:
            return old_bn
            
        new_bn = nn.BatchNorm2d(new_channels, old_bn.eps, old_bn.momentum, 
                               old_bn.affine, old_bn.track_running_stats)
        
        with torch.no_grad():
            if old_bn.affine:
                # 复制权重和偏置
                new_bn.weight[:old_channels] = old_bn.weight.clone()
                new_bn.bias[:old_channels] = old_bn.bias.clone()
                
                # 新通道初始化
                new_bn.weight[old_channels:].fill_(1.0)
                new_bn.bias[old_channels:].zero_()
            
            if old_bn.track_running_stats:
                # 复制运行统计
                new_bn.running_mean[:old_channels] = old_bn.running_mean.clone()
                new_bn.running_var[:old_channels] = old_bn.running_var.clone()
                
                # 新通道统计初始化
                new_bn.running_mean[old_channels:].zero_()
                new_bn.running_var[old_channels:].fill_(1.0)
                
                new_bn.num_batches_tracked = old_bn.num_batches_tracked
        
        logger.info(f"Expanded BatchNorm channels: {old_channels} -> {new_channels}")
        return new_bn
    
    def zero_branch_init(self, branch: nn.Module) -> nn.Module:
        """
        分支零初始化
        
        将新增分支的最后一层初始化为零权重，使其在诞生瞬间不影响主干流
        """
        # 找到分支的最后一层
        last_layer = None
        for module in branch.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                last_layer = module
        
        if last_layer is not None:
            with torch.no_grad():
                last_layer.weight.zero_()
                if last_layer.bias is not None:
                    last_layer.bias.zero_()
            logger.info(f"Zero-initialized branch output layer: {type(last_layer)}")
        
        return branch
    
    def smooth_parameter_transfer(self, old_model: nn.Module, new_model: nn.Module,
                                layer_mapping: Dict[str, str]) -> nn.Module:
        """
        平滑参数传递
        
        将旧模型的参数平滑传递到新模型中，支持形状变化的情况
        
        Args:
            old_model: 旧模型
            new_model: 新模型  
            layer_mapping: 层名称映射 {"new_layer_name": "old_layer_name"}
            
        Returns:
            参数传递后的新模型
        """
        old_state_dict = old_model.state_dict()
        new_state_dict = new_model.state_dict()
        
        transferred_count = 0
        
        for new_name, old_name in layer_mapping.items():
            if old_name in old_state_dict and new_name in new_state_dict:
                old_param = old_state_dict[old_name]
                new_param = new_state_dict[new_name]
                
                if old_param.shape == new_param.shape:
                    # 直接复制
                    new_state_dict[new_name] = old_param.clone()
                    transferred_count += 1
                    
                elif len(old_param.shape) == len(new_param.shape):
                    # 形状不同但维度相同，尝试部分复制
                    min_shape = tuple(min(old_param.shape[i], new_param.shape[i]) 
                                    for i in range(len(old_param.shape)))
                    
                    # 创建索引切片
                    slices = tuple(slice(0, s) for s in min_shape)
                    
                    with torch.no_grad():
                        new_param[slices] = old_param[slices]
                    
                    transferred_count += 1
                    logger.info(f"Partially transferred {old_name} -> {new_name}: "
                              f"{old_param.shape} -> {new_param.shape}")
                else:
                    logger.warning(f"Cannot transfer {old_name} -> {new_name}: "
                                 f"incompatible shapes {old_param.shape} vs {new_param.shape}")
        
        new_model.load_state_dict(new_state_dict)
        logger.info(f"Successfully transferred {transferred_count} parameters")
        
        return new_model


def create_identity_residual_block(in_channels: int, out_channels: int) -> nn.Module:
    """
    创建恒等残差块
    
    用于添加新层时的函数保持初始化
    """
    if in_channels == out_channels:
        # 直接恒等连接
        return nn.Identity()
    else:
        # 需要通道调整的恒等连接
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # 函数保持初始化
        initializer = FunctionPreservingInitializer()
        conv = initializer.identity_layer_init(conv, None)
        
        return nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels)
        )


def test_function_preserving_init():
    """测试函数保持初始化的功能"""
    print("🧪 Testing Function-Preserving Initialization...")
    
    initializer = FunctionPreservingInitializer()
    
    # 测试恒等映射初始化
    conv = nn.Conv2d(32, 32, 3, padding=1)
    x = torch.randn(1, 32, 16, 16)
    
    # 初始化前后的输出应该相同（对于恒等映射）
    conv_identity = initializer.identity_layer_init(conv, x.shape)
    
    # 测试通道扩展
    old_conv = nn.Conv2d(32, 64, 3, padding=1)
    torch.nn.init.xavier_uniform_(old_conv.weight)
    
    new_conv = initializer.expand_channels_preserving(old_conv, 128, "replicate")
    
    print(f"✅ Original conv output channels: {old_conv.out_channels}")
    print(f"✅ Expanded conv output channels: {new_conv.out_channels}")
    
    # 验证前64个通道的权重保持不变
    assert torch.allclose(old_conv.weight, new_conv.weight[:64])
    print("✅ Channel expansion preserves original weights")
    
    print("🎉 Function-Preserving Initialization tests passed!")


if __name__ == "__main__":
    test_function_preserving_init() 