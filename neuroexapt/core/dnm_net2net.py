#!/usr/bin/env python3
"""
DNM Net2Net 参数平滑迁移模块

基于Net2Net论文思想，实现神经网络架构变异时的参数平滑迁移：
1. 网络加宽(Net2WiderNet): 增加神经元数量
2. 网络加深(Net2DeeperNet): 增加网络层数
3. 分支分裂: 一层分裂成多个并行分支
4. 操作变异: 卷积层变异成不同操作组合

确保架构变异时保持函数等价性和训练稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class Net2NetTransformer:
    """Net2Net变换器 - 实现平滑的架构变异"""
    
    def __init__(self, noise_scale: float = 1e-5):
        self.noise_scale = noise_scale
        
    def wider_conv2d(self, layer: nn.Conv2d, new_out_channels: int, 
                     next_layer: Optional[nn.Module] = None) -> Tuple[nn.Conv2d, Optional[nn.Module]]:
        """
        Net2WiderNet: 扩展卷积层的输出通道数
        
        Args:
            layer: 原始卷积层
            new_out_channels: 新的输出通道数
            next_layer: 下一层(用于权重调整)
            
        Returns:
            (新卷积层, 调整后的下一层)
        """
        assert new_out_channels > layer.out_channels, "新通道数必须大于原通道数"
        
        # 创建新的卷积层
        new_layer = nn.Conv2d(
            in_channels=layer.in_channels,
            out_channels=new_out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode
        )
        
        # 初始化权重
        with torch.no_grad():
            # 复制原有权重
            new_layer.weight[:layer.out_channels] = layer.weight.data
            if layer.bias is not None:
                new_layer.bias[:layer.out_channels] = layer.bias.data
            
            # 为新增通道随机选择复制源
            additional_channels = new_out_channels - layer.out_channels
            for i in range(additional_channels):
                # 随机选择一个原有通道进行复制
                source_idx = np.random.randint(0, layer.out_channels)
                target_idx = layer.out_channels + i
                
                # 复制权重并添加小扰动
                new_layer.weight[target_idx] = layer.weight[source_idx].clone()
                new_layer.weight[target_idx] += torch.randn_like(new_layer.weight[target_idx]) * self.noise_scale
                
                if layer.bias is not None:
                    new_layer.bias[target_idx] = layer.bias[source_idx].clone()
                    new_layer.bias[target_idx] += torch.randn_like(new_layer.bias[target_idx]) * self.noise_scale
        
        # 调整下一层
        new_next_layer = None
        if next_layer is not None:
            new_next_layer = self._adjust_next_layer_for_wider(next_layer, layer.out_channels, new_out_channels)
        
        logger.info(f"Net2WiderNet: {layer.out_channels} -> {new_out_channels} channels")
        return new_layer, new_next_layer
    
    def wider_linear(self, layer: nn.Linear, new_out_features: int, 
                     next_layer: Optional[nn.Module] = None) -> Tuple[nn.Linear, Optional[nn.Module]]:
        """
        Net2WiderNet: 扩展线性层的输出特征数
        """
        assert new_out_features > layer.out_features, "新特征数必须大于原特征数"
        
        # 创建新的线性层
        new_layer = nn.Linear(
            in_features=layer.in_features,
            out_features=new_out_features,
            bias=layer.bias is not None
        )
        
        # 初始化权重
        with torch.no_grad():
            # 复制原有权重
            new_layer.weight[:layer.out_features] = layer.weight.data
            if layer.bias is not None:
                new_layer.bias[:layer.out_features] = layer.bias.data
            
            # 为新增特征复制权重
            additional_features = new_out_features - layer.out_features
            for i in range(additional_features):
                source_idx = np.random.randint(0, layer.out_features)
                target_idx = layer.out_features + i
                
                new_layer.weight[target_idx] = layer.weight[source_idx].clone()
                new_layer.weight[target_idx] += torch.randn_like(new_layer.weight[target_idx]) * self.noise_scale
                
                if layer.bias is not None:
                    new_layer.bias[target_idx] = layer.bias[source_idx].clone()
        
        # 调整下一层
        new_next_layer = None
        if next_layer is not None:
            new_next_layer = self._adjust_next_layer_for_wider(next_layer, layer.out_features, new_out_features)
        
        logger.info(f"Net2WiderNet: {layer.out_features} -> {new_out_features} features")
        return new_layer, new_next_layer
    
    def deeper_conv2d(self, layer: nn.Conv2d, position: str = 'after') -> nn.Module:
        """
        Net2DeeperNet: 在卷积层后插入新层
        
        Args:
            layer: 原始卷积层
            position: 'before' 或 'after'
            
        Returns:
            新插入的层（恒等变换）
        """
        if position == 'after':
            # 在后面插入1x1恒等卷积
            new_layer = nn.Conv2d(
                in_channels=layer.out_channels,
                out_channels=layer.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
            
            # 初始化为恒等变换
            with torch.no_grad():
                nn.init.eye_(new_layer.weight.squeeze())
                if new_layer.bias is not None:
                    nn.init.zeros_(new_layer.bias)
                    
        else:  # before
            # 在前面插入恒等卷积
            new_layer = nn.Conv2d(
                in_channels=layer.in_channels,
                out_channels=layer.in_channels,
                kernel_size=layer.kernel_size,
                stride=1,
                padding=layer.padding,
                bias=True
            )
            
            # 初始化为恒等变换
            with torch.no_grad():
                # 对于3x3卷积，中心设为1，其余为0
                nn.init.zeros_(new_layer.weight)
                if layer.kernel_size == (3, 3):
                    for i in range(layer.in_channels):
                        new_layer.weight[i, i, 1, 1] = 1.0
                elif layer.kernel_size == (1, 1):
                    nn.init.eye_(new_layer.weight.squeeze())
                
                if new_layer.bias is not None:
                    nn.init.zeros_(new_layer.bias)
        
        logger.info(f"Net2DeeperNet: Added identity layer {position} conv layer")
        return new_layer
    
    def deeper_linear(self, layer: nn.Linear, position: str = 'after') -> nn.Module:
        """
        Net2DeeperNet: 在线性层附近插入新层
        """
        if position == 'after':
            new_layer = nn.Linear(layer.out_features, layer.out_features, bias=True)
        else:
            new_layer = nn.Linear(layer.in_features, layer.in_features, bias=True)
        
        # 初始化为恒等变换
        with torch.no_grad():
            nn.init.eye_(new_layer.weight)
            nn.init.zeros_(new_layer.bias)
        
        logger.info(f"Net2DeeperNet: Added identity linear layer {position} existing layer")
        return new_layer
    
    def _adjust_next_layer_for_wider(self, next_layer: nn.Module, old_channels: int, new_channels: int) -> nn.Module:
        """调整下一层以适应扩宽的前一层"""
        
        if isinstance(next_layer, nn.Conv2d):
            # 调整卷积层的输入通道
            new_next = nn.Conv2d(
                in_channels=new_channels,
                out_channels=next_layer.out_channels,
                kernel_size=next_layer.kernel_size,
                stride=next_layer.stride,
                padding=next_layer.padding,
                dilation=next_layer.dilation,
                groups=next_layer.groups,
                bias=next_layer.bias is not None,
                padding_mode=next_layer.padding_mode
            )
            
            with torch.no_grad():
                # 复制原有权重
                new_next.weight[:, :old_channels] = next_layer.weight.data
                
                # 新增通道的权重设为0（保持函数等价性）
                new_next.weight[:, old_channels:] = 0
                
                if next_layer.bias is not None:
                    new_next.bias[:] = next_layer.bias.data
            
            return new_next
            
        elif isinstance(next_layer, nn.Linear):
            # 调整线性层的输入特征
            new_next = nn.Linear(
                in_features=new_channels,
                out_features=next_layer.out_features,
                bias=next_layer.bias is not None
            )
            
            with torch.no_grad():
                # 复制原有权重
                new_next.weight[:, :old_channels] = next_layer.weight.data
                
                # 新增特征的权重设为0
                new_next.weight[:, old_channels:] = 0
                
                if next_layer.bias is not None:
                    new_next.bias[:] = next_layer.bias.data
            
            return new_next
            
        elif isinstance(next_layer, nn.BatchNorm2d):
            # 调整BatchNorm层
            new_next = nn.BatchNorm2d(new_channels)
            
            with torch.no_grad():
                # 复制原有参数
                new_next.weight[:old_channels] = next_layer.weight.data
                new_next.bias[:old_channels] = next_layer.bias.data
                new_next.running_mean[:old_channels] = next_layer.running_mean.data
                new_next.running_var[:old_channels] = next_layer.running_var.data
                
                # 新增通道初始化
                new_next.weight[old_channels:] = 1.0
                new_next.bias[old_channels:] = 0.0
                new_next.running_mean[old_channels:] = 0.0
                new_next.running_var[old_channels:] = 1.0
                
                new_next.num_batches_tracked = next_layer.num_batches_tracked
            
            return new_next
        
        return next_layer


class DNMArchitectureMutator:
    """DNM架构变异器 - 智能的架构变异策略"""
    
    def __init__(self, transformer: Net2NetTransformer):
        self.transformer = transformer
        
    def split_conv_layer(self, layer: nn.Conv2d, split_type: str = 'parallel') -> nn.Module:
        """
        分裂卷积层
        
        Args:
            layer: 原始卷积层
            split_type: 'parallel' (并行分支) 或 'sequential' (串行分层)
            
        Returns:
            分裂后的模块
        """
        if split_type == 'parallel':
            return self._split_conv_parallel(layer)
        elif split_type == 'sequential':
            return self._split_conv_sequential(layer)
        else:
            raise ValueError(f"Unsupported split_type: {split_type}")
    
    def _split_conv_parallel(self, layer: nn.Conv2d) -> nn.Module:
        """将卷积层分裂成两个并行分支"""
        
        # 计算每个分支的通道数
        branch1_channels = layer.out_channels // 2
        branch2_channels = layer.out_channels - branch1_channels
        
        # 创建两个分支
        branch1 = nn.Conv2d(
            layer.in_channels, branch1_channels,
            layer.kernel_size, layer.stride, layer.padding,
            layer.dilation, layer.groups, layer.bias is not None
        )
        
        branch2 = nn.Conv2d(
            layer.in_channels, branch2_channels,
            layer.kernel_size, layer.stride, layer.padding,
            layer.dilation, layer.groups, layer.bias is not None
        )
        
        # 分配权重
        with torch.no_grad():
            branch1.weight.data = layer.weight[:branch1_channels].clone()
            branch2.weight.data = layer.weight[branch1_channels:].clone()
            
            if layer.bias is not None:
                branch1.bias.data = layer.bias[:branch1_channels].clone()
                branch2.bias.data = layer.bias[branch1_channels:].clone()
        
        # 创建并行模块
        class ParallelBranches(nn.Module):
            def __init__(self, branch1, branch2):
                super().__init__()
                self.branch1 = branch1
                self.branch2 = branch2
            
            def forward(self, x):
                out1 = self.branch1(x)
                out2 = self.branch2(x)
                return torch.cat([out1, out2], dim=1)
        
        logger.info(f"Split conv layer into parallel branches: {branch1_channels} + {branch2_channels}")
        return ParallelBranches(branch1, branch2)
    
    def _split_conv_sequential(self, layer: nn.Conv2d) -> nn.Module:
        """将卷积层分裂成两个串行层"""
        
        # 计算中间通道数
        intermediate_channels = max(layer.out_channels, layer.in_channels)
        
        # 第一层：输入 -> 中间通道
        layer1 = nn.Conv2d(
            layer.in_channels, intermediate_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        
        # 第二层：中间通道 -> 输出
        layer2 = nn.Conv2d(
            intermediate_channels, layer.out_channels,
            layer.kernel_size, layer.stride, layer.padding,
            layer.dilation, bias=layer.bias is not None
        )
        
        # BatchNorm层
        bn = nn.BatchNorm2d(intermediate_channels)
        
        # 权重初始化
        with torch.no_grad():
            # 第一层使用随机初始化
            nn.init.kaiming_normal_(layer1.weight, mode='fan_out', nonlinearity='relu')
            
            # 第二层尽量保持原有功能
            if intermediate_channels == layer.out_channels:
                # 如果通道数匹配，可以直接复制权重
                layer2.weight.data = layer.weight.data.clone()
                if layer.bias is not None:
                    layer2.bias.data = layer.bias.data.clone()
            else:
                # 否则使用He初始化
                nn.init.kaiming_normal_(layer2.weight, mode='fan_out', nonlinearity='relu')
                if layer2.bias is not None:
                    nn.init.zeros_(layer2.bias)
        
        # 创建串行模块
        sequential_module = nn.Sequential(layer1, bn, nn.ReLU(inplace=True), layer2)
        
        logger.info(f"Split conv layer into sequential: {layer.in_channels} -> {intermediate_channels} -> {layer.out_channels}")
        return sequential_module
    
    def mutate_conv_to_depthwise_separable(self, layer: nn.Conv2d) -> nn.Module:
        """将标准卷积变异为深度可分离卷积"""
        
        if layer.groups != 1:
            logger.warning("Layer is already grouped, skipping depthwise separable mutation")
            return layer
        
        # 深度卷积
        depthwise = nn.Conv2d(
            layer.in_channels, layer.in_channels,
            layer.kernel_size, layer.stride, layer.padding,
            groups=layer.in_channels, bias=False
        )
        
        # 逐点卷积
        pointwise = nn.Conv2d(
            layer.in_channels, layer.out_channels,
            kernel_size=1, stride=1, padding=0,
            bias=layer.bias is not None
        )
        
        # 权重初始化
        with torch.no_grad():
            # 深度卷积：每个输入通道对应一个滤波器
            for i in range(layer.in_channels):
                if i < layer.out_channels:
                    # 从原层复制对应的权重
                    depthwise.weight[i, 0] = layer.weight[i % layer.out_channels, i]
                else:
                    # 随机初始化
                    nn.init.kaiming_normal_(depthwise.weight[i:i+1], mode='fan_out', nonlinearity='relu')
            
            # 逐点卷积初始化
            nn.init.kaiming_normal_(pointwise.weight, mode='fan_out', nonlinearity='relu')
            if pointwise.bias is not None and layer.bias is not None:
                pointwise.bias.data = layer.bias.data.clone()
        
        # 组合模块
        module = nn.Sequential(depthwise, pointwise)
        
        logger.info(f"Mutated conv to depthwise separable: {layer.in_channels}x{layer.out_channels}")
        return module
    
    def add_residual_connection(self, layer: nn.Module, input_channels: int, output_channels: int) -> nn.Module:
        """为层添加残差连接"""
        
        # 创建跳跃连接（如果通道数不匹配，使用1x1卷积调整）
        if input_channels == output_channels:
            shortcut = nn.Identity()
        else:
            shortcut = nn.Conv2d(input_channels, output_channels, 1, bias=False)
            with torch.no_grad():
                nn.init.kaiming_normal_(shortcut.weight, mode='fan_out', nonlinearity='relu')
        
        # 创建残差模块
        class ResidualBlock(nn.Module):
            def __init__(self, main_layer, shortcut):
                super().__init__()
                self.main_layer = main_layer
                self.shortcut = shortcut
            
            def forward(self, x):
                identity = self.shortcut(x)
                out = self.main_layer(x)
                
                # 确保尺寸匹配
                if out.shape != identity.shape:
                    # 使用自适应池化调整空间尺寸
                    identity = F.adaptive_avg_pool2d(identity, out.shape[2:])
                
                return out + identity
        
        logger.info(f"Added residual connection: {input_channels} -> {output_channels}")
        return ResidualBlock(layer, shortcut)


# 使用示例和测试函数
def test_net2net_transforms():
    """测试Net2Net变换"""
    print("🧪 Testing Net2Net Transforms")
    
    transformer = Net2NetTransformer()
    mutator = DNMArchitectureMutator(transformer)
    
    # 测试卷积层扩宽
    conv = nn.Conv2d(32, 64, 3, padding=1)
    new_conv, _ = transformer.wider_conv2d(conv, 96)
    print(f"✅ Conv wider: {conv.out_channels} -> {new_conv.out_channels}")
    
    # 测试卷积层加深
    deeper_layer = transformer.deeper_conv2d(conv, 'after')
    print(f"✅ Conv deeper: Added layer after conv")
    
    # 测试并行分裂
    parallel_split = mutator.split_conv_layer(conv, 'parallel')
    print(f"✅ Parallel split: Created branched structure")
    
    # 测试串行分裂
    sequential_split = mutator.split_conv_layer(conv, 'sequential')
    print(f"✅ Sequential split: Created layered structure")
    
    # 测试深度可分离卷积变异
    depthwise_sep = mutator.mutate_conv_to_depthwise_separable(conv)
    print(f"✅ Depthwise separable: Converted standard conv")
    
    print("🎉 Net2Net transforms test completed!")


if __name__ == "__main__":
    test_net2net_transforms()