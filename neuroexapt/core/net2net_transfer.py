#!/usr/bin/env python3
"""
defgroup group_net2net_transfer Net2Net Transfer
ingroup core
Net2Net Transfer module for NeuroExapt framework.
"""

ASO-SE Net2Net参数平滑迁移模块

基于Net2Net论文的Function-Preserving Transformations理念：
- Net2Wider: 宽度扩展时的参数复制和权重分配
- Net2Deeper: 深度扩展时的恒等映射初始化  
- Net2Branch: 分支扩展时的权重共享

核心原理：
1. 保持网络输出函数不变: f_student(x) = f_teacher(x)
2. 通过权重复制和缩放实现平滑迁移
3. 支持卷积层、线性层、批归一化层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class Net2NetTransfer:
    """Net2Net风格的平滑参数迁移工具"""
    
    def __init__(self):
        self.transfer_records = []  # 记录迁移历史
    
    def net2wider_conv(self, 
                      conv_layer: nn.Conv2d,
                      next_layer: Optional[nn.Module],
                      new_width: int,
                      noise_std: float = 1e-7) -> Tuple[nn.Conv2d, Optional[nn.Module]]:
        """
        卷积层宽度扩展 (增加输出通道数)
        
        Args:
            conv_layer: 要扩展的卷积层
            next_layer: 后续层(需要相应调整输入通道)
            new_width: 新的输出通道数
            noise_std: 添加的噪声标准差(用于打破对称性)
        
        Returns:
            (新的卷积层, 新的后续层)
        """
        old_width = conv_layer.out_channels
        if new_width <= old_width:
            raise ValueError(f"New width {new_width} must be greater than old width {old_width}")
        
        # 创建新的卷积层
        new_conv = nn.Conv2d(
            in_channels=conv_layer.in_channels,
            out_channels=new_width,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=conv_layer.bias is not None,
            padding_mode=conv_layer.padding_mode
        )
        
        # 随机映射函数 g: {1,...,new_width} -> {1,...,old_width}
        # 前old_width个直接对应，后面的随机选择
        mapping = list(range(old_width)) + [
            np.random.randint(0, old_width) for _ in range(new_width - old_width)
        ]
        
        # 计算每个旧通道被复制的次数
        replication_count = {}
        for i, src_idx in enumerate(mapping):
            replication_count[src_idx] = replication_count.get(src_idx, 0) + 1
        
        # 复制权重
        with torch.no_grad():
            for new_idx, old_idx in enumerate(mapping):
                new_conv.weight[new_idx] = conv_layer.weight[old_idx].clone()
                # 添加小噪声打破对称性
                if new_idx >= old_width:  # 只对复制的权重添加噪声
                    new_conv.weight[new_idx] += torch.randn_like(new_conv.weight[new_idx]) * noise_std
            
            if conv_layer.bias is not None:
                for new_idx, old_idx in enumerate(mapping):
                    new_conv.bias[new_idx] = conv_layer.bias[old_idx].clone()
                    if new_idx >= old_width:
                        new_conv.bias[new_idx] += torch.randn_like(new_conv.bias[new_idx]) * noise_std
        
        # 处理后续层
        new_next_layer = None
        if next_layer is not None:
            if isinstance(next_layer, nn.Conv2d):
                new_next_layer = self._wider_next_conv(next_layer, mapping, replication_count)
            elif isinstance(next_layer, nn.Linear):
                new_next_layer = self._wider_next_linear(next_layer, mapping, replication_count)
            elif isinstance(next_layer, nn.BatchNorm2d):
                new_next_layer = self._wider_next_batchnorm2d(next_layer, mapping, replication_count)
        
        return new_conv, new_next_layer
    
    def _wider_next_conv(self, conv_layer: nn.Conv2d, mapping: List[int], 
                        replication_count: Dict[int, int]) -> nn.Conv2d:
        """扩展后续卷积层的输入通道"""
        old_in_channels = conv_layer.in_channels
        new_in_channels = len(mapping)
        
        new_conv = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=conv_layer.bias is not None,
            padding_mode=conv_layer.padding_mode
        )
        
        with torch.no_grad():
            for new_idx, old_idx in enumerate(mapping):
                # 权重需要按复制次数缩放
                scale_factor = 1.0 / replication_count[old_idx]
                new_conv.weight[:, new_idx] = conv_layer.weight[:, old_idx] * scale_factor
            
            if conv_layer.bias is not None:
                new_conv.bias.copy_(conv_layer.bias)
        
        return new_conv
    
    def _wider_next_linear(self, linear_layer: nn.Linear, mapping: List[int],
                          replication_count: Dict[int, int]) -> nn.Linear:
        """扩展后续线性层的输入特征"""
        old_in_features = linear_layer.in_features
        new_in_features = len(mapping)
        
        new_linear = nn.Linear(new_in_features, linear_layer.out_features, 
                              bias=linear_layer.bias is not None)
        
        with torch.no_grad():
            for new_idx, old_idx in enumerate(mapping):
                scale_factor = 1.0 / replication_count[old_idx]
                new_linear.weight[:, new_idx] = linear_layer.weight[:, old_idx] * scale_factor
            
            if linear_layer.bias is not None:
                new_linear.bias.copy_(linear_layer.bias)
        
        return new_linear
    
    def _wider_next_batchnorm2d(self, bn_layer: nn.BatchNorm2d, mapping: List[int],
                               replication_count: Dict[int, int]) -> nn.BatchNorm2d:
        """扩展后续BatchNorm层"""
        new_num_features = len(mapping)
        new_bn = nn.BatchNorm2d(new_num_features, eps=bn_layer.eps, 
                               momentum=bn_layer.momentum, affine=bn_layer.affine)
        
        if bn_layer.affine:
            with torch.no_grad():
                for new_idx, old_idx in enumerate(mapping):
                    new_bn.weight[new_idx] = bn_layer.weight[old_idx]
                    new_bn.bias[new_idx] = bn_layer.bias[old_idx]
        
        # 复制运行时统计信息
        if bn_layer.running_mean is not None:
            new_bn.running_mean = torch.zeros(new_num_features)
            new_bn.running_var = torch.ones(new_num_features)
            for new_idx, old_idx in enumerate(mapping):
                new_bn.running_mean[new_idx] = bn_layer.running_mean[old_idx]
                new_bn.running_var[new_idx] = bn_layer.running_var[old_idx]
        
        return new_bn
    
    def net2deeper_conv(self, reference_layer: nn.Conv2d, 
                       activation: str = 'relu') -> nn.Conv2d:
        """
        在卷积层后插入恒等映射层
        
        Args:
            reference_layer: 参考层(用于确定新层的通道数)
            activation: 激活函数类型
        
        Returns:
            恒等映射的新卷积层
        """
        # 创建恒等卷积层
        identity_conv = nn.Conv2d(
            in_channels=reference_layer.out_channels,
            out_channels=reference_layer.out_channels,
            kernel_size=3,  # 使用3x3卷积
            stride=1,
            padding=1,  # 保持尺寸不变
            bias=True
        )
        
        # 初始化为恒等映射
        with torch.no_grad():
            nn.init.zeros_(identity_conv.weight)
            # 设置中心权重为1(恒等核)
            for i in range(reference_layer.out_channels):
                identity_conv.weight[i, i, 1, 1] = 1.0
            
            # 偏置设为0
            if identity_conv.bias is not None:
                nn.init.zeros_(identity_conv.bias)
        
        return identity_conv
    
    def net2deeper_linear(self, reference_layer: nn.Linear) -> nn.Linear:
        """
        在线性层后插入恒等映射层
        """
        identity_linear = nn.Linear(
            in_features=reference_layer.out_features,
            out_features=reference_layer.out_features,
            bias=True
        )
        
        with torch.no_grad():
            # 恒等权重矩阵
            nn.init.eye_(identity_linear.weight)
            # 零偏置
            nn.init.zeros_(identity_linear.bias)
        
        return identity_linear
    
    def net2branch(self, base_layer: nn.Module, 
                  num_branches: int = 2) -> List[nn.Module]:
        """
        创建分支结构，每个分支初始化为与基础层相同
        
        Args:
            base_layer: 基础层
            num_branches: 分支数量
        
        Returns:
            分支层列表
        """
        branches = []
        
        for i in range(num_branches):
            if isinstance(base_layer, nn.Conv2d):
                branch = nn.Conv2d(
                    in_channels=base_layer.in_channels,
                    out_channels=base_layer.out_channels,
                    kernel_size=base_layer.kernel_size,
                    stride=base_layer.stride,
                    padding=base_layer.padding,
                    dilation=base_layer.dilation,
                    groups=base_layer.groups,
                    bias=base_layer.bias is not None,
                    padding_mode=base_layer.padding_mode
                )
            elif isinstance(base_layer, nn.Linear):
                branch = nn.Linear(
                    in_features=base_layer.in_features,
                    out_features=base_layer.out_features,
                    bias=base_layer.bias is not None
                )
            else:
                raise NotImplementedError(f"Branch creation not implemented for {type(base_layer)}")
            
            # 复制权重
            with torch.no_grad():
                branch.weight.copy_(base_layer.weight)
                if base_layer.bias is not None:
                    branch.bias.copy_(base_layer.bias)
                
                # 第二个分支之后添加小噪声
                if i > 0:
                    noise_std = 1e-7
                    branch.weight += torch.randn_like(branch.weight) * noise_std
                    if branch.bias is not None:
                        branch.bias += torch.randn_like(branch.bias) * noise_std
            
            branches.append(branch)
        
        return branches
    
    def smooth_transition_loss(self, student_output: torch.Tensor, 
                              teacher_output: torch.Tensor, 
                              alpha: float = 0.1) -> torch.Tensor:
        """
        平滑过渡损失函数
        
        Args:
            student_output: 学生网络输出
            teacher_output: 教师网络输出
            alpha: 过渡损失权重
        
        Returns:
            平滑过渡损失
        """
        mse_loss = F.mse_loss(student_output, teacher_output.detach())
        return alpha * mse_loss
    
    def verify_function_preserving(self, teacher_model: nn.Module,
                                 student_model: nn.Module,
                                 test_input: torch.Tensor,
                                 tolerance: float = 1e-5) -> bool:
        """
        验证函数保持性质
        
        Args:
            teacher_model: 教师模型
            student_model: 学生模型  
            test_input: 测试输入
            tolerance: 容差
        
        Returns:
            是否保持函数相等
        """
        teacher_model.eval()
        student_model.eval()
        
        with torch.no_grad():
            teacher_output = teacher_model(test_input)
            student_output = student_model(test_input)
            
            max_diff = torch.max(torch.abs(teacher_output - student_output))
            return max_diff.item() < tolerance
    
    def record_transfer(self, transfer_type: str, details: Dict):
        """记录迁移操作"""
        record = {
            'type': transfer_type,
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None,
            'details': details
        }
        self.transfer_records.append(record)
    
    def get_transfer_history(self) -> List[Dict]:
        """获取迁移历史"""
        return self.transfer_records