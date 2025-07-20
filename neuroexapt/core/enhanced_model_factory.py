"""
增强模型工厂
Enhanced Model Factory

为CIFAR-10数据集优化的高性能模型架构，支持：
1. 增强ResNet架构（SE注意力、残差连接优化）
2. 自适应池化和归一化
3. 可进化的模块化设计
4. 95%准确率目标优化

作者：基于用户要求为CIFAR-10优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EnhancedBasicBlock(nn.Module):
    """增强的ResNet基础块"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, use_se=True, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # SE注意力机制
        self.se = SEBlock(planes) if use_se else None
        
        # Dropout正则化
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        if self.dropout:
            out = self.dropout(out)
        
        out = self.bn2(self.conv2(out))
        
        if self.se:
            out = self.se(out)
        
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EnhancedBottleneck(nn.Module):
    """增强的ResNet瓶颈块"""
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, use_se=True, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
        # SE注意力机制
        self.se = SEBlock(self.expansion * planes) if use_se else None
        
        # Dropout正则化
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        
        if self.dropout:
            out = self.dropout(out)
        
        out = self.bn3(self.conv3(out))
        
        if self.se:
            out = self.se(out)
        
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EnhancedResNet(nn.Module):
    """增强的ResNet架构"""
    
    def __init__(self, block, num_blocks, num_classes=10, use_se=True, 
                 dropout_rate=0.1, width_multiplier=1.0):
        super().__init__()
        self.in_planes = int(64 * width_multiplier)
        self.use_se = use_se
        self.dropout_rate = dropout_rate
        
        # CIFAR-10优化的输入层
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        # 构建ResNet层
        self.layer1 = self._make_layer(block, int(64 * width_multiplier), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128 * width_multiplier), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * width_multiplier), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * width_multiplier), num_blocks[3], stride=2)
        
        # 自适应全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头
        self.fc = nn.Linear(int(512 * width_multiplier) * block.expansion, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, 
                               self.use_se, self.dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class DenseBlock(nn.Module):
    """DenseNet风格的密集块"""
    
    def __init__(self, in_channels, growth_rate, num_layers, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 
                         kernel_size=3, padding=1, bias=False),
                nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
            )
            self.layers.append(layer)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


class HybridResNetDense(nn.Module):
    """混合ResNet-DenseNet架构"""
    
    def __init__(self, num_classes=10, growth_rate=32, num_dense_layers=6):
        super().__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet层
        self.layer1 = self._make_resnet_layer(64, 64, 2)
        self.layer2 = self._make_resnet_layer(64, 128, 2, stride=2)
        
        # DenseNet块
        self.dense_block = DenseBlock(128, growth_rate, num_dense_layers)
        dense_out_channels = 128 + num_dense_layers * growth_rate
        
        # 过渡层
        self.transition = nn.Sequential(
            nn.BatchNorm2d(dense_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(dense_out_channels, dense_out_channels // 2, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # 最后的ResNet层
        self.layer3 = self._make_resnet_layer(dense_out_channels // 2, 256, 2, stride=2)
        
        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        self._initialize_weights()
    
    def _make_resnet_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(EnhancedBasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(EnhancedBasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.dense_block(out)
        out = self.transition(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def create_enhanced_model(model_type='enhanced_resnet34', num_classes=10, **kwargs):
    """
    创建增强模型
    
    Args:
        model_type: 模型类型
        num_classes: 分类数
        **kwargs: 其他参数
        
    Returns:
        nn.Module: 创建的模型
    """
    
    if model_type == 'enhanced_resnet18':
        return EnhancedResNet(EnhancedBasicBlock, [2, 2, 2, 2], num_classes, **kwargs)
    elif model_type == 'enhanced_resnet34':
        return EnhancedResNet(EnhancedBasicBlock, [3, 4, 6, 3], num_classes, **kwargs)
    elif model_type == 'enhanced_resnet50':
        return EnhancedResNet(EnhancedBottleneck, [3, 4, 6, 3], num_classes, **kwargs)
    elif model_type == 'enhanced_resnet101':
        return EnhancedResNet(EnhancedBottleneck, [3, 4, 23, 3], num_classes, **kwargs)
    elif model_type == 'enhanced_resnet152':
        return EnhancedResNet(EnhancedBottleneck, [3, 8, 36, 3], num_classes, **kwargs)
    elif model_type == 'hybrid_resdense':
        return HybridResNetDense(num_classes, **kwargs)
    elif model_type == 'wide_resnet':
        # 宽ResNet变体
        return EnhancedResNet(EnhancedBasicBlock, [3, 4, 6, 3], num_classes, 
                            width_multiplier=2.0, **kwargs)
    else:
        # 默认使用增强ResNet34
        return EnhancedResNet(EnhancedBasicBlock, [3, 4, 6, 3], num_classes, **kwargs)


def get_model_stats(model):
    """
    获取模型统计信息
    
    Args:
        model: PyTorch模型
        
    Returns:
        Dict: 模型统计信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算模型大小（MB）
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'param_size_mb': param_size / (1024 ** 2),
        'buffer_size_mb': buffer_size / (1024 ** 2)
    }


# 支持的增强训练配置
class EnhancedTrainingConfig:
    """增强训练配置"""
    
    def __init__(self):
        # 数据增广配置
        self.use_cutmix = True
        self.use_mixup = True
        self.use_random_erasing = True
        self.use_auto_augment = True
        
        # 正则化配置
        self.label_smoothing = 0.1
        self.dropout_rate = 0.1
        self.weight_decay = 5e-4
        
        # 优化器配置
        self.optimizer_type = 'sgd'  # 'sgd', 'adam', 'adamw'
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.nesterov = True
        
        # 学习率调度
        self.scheduler_type = 'cosine'  # 'cosine', 'step', 'multi_step'
        self.warmup_epochs = 5
        self.min_lr = 1e-6
        
        # 训练配置
        self.epochs = 200
        self.batch_size = 128
        self.gradient_clip = 1.0
        
        # 模型配置
        self.model_ema = True  # 指数移动平均
        self.ema_decay = 0.9999


def get_enhanced_transforms(config=None):
    """
    获取增强的数据变换
    
    Args:
        config: 训练配置
        
    Returns:
        Tuple: (训练变换, 测试变换)
    """
    import torchvision.transforms as transforms
    
    if config is None:
        config = EnhancedTrainingConfig()
    
    # 基础变换
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    
    # 高级数据增广
    if config.use_auto_augment:
        try:
            from torchvision.transforms import AutoAugment, AutoAugmentPolicy
            train_transforms.append(AutoAugment(AutoAugmentPolicy.CIFAR10))
        except ImportError:
            # 如果没有AutoAugment，使用手动增广
            train_transforms.extend([
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
    
    # 标准化
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 随机擦除
    if config.use_random_erasing:
        train_transforms.append(transforms.RandomErasing(p=0.1))
    
    # 测试变换
    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    
    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)
    
    return train_transform, test_transform


def mixup_data(x, y, alpha=1.0):
    """Mixup数据增广"""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample()
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# 导出主要函数和类
__all__ = [
    'create_enhanced_model',
    'get_model_stats', 
    'EnhancedTrainingConfig',
    'get_enhanced_transforms',
    'mixup_data',
    'mixup_criterion',
    'LabelSmoothingCrossEntropy',
    'EnhancedResNet',
    'HybridResNetDense',
    'SEBlock'
]