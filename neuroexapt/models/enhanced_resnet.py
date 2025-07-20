"""
增强版ResNet架构 - 针对CIFAR-10优化
Enhanced ResNet Architecture - Optimized for CIFAR-10

目标：达到95%+的准确率
优化策略：
1. 改进的残差块设计（预激活 + SE注意力）
2. 更好的正则化（Dropout + BatchNorm + 数据增强）
3. 自适应学习率调度
4. 混合精度训练支持
5. 渐进式训练策略

Args:
    use_dropout (bool): Whether to apply dropout after global average pooling. Default: True.
    dropout_rate (float): Dropout probability if dropout is used. Default: 0.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Optional


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PreActBlock(nn.Module):
    """Pre-activation Residual Block with SE attention"""
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1, 
                 dropout_rate: float = 0.0, use_se: bool = True):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        self.se = SEBlock(planes) if use_se else None
        
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)
        
        if self.se is not None:
            out = self.se(out)
            
        out += shortcut
        return out


class EnhancedResNet(nn.Module):
    """
    增强版ResNet，针对CIFAR-10优化
    
    改进点：
    - 使用Pre-activation ResBlock
    - 添加SE注意力机制
    - 改进的数据流和正则化
    - 更深的网络结构
    """
    
    def __init__(self, num_blocks: List[int] = [3, 4, 6, 3], 
                 num_classes: int = 10, dropout_rate: float = 0.1,
                 use_se: bool = True, widen_factor: int = 1, 
                 use_dropout: bool = True):
        super(EnhancedResNet, self).__init__()
        self.in_planes = 64 * widen_factor
        self.dropout_rate = dropout_rate
        self.use_se = use_se
        self.use_dropout = use_dropout
        
        # Initial convolution - 针对CIFAR-10的小图像优化
        self.conv1 = nn.Conv2d(3, 64 * widen_factor, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64 * widen_factor)
        
        # Residual layers
        self.layer1 = self._make_layer(64 * widen_factor, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128 * widen_factor, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256 * widen_factor, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512 * widen_factor, num_blocks[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * widen_factor, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(self.in_planes, planes, stride, 
                                    self.dropout_rate, self.use_se))
            self.in_planes = planes
        return nn.Sequential(*layers)
        
    def _initialize_weights(self):
        """权重初始化 - 修复bias为None的问题"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:  # 🔧 修复：检查bias是否存在
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:  # 🔧 修复：检查bias是否存在
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:  # 🔧 修复：检查bias是否存在
                    nn.init.constant_(m.bias, 0)
                
    def forward(self, x, return_features: bool = False):
        # Initial conv
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Residual layers
        features = {}
        out = self.layer1(out)
        features['layer1'] = out
        
        out = self.layer2(out)
        features['layer2'] = out
        
        out = self.layer3(out)
        features['layer3'] = out
        
        out = self.layer4(out)
        features['layer4'] = out
        
        # Global pooling and classification
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        features['avgpool'] = out
        
        # 🔧 修复：条件性应用dropout
        if self.use_dropout:
            out = self.dropout(out)
        out = self.fc(out)
        features['fc'] = out
        
        if return_features:
            return out, features
        return out


def enhanced_resnet18(**kwargs):
    """Enhanced ResNet-18"""
    return EnhancedResNet([2, 2, 2, 2], **kwargs)


def enhanced_resnet34(**kwargs):
    """Enhanced ResNet-34"""
    return EnhancedResNet([3, 4, 6, 3], **kwargs)


def enhanced_resnet50(**kwargs):
    """Enhanced ResNet-50 (using PreActBlock)"""
    return EnhancedResNet([3, 4, 6, 3], **kwargs)


def enhanced_wide_resnet(**kwargs):
    """Enhanced Wide ResNet"""
    return EnhancedResNet([3, 4, 6, 3], widen_factor=2, **kwargs)


class EnhancedTrainingConfig:
    """Enhanced training configuration for 95% accuracy"""
    
    def __init__(self):
        # Data augmentation
        self.use_strong_augmentation = True
        self.cutmix_prob = 0.5
        self.mixup_alpha = 0.2
        
        # Training hyperparameters
        self.initial_lr = 0.1
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.epochs = 300
        self.batch_size = 128
        
        # Learning rate schedule
        self.lr_schedule = 'cosine_annealing'  # or 'multistep'
        self.lr_milestones = [150, 225]
        self.lr_gamma = 0.1
        
        # Regularization
        self.dropout_rate = 0.1
        self.label_smoothing = 0.1
        self.use_dropout = True  # 🔧 新增：dropout配置
        
        # Advanced techniques
        self.use_ema = True  # Exponential Moving Average
        self.ema_decay = 0.9999
        self.use_sam = True  # Sharpness-Aware Minimization
        self.use_mixup = True
        self.use_cutmix = True
        
        # Model architecture
        self.model_type = 'enhanced_resnet34'
        self.use_se = True
        self.widen_factor = 1


def create_enhanced_model(config: EnhancedTrainingConfig = None) -> nn.Module:
    """Create enhanced model with optimal configuration for CIFAR-10"""
    
    if config is None:
        config = EnhancedTrainingConfig()
        
    model_kwargs = {
        'num_classes': 10,
        'dropout_rate': config.dropout_rate,
        'use_se': config.use_se,
        'widen_factor': config.widen_factor,
        'use_dropout': config.use_dropout  # 🔧 新增：dropout配置传递
    }
    
    if config.model_type == 'enhanced_resnet18':
        return enhanced_resnet18(**model_kwargs)
    elif config.model_type == 'enhanced_resnet34':
        return enhanced_resnet34(**model_kwargs)
    elif config.model_type == 'enhanced_resnet50':
        return enhanced_resnet50(**model_kwargs)
    elif config.model_type == 'enhanced_wide_resnet':
        return enhanced_wide_resnet(**model_kwargs)
    else:
        return enhanced_resnet34(**model_kwargs)  # default


def get_enhanced_transforms():
    """Get enhanced data augmentation transforms for CIFAR-10"""
    import torchvision.transforms as transforms
    
    # Training transforms with strong augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2023, 0.1994, 0.2010]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])
    
    # Test transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2023, 0.1994, 0.2010])
    ])
    
    return train_transform, test_transform


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
        log_prob = F.log_softmax(pred, dim=1)
        return F.kl_div(log_prob, one_hot, reduction='batchmean')


def mixup_data(x, y, alpha=1.0):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)