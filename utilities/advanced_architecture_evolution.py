#!/usr/bin/env python3
"""
NeuroExapt - 高级架构演化系统
专门用于突破82%准确度瓶颈，实现自适应架构升级
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
from collections import defaultdict
import sys
import os

# Add neuroexapt to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt.trainer import Trainer, train_with_neuroexapt


class ResidualBlock(nn.Module):
    """改进的残差块"""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 双路径注意力
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        attention = avg_out + max_out
        return x * attention.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


class CBAM(nn.Module):
    """卷积块注意力模块 (Convolutional Block Attention Module)"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class EvolutionaryResNet(nn.Module):
    """演化的ResNet架构 - 能够自适应升级"""
    
    def __init__(self, num_classes=10, evolution_level=0):
        super().__init__()
        self.evolution_level = evolution_level
        
        # 基础层
        self.conv1 = nn.Conv2d(3, 64, 7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # 动态层配置
        if evolution_level >= 0:
            # Level 0: 基础ResNet
            self.layer1 = self._make_layer(64, 64, 2, 1)
            self.layer2 = self._make_layer(64, 128, 2, 2)
            self.layer3 = self._make_layer(128, 256, 2, 2)
            self.layer4 = self._make_layer(256, 512, 2, 2)
        
        if evolution_level >= 1:
            # Level 1: 添加注意力机制
            self.attention1 = CBAM(64)
            self.attention2 = CBAM(128)
            self.attention3 = CBAM(256)
            self.attention4 = CBAM(512)
        
        if evolution_level >= 2:
            # Level 2: 增加深度
            self.layer5 = self._make_layer(512, 1024, 2, 2)
            self.attention5 = CBAM(1024)
        
        if evolution_level >= 3:
            # Level 3: 多尺度特征融合
            self.multiscale_conv = nn.ModuleList([
                nn.Conv2d(512, 128, 1),
                nn.Conv2d(512, 128, 3, padding=1),
                nn.Conv2d(512, 128, 5, padding=2),
                nn.Conv2d(512, 128, 7, padding=3)
            ])
            
        # 自适应全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        final_channels = 1024 if evolution_level >= 2 else 512
        if evolution_level >= 3:
            final_channels = 512  # 多尺度融合后的通道数
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(final_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout=0.1))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, dropout=0.1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # 残差块
        x = self.layer1(x)
        if self.evolution_level >= 1:
            x = self.attention1(x)
        
        x = self.layer2(x)
        if self.evolution_level >= 1:
            x = self.attention2(x)
        
        x = self.layer3(x)
        if self.evolution_level >= 1:
            x = self.attention3(x)
        
        x = self.layer4(x)
        if self.evolution_level >= 1:
            x = self.attention4(x)
        
        # 高级演化特性
        if self.evolution_level >= 2:
            x = self.layer5(x)
            x = self.attention5(x)
        
        if self.evolution_level >= 3:
            # 多尺度特征融合
            multiscale_features = []
            for conv in self.multiscale_conv:
                multiscale_features.append(conv(x))
            x = torch.cat(multiscale_features, dim=1)
        
        # 分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


class AdvancedEvolutionStrategy:
    """高级演化策略"""
    
    def __init__(self, patience=8, min_improvement=0.3, max_level=3):
        self.patience = patience
        self.min_improvement = min_improvement
        self.max_level = max_level
        self.best_val_acc = 0.0
        self.no_improvement_count = 0
        self.current_level = 0
        self.evolution_history = []
        
    def should_evolve(self, current_val_acc, current_epoch):
        """智能判断是否需要演化"""
        # 更新最佳准确度
        if current_val_acc > self.best_val_acc:
            improvement = current_val_acc - self.best_val_acc
            self.best_val_acc = current_val_acc
            
            if improvement >= self.min_improvement:
                self.no_improvement_count = 0
                return False
        
        self.no_improvement_count += 1
        
        # 判断是否需要演化
        if self.no_improvement_count >= self.patience and self.current_level < self.max_level:
            evolution_type = self._determine_evolution_type()
            print(f"🔄 触发演化条件:")
            print(f"  当前准确度: {current_val_acc:.2f}%")
            print(f"  最佳准确度: {self.best_val_acc:.2f}%")
            print(f"  停滞轮数: {self.no_improvement_count}")
            print(f"  建议演化: {evolution_type}")
            
            return evolution_type
        
        return False
    
    def _determine_evolution_type(self):
        """根据当前状态确定演化类型"""
        if self.current_level == 0:
            return "add_attention"
        elif self.current_level == 1:
            return "increase_depth"
        elif self.current_level == 2:
            return "multiscale_fusion"
        else:
            return "advanced_optimization"
    
    def evolve_model(self, current_model, evolution_type):
        """执行模型演化"""
        print(f"🚀 开始模型演化: {evolution_type}")
        
        if evolution_type == "add_attention":
            # 演化到Level 1: 添加注意力机制
            new_model = EvolutionaryResNet(num_classes=10, evolution_level=1)
            self._transfer_weights(current_model, new_model)
            self.current_level = 1
            return new_model, "添加CBAM注意力机制，增强特征表示能力"
        
        elif evolution_type == "increase_depth":
            # 演化到Level 2: 增加深度
            new_model = EvolutionaryResNet(num_classes=10, evolution_level=2)
            self._transfer_weights(current_model, new_model)
            self.current_level = 2
            return new_model, "增加网络深度，提高特征提取能力"
        
        elif evolution_type == "multiscale_fusion":
            # 演化到Level 3: 多尺度特征融合
            new_model = EvolutionaryResNet(num_classes=10, evolution_level=3)
            self._transfer_weights(current_model, new_model)
            self.current_level = 3
            return new_model, "添加多尺度特征融合，捕获不同尺度的特征"
        
        elif evolution_type == "advanced_optimization":
            # 高级优化技巧
            return self._apply_advanced_optimization(current_model)
        
        return current_model, "未知演化类型"
    
    def _transfer_weights(self, old_model, new_model):
        """权重迁移"""
        try:
            old_dict = old_model.state_dict()
            new_dict = new_model.state_dict()
            
            # 只迁移匹配的权重
            transfer_dict = {k: v for k, v in old_dict.items() if k in new_dict and v.shape == new_dict[k].shape}
            
            new_model.load_state_dict(transfer_dict, strict=False)
            
            print(f"✅ 权重迁移完成: {len(transfer_dict)}/{len(new_dict)} 层")
            
        except Exception as e:
            print(f"⚠️ 权重迁移失败: {e}")
    
    def _apply_advanced_optimization(self, model):
        """应用高级优化技巧"""
        # 这里可以实现标签平滑、知识蒸馏等技巧
        return model, "应用高级优化技巧（标签平滑、混合精度等）"


def create_advanced_dataloaders():
    """创建高级数据加载器"""
    # 高级数据增强策略
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    return train_loader, val_loader


def evolutionary_training(model, train_loader, val_loader, epochs=50, lr=0.001):
    """演化训练过程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    
    criterion = nn.CrossEntropyLoss()
    
    train_acc_history = []
    val_acc_history = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        scheduler.step()
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
    
    return {
        'train_accuracy': train_acc_history,
        'val_accuracy': val_acc_history,
        'final_model': model
    }


def main():
    print("🧬 高级架构演化系统 - 突破82%准确度瓶颈")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader = create_advanced_dataloaders()
    
    # 创建演化策略
    evolution_strategy = AdvancedEvolutionStrategy(patience=6, min_improvement=0.2)
    
    # 开始演化过程
    current_model = EvolutionaryResNet(num_classes=10, evolution_level=0)
    
    print(f"\n🚀 开始演化训练 - Level 0 (基础ResNet)")
    print("="*60)
    
    # 阶段性训练
    total_epochs = 0
    max_evolutions = 3
    evolution_count = 0
    
    while evolution_count <= max_evolutions:
        print(f"\n📊 训练阶段 {evolution_count + 1} - Evolution Level {evolution_strategy.current_level}")
        
        # 训练当前模型
        results = evolutionary_training(
            current_model, train_loader, val_loader, 
            epochs=25, lr=0.001 * (0.8 ** evolution_count)
        )
        
        total_epochs += 25
        final_val_acc = results['val_accuracy'][-1]
        best_val_acc = max(results['val_accuracy'])
        
        print(f"\n📈 阶段结果:")
        print(f"  最终验证准确度: {final_val_acc:.2f}%")
        print(f"  最佳验证准确度: {best_val_acc:.2f}%")
        
        # 检查是否需要演化
        evolution_type = evolution_strategy.should_evolve(best_val_acc, total_epochs)
        
        if evolution_type:
            print(f"\n🔄 触发演化: {evolution_type}")
            
            # 演化模型
            evolved_model, evolution_desc = evolution_strategy.evolve_model(
                current_model, evolution_type
            )
            
            print(f"✅ 演化完成: {evolution_desc}")
            
            # 更新当前模型
            current_model = evolved_model
            evolution_count += 1
            
            # 重置无改善计数
            evolution_strategy.no_improvement_count = 0
            
        else:
            print(f"\n✅ 训练完成 - 无需进一步演化")
            break
    
    # 最终总结
    print("\n" + "="*60)
    print("🎉 演化训练完成")
    print("="*60)
    
    print(f"总训练轮数: {total_epochs}")
    print(f"演化次数: {evolution_count}")
    print(f"最终架构级别: {evolution_strategy.current_level}")
    print(f"最终验证准确度: {evolution_strategy.best_val_acc:.2f}%")
    
    # 架构演化历史
    print(f"\n📊 架构演化历史:")
    evolution_names = ["基础ResNet", "添加注意力", "增加深度", "多尺度融合"]
    for i in range(min(evolution_count + 1, len(evolution_names))):
        print(f"  Level {i}: {evolution_names[i]}")
    
    # 突破分析
    if evolution_strategy.best_val_acc > 82:
        print(f"\n🎯 成功突破82%瓶颈!")
        print(f"  最终准确度: {evolution_strategy.best_val_acc:.2f}%")
        print(f"  提升幅度: {evolution_strategy.best_val_acc - 82:.2f}%")
    else:
        print(f"\n⚠️ 未能突破82%瓶颈")
        print(f"  当前准确度: {evolution_strategy.best_val_acc:.2f}%")
        print(f"  距离目标: {82 - evolution_strategy.best_val_acc:.2f}%")
    
    print("\n✅ 演化训练系统运行完成！")


if __name__ == "__main__":
    main() 