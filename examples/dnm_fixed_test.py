#!/usr/bin/env python3
"""
DNM (Dynamic Neural Morphogenesis) 重构测试

🧬 新特性：
1. 多理论支撑的触发机制
2. 更智能的神经元分裂策略
3. 自适应的形态发生判断
4. 更激进的早期干预
5. 突破90%准确率的设计

🎯 目标：验证重构后的DNM框架能够有效突破性能瓶颈
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import logging
from tqdm import tqdm
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.core.dnm_framework import DNMFramework
from neuroexapt.core.dnm_neuron_division import AdaptiveNeuronDivision

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(name)s:%(message)s')
logger = logging.getLogger('neuroexapt.core.dnm_framework')

class AdvancedCNNModel(nn.Module):
    """增强的CNN模型，为DNM优化"""
    
    def __init__(self, num_classes=10):
        super(AdvancedCNNModel, self).__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # 第三个卷积块  
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ActivationHook:
    """激活值捕获钩子"""
    
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def hook_fn(self, name):
        def fn(module, input, output):
            self.activations[name] = output.detach()
        return fn
    
    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 只为主要层注册钩子，避免过多的激活值
                if ('classifier' in name and isinstance(module, nn.Linear)) or \
                   ('features' in name and isinstance(module, nn.Conv2d) and 'features.17' in name):
                    hook = module.register_forward_hook(self.hook_fn(name))
                    self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class GradientHook:
    """梯度捕获钩子"""
    
    def __init__(self):
        self.gradients = {}
    
    def capture_gradients(self, model):
        self.gradients.clear()
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.gradients[name] = param.grad.detach().clone()

def load_cifar10_data(batch_size=128):
    """加载CIFAR-10数据集"""
    
    # 数据增强和标准化
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return trainloader, testloader

def train_epoch(model, train_loader, optimizer, criterion, device, activation_hook, gradient_hook):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 捕获梯度
        gradient_hook.capture_gradients(model)
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 50 == 0:
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss / len(train_loader), 100. * correct / total

def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return val_loss / len(val_loader), 100. * correct / total

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    """主训练循环"""
    print("🧬 DNM (Dynamic Neural Morphogenesis) 重构测试")
    print("="*60)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    # 数据加载
    print("📁 加载CIFAR-10数据集...")
    train_loader, val_loader = load_cifar10_data(batch_size=128)
    
    # 模型初始化
    print("🏗️ 初始化模型...")
    model = AdvancedCNNModel(num_classes=10).to(device)
    initial_params = count_parameters(model)
    print(f"初始参数数量: {initial_params:,}")
    
    # DNM框架配置
    dnm_config = {
        'morphogenesis_interval': 3,  # 每3个epoch检查一次
        'max_morphogenesis_per_epoch': 1,  # 每次最多1次形态发生
        'performance_improvement_threshold': 0.01,  # 性能改善阈值
    }
    
    # 初始化DNM框架
    print("🧬 初始化DNM框架...")
    dnm = DNMFramework(model, dnm_config)
    
    # 优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()
    
    # 钩子设置
    activation_hook = ActivationHook()
    activation_hook.register_hooks(model)
    gradient_hook = GradientHook()
    
    # 训练配置
    epochs = 100
    best_acc = 0.0
    patience = 25
    patience_counter = 0
    morphogenesis_events = 0
    
    print("\n🚀 开始训练...")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n🧬 Epoch {epoch+1}/{epochs} - Dynamic Morphogenesis")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, 
                                          device, activation_hook, gradient_hook)
        
        # 验证
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # 更新DNM框架状态
        dnm.update_caches(activation_hook.activations, gradient_hook.gradients)
        dnm.record_performance(val_acc)
        
        # 检查是否需要形态发生
        train_metrics = {'accuracy': train_acc, 'loss': train_loss, 'learning_rate': current_lr}
        val_metrics = {'accuracy': val_acc, 'loss': val_loss}
        
        should_trigger, reasons = dnm.should_trigger_morphogenesis(epoch+1, train_metrics, val_metrics)
        
        if should_trigger:
            print(f"  🔄 Triggering morphogenesis analysis...")
            for reason in reasons:
                print(f"    - {reason}")
                
            # 执行形态发生
            results = dnm.execute_morphogenesis(epoch+1)
            if results['neuron_divisions'] > 0:
                model = dnm.model  # 更新模型引用
                morphogenesis_events += 1
                
                # 重新设置优化器（因为模型参数变了）
                optimizer = optim.SGD(model.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
                # 创建新的调度器，但要保持当前的步数
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, 100-epoch))
                # 快进调度器到当前epoch
                for _ in range(epoch + 1):
                    scheduler.step()
                
                # 重新注册钩子
                activation_hook.remove_hooks()
                activation_hook.register_hooks(model)
                
                print(f"    ✅ 形态发生完成: {results['neuron_divisions']} 次神经元分裂")
                print(f"    📊 新增参数: {results['parameters_added']:,}")
        
        current_params = count_parameters(model)
        param_growth = ((current_params - initial_params) / initial_params) * 100
        
        print(f"  📊 Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Params: {current_params:,} | ")
        
        # 详细日志输出（每3个epoch）
        if (epoch + 1) % 3 == 0:
            print(f"📈 Epoch {epoch+1:3d}: Train Acc={train_acc:.2f}% Loss={train_loss:.4f} | "
                  f"Val Acc={val_acc:.2f}% Loss={val_loss:.4f} | "
                  f"Params={current_params:,} (+{param_growth:.1f}%) | LR={current_lr:.6f}")
        
        # 更新最佳准确率
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 学习率调度（在早停检查之前）
        scheduler.step()
        
        # 早停检查
        if patience_counter >= patience:
            print(f"  🛑 Early stopping triggered (patience: {patience})")
            break
    
    # 训练完成
    print("✅ DNM training completed")
    
    training_time = time.time() - start_time
    final_params = count_parameters(model)
    param_growth = ((final_params - initial_params) / initial_params) * 100
    
    print(f"   Final accuracy: {val_acc:.2f}% | Best: {best_acc:.2f}%")
    print(f"   Morphogenesis events: {morphogenesis_events}")
    
    # 获取DNM摘要
    summary = dnm.get_morphogenesis_summary()
    
    print(f"\n🎉 DNM训练完成!")
    print("="*60)
    print(f"📊 最终结果:")
    print(f"   最佳验证准确率: {best_acc:.2f}%")
    print(f"   最终验证准确率: {val_acc:.2f}%")
    print(f"   参数增长: +{param_growth:.1f}% ({initial_params:,} → {final_params:,})")
    print(f"   训练时间: {training_time/60:.1f}分钟")
    print(f"   形态发生事件: {summary['total_events']}")
    print(f"   总神经元分裂: {summary['total_neuron_divisions']}")
    print(f"   总连接生长: 0")  # 未实现
    print(f"   总优化次数: 0")  # 未实现
    
    # 形态发生事件分析
    if summary['events_detail']:
        print(f"\n🧬 形态发生事件分析:")
        for i, event in enumerate(summary['events_detail']):
            print(f"   事件 {i+1} (Epoch {event['epoch']}):")
            print(f"     神经元分裂: {event['params_added']}")
            print(f"     连接生长: 0")
            print(f"     优化触发: False")
            print(f"     触发前性能: {event.get('performance_before', 0):.2f}%")
    
    # 性能评估
    if best_acc >= 90.0:
        print(f"\n🏆 BREAKTHROUGH: 成功突破90%准确率大关! ({best_acc:.2f}%)")
    elif best_acc >= 85.0:
        print(f"\n🔄 IMPROVING: 需要进一步优化 ({best_acc:.2f}%)")
    else:
        print(f"\n⚠️ NEEDS WORK: 需要重大改进 ({best_acc:.2f}%)")
    
    print(f"\n✅ 测试完成!")
    print(f"   DNM框架运行正常")
    print(f"   最终准确率: {val_acc:.2f}%")
    
    # 清理钩子
    activation_hook.remove_hooks()

if __name__ == "__main__":
    main()