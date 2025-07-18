#!/usr/bin/env python3
"""
DNM Enhanced Test - 增强版 DNM 框架测试

🧬 特性:
1. 集成增强的瓶颈检测器
2. 基于性能导向的神经元分裂
3. 多维度性能监控
4. 智能触发机制
5. 详细的分析报告

🎯 目标: 验证增强的 DNM 框架能够显著提升分裂效果和准确率
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
from neuroexapt.core.enhanced_bottleneck_detector import EnhancedBottleneckDetector
from neuroexapt.core.performance_guided_division import PerformanceGuidedDivision, DivisionStrategy

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger('dnm_enhanced_test')

class EnhancedCNNModel(nn.Module):
    """增强的CNN模型，专为DNM优化"""
    
    def __init__(self, num_classes=10):
        super(EnhancedCNNModel, self).__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            # 第一个卷积块 - 较小的初始通道数，便于观察分裂效果
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
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
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        
        # 分类器部分 - 较小的隐藏层，便于观察分裂
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
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
    """改进的激活值捕获钩子"""
    
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def hook_fn(self, name):
        def fn(module, input, output):
            self.activations[name] = output.detach().clone()
        return fn
    
    def register_hooks(self, model):
        self.remove_hooks()  # 清理之前的钩子
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 为所有卷积层和线性层注册钩子
                hook = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()

class GradientHook:
    """改进的梯度捕获钩子"""
    
    def __init__(self):
        self.gradients = {}
        self.last_targets = None
    
    def capture_gradients(self, model):
        self.gradients.clear()
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.gradients[name] = param.grad.detach().clone()

def load_cifar10_data(batch_size=64):
    """加载CIFAR-10数据集，使用较小的batch size以更好地观察分裂效果"""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
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
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        gradient_hook.last_targets = target
        
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
        
        if batch_idx % 100 == 0:
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # 为了观察分裂效果，只训练一部分数据
        if batch_idx >= 200:  # 限制每个epoch的批次数
            break
    
    return running_loss / min(len(train_loader), 201), 100. * correct / total

def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 限制验证数据量
            if batch_idx >= 50:
                break
    
    return val_loss / min(len(val_loader), 51), 100. * correct / total

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_model_structure(model):
    """分析模型结构"""
    conv_layers = 0
    linear_layers = 0
    total_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers += 1
            total_params += sum(p.numel() for p in module.parameters())
        elif isinstance(module, nn.Linear):
            linear_layers += 1
            total_params += sum(p.numel() for p in module.parameters())
    
    return {
        'conv_layers': conv_layers,
        'linear_layers': linear_layers,
        'total_params': total_params
    }

def main():
    """主训练循环"""
    print("🧬 DNM Enhanced Framework Test")
    print("=" * 60)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 设备: {device}")
    
    # 数据加载
    print("📁 加载CIFAR-10数据集...")
    train_loader, val_loader = load_cifar10_data(batch_size=64)
    
    # 模型初始化
    print("🏗️ 初始化增强模型...")
    model = EnhancedCNNModel(num_classes=10).to(device)
    initial_params = count_parameters(model)
    initial_structure = analyze_model_structure(model)
    print(f"初始参数数量: {initial_params:,}")
    print(f"初始结构: Conv={initial_structure['conv_layers']}, Linear={initial_structure['linear_layers']}")
    
    # 增强的瓶颈检测器配置
    bottleneck_detector = EnhancedBottleneckDetector(
        sensitivity_threshold=0.05,   # 更敏感的检测
        diversity_threshold=0.2,      # 较低的多样性阈值
        gradient_threshold=1e-7,      # 更敏感的梯度检测
        info_flow_threshold=0.3       # 较低的信息流阈值
    )
    
    # 性能导向分裂器配置
    guided_division = PerformanceGuidedDivision(
        noise_scale=0.05,            # 较小的噪声，保持稳定性
        progressive_epochs=3,         # 较快的渐进激活
        diversity_threshold=0.7,      # 多样性阈值
        performance_monitoring=True   # 启用性能监控
    )
    
    # DNM框架配置
    dnm_config = {
        'morphogenesis_interval': 2,   # 每2个epoch检查一次，更频繁
        'max_morphogenesis_per_epoch': 2,  # 每次最多2次形态发生
        'performance_improvement_threshold': 0.005,  # 更敏感的阈值
        'enhanced_bottleneck_detector': bottleneck_detector,
        'performance_guided_division': guided_division,
        'division_strategy': DivisionStrategy.HYBRID,  # 使用混合策略
    }
    
    # 初始化DNM框架
    print("🧬 初始化增强DNM框架...")
    dnm = DNMFramework(model, dnm_config)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss()
    
    # 钩子设置
    activation_hook = ActivationHook()
    activation_hook.register_hooks(model)
    gradient_hook = GradientHook()
    
    # 训练配置
    epochs = 30  # 较少的epochs以观察快速效果
    best_acc = 0.0
    patience = 15
    patience_counter = 0
    morphogenesis_events = 0
    
    # 性能历史记录
    performance_history = []
    bottleneck_history = []
    division_history = []
    
    print("\n🚀 开始增强训练...")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n🧬 Epoch {epoch+1}/{epochs} - Enhanced DNM")
        
        # 更新组件的epoch信息
        guided_division.update_epoch(epoch)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, 
                                          device, activation_hook, gradient_hook)
        
        # 验证
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # 记录性能
        performance_history.append(val_acc)
        bottleneck_detector.update_performance_history(val_acc)
        
        # 使用增强的瓶颈检测
        bottleneck_scores = bottleneck_detector.detect_bottlenecks(
            model, activation_hook.activations, gradient_hook.gradients, gradient_hook.last_targets
        )
        bottleneck_history.append(bottleneck_scores)
        
        # 获取瓶颈分析摘要
        bottleneck_summary = bottleneck_detector.get_analysis_summary(bottleneck_scores)
        
        # 智能触发判断
        should_trigger, reasons = bottleneck_detector.should_trigger_division(
            bottleneck_scores, performance_history[-5:]  # 使用最近5个epoch的性能
        )
        
        print(f"  📊 Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Loss: {val_loss:.4f}")
        print(f"  🔍 瓶颈分析: 最高={bottleneck_summary.get('max_score', 0):.3f}, 平均={bottleneck_summary.get('mean_score', 0):.3f}")
        
        if should_trigger:
            print(f"  🔄 触发增强形态发生:")
            for reason in reasons:
                print(f"    - {reason}")
            
            # 获取Top瓶颈层
            top_bottlenecks = bottleneck_detector.get_top_bottlenecks(bottleneck_scores, 2)
            
            for layer_name, score in top_bottlenecks:
                print(f"    🎯 目标层: {layer_name} (分数: {score:.3f})")
                
                # 找到对应的层和神经元
                for name, module in model.named_modules():
                    if name == layer_name and isinstance(module, (nn.Conv2d, nn.Linear)):
                        # 选择分裂的神经元（这里简单选择中间的神经元）
                        if isinstance(module, nn.Conv2d):
                            neuron_idx = module.out_channels // 2
                        else:
                            neuron_idx = module.out_features // 2
                        
                        # 执行性能导向分裂
                        success, division_info = guided_division.divide_neuron(
                            module, neuron_idx, DivisionStrategy.HYBRID,
                            activation_hook.activations.get(name),
                            gradient_hook.gradients.get(name + '.weight'),
                            gradient_hook.last_targets
                        )
                        
                        if success:
                            morphogenesis_events += 1
                            division_history.append(division_info)
                            print(f"    ✅ 分裂成功: {division_info.get('division_type', 'unknown')} 策略")
                            
                            # 重新注册钩子（因为模型结构可能变化）
                            activation_hook.register_hooks(model)
                            
                            # 更新优化器（如果模型参数变化）
                            optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=1e-4)
                        else:
                            print(f"    ❌ 分裂失败: {division_info.get('error', 'unknown error')}")
                        
                        break
        
        # 计算当前模型状态
        current_params = count_parameters(model)
        current_structure = analyze_model_structure(model)
        param_growth = ((current_params - initial_params) / initial_params) * 100
        
        print(f"  📈 参数: {current_params:,} (+{param_growth:.1f}%) | 形态发生: {morphogenesis_events}")
        
        # 更新最佳准确率
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 学习率调度
        scheduler.step()
        
        # 早停检查
        if patience_counter >= patience:
            print(f"  🛑 早停触发 (patience: {patience})")
            break
    
    # 训练完成分析
    training_time = time.time() - start_time
    final_params = count_parameters(model)
    final_structure = analyze_model_structure(model)
    param_growth = ((final_params - initial_params) / initial_params) * 100
    
    print(f"\n🎉 增强DNM训练完成!")
    print("=" * 60)
    print(f"📊 最终结果:")
    print(f"   最佳验证准确率: {best_acc:.2f}%")
    print(f"   最终验证准确率: {val_acc:.2f}%")
    print(f"   参数增长: +{param_growth:.1f}% ({initial_params:,} → {final_params:,})")
    print(f"   训练时间: {training_time/60:.1f}分钟")
    print(f"   形态发生事件: {morphogenesis_events}")
    
    # 获取增强组件的摘要
    division_summary = guided_division.get_division_summary()
    
    print(f"\n🧬 增强组件分析:")
    print(f"   瓶颈检测事件: {len(bottleneck_history)}")
    print(f"   总分裂次数: {division_summary.get('total_divisions', 0)}")
    if division_summary.get('strategies_used'):
        print(f"   分裂策略分布:")
        for strategy, count in division_summary['strategies_used'].items():
            print(f"     - {strategy}: {count} 次")
    
    # 性能趋势分析
    if len(performance_history) > 5:
        early_avg = np.mean(performance_history[:3])
        late_avg = np.mean(performance_history[-3:])
        improvement = late_avg - early_avg
        print(f"   性能改善: {improvement:.2f}% (从 {early_avg:.2f}% 到 {late_avg:.2f}%)")
    
    # 性能评估
    if best_acc >= 75.0:
        print(f"\n🏆 优秀表现: 成功达到 {best_acc:.2f}% 准确率!")
    elif best_acc >= 65.0:
        print(f"\n🔄 良好表现: 达到 {best_acc:.2f}% 准确率，有改进空间")
    else:
        print(f"\n⚠️ 需要改进: 当前 {best_acc:.2f}% 准确率")
    
    print(f"\n✅ 增强DNM测试完成!")
    print(f"   瓶颈检测器工作正常: {len(bottleneck_history)} 次分析")
    print(f"   性能导向分裂器工作正常: {division_summary.get('total_divisions', 0)} 次分裂")
    print(f"   最终准确率: {val_acc:.2f}%")
    
    # 清理资源
    activation_hook.remove_hooks()

if __name__ == "__main__":
    main()