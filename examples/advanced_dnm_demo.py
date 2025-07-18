#!/usr/bin/env python3
"""
高级DNM形态发生演示
Advanced DNM Morphogenesis Demo

🧬 演示内容：
1. 串行分裂 (Serial Division) - 增加网络深度，提升表达能力
2. 并行分裂 (Parallel Division) - 创建多分支结构，增强特征多样性  
3. 混合分裂 (Hybrid Division) - 组合不同层类型，探索复杂架构
4. 智能瓶颈识别和决策制定
5. 性能对比分析

🎯 目标：在CIFAR-10上实现90%+准确率
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import logging

# 导入增强的DNM组件
from neuroexapt.core import (
    EnhancedDNMFramework,
    AdvancedBottleneckAnalyzer,
    AdvancedMorphogenesisExecutor,
    IntelligentMorphogenesisDecisionMaker,
    MorphogenesisType,
    MorphogenesisDecision
)

# 配置日志
logging.basicConfig(level=logging.INFO)

class AdaptiveResNet(nn.Module):
    """增强的自适应ResNet - 冲刺95%准确率"""
    
    def __init__(self, num_classes=10):
        super(AdaptiveResNet, self).__init__()
        
        # 🚀 增强的初始特征提取
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)  # 增加初始通道数
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 🚀 更深的特征提取网络
        self.feature_block1 = self._make_resnet_block(64, 128, 2, 2)    # 2个残差块
        self.feature_block2 = self._make_resnet_block(128, 256, 2, 2)   # 2个残差块  
        self.feature_block3 = self._make_resnet_block(256, 512, 2, 2)   # 2个残差块
        self.feature_block4 = self._make_resnet_block(512, 512, 1, 2)   # 2个残差块，不降采样
        
        # 🚀 增强的全局特征聚合
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # 🚀 更强的分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2, 1024),  # 结合avg和max pooling
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, num_classes)
        )
        
    def _make_resnet_block(self, in_channels, out_channels, stride, num_blocks):
        """创建残差块组"""
        layers = []
        
        # 第一个块可能有降采样
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # 后续块保持相同尺寸
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 🚀 增强的初始特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 🚀 深度残差特征提取
        x = self.feature_block1(x)
        x = self.feature_block2(x)
        x = self.feature_block3(x)
        x = self.feature_block4(x)
        
        # 🚀 双重全局池化特征聚合
        avg_pool = self.global_pool(x)
        max_pool = self.global_max_pool(x)
        x = torch.cat([avg_pool, max_pool], dim=1)  # 特征融合
        
        # 分类
        x = self.classifier(x)
        
        return x

class ResidualBlock(nn.Module):
    """残差块实现"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = self.relu(out)
        
        return out

class AdvancedDNMTrainer:
    """高级DNM训练器"""
    
    def __init__(self, model, device, train_loader, test_loader):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # 🚀 增强的DNM框架配置 - 冲刺95%准确率
        self.dnm_config = {
            'trigger_interval': 8,  # 每8个epoch检查一次，更稳定
            'complexity_threshold': 0.5,  # 降低阈值，更容易触发
            'enable_serial_division': True,
            'enable_parallel_division': True,
            'enable_hybrid_division': True,
            'max_parameter_growth_ratio': 3.0  # 允许更多参数增长
        }
        
        self.dnm_framework = EnhancedDNMFramework(self.dnm_config)
        
        # 训练历史
        self.train_history = []
        self.test_history = []
        self.morphogenesis_history = []
        self.parameter_history = []
        
    def capture_network_state(self):
        """捕获网络状态（激活值和梯度）"""
        activations = {}
        gradients = {}
        
        # 注册钩子函数
        def forward_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach().cpu()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    gradients[name] = grad_output[0].detach().cpu()
            return hook
        
        # 注册钩子
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(forward_hook(name)))
                hooks.append(module.register_backward_hook(backward_hook(name)))
        
        # 执行一次前向和反向传播
        self.model.train()
        data, target = next(iter(self.train_loader))
        data, target = data.to(self.device), target.to(self.device)
        
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        
        # 清空之前的梯度
        self.model.zero_grad()
        loss.backward()
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return activations, gradients
    
    def train_epoch(self, optimizer, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'    Train Batch: {batch_idx:3d}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.6f}, Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def test_epoch(self):
        """测试一个epoch"""
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        
        return test_loss, accuracy
    
    def train_with_morphogenesis(self, epochs=80):  # 🚀 增加到80个epoch
        """带形态发生的训练 - 冲刺95%准确率"""
        print("🧬 开始高级DNM训练 - 冲刺95%准确率...")
        print("=" * 60)
        
        # 🚀 增强的优化器配置
        # 使用SGD + Momentum，对CIFAR-10更有效
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=0.1,              # 较高的初始学习率
            momentum=0.9,        # 强动量
            weight_decay=5e-4,   # 适中的权重衰减
            nesterov=True        # Nesterov动量
        )
        
        # 🚀 多阶段学习率调度
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[30, 60, 75],  # 在30, 60, 75 epoch降低学习率
            gamma=0.1                 # 每次降低10倍
        )
        
        # 记录初始参数数量
        initial_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 初始参数数量: {initial_params:,}")
        self.parameter_history.append(initial_params)
        
        best_test_acc = 0.0
        patience_counter = 0
        
        # 🚀 添加学习率预热
        warmup_epochs = 5
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=warmup_epochs
        )
        
        for epoch in range(epochs):
            print(f"\n🧬 Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(optimizer, epoch)
            
            # 测试
            test_loss, test_acc = self.test_epoch()
            
            # 更新历史
            self.train_history.append((train_loss, train_acc))
            self.test_history.append((test_loss, test_acc))
            self.dnm_framework.update_performance_history(test_acc / 100.0)
            
            print(f"  📊 Train: {train_acc:.2f}% (Loss: {train_loss:.4f}) | "
                  f"Test: {test_acc:.2f}% (Loss: {test_loss:.4f})")
            
            # 🚀 智能学习率调度
            if epoch < warmup_epochs:
                warmup_scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  🔥 预热阶段: LR={current_lr:.6f}")
            else:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  📈 当前学习率: {current_lr:.6f}")
            
            # 检查是否需要形态发生
            if epoch >= 10:  # 让网络稳定训练更长时间
                activations, gradients = self.capture_network_state()
                
                context = {
                    'epoch': epoch,
                    'activations': activations,
                    'gradients': gradients,
                    'performance_history': self.dnm_framework.performance_history
                }
                
                # 执行形态发生
                results = self.dnm_framework.execute_morphogenesis(self.model, context)
                
                if results['model_modified']:
                    print(f"  🎉 形态发生成功!")
                    print(f"    类型: {results['morphogenesis_type']}")
                    print(f"    新增参数: {results['parameters_added']:,}")
                    print(f"    置信度: {results.get('decision_confidence', 0):.3f}")
                    
                    # 更新模型
                    self.model = results['new_model']
                    
                    # 🚀 重新创建优化器以包含新参数，保持当前学习率
                    current_lr = optimizer.param_groups[0]['lr']
                    optimizer = optim.SGD(
                        self.model.parameters(), 
                        lr=current_lr,
                        momentum=0.9,
                        weight_decay=5e-4,
                        nesterov=True
                    )
                    
                    # 重新创建调度器
                    remaining_epochs = epochs - epoch
                    if remaining_epochs > 0:
                        milestones = [m - epoch for m in [30, 60, 75] if m > epoch]
                        if milestones:
                            scheduler = optim.lr_scheduler.MultiStepLR(
                                optimizer, milestones=milestones, gamma=0.1
                            )
                    
                    # 记录形态发生事件
                    current_params = sum(p.numel() for p in self.model.parameters())
                    self.parameter_history.append(current_params)
                    
                    self.morphogenesis_history.append({
                        'epoch': epoch,
                        'type': results['morphogenesis_type'],
                        'parameters_added': results['parameters_added'],
                        'test_acc_before': test_acc,
                        'total_params': current_params
                    })
                    
                    print(f"    总参数: {current_params:,} "
                          f"(+{((current_params-initial_params)/initial_params*100):.1f}%)")
                else:
                    # 没有形态发生时也记录参数数量
                    current_params = sum(p.numel() for p in self.model.parameters())
                    self.parameter_history.append(current_params)
            else:
                # 前几个epoch也记录参数数量
                current_params = sum(p.numel() for p in self.model.parameters())
                self.parameter_history.append(current_params)
            
            # 🚀 性能监控和早停检查
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
                print(f"  🎯 新的最佳准确率: {best_test_acc:.2f}%!")
                
                # 🏆 里程碑提示
                if best_test_acc >= 95.0:
                    print("  🏆 恭喜！达到95%+准确率目标!")
                elif best_test_acc >= 90.0:
                    print("  🌟 很好！达到90%+准确率!")
                elif best_test_acc >= 85.0:
                    print("  ✨ 不错！达到85%+准确率!")
            else:
                patience_counter += 1
                
            # 增加耐心值，给更多时间训练
            if patience_counter >= 15:
                print(f"  🛑 Early stopping triggered at epoch {epoch+1}")
                break
                
            # 🚀 进度提示
            progress = (epoch + 1) / epochs * 100
            if progress % 25 == 0:
                print(f"  📊 训练进度: {progress:.0f}% 完成")
        
        print(f"\n✅ 训练完成!")
        print(f"📊 最佳测试准确率: {best_test_acc:.2f}%")
        
        return best_test_acc
    
    def analyze_morphogenesis_effects(self):
        """分析形态发生效果"""
        print("\n🔬 形态发生效果分析")
        print("=" * 50)
        
        summary = self.dnm_framework.get_morphogenesis_summary()
        
        print(f"📊 总体统计:")
        print(f"  形态发生事件: {summary['total_events']}")
        print(f"  新增参数: {summary['total_parameters_added']:,}")
        print(f"  形态发生类型分布: {summary['morphogenesis_types']}")
        
        if self.morphogenesis_history:
            print(f"\n📈 性能改进分析:")
            
            for i, event in enumerate(self.morphogenesis_history):
                # 计算形态发生后的性能变化
                epoch = event['epoch']
                if epoch + 5 < len(self.test_history):
                    acc_before = event['test_acc_before']
                    acc_after = self.test_history[epoch + 5][1]  # 5个epoch后的准确率
                    improvement = acc_after - acc_before
                    
                    print(f"  事件 {i+1} (Epoch {epoch}):")
                    print(f"    类型: {event['type']}")
                    print(f"    新增参数: {event['parameters_added']:,}")
                    print(f"    性能变化: {acc_before:.2f}% → {acc_after:.2f}% "
                          f"({improvement:+.2f}%)")
        
        return summary
    
    def plot_training_progress(self):
        """绘制训练进度图"""
        if len(self.train_history) == 0:
            return
            
        epochs = range(1, len(self.train_history) + 1)
        train_accs = [acc for _, acc in self.train_history]
        test_accs = [acc for _, acc in self.test_history]
        
        # 确保参数历史长度与epoch匹配
        param_history_aligned = self.parameter_history[:len(self.train_history)]
        if len(param_history_aligned) < len(self.train_history):
            # 如果参数历史不够长，用最后一个值填充
            last_param = param_history_aligned[-1] if param_history_aligned else 0
            param_history_aligned.extend([last_param] * (len(self.train_history) - len(param_history_aligned)))
        
        plt.figure(figsize=(15, 5))
        
        # 准确率曲线
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_accs, label='Train Accuracy', color='blue')
        plt.plot(epochs, test_accs, label='Test Accuracy', color='red')
        
        # 标记形态发生事件
        for event in self.morphogenesis_history:
            if event['epoch'] <= len(self.train_history):
                plt.axvline(x=event['epoch'], color='green', linestyle='--', alpha=0.7)
                plt.text(event['epoch'], max(test_accs) * 0.9, 
                        event['type'].split('_')[0], rotation=90, fontsize=8)
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 参数增长曲线
        plt.subplot(1, 3, 2)
        param_growth = [(p - param_history_aligned[0]) / param_history_aligned[0] * 100 
                       for p in param_history_aligned]
        plt.plot(epochs, param_growth, color='purple', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Parameter Growth (%)')
        plt.title('Parameter Growth')
        plt.grid(True, alpha=0.3)
        
        # 形态发生类型分布
        plt.subplot(1, 3, 3)
        if self.morphogenesis_history:
            types = [event['type'] for event in self.morphogenesis_history]
            type_counts = {t: types.count(t) for t in set(types)}
            
            plt.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
            plt.title('Morphogenesis Types')
        else:
            plt.text(0.5, 0.5, 'No Morphogenesis\nEvents', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Morphogenesis Types')
        
        plt.tight_layout()
        plt.savefig('advanced_dnm_training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()

def prepare_data():
    """准备CIFAR-10数据 - 增强版数据增强策略"""
    # 🚀 强化数据增强策略 - 冲刺95%准确率
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),  # 随机高斯模糊
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))  # 随机擦除
    ])
    
    # 测试时使用标准预处理
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # 增加批次大小以提升训练效率
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

def compare_with_fixed_architecture():
    """与固定架构进行对比"""
    print("\n⚖️ 对比固定架构 vs 自适应架构")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = prepare_data()
    
    # 1. 训练固定架构
    print("🔧 训练固定架构...")
    fixed_model = AdaptiveResNet()
    fixed_trainer = AdvancedDNMTrainer(fixed_model, device, train_loader, test_loader)
    
    # 禁用形态发生
    fixed_trainer.dnm_framework.config['trigger_interval'] = 999  # 永不触发
    
    fixed_acc = fixed_trainer.train_with_morphogenesis(epochs=30)
    
    # 2. 训练自适应架构
    print("\n🧬 训练自适应架构...")
    adaptive_model = AdaptiveResNet()
    adaptive_trainer = AdvancedDNMTrainer(adaptive_model, device, train_loader, test_loader)
    
    adaptive_acc = adaptive_trainer.train_with_morphogenesis(epochs=30)
    
    # 3. 分析结果
    print("\n📊 对比结果:")
    print(f"  固定架构最佳准确率: {fixed_acc:.2f}%")
    print(f"  自适应架构最佳准确率: {adaptive_acc:.2f}%")
    print(f"  性能提升: {adaptive_acc - fixed_acc:+.2f}%")
    
    # 分析形态发生效果
    adaptive_summary = adaptive_trainer.analyze_morphogenesis_effects()
    
    return fixed_acc, adaptive_acc, adaptive_summary

def demonstrate_morphogenesis_types():
    """演示不同形态发生类型"""
    print("\n🎭 演示不同形态发生类型")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试每种形态发生类型
    morphogenesis_types = [
        MorphogenesisType.SERIAL_DIVISION,
        MorphogenesisType.PARALLEL_DIVISION,
        MorphogenesisType.HYBRID_DIVISION
    ]
    
    results = {}
    
    for morph_type in morphogenesis_types:
        print(f"\n🔬 测试 {morph_type.value}...")
        
        model = AdaptiveResNet().to(device)
        original_params = sum(p.numel() for p in model.parameters())
        
        # 创建决策
        decision = MorphogenesisDecision(
            morphogenesis_type=morph_type,
            target_location='classifier.1',
            confidence=0.8,
            expected_improvement=0.05,
            complexity_cost=0.3,
            parameters_added=5000,
            reasoning=f"演示{morph_type.value}"
        )
        
        # 执行形态发生
        executor = AdvancedMorphogenesisExecutor()
        try:
            new_model, params_added = executor.execute_morphogenesis(model, decision)
            new_params = sum(p.numel() for p in new_model.parameters())
            
            # 测试功能
            test_input = torch.randn(4, 3, 32, 32).to(device)
            with torch.no_grad():
                output = new_model(test_input)
            
            results[morph_type.value] = {
                'success': True,
                'original_params': original_params,
                'new_params': new_params,
                'params_added': params_added,
                'growth_ratio': (new_params - original_params) / original_params,
                'output_shape': output.shape
            }
            
            print(f"  ✅ 成功")
            print(f"    原始参数: {original_params:,}")
            print(f"    新增参数: {params_added:,}")
            print(f"    总参数: {new_params:,}")
            print(f"    增长率: {results[morph_type.value]['growth_ratio']:.1%}")
            
        except Exception as e:
            results[morph_type.value] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ❌ 失败: {e}")
    
    return results

def main():
    """主演示函数"""
    print("🧬 高级DNM形态发生演示")
    print("=" * 60)
    
    try:
        # 1. 演示不同形态发生类型
        morphogenesis_results = demonstrate_morphogenesis_types()
        
        # 2. 完整训练演示
        print(f"\n🚀 完整训练演示")
        print("=" * 50)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 使用设备: {device}")
        
        # 准备数据
        train_loader, test_loader = prepare_data()
        
        # 创建模型和训练器
        model = AdaptiveResNet()
        trainer = AdvancedDNMTrainer(model, device, train_loader, test_loader)
        
        # 🚀 冲刺95%准确率 - 增加训练轮数
        best_acc = trainer.train_with_morphogenesis(epochs=80)
        
        # 分析结果
        summary = trainer.analyze_morphogenesis_effects()
        
        # 绘制图表
        trainer.plot_training_progress()
        
        print(f"\n🎉 演示完成!")
        print("=" * 60)
        
        print(f"\n📊 最终结果:")
        print(f"  最佳测试准确率: {best_acc:.2f}%")
        print(f"  形态发生事件: {summary['total_events']}")
        print(f"  新增参数: {summary['total_parameters_added']:,}")
        print(f"  支持的形态发生类型: {len([r for r in morphogenesis_results.values() if r['success']])}/3")
        
        if best_acc >= 95.0:
            print("  🏆 恭喜！成功达到95%+准确率目标!")
        elif best_acc >= 90.0:
            print("  🌟 很好！达到90%+准确率，接近目标!")
        elif best_acc >= 85.0:
            print("  ✨ 不错！达到85%+准确率，继续优化中...")
        elif summary['total_events'] > 0:
            print("  🔧 形态发生功能正常，需要更多训练时间")
        else:
            print("  ⚠️ 建议调整触发阈值以激活更多形态发生")
            
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()