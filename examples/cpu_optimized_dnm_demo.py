#!/usr/bin/env python3
"""
CPU优化的高级DNM演示 - 解决内存问题
修复被killed的问题，优化资源使用
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import logging
import gc
import sys
import os

# 设置路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.core import (
    AdvancedBottleneckAnalyzer,
    AdvancedMorphogenesisExecutor, 
    IntelligentMorphogenesisDecisionMaker,
    EnhancedDNMFramework
)

# 配置日志
logging.basicConfig(level=logging.INFO)

class CPUOptimizedResNet(nn.Module):
    """CPU优化的ResNet - 减少参数量和计算复杂度"""
    
    def __init__(self, num_classes=10):
        super(CPUOptimizedResNet, self).__init__()
        
        # 🚀 CPU友好的初始特征提取 - 减少通道数
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)  # 64 → 32
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # 🚀 轻量级残差块组
        self.feature_block1 = self._make_resnet_block(32, 64, 2, 1)    # 减少块数
        self.feature_block2 = self._make_resnet_block(64, 128, 2, 1)   
        self.feature_block3 = self._make_resnet_block(128, 256, 2, 1)   
        
        # 🚀 简化的全局特征聚合
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 🚀 CPU友好的分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
    
    def _make_resnet_block(self, in_channels, out_channels, stride, num_blocks):
        """创建轻量级残差块组"""
        layers = []
        
        # 第一个块可能有降采样
        layers.append(LightResidualBlock(in_channels, out_channels, stride))
        
        # 后续块保持相同尺寸
        for _ in range(1, num_blocks):
            layers.append(LightResidualBlock(out_channels, out_channels, 1))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 🚀 CPU优化的前向传播
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 残差特征提取
        x = self.feature_block1(x)
        x = self.feature_block2(x)
        x = self.feature_block3(x)
        
        # 全局池化和分类
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x

class LightResidualBlock(nn.Module):
    """轻量级残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(LightResidualBlock, self).__init__()
        
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

class CPUOptimizedDNMTrainer:
    """CPU优化的DNM训练器"""
    
    def __init__(self, model, device, train_loader, test_loader):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # 🚀 CPU优化的DNM框架配置
        self.dnm_config = {
            'trigger_interval': 10,  # 增加间隔，减少触发频率
            'complexity_threshold': 0.6,  # 提高阈值，减少不必要的变异
            'enable_serial_division': True,
            'enable_parallel_division': False,  # 暂时禁用并行分裂
            'enable_hybrid_division': False,    # 暂时禁用混合分裂
            'max_parameter_growth_ratio': 1.5   # 限制参数增长
        }
        
        self.dnm_framework = EnhancedDNMFramework(self.dnm_config)
        
        # 训练历史
        self.train_history = []
        self.test_history = []
        self.parameter_history = []
        self.morphogenesis_history = []
    
    def train_epoch(self, optimizer):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            running_acc += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
            
            # 🚀 显示进度，减少内存占用
            if batch_idx % 50 == 0:
                current_acc = 100. * running_acc / total_samples
                print(f"    Train Batch: {batch_idx:3d}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.6f}, Acc: {current_acc:.2f}%")
                
                # 🚀 强制垃圾回收
                if batch_idx % 100 == 0:
                    gc.collect()
        
        avg_loss = running_loss / total_samples
        avg_acc = 100. * running_acc / total_samples
        
        return avg_loss, avg_acc
    
    def test_epoch(self):
        """测试一个epoch"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_loss /= total
        accuracy = 100. * correct / total
        
        return test_loss, accuracy
    
    def capture_network_state(self):
        """捕获网络状态 - 优化内存使用"""
        activations = {}
        gradients = {}
        
        # 🚀 只捕获关键层的状态，减少内存占用
        key_modules = ['feature_block3', 'classifier']
        
        for name, module in self.model.named_modules():
            if any(key in name for key in key_modules):
                # 简化的激活统计
                if hasattr(module, 'weight') and module.weight is not None:
                    activations[name] = {
                        'mean': float(module.weight.data.mean()),
                        'std': float(module.weight.data.std())
                    }
                    
                    if module.weight.grad is not None:
                        gradients[name] = {
                            'mean': float(module.weight.grad.mean()),
                            'std': float(module.weight.grad.std())
                        }
        
        return activations, gradients
    
    def train_with_morphogenesis(self, epochs=50):  # 减少默认轮数
        """CPU优化的形态发生训练"""
        print("🧬 开始CPU优化的DNM训练...")
        print("=" * 60)
        
        # 🚀 CPU优化的训练配置
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=0.05,             # 稍微降低学习率
            momentum=0.9,
            weight_decay=1e-4,   # 减少权重衰减
            nesterov=True
        )
        
        # 🚀 简化的学习率调度
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # 记录初始参数数量
        initial_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 初始参数数量: {initial_params:,}")
        self.parameter_history.append(initial_params)
        
        best_test_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\n🧬 Epoch {epoch+1}/{epochs}")
            
            # 训练和测试
            train_loss, train_acc = self.train_epoch(optimizer)
            test_loss, test_acc = self.test_epoch()
            
            # 记录历史
            self.train_history.append((train_loss, train_acc))
            self.test_history.append((test_loss, test_acc))
            
            print(f"  📊 Train: {train_acc:.2f}% (Loss: {train_loss:.4f}) | "
                  f"Test: {test_acc:.2f}% (Loss: {test_loss:.4f})")
            
            # 学习率调度
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  📈 当前学习率: {current_lr:.6f}")
            
            # 🚀 减少形态发生频率，节省资源
            if epoch >= 15 and epoch % self.dnm_config['trigger_interval'] == 0:
                print("  🔬 检查形态发生需求...")
                
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
                    
                    # 更新模型
                    self.model = results['new_model']
                    
                    # 重新创建优化器
                    current_lr = optimizer.param_groups[0]['lr']
                    optimizer = optim.SGD(
                        self.model.parameters(), 
                        lr=current_lr,
                        momentum=0.9,
                        weight_decay=1e-4,
                        nesterov=True
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
                    
                    # 🚀 强制垃圾回收
                    gc.collect()
                else:
                    current_params = sum(p.numel() for p in self.model.parameters())
                    self.parameter_history.append(current_params)
            else:
                current_params = sum(p.numel() for p in self.model.parameters())
                self.parameter_history.append(current_params)
            
            # 性能监控
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
                print(f"  🎯 新的最佳准确率: {best_test_acc:.2f}%!")
                
                if best_test_acc >= 95.0:
                    print("  🏆 恭喜！达到95%+准确率目标!")
                elif best_test_acc >= 90.0:
                    print("  🌟 很好！达到90%+准确率!")
                elif best_test_acc >= 85.0:
                    print("  ✨ 不错！达到85%+准确率!")
            else:
                patience_counter += 1
            
            # 早停检查
            if patience_counter >= 20:  # 增加耐心值
                print(f"  🛑 Early stopping triggered at epoch {epoch+1}")
                break
            
            # 🚀 定期垃圾回收
            if epoch % 5 == 0:
                gc.collect()
        
        print(f"\n✅ 训练完成!")
        print(f"📊 最佳测试准确率: {best_test_acc:.2f}%")
        
        return best_test_acc
    
    def analyze_morphogenesis_effects(self):
        """分析形态发生效果"""
        print("\n🔬 形态发生效果分析")
        print("=" * 50)
        
        total_events = len(self.morphogenesis_history)
        total_params_added = sum(event['parameters_added'] for event in self.morphogenesis_history)
        
        print(f"📊 总体统计:")
        print(f"  形态发生事件: {total_events}")
        print(f"  新增参数: {total_params_added:,}")
        
        if total_events > 0:
            types = [event['type'] for event in self.morphogenesis_history]
            type_counts = {t: types.count(t) for t in set(types)}
            print(f"  形态发生类型分布: {type_counts}")
            
            print(f"\n📈 性能改进分析:")
            for i, event in enumerate(self.morphogenesis_history):
                if i < len(self.morphogenesis_history) - 1:
                    next_event = self.morphogenesis_history[i + 1]
                    acc_improvement = next_event['test_acc_before'] - event['test_acc_before']
                    print(f"  事件 {i+1} (Epoch {event['epoch']}):")
                    print(f"    类型: {event['type']}")
                    print(f"    新增参数: {event['parameters_added']:,}")
                    print(f"    性能变化: {event['test_acc_before']:.2f}% → "
                          f"{next_event['test_acc_before']:.2f}% ({acc_improvement:+.2f}%)")
        
        return {
            'total_events': total_events,
            'total_parameters_added': total_params_added,
            'morphogenesis_types': type_counts if total_events > 0 else {}
        }

def prepare_cpu_optimized_data():
    """准备CPU优化的CIFAR-10数据"""
    # 🚀 适度的数据增强 - 避免过度计算
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
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
    
    # 🚀 CPU友好的数据加载配置
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128,      # 减少批次大小
        shuffle=True, 
        num_workers=2,       # 减少worker数量
        pin_memory=False     # CPU环境下关闭
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=100, 
        shuffle=False, 
        num_workers=2,
        pin_memory=False
    )
    
    return train_loader, test_loader

def main():
    """主函数"""
    try:
        print("🚀 CPU优化的高级DNM演示")
        print("=" * 60)
        
        # 设备配置
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 使用设备: {device}")
        
        # 🚀 设置CPU优化
        if device.type == 'cpu':
            torch.set_num_threads(4)  # 限制线程数
            print("🔧 CPU优化: 设置线程数为4")
        
        # 准备数据
        print("📊 准备数据...")
        train_loader, test_loader = prepare_cpu_optimized_data()
        
        # 创建模型和训练器
        print("🏗️ 创建CPU优化模型...")
        model = CPUOptimizedResNet()
        trainer = CPUOptimizedDNMTrainer(model, device, train_loader, test_loader)
        
        # 🚀 CPU友好的训练设置
        print("🏃 开始训练...")
        best_acc = trainer.train_with_morphogenesis(epochs=40)  # 减少轮数
        
        # 分析结果
        summary = trainer.analyze_morphogenesis_effects()
        
        print(f"\n🎯 最终结果:")
        print(f"  最佳准确率: {best_acc:.2f}%")
        print(f"  形态发生事件: {summary['total_events']}")
        print(f"  新增参数: {summary['total_parameters_added']:,}")
        
        if best_acc >= 90.0:
            print("  🏆 优秀！CPU环境下达到90%+准确率!")
        elif best_acc >= 85.0:
            print("  🌟 很好！CPU环境下达到85%+准确率!")
        elif summary['total_events'] > 0:
            print("  🔧 形态发生功能正常，继续优化中...")
        else:
            print("  ⚠️ 建议调整配置以激活更多形态发生")
            
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 🚀 确保清理资源
        gc.collect()
        print("🧹 资源清理完成")

if __name__ == "__main__":
    main()