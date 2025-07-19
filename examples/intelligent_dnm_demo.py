#!/usr/bin/env python3
"""
智能瓶颈驱动的DNM形态发生演示
Intelligent Bottleneck-Driven DNM Morphogenesis Demo

🧬 演示内容：
1. 智能瓶颈检测 - 无需固定间隔，实时监控网络瓶颈
2. Net2Net输出反向投影分析 - 检测哪一层阻碍了准确率提升
3. 多优先级决策制定 - 基于瓶颈严重程度和改进潜力
4. 精确的形态发生策略 - 针对性地解决特定瓶颈

🎯 目标：让神经网络像活过来一样自适应生长，突破性能瓶颈
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

class IntelligentResNet(nn.Module):
    """智能自适应ResNet - 专为演示瓶颈检测设计"""
    
    def __init__(self, num_classes=10):
        super(IntelligentResNet, self).__init__()
        
        # 故意设计一些瓶颈来演示智能检测
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)  # 较小的初始通道数
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # 特意创建深度瓶颈
        self.shallow_block = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 16x16
        )
        
        # 特意创建宽度瓶颈 - 通道数过少
        self.narrow_block = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),  # 减少通道数
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 8x8
        )
        
        # 信息流瓶颈 - 单一路径处理
        self.bottleneck_conv = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bottleneck_bn = nn.BatchNorm2d(64)
        
        # 分类器瓶颈 - 较小的隐藏层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),  # 很小的隐藏层
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # 初始特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 浅层处理 (深度瓶颈)
        x = self.shallow_block(x)
        
        # 窄通道处理 (宽度瓶颈) 
        x = self.narrow_block(x)
        
        # 单路径处理 (信息流瓶颈)
        x = self.relu(self.bottleneck_bn(self.bottleneck_conv(x)))
        
        # 分类 (容量瓶颈)
        x = self.classifier(x)
        
        return x

class IntelligentDNMTrainer:
    """智能DNM训练器 - 展示智能瓶颈检测"""
    
    def __init__(self, model, device, train_loader, test_loader):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # 🧠 智能DNM框架配置 - 无固定间隔触发
        self.dnm_config = {
            # 移除固定触发间隔，采用智能检测
            'trigger_interval': 1,  # 每轮都检查，但由智能算法决定是否触发
            'complexity_threshold': 0.3,  # 降低阈值，更敏感
            'enable_serial_division': True,
            'enable_parallel_division': True,
            'enable_hybrid_division': True,
            'max_parameter_growth_ratio': 5.0,  # 允许更大增长
            
            # 🧠 智能瓶颈检测配置
            'enable_intelligent_bottleneck_detection': True,
            'bottleneck_severity_threshold': 0.5,  # 瓶颈严重程度阈值
            'stagnation_threshold': 0.005,  # 0.5% 停滞阈值
            'net2net_improvement_threshold': 0.3,  # Net2Net改进潜力阈值
            
                         # 激进模式配置 (作为备用) - 暂时关闭避免导入问题
             'enable_aggressive_mode': False,
             'accuracy_plateau_threshold': 0.001,
             'plateau_detection_window': 3,
             'aggressive_trigger_accuracy': 0.88,
             'max_concurrent_mutations': 2,
             'morphogenesis_budget': 15000
        }
        
        self.dnm_framework = EnhancedDNMFramework(self.dnm_config)
        
        # 训练历史
        self.train_history = []
        self.test_history = []
        self.morphogenesis_history = []
        self.parameter_history = []
        self.bottleneck_history = []  # 记录瓶颈检测历史
        
    def capture_network_state(self):
        """捕获网络状态用于瓶颈分析"""
        print("      🔍 智能网络状态捕获...")
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
        
        # 执行前向和反向传播
        try:
            self.model.train()
            data, target = next(iter(self.train_loader))
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            
            self.model.zero_grad()
            loss.backward()
            
        except Exception as e:
            print(f"        ❌ 状态捕获失败: {e}")
        
        # 清理钩子
        for hook in hooks:
            hook.remove()
        
        print(f"        ✅ 捕获完成: {len(activations)}个激活, {len(gradients)}个梯度")
        return activations, gradients, target.detach().cpu()
    
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
                print(f'    Batch: {batch_idx:3d}/{len(self.train_loader)}, '
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
    
    def train_with_intelligent_morphogenesis(self, epochs=50):
        """带智能形态发生的训练"""
        print("🧠 开始智能DNM训练...")
        print("=" * 60)
        
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        
        # 记录初始参数数量
        initial_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 初始模型: {initial_params:,} 参数")
        self.parameter_history.append(initial_params)
        
        best_test_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\n🧠 Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(optimizer, epoch)
            
            # 测试
            test_loss, test_acc = self.test_epoch()
            
            # 更新历史
            self.train_history.append((train_loss, train_acc))
            self.test_history.append((test_loss, test_acc))
            self.dnm_framework.update_performance_history(test_acc / 100.0)
            
            print(f"  📊 Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
            
            # 智能形态发生检查 - 每轮都检查但由算法决定
            print(f"  🧠 智能瓶颈检测分析...")
            print(f"    📊 当前模型: {sum(p.numel() for p in self.model.parameters()):,} 参数")
            
            try:
                # 捕获网络状态
                activations, gradients, targets = self.capture_network_state()
                
                # 构建分析上下文
                context = {
                    'epoch': epoch,
                    'activations': activations,
                    'gradients': gradients,
                    'performance_history': self.dnm_framework.performance_history,
                    'targets': targets
                }
                
                # 执行智能形态发生
                print("  🚀 执行智能形态发生分析...")
                results = self.dnm_framework.execute_morphogenesis(
                    model=self.model,
                    activations_or_context=context,
                    gradients=None,  # context中已包含
                    performance_history=None,  # context中已包含
                    epoch=None,  # context中已包含
                    targets=targets
                )
                
                print(f"    ✅ 分析完成: 模型{'已修改' if results.get('model_modified', False) else '未修改'}")
                
                # 记录瓶颈分析历史
                bottleneck_info = {
                    'epoch': epoch,
                    'model_modified': results.get('model_modified', False),
                    'morphogenesis_type': results.get('morphogenesis_type', 'none'),
                    'trigger_reasons': results.get('trigger_reasons', []),
                    'intelligent_decision': results.get('intelligent_decision', False)
                }
                self.bottleneck_history.append(bottleneck_info)
                
                if results['model_modified']:
                    print(f"  🎉 智能形态发生触发!")
                    print(f"    类型: {results['morphogenesis_type']}")
                    print(f"    新增参数: {results['parameters_added']:,}")
                    print(f"    触发原因:")
                    for reason in results.get('trigger_reasons', []):
                        print(f"      • {reason}")
                    
                    # 更新模型
                    old_param_count = sum(p.numel() for p in self.model.parameters())
                    self.model = results['new_model']
                    new_param_count = sum(p.numel() for p in self.model.parameters())
                    
                    print(f"    📈 参数增长: {old_param_count:,} → {new_param_count:,}")
                    
                    # 重建优化器
                    current_lr = optimizer.param_groups[0]['lr']
                    optimizer = optim.SGD(
                        self.model.parameters(), 
                        lr=current_lr,
                        momentum=0.9,
                        weight_decay=1e-4
                    )
                    
                    # 记录形态发生事件
                    morphogenesis_event = {
                        'epoch': epoch,
                        'type': results['morphogenesis_type'],
                        'parameters_added': results['parameters_added'],
                        'test_acc_before': test_acc,
                        'total_params': new_param_count,
                        'trigger_reasons': results.get('trigger_reasons', []),
                        'intelligent': results.get('intelligent_decision', False)
                    }
                    self.morphogenesis_history.append(morphogenesis_event)
                else:
                    print(f"  ✅ 瓶颈检测: 当前无需形态发生")
                    if results.get('trigger_reasons'):
                        print(f"    未触发原因: {results.get('trigger_reasons', [])}")
                
            except Exception as e:
                print(f"    ❌ 智能分析失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 记录参数历史
            current_params = sum(p.numel() for p in self.model.parameters())
            self.parameter_history.append(current_params)
            
            # 更新学习率
            scheduler.step()
            
            # 性能监控
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                print(f"  🎯 新最佳准确率: {best_test_acc:.2f}%!")
                
                if best_test_acc >= 95.0:
                    print("  🏆 达到95%+准确率目标!")
                elif best_test_acc >= 90.0:
                    print("  🌟 达到90%+准确率!")
        
        print(f"\n✅ 智能训练完成!")
        print(f"📊 最佳测试准确率: {best_test_acc:.2f}%")
        
        return best_test_acc
    
    def analyze_intelligent_morphogenesis(self):
        """分析智能形态发生效果"""
        print("\n🧠 智能形态发生分析")
        print("=" * 50)
        
        # 总体统计
        total_events = len(self.morphogenesis_history)
        intelligent_events = len([e for e in self.morphogenesis_history if e.get('intelligent', False)])
        total_params_added = sum(e['parameters_added'] for e in self.morphogenesis_history)
        
        print(f"📊 智能形态发生统计:")
        print(f"  总事件数: {total_events}")
        print(f"  智能决策: {intelligent_events}")
        print(f"  传统决策: {total_events - intelligent_events}")
        print(f"  新增参数: {total_params_added:,}")
        
        # 触发原因分析
        if self.morphogenesis_history:
            print(f"\n🎯 触发原因分析:")
            all_reasons = []
            for event in self.morphogenesis_history:
                all_reasons.extend(event.get('trigger_reasons', []))
            
            reason_types = defaultdict(int)
            for reason in all_reasons:
                if 'Net2Net' in reason:
                    reason_types['Net2Net建议'] += 1
                elif '瓶颈检测' in reason:
                    reason_types['瓶颈检测'] += 1
                elif '停滞' in reason:
                    reason_types['性能停滞'] += 1
                elif '激进' in reason:
                    reason_types['激进模式'] += 1
                else:
                    reason_types['其他'] += 1
            
            for reason_type, count in reason_types.items():
                print(f"  {reason_type}: {count}次")
        
        # 瓶颈检测效果分析
        print(f"\n🔬 瓶颈检测效果:")
        detection_cycles = len(self.bottleneck_history)
        triggered_cycles = len([b for b in self.bottleneck_history if b['model_modified']])
        detection_rate = triggered_cycles / detection_cycles if detection_cycles > 0 else 0
        
        print(f"  检测周期: {detection_cycles}")
        print(f"  触发周期: {triggered_cycles}")
        print(f"  触发率: {detection_rate:.1%}")
        
        # 性能改进分析
        if self.morphogenesis_history:
            print(f"\n📈 性能改进分析:")
            for i, event in enumerate(self.morphogenesis_history):
                epoch = event['epoch']
                acc_before = event['test_acc_before']
                
                # 查找5轮后的准确率
                if epoch + 5 < len(self.test_history):
                    acc_after = self.test_history[epoch + 5][1]
                    improvement = acc_after - acc_before
                    
                    print(f"  事件 {i+1} (Epoch {epoch}):")
                    print(f"    类型: {event['type']}")
                    print(f"    智能决策: {'是' if event.get('intelligent', False) else '否'}")
                    print(f"    性能变化: {acc_before:.2f}% → {acc_after:.2f}% ({improvement:+.2f}%)")
                    print(f"    主要原因: {event.get('trigger_reasons', ['未知'])[0]}")
        
        return {
            'total_events': total_events,
            'intelligent_events': intelligent_events,
            'total_parameters_added': total_params_added,
            'detection_rate': detection_rate,
            'reason_distribution': dict(reason_types) if 'reason_types' in locals() else {}
        }
    
    def plot_intelligent_training_progress(self):
        """绘制智能训练进度图"""
        if len(self.train_history) == 0:
            return
            
        epochs = range(1, len(self.train_history) + 1)
        train_accs = [acc for _, acc in self.train_history]
        test_accs = [acc for _, acc in self.test_history]
        
        plt.figure(figsize=(18, 6))
        
        # 准确率曲线
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_accs, label='Train Accuracy', color='blue', alpha=0.7)
        plt.plot(epochs, test_accs, label='Test Accuracy', color='red', linewidth=2)
        
        # 标记智能形态发生事件
        for event in self.morphogenesis_history:
            if event['epoch'] <= len(self.train_history):
                color = 'green' if event.get('intelligent', False) else 'orange'
                style = '-' if event.get('intelligent', False) else '--'
                plt.axvline(x=event['epoch'], color=color, linestyle=style, alpha=0.8)
                
                # 添加事件标签
                plt.text(event['epoch'], max(test_accs) * 0.95, 
                        '🧠' if event.get('intelligent', False) else '🔄', 
                        rotation=0, fontsize=12, ha='center')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Intelligent DNM Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 参数增长曲线
        plt.subplot(1, 3, 2)
        param_growth = [(p - self.parameter_history[0]) / self.parameter_history[0] * 100 
                       for p in self.parameter_history[:len(epochs)]]
        plt.plot(epochs, param_growth, color='purple', linewidth=2)
        
        # 标记智能vs传统决策
        for event in self.morphogenesis_history:
            if event['epoch'] <= len(epochs):
                color = 'green' if event.get('intelligent', False) else 'orange'
                marker = 'o' if event.get('intelligent', False) else 's'
                epoch_idx = event['epoch']
                if epoch_idx < len(param_growth):
                    plt.scatter(epoch_idx, param_growth[epoch_idx], 
                              color=color, marker=marker, s=100, alpha=0.8)
        
        plt.xlabel('Epoch')
        plt.ylabel('Parameter Growth (%)')
        plt.title('Intelligent Parameter Growth')
        plt.grid(True, alpha=0.3)
        
        # 触发原因分布
        plt.subplot(1, 3, 3)
        if self.morphogenesis_history:
            all_reasons = []
            for event in self.morphogenesis_history:
                for reason in event.get('trigger_reasons', []):
                    if 'Net2Net' in reason:
                        all_reasons.append('Net2Net')
                    elif '瓶颈' in reason:
                        all_reasons.append('瓶颈检测')
                    elif '停滞' in reason:
                        all_reasons.append('停滞检测')
                    elif '激进' in reason:
                        all_reasons.append('激进模式')
                    else:
                        all_reasons.append('其他')
            
            if all_reasons:
                reason_counts = defaultdict(int)
                for reason in all_reasons:
                    reason_counts[reason] += 1
                
                colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
                plt.pie(reason_counts.values(), labels=reason_counts.keys(), 
                       colors=colors[:len(reason_counts)], autopct='%1.1f%%')
                plt.title('Trigger Reason Distribution')
            else:
                plt.text(0.5, 0.5, 'No Morphogenesis\nEvents', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Trigger Reason Distribution')
        
        plt.tight_layout()
        plt.savefig('intelligent_dnm_training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()

def prepare_data():
    """准备CIFAR-10数据"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
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
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def main():
    """主演示函数"""
    print("🧠 智能瓶颈驱动的DNM形态发生演示")
    print("=" * 60)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 使用设备: {device}")
        
        # 准备数据
        train_loader, test_loader = prepare_data()
        
        # 创建故意有瓶颈的模型
        model = IntelligentResNet()
        trainer = IntelligentDNMTrainer(model, device, train_loader, test_loader)
        
        print(f"\n📊 初始模型架构分析:")
        print(f"  总参数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  设计瓶颈:")
        print(f"    - 深度瓶颈: 浅层网络设计")
        print(f"    - 宽度瓶颈: 过窄的通道数")
        print(f"    - 信息流瓶颈: 单一处理路径")
        print(f"    - 容量瓶颈: 小分类器隐藏层")
        
        # 智能训练
        print(f"\n🧠 开始智能瓶颈驱动训练...")
        best_acc = trainer.train_with_intelligent_morphogenesis(epochs=30)
        
        # 分析结果
        summary = trainer.analyze_intelligent_morphogenesis()
        
        # 绘制图表
        trainer.plot_intelligent_training_progress()
        
        print(f"\n🎉 智能演示完成!")
        print("=" * 60)
        
        print(f"\n📊 最终结果:")
        print(f"  最佳准确率: {best_acc:.2f}%")
        print(f"  形态发生事件: {summary['total_events']}")
        print(f"  智能决策: {summary['intelligent_events']}")
        print(f"  新增参数: {summary['total_parameters_added']:,}")
        print(f"  智能触发率: {summary['detection_rate']:.1%}")
        
        print(f"\n🧠 智能特性展示:")
        print(f"  ✅ 无固定间隔限制 - 实时瓶颈检测")
        print(f"  ✅ Net2Net输出分析 - 精确定位问题层")
        print(f"  ✅ 多优先级决策 - 智能选择最优策略")
        print(f"  ✅ 自适应生长 - 网络像活过来一样进化")
        
        if best_acc >= 90.0:
            print(f"  🏆 成功突破瓶颈，达到高准确率!")
        elif summary['total_events'] > 0:
            print(f"  🔧 智能检测正常工作，网络在持续进化")
        else:
            print(f"  ⚠️ 未触发形态发生，可能需要调整敏感度")
            
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()