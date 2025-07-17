#!/usr/bin/env python3
"""
CIFAR-10 DNM Benchmark - 目标准确率 95%

使用DNM框架在CIFAR-10数据集上进行基准测试
初始架构采用ResNet-18变体，通过DNM演化实现95%准确率突破
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
import sys
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入DNM框架
from neuroexapt.core.dnm_framework import train_with_dnm
from neuroexapt.core.dnm_neuron_division import DNMNeuronDivision
from neuroexapt.core.dnm_connection_growth import DNMConnectionGrowth
from neuroexapt.math.pareto_optimization import DNMMultiObjectiveOptimization


# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 改为INFO级别以显示BatchNorm同步日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dnm_benchmark.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class EvolvableResNet18(nn.Module):
    """可演化的ResNet-18架构 - 专门设计用于DNM演化"""
    
    def __init__(self, num_classes=10, initial_channels=64):
        super().__init__()
        self.initial_channels = initial_channels
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 可演化的残差块
        self.layer1 = self._make_layer(initial_channels, initial_channels, 2, stride=1)
        self.layer2 = self._make_layer(initial_channels, initial_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(initial_channels * 2, initial_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(initial_channels * 4, initial_channels * 8, 2, stride=2)
        
        # 可演化的分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(initial_channels * 8, initial_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(initial_channels * 4, initial_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(initial_channels * 2, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # 第一个块可能需要降采样
        layers.append(EvolvableBasicBlock(in_channels, out_channels, stride))
        
        # 其余块
        for _ in range(1, blocks):
            layers.append(EvolvableBasicBlock(out_channels, out_channels, 1))
        
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
        # 初始特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 分类
        x = self.avgpool(x)
        x = self.fc_layers(x)
        
        return x


class EvolvableBasicBlock(nn.Module):
    """可演化的基础残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 跳跃连接
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


def prepare_cifar10_data(batch_size=128, num_workers=4):
    """准备CIFAR-10数据集"""
    
    # 数据增强策略
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"📊 CIFAR-10 Dataset loaded:")
    print(f"   Training samples: {len(trainset)}")
    print(f"   Test samples: {len(testset)}")
    print(f"   Batch size: {batch_size}")
    
    return train_loader, test_loader


def create_optimized_dnm_config():
    """创建优化的DNM配置，目标95%准确率"""
    
    return {
        'neuron_division': {
            'splitter': {
                'entropy_threshold': 0.6,        # 适中的分裂阈值
                'overload_threshold': 0.5,       # 较低的过载阈值，更积极分裂
                'split_probability': 0.6,        # 提高分裂概率
                'max_splits_per_layer': 4,       # 允许更多分裂
                'inheritance_noise': 0.05        # 较小的继承噪声
            },
            'monitoring': {
                'target_layers': ['conv', 'linear'],
                'analysis_frequency': 4,         # 更频繁的分析
                'min_epoch_before_split': 8      # 更早开始分裂
            }
        },
        'connection_growth': {
            'analyzer': {
                'correlation_threshold': 0.12,   # 较低的相关性阈值
                'history_length': 10             # 更长的历史记录
            },
            'growth': {
                'max_new_connections': 4,        # 更多连接
                'min_correlation_threshold': 0.08,
                'growth_frequency': 6,           # 更频繁的连接生长
                'connection_types': ['skip_connection', 'attention_connection']
            },
            'filtering': {
                'min_layer_distance': 1,         # 允许相邻层连接
                'max_layer_distance': 8,         # 更大的连接范围
                'avoid_redundant_connections': True
            }
        },
        'multi_objective': {
            'evolution': {
                'population_size': 10,           # 适中的种群大小
                'max_generations': 12,           # 适中的代数
                'mutation_rate': 0.4,
                'crossover_rate': 0.8,
                'elitism_ratio': 0.2
            },
            'optimization': {
                'trigger_frequency': 15,         # 适中的触发频率
                'performance_plateau_threshold': 0.005,
                'min_improvement_epochs': 3
            }
        },
        'framework': {
            'morphogenesis_frequency': 4,       # 更频繁的形态发生
            'performance_tracking_window': 8,
            'early_stopping_patience': 25,
            'target_accuracy_threshold': 95.0, # 目标95%
            'enable_architecture_snapshots': True,
            'adaptive_morphogenesis': True
        }
    }


def run_cifar10_dnm_benchmark():
    """运行CIFAR-10 DNM基准测试"""
    
    print("🧬 CIFAR-10 DNM Benchmark - Target: 95% Accuracy")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # 准备数据
    train_loader, test_loader = prepare_cifar10_data(batch_size=128)
    
    # 创建可演化模型
    model = EvolvableResNet18(num_classes=10, initial_channels=64)
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️ Initial model: {initial_params:,} parameters")
    
    # 创建优化的DNM配置
    dnm_config = create_optimized_dnm_config()
    
    # 创建优化器和学习率调度器
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print(f"\n🚀 Starting DNM Training (Target: 95% accuracy)")
    print("=" * 60)
    
    # 记录开始时间
    start_time = time.time()
    
    # 回调函数
    def progress_callback(dnm_framework, model, epoch_record):
        epoch = epoch_record['epoch']
        val_acc = epoch_record['val_acc']
        train_acc = epoch_record['train_acc']
        params = epoch_record['model_params']
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        if epoch % 5 == 0:
            param_growth = (params - initial_params) / initial_params * 100
            print(f"📈 Epoch {epoch}: Train={train_acc:.2f}%, Val={val_acc:.2f}%, "
                  f"Params={params:,} (+{param_growth:.1f}%), LR={current_lr:.6f}")
    
    # 使用DNM训练
    try:
        result = train_with_dnm(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=200,
            optimizer=optimizer,
            criterion=criterion,
            config=dnm_config,
            callbacks=[progress_callback]
        )
        
        # 训练完成统计
        training_time = time.time() - start_time
        final_params = sum(p.numel() for p in result['model'].parameters())
        param_growth = (final_params - initial_params) / initial_params * 100
        
        print("\n🎉 DNM Training Completed!")
        print("=" * 60)
        print(f"📊 Results Summary:")
        print(f"   Best Validation Accuracy: {result['best_val_accuracy']:.2f}%")
        print(f"   Final Validation Accuracy: {result['final_val_accuracy']:.2f}%")
        print(f"   Parameter Growth: +{param_growth:.1f}% ({initial_params:,} → {final_params:,})")
        print(f"   Training Time: {training_time/60:.1f} minutes")
        print(f"   Morphogenesis Events: {len(result['morphogenesis_events'])}")
        print(f"   Total Neuron Splits: {result['statistics']['total_neuron_splits']}")
        print(f"   Total Connections Grown: {result['statistics']['total_connections_grown']}")
        
        # 详细分析形态发生事件
        if result['morphogenesis_events']:
            print(f"\n🧬 Morphogenesis Analysis:")
            for i, event in enumerate(result['morphogenesis_events'][-5:]):  # 显示最后5个事件
                print(f"   Event {i+1} (Epoch {event['epoch']}):")
                print(f"     Neuron splits: {event['neuron_splits']}")
                print(f"     Connections grown: {event['connections_grown']}")
                print(f"     Performance before: {event['performance_before']:.2f}%")
        
        # 保存模型
        model_path = f"cifar10_dnm_evolved_{result['best_val_accuracy']:.1f}percent.pth"
        torch.save({
            'model_state_dict': result['model'].state_dict(),
            'config': dnm_config,
            'results': result,
            'final_accuracy': result['best_val_accuracy']
        }, model_path)
        print(f"💾 Model saved: {model_path}")
        
        # 成功标志
        if result['best_val_accuracy'] >= 95.0:
            print(f"\n🏆 SUCCESS: Achieved target 95% accuracy! ({result['best_val_accuracy']:.2f}%)")
        elif result['best_val_accuracy'] >= 90.0:
            print(f"\n🎯 GOOD: Close to target! ({result['best_val_accuracy']:.2f}%)")
        else:
            print(f"\n🔄 IMPROVING: Need more optimization ({result['best_val_accuracy']:.2f}%)")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # 运行基准测试
    result = run_cifar10_dnm_benchmark()
    
    if result:
        print(f"\n✅ Benchmark completed successfully!")
    else:
        print(f"\n❌ Benchmark failed!")