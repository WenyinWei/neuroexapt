#!/usr/bin/env python3
"""
修复版DNM测试 - 解决所有导入问题

参考examples/aso_se_classification.py的数据加载逻辑
目标：CIFAR-10数据集突破95%准确率
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# 直接导入DNM模块，绕过__init__.py的问题
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接导入DNM核心模块
try:
    from neuroexapt.core.dnm_framework import DNMFramework
    from neuroexapt.core.dnm_neuron_division import DNMNeuronDivision
    from neuroexapt.core.dnm_connection_growth import DNMConnectionGrowth
    print("✅ DNM模块导入成功")
except ImportError as e:
    print(f"❌ DNM模块导入失败: {e}")
    sys.exit(1)


def setup_cifar10_data(batch_size=128, data_dir='./data'):
    """
    设置CIFAR-10数据集 - 参考aso_se_classification.py的逻辑
    包含强化的数据增强策略
    """
    print(f"📊 设置CIFAR-10数据集...")
    
    # 强化的训练数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        # 随机擦除增强
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])
    
    # 测试数据标准化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 尝试加载数据集
    try:
        # 首先尝试从本地加载
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=False, transform=test_transform
        )
        print(f"✅ 从本地加载CIFAR-10数据集")
        
    except:
        print(f"📥 本地数据不存在，开始下载...")
        # 自动下载
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform
        )
        print(f"✅ CIFAR-10数据集下载完成")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=4, pin_memory=True,
        drop_last=True  # 确保batch大小一致
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    print(f"📊 数据集信息:")
    print(f"   训练样本: {len(train_dataset):,}")
    print(f"   测试样本: {len(test_dataset):,}")
    print(f"   批次大小: {batch_size}")
    print(f"   训练批次: {len(train_loader)}")
    print(f"   测试批次: {len(test_loader)}")
    
    return train_loader, test_loader


class EnhancedCIFAR10Net(nn.Module):
    """
    增强的CIFAR-10网络 - 专为DNM演化设计
    
    采用现代架构设计原则：
    1. 残差连接
    2. 批量归一化
    3. 适当的Dropout
    4. 全局平均池化
    """
    
    def __init__(self, num_classes=10, base_channels=64):
        super().__init__()
        self.base_channels = base_channels
        
        # 输入处理
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 特征提取块
        self.block1 = self._make_block(base_channels, base_channels, stride=1)
        self.block2 = self._make_block(base_channels, base_channels * 2, stride=2)
        self.block3 = self._make_block(base_channels * 2, base_channels * 4, stride=2)
        self.block4 = self._make_block(base_channels * 4, base_channels * 8, stride=2)
        
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels * 2, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
        
    def _make_block(self, in_channels, out_channels, stride):
        """创建基础块"""
        layers = []
        
        # 主卷积
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # 第二个卷积
        layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        # 残差连接
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            shortcut = nn.Identity()
        
        return ResidualBlock(nn.Sequential(*layers), shortcut)
    
    def _initialize_weights(self):
        """权重初始化"""
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
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, main_path, shortcut):
        super().__init__()
        self.main_path = main_path
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.main_path(x)
        out += identity
        out = self.relu(out)
        return out


def create_optimized_dnm_config():
    """创建优化的DNM配置 - 针对CIFAR-10 95%目标"""
    
    return {
        'neuron_division': {
            'splitter': {
                'entropy_threshold': 0.5,        # 较低阈值，促进分裂
                'overload_threshold': 0.4,       # 较低过载阈值
                'split_probability': 0.7,        # 较高分裂概率
                'max_splits_per_layer': 3,       # 允许适量分裂
                'inheritance_noise': 0.08        # 适中的继承噪声
            },
            'monitoring': {
                'target_layers': ['conv', 'linear'],
                'analysis_frequency': 4,         # 更频繁的分析
                'min_epoch_before_split': 8      # 适中的开始时机
            }
        },
        'connection_growth': {
            'analyzer': {
                'correlation_threshold': 0.12,   # 适中的相关性阈值
                'history_length': 8              # 适中的历史长度
            },
            'growth': {
                'max_new_connections': 3,        # 适量新连接
                'min_correlation_threshold': 0.08,
                'growth_frequency': 6,           # 适中的生长频率
                'connection_types': ['skip_connection', 'attention_connection']
            },
            'filtering': {
                'min_layer_distance': 1,         # 允许相邻层连接
                'max_layer_distance': 6,         # 适中的连接范围
                'avoid_redundant_connections': True
            }
        },
        'multi_objective': {
            'evolution': {
                'population_size': 8,            # 适中的种群大小
                'max_generations': 10,           # 适中的代数
                'mutation_rate': 0.4,
                'crossover_rate': 0.7,
                'elitism_ratio': 0.2
            },
            'optimization': {
                'trigger_frequency': 15,         # 适中的触发频率
                'performance_plateau_threshold': 0.005,
                'min_improvement_epochs': 3
            }
        },
        'framework': {
            'morphogenesis_frequency': 4,       # 较频繁的形态发生
            'performance_tracking_window': 8,
            'early_stopping_patience': 25,
            'target_accuracy_threshold': 95.0, # 目标95%
            'enable_architecture_snapshots': True,
            'adaptive_morphogenesis': True
        }
    }


def run_dnm_cifar10_test():
    """运行CIFAR-10 DNM测试"""
    
    print("🧬 DNM CIFAR-10 Test - Target: 95% Accuracy")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # 数据准备
    train_loader, test_loader = setup_cifar10_data(batch_size=128)
    
    # 模型创建
    model = EnhancedCIFAR10Net(num_classes=10, base_channels=64)
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"\n🏗️ 模型信息:")
    print(f"   初始参数: {initial_params:,}")
    print(f"   初始模型大小: {initial_params * 4 / 1024 / 1024:.2f} MB")
    
    # DNM配置
    dnm_config = create_optimized_dnm_config()
    
    # 优化器和调度器
    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    
    # 余弦退火学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    # 标签平滑的交叉熵
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print(f"\n🚀 开始DNM训练")
    print("=" * 60)
    
    # 训练回调
    def training_callback(dnm_framework, model, epoch_record):
        epoch = epoch_record['epoch']
        train_acc = epoch_record['train_acc']
        val_acc = epoch_record['val_acc']
        train_loss = epoch_record['train_loss']
        val_loss = epoch_record['val_loss']
        params = epoch_record['model_params']
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 计算参数增长
        param_growth = (params - initial_params) / initial_params * 100
        
        # 定期输出详细信息
        if epoch % 3 == 0 or epoch < 10:
            print(f"📈 Epoch {epoch:3d}: "
                  f"Train Acc={train_acc:5.2f}% Loss={train_loss:.4f} | "
                  f"Val Acc={val_acc:5.2f}% Loss={val_loss:.4f} | "
                  f"Params={params:,} (+{param_growth:4.1f}%) | "
                  f"LR={current_lr:.6f}")
    
    # 开始训练
    start_time = time.time()
    
    try:
        result = DNMFramework(dnm_config).train_with_morphogenesis(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=100,
            optimizer=optimizer,
            criterion=criterion,
            callbacks=[training_callback]
        )
        
        # 结果分析
        training_time = time.time() - start_time
        final_params = sum(p.numel() for p in result['model'].parameters())
        param_growth = (final_params - initial_params) / initial_params * 100
        
        print(f"\n🎉 DNM训练完成!")
        print("=" * 60)
        print(f"📊 最终结果:")
        print(f"   最佳验证准确率: {result['best_val_accuracy']:.2f}%")
        print(f"   最终验证准确率: {result['final_val_accuracy']:.2f}%")
        print(f"   参数增长: +{param_growth:.1f}% ({initial_params:,} → {final_params:,})")
        print(f"   训练时间: {training_time/60:.1f}分钟")
        print(f"   形态发生事件: {len(result['morphogenesis_events'])}")
        print(f"   总神经元分裂: {result['statistics']['total_neuron_splits']}")
        print(f"   总连接生长: {result['statistics']['total_connections_grown']}")
        print(f"   总优化次数: {result['statistics']['total_optimizations']}")
        
        # 形态发生分析
        if result['morphogenesis_events']:
            print(f"\n🧬 形态发生事件分析:")
            for i, event in enumerate(result['morphogenesis_events'][-5:]):
                print(f"   事件 {i+1} (Epoch {event['epoch']}):")
                print(f"     神经元分裂: {event['neuron_splits']}")
                print(f"     连接生长: {event['connections_grown']}")
                print(f"     优化触发: {event['optimization_triggered']}")
                print(f"     触发前性能: {event['performance_before']:.2f}%")
        
        # 成功评估
        if result['best_val_accuracy'] >= 95.0:
            print(f"\n🏆 SUCCESS: 达到目标95%准确率! ({result['best_val_accuracy']:.2f}%)")
        elif result['best_val_accuracy'] >= 90.0:
            print(f"\n🎯 GOOD: 接近目标! ({result['best_val_accuracy']:.2f}%)")
            print(f"   建议: 调整配置参数或增加训练轮数")
        else:
            print(f"\n🔄 IMPROVING: 需要进一步优化 ({result['best_val_accuracy']:.2f}%)")
        
        # 保存最佳模型
        if result['best_val_accuracy'] >= 85.0:
            model_path = f"cifar10_dnm_{result['best_val_accuracy']:.1f}percent.pth"
            torch.save({
                'model_state_dict': result['model'].state_dict(),
                'config': dnm_config,
                'accuracy': result['best_val_accuracy'],
                'morphogenesis_events': result['morphogenesis_events']
            }, model_path)
            print(f"💾 模型已保存: {model_path}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🚀 启动DNM CIFAR-10测试")
    
    result = run_dnm_cifar10_test()
    
    if result:
        print(f"\n✅ 测试完成!")
        print(f"   DNM框架运行正常")
        print(f"   最终准确率: {result['final_val_accuracy']:.2f}%")
        if result['best_val_accuracy'] >= 95.0:
            print(f"   🎯 成功达到95%目标!")
    else:
        print(f"\n❌ 测试失败")
        print(f"   检查错误信息进行调试")