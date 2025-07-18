#!/usr/bin/env python3
"""
简化的DNM测试 - 修复原始问题

解决的问题：
1. view size is not compatible 错误
2. 架构实际未演化的问题
3. 提升性能表现
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

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入修复后的DNM框架
from neuroexapt.core.dnm_framework import DNMFramework


class SimpleCNN(nn.Module):
    """简单但可演化的CNN模型"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 特征提取层
        self.features = nn.Sequential(
            # 第一组卷积
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第二组卷积
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第三组卷积
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def prepare_small_cifar10(batch_size=64):
    """准备CIFAR-10数据集（小规模测试）"""
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
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
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    print(f"📊 CIFAR-10 loaded: {len(trainset)} train, {len(testset)} test samples")
    return train_loader, test_loader


def create_aggressive_dnm_config():
    """创建更激进的DNM配置来确保演化发生"""
    
    return {
        'neuron_division': {
            'splitter': {
                'entropy_threshold': 0.4,        # 降低阈值，更容易分裂
                'overload_threshold': 0.3,       # 降低过载阈值
                'split_probability': 0.8,        # 提高分裂概率
                'max_splits_per_layer': 3,       # 允许分裂
                'inheritance_noise': 0.1         # 适中的噪声
            },
            'monitoring': {
                'target_layers': ['conv', 'linear'],
                'analysis_frequency': 3,         # 更频繁分析
                'min_epoch_before_split': 5      # 更早开始
            }
        },
        'connection_growth': {
            'analyzer': {
                'correlation_threshold': 0.1,    # 降低相关性阈值
                'history_length': 6              # 较短历史
            },
            'growth': {
                'max_new_connections': 2,        # 适中连接数
                'min_correlation_threshold': 0.05,
                'growth_frequency': 4,           # 更频繁
                'connection_types': ['skip_connection']  # 简化连接类型
            },
            'filtering': {
                'min_layer_distance': 1,
                'max_layer_distance': 4,
                'avoid_redundant_connections': True
            }
        },
        'multi_objective': {
            'evolution': {
                'population_size': 6,            # 较小种群
                'max_generations': 8,            # 较少代数
                'mutation_rate': 0.5,
                'crossover_rate': 0.7,
                'elitism_ratio': 0.3
            },
            'optimization': {
                'trigger_frequency': 12,         # 适中频率
                'performance_plateau_threshold': 0.01,
                'min_improvement_epochs': 2
            }
        },
        'framework': {
            'morphogenesis_frequency': 3,       # 更频繁的形态发生
            'performance_tracking_window': 5,
            'early_stopping_patience': 20,
            'target_accuracy_threshold': 85.0,  # 适中目标
            'enable_architecture_snapshots': True,
            'adaptive_morphogenesis': True
        }
    }


def run_dnm_simple_test():
    """运行简化的DNM测试"""
    
    print("🧬 DNM Simple Test - Fixed Version")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # 准备数据
    train_loader, test_loader = prepare_small_cifar10(batch_size=64)
    
    # 创建模型
    model = SimpleCNN(num_classes=10)
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️ Initial model: {initial_params:,} parameters")
    
    # 创建DNM框架
    dnm_config = create_aggressive_dnm_config()
    dnm = DNMFramework(dnm_config)
    
    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n🚀 Starting DNM Training")
    print("=" * 50)
    
    # 进度回调
    def simple_callback(dnm_framework, model, epoch_record):
        epoch = epoch_record['epoch']
        train_acc = epoch_record['train_acc']
        val_acc = epoch_record['val_acc']
        params = epoch_record['model_params']
        
        param_growth = (params - initial_params) / initial_params * 100
        
        if epoch % 2 == 0:
            print(f"📈 Epoch {epoch}: Train={train_acc:.2f}%, "
                  f"Val={val_acc:.2f}%, Params={params:,} (+{param_growth:.1f}%)")
    
    # 开始训练
    start_time = time.time()
    
    try:
        result = dnm.train_with_morphogenesis(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=30,
            optimizer=optimizer,
            criterion=criterion,
            callbacks=[simple_callback]
        )
        
        # 结果分析
        training_time = time.time() - start_time
        final_params = sum(p.numel() for p in result['model'].parameters())
        param_growth = (final_params - initial_params) / initial_params * 100
        
        print(f"\n🎉 DNM Test Completed!")
        print("=" * 50)
        print(f"📊 Results:")
        print(f"   Best Accuracy: {result['best_val_accuracy']:.2f}%")
        print(f"   Final Accuracy: {result['final_val_accuracy']:.2f}%")
        print(f"   Parameter Growth: +{param_growth:.1f}%")
        print(f"   Training Time: {training_time:.1f}s")
        print(f"   Morphogenesis Events: {len(result['morphogenesis_events'])}")
        
        # 形态发生分析
        if result['morphogenesis_events']:
            print(f"\n🧬 Morphogenesis Events:")
            for event in result['morphogenesis_events']:
                print(f"   Epoch {event['epoch']}: "
                      f"{event['neuron_splits']} splits, "
                      f"{event['connections_grown']} connections")
        else:
            print(f"\n⚠️ No morphogenesis events occurred")
            print(f"   Try lowering thresholds or increasing frequencies")
        
        # 成功评估
        if param_growth > 0:
            print(f"\n✅ SUCCESS: Model actually evolved! (+{param_growth:.1f}% parameters)")
        else:
            print(f"\n❌ No evolution detected. Model structure unchanged.")
        
        if result['best_val_accuracy'] >= 60.0:
            print(f"✅ GOOD: Achieved decent accuracy ({result['best_val_accuracy']:.2f}%)")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    
    # 运行测试
    result = run_dnm_simple_test()
    
    if result:
        print(f"\n🎯 Test Summary:")
        print(f"   The DNM framework is working!")
        print(f"   Check morphogenesis events for evolution details.")
    else:
        print(f"\n🔧 Need debugging:")
        print(f"   Check error messages above for issues to fix.")