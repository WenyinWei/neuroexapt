#!/usr/bin/env python3
"""
ASO-SE 简化版本 - 解决基础性能问题

重点：
1. Warmup阶段使用真正的固定架构
2. 合理的网络架构设计
3. 正常的训练速度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import sys
import argparse
from datetime import datetime
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

class SimpleBlock(nn.Module):
    """简单的残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SimpleNetwork(nn.Module):
    """简单的基准网络"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(SimpleBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(SimpleBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ASOSESimpleTrainer:
    """简化的ASO-SE训练器"""
    
    def __init__(self, experiment_name="aso_se_simple"):
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 训练参数
        self.batch_size = 128
        self.num_epochs = 50
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        
        print(f"🚀 ASO-SE 简化训练器初始化")
        print(f"   实验名称: {experiment_name}")
        print(f"   设备: {self.device}")
    
    def setup_data(self):
        """设置数据加载器"""
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
            root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                     shuffle=True, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, 
                                    shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"📊 数据加载完成: 训练集 {len(train_dataset)}, 测试集 {len(test_dataset)}")
    
    def setup_model(self):
        """设置模型"""
        self.network = SimpleNetwork(num_classes=10).to(self.device)
        
        total_params = sum(p.numel() for p in self.network.parameters())
        print(f"🏗️ 模型创建完成")
        print(f"   参数量: {total_params:,}")
    
    def setup_optimizer(self):
        """设置优化器"""
        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs, eta_min=1e-4)
        
        print(f"⚙️ 优化器设置完成")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.network.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.network(data)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.3f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        return total_loss / len(self.train_loader), accuracy
    
    def evaluate(self):
        """评估模型"""
        self.network.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.network(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def train(self):
        """完整训练流程"""
        print(f"\n🔧 简单网络训练开始 - 验证基础性能")
        print(f"{'='*60}")
        
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        
        best_accuracy = 0.0
        
        for epoch in range(self.num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 评估
            test_acc = self.evaluate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 更新最佳精度
            if test_acc > best_accuracy:
                best_accuracy = test_acc
            
            # 打印进度
            if (epoch + 1) % 5 == 0:
                print(f"\n📊 Epoch {epoch+1}/{self.num_epochs}")
                print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"   Test Acc: {test_acc:.2f}% | Best: {best_accuracy:.2f}%")
        
        print(f"\n🎉 简单网络训练完成!")
        print(f"   最佳精度: {best_accuracy:.2f}%")
        print(f"   预期: 应该达到85%+的准确率")
        
        return best_accuracy

def main():
    parser = argparse.ArgumentParser(description='ASO-SE 简化基准测试')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.1, help='学习率')
    
    args = parser.parse_args()
    
    print("🔧 ASO-SE: 简化基准测试")
    print(f"   目标: 验证网络基础训练能力")
    print(f"   预期: CIFAR-10 85%+准确率")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建训练器并开始训练
    trainer = ASOSESimpleTrainer("aso_se_simple_baseline")
    trainer.num_epochs = args.epochs
    trainer.batch_size = args.batch_size
    trainer.learning_rate = args.lr
    
    best_acc = trainer.train()
    
    print(f"\n✨ 基准测试完成!")
    print(f"   最终精度: {best_acc:.2f}%")
    print(f"   评价: {'✅ 基础训练正常' if best_acc > 80 else '❌ 基础训练异常'}")

if __name__ == '__main__':
    main()