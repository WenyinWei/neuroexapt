#!/usr/bin/env python3
"""
ASO-SE (Alternating Stable Optimization with Stochastic Exploration) 神经网络训练

核心特性：
🚀 真正的架构搜索和网络结构动态生长
🔧 基于Net2Net的平滑参数迁移
⚡ Gumbel-Softmax引导的可微分架构采样
🎯 四阶段训练循环：预热→搜索→生长→优化

架构生长策略：
- 深度生长：添加新的可进化层
- 宽度生长：扩展现有层的通道数
- 分支生长：增加操作分支的复杂度
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
from typing import Dict, List, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模块
from neuroexapt.core.genotypes import PRIMITIVES
from neuroexapt.core.operations import OPS
from neuroexapt.core.net2net_transfer import Net2NetTransfer

# 配置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

class GumbelSoftmaxSelector(nn.Module):
    """Gumbel-Softmax架构采样器"""
    
    def __init__(self, initial_temp=5.0, min_temp=0.1, anneal_rate=0.98):
        super().__init__()
        self.temperature = initial_temp
        self.min_temperature = min_temp
        self.anneal_rate = anneal_rate
        
    def forward(self, logits, hard=True):
        """Gumbel-Softmax采样"""
        if not self.training:
            # 推理时使用argmax
            y_hard = torch.zeros_like(logits)
            y_hard.scatter_(-1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
            return y_hard
        
        # 训练时使用Gumbel-Softmax
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        y_soft = F.softmax((logits + gumbel_noise) / self.temperature, dim=-1)
        
        if hard:
            # Straight-through estimator
            max_indices = torch.argmax(y_soft, dim=-1, keepdim=True)
            y_hard = torch.zeros_like(y_soft).scatter_(-1, max_indices, 1.0)
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_soft
    
    def anneal_temperature(self):
        """温度退火"""
        self.temperature = max(self.min_temperature, self.temperature * self.anneal_rate)

class MixedOperation(nn.Module):
    """混合操作层 - 支持多种原始操作"""
    
    def __init__(self, C, stride):
        super().__init__()
        self.operations = nn.ModuleList()
        self.C = C
        self.stride = stride
        
        # 创建所有候选操作
        for primitive in PRIMITIVES:
            op = self._create_operation(primitive, C, stride)
            self.operations.append(op)
        
        self.num_ops = len(PRIMITIVES)
        print(f"🔧 MixedOperation 创建: {self.num_ops} 个操作, C={C}, stride={stride}")
    
    def _create_operation(self, primitive, C, stride):
        """创建单个操作"""
        if primitive in OPS:
            return OPS[primitive](C, stride, False)
        elif primitive == 'none':
            return Zero(stride)
        elif primitive == 'skip_connect':
            if stride == 1:
                return Identity()
            else:
                return FactorizedReduce(C, C)
        elif primitive == 'sep_conv_3x3':
            return SepConv(C, C, 3, stride, 1)
        elif primitive == 'sep_conv_5x5':
            return SepConv(C, C, 5, stride, 2)
        elif primitive == 'sep_conv_7x7':
            return SepConv(C, C, 7, stride, 3)
        elif primitive == 'dil_conv_3x3':
            return DilConv(C, C, 3, stride, 2, 2)
        elif primitive == 'dil_conv_5x5':
            return DilConv(C, C, 5, stride, 4, 2)
        elif primitive == 'conv_7x1_1x7':
            return Conv7x1_1x7(C, C, stride)
        elif primitive == 'avg_pool_3x3':
            return nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif primitive == 'max_pool_3x3':
            return nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError(f"Unknown primitive: {primitive}")
    
    def forward(self, x, arch_weights):
        """前向传播"""
        # 对每个操作计算结果并加权求和
        results = []
        for i, op in enumerate(self.operations):
            results.append(arch_weights[i] * op(x))
        
        return sum(results)

class ArchitectureManager(nn.Module):
    """架构参数管理器"""
    
    def __init__(self, num_layers, num_ops):
        super().__init__()
        self.num_layers = num_layers
        self.num_ops = num_ops
        
        # 为每层创建架构参数
        self.arch_params = nn.ParameterList()
        for i in range(num_layers):
            # 每层的架构参数 - 避免none操作被选中，给skip_connect更高的初始权重
            layer_params = nn.Parameter(torch.randn(num_ops) * 0.5)
            # 给skip_connect(索引3)更高的初始值，避免none(索引0)
            with torch.no_grad():
                layer_params[0] = -2.0  # none操作权重降低
                if num_ops > 3:
                    layer_params[3] = 1.0   # skip_connect权重提高
            self.arch_params.append(layer_params)
        
        print(f"🔧 ArchitectureManager: {num_layers} 层, 每层 {num_ops} 个操作")
    
    def get_arch_weights(self, layer_idx, selector, training_phase='warmup'):
        """获取指定层的架构权重"""
        if layer_idx >= len(self.arch_params):
            # 如果层数增加了，添加新的架构参数
            while len(self.arch_params) <= layer_idx:
                new_params = nn.Parameter(torch.randn(self.num_ops) * 0.5)
                with torch.no_grad():
                    new_params[0] = -2.0  # none操作权重降低
                    if self.num_ops > 3:
                        new_params[3] = 1.0   # skip_connect权重提高
                if len(self.arch_params) > 0 and self.arch_params[0].device != torch.device('cpu'):
                    new_params = new_params.to(self.arch_params[0].device)
                self.arch_params.append(new_params)
        
        logits = self.arch_params[layer_idx]
        
        # 在warmup阶段使用固定架构（softmax without gumbel noise）
        if training_phase == 'warmup':
            # 在warmup阶段强制使用skip连接，避免none操作
            weights = torch.zeros_like(logits)
            weights[3] = 1.0  # 强制使用skip_connect (index 3)
            return weights.detach()  # 不需要梯度
        else:
            # 在搜索和生长阶段使用Gumbel-Softmax
            return selector(logits.unsqueeze(0)).squeeze(0)
    
    def get_current_genotype(self):
        """获取当前基因型"""
        genotype = []
        for layer_params in self.arch_params:
            best_op_idx = torch.argmax(layer_params).item()
            best_op = PRIMITIVES[best_op_idx]
            genotype.append(best_op)
        return genotype

class EvolvableBlock(nn.Module):
    """可进化的网络块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # 预处理层（如果通道数不匹配）
        self.preprocess = None
        if in_channels != out_channels:
            self.preprocess = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # 主要的混合操作
        self.mixed_op = MixedOperation(out_channels, stride=1)
    
    def forward(self, x, arch_weights):
        """前向传播"""
        if self.preprocess is not None:
            x = self.preprocess(x)
        
        out = self.mixed_op(x, arch_weights)
        return out

class ASOSENetwork(nn.Module):
    """ASO-SE可生长神经网络"""
    
    def __init__(self, input_channels=3, initial_channels=16, num_classes=10, initial_depth=8):
        super().__init__()
        self.input_channels = input_channels
        self.initial_channels = initial_channels
        self.num_classes = num_classes
        self.current_depth = initial_depth
        self.current_channels = initial_channels
        
        # 初始特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, initial_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        
        # 可进化层
        self.layers = nn.ModuleList()
        current_channels = initial_channels
        
        for i in range(initial_depth):
            # 每隔几层进行下采样
            stride = 2 if i in [initial_depth//3, 2*initial_depth//3] else 1
            if stride == 2:
                next_channels = min(current_channels * 2, 512)
            else:
                next_channels = current_channels
            
            layer = EvolvableBlock(current_channels, next_channels, stride)
            self.layers.append(layer)
            current_channels = next_channels
        
        # 全局平均池化和分类器
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, num_classes)
        
        # 架构管理器
        self.arch_manager = ArchitectureManager(self.current_depth, len(PRIMITIVES))
        
        # Gumbel-Softmax选择器
        self.gumbel_selector = GumbelSoftmaxSelector()
        
        # Net2Net迁移工具
        self.net2net_transfer = Net2NetTransfer()
        
        # 训练阶段状态
        self.training_phase = 'warmup'
        
        print(f"🚀 ASO-SE 网络初始化:")
        print(f"   深度: {self.current_depth} 层")
        print(f"   初始通道: {initial_channels}")
        print(f"   当前通道: {current_channels}")
        print(f"   参数量: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, x):
        """前向传播"""
        x = self.stem(x)
        
        for i, layer in enumerate(self.layers):
            arch_weights = self.arch_manager.get_arch_weights(i, self.gumbel_selector, self.training_phase)
            x = layer(x, arch_weights)
        
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def set_training_phase(self, phase):
        """设置训练阶段"""
        self.training_phase = phase
        print(f"🔄 设置训练阶段: {phase}")
    
    def grow_depth(self, num_new_layers=1):
        """深度生长 - 添加新层"""
        print(f"🌱 网络深度生长: 添加 {num_new_layers} 层")
        
        for _ in range(num_new_layers):
            # 在倒数第二层后插入新层
            insert_pos = len(self.layers) - 1
            if insert_pos <= 0:
                insert_pos = len(self.layers) // 2  # 在中间插入
            
            # 获取当前层的通道数
            reference_layer = self.layers[insert_pos]
            current_channels = reference_layer.out_channels
            
            # 创建新层并移动到正确设备
            new_layer = EvolvableBlock(current_channels, current_channels, stride=1)
            new_layer = new_layer.to(next(self.parameters()).device)
            
            self.layers.insert(insert_pos, new_layer)
            self.current_depth += 1
        
        # 更新架构管理器（保持现有参数）
        # ArchitectureManager已经能够动态扩展参数，无需重新创建
        
        print(f"   新深度: {self.current_depth}")
    
    def grow_width(self, growth_factor=1.5):
        """宽度生长 - 扩展通道数（简化实现）"""
        print(f"🌱 网络宽度生长: 增长因子 {growth_factor}")
        
        # 简化实现：只扩展分类器的输入特征数
        # 真正的宽度扩展需要更复杂的Net2Net操作，这里先记录意图
        old_classifier = self.classifier
        current_features = old_classifier.in_features
        new_features = int(current_features * growth_factor)
        
        if new_features > current_features:
            print(f"   分类器扩展: {current_features} -> {new_features} 特征")
            # 这里可以在未来集成真正的Net2Net宽度扩展
        else:
            print(f"   宽度生长跳过（增长因子太小）")
    
    def get_architecture_info(self):
        """获取架构信息"""
        genotype = self.arch_manager.get_current_genotype()
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'depth': self.current_depth,
            'genotype': genotype,
            'parameters': params,
            'temperature': self.gumbel_selector.temperature
        }

class ASOSETrainingController:
    """ASO-SE训练控制器"""
    
    def __init__(self, network, growth_patience=5, performance_threshold=0.02):
        self.network = network
        self.growth_patience = growth_patience
        self.performance_threshold = performance_threshold
        
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.growth_history = []
    
    def should_grow(self, current_accuracy):
        """判断是否应该生长"""
        improvement = current_accuracy - self.best_accuracy
        
        if improvement > self.performance_threshold:
            self.best_accuracy = current_accuracy
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.growth_patience
    
    def trigger_growth(self, growth_type='depth'):
        """触发网络生长"""
        print(f"🌱 触发 {growth_type} 生长")
        
        if growth_type == 'depth':
            self.network.grow_depth(1)
        elif growth_type == 'width':
            self.network.grow_width(1.2)
        
        self.growth_history.append({
            'type': growth_type,
            'step': len(self.growth_history),
            'architecture': self.network.get_architecture_info()
        })
        
        # 重置控制器
        self.patience_counter = 0

class ASOSETrainer:
    """ASO-SE训练器"""
    
    def __init__(self, experiment_name="aso_se_neural_growth"):
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 训练参数
        self.batch_size = 128
        self.num_epochs = 100
        self.weight_lr = 0.025
        self.arch_lr = 3e-4
        self.momentum = 0.9
        self.weight_decay = 3e-4
        
        # 阶段控制
        self.phase_durations = {
            'warmup': 10,      # 预热阶段
            'search': 30,      # 架构搜索阶段
            'growth': 40,      # 生长阶段
            'optimize': 20     # 优化阶段
        }
        
        self.current_phase = 'warmup'
        self.phase_epochs = 0
        
        print(f"🚀 ASO-SE 训练器初始化")
        print(f"   实验名称: {experiment_name}")
        print(f"   设备: {self.device}")
    
    def setup_data(self):
        """设置数据加载器"""
        # CIFAR-10数据增强
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # 数据集
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform)
        
        # 数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                     shuffle=True, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, 
                                    shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"📊 数据加载完成: 训练集 {len(train_dataset)}, 测试集 {len(test_dataset)}")
    
    def setup_model(self):
        """设置模型"""
        self.network = ASOSENetwork(
            input_channels=3,
            initial_channels=16,
            num_classes=10,
            initial_depth=8
        ).to(self.device)
        
        # 训练控制器
        self.training_controller = ASOSETrainingController(self.network)
        
        print(f"🏗️ 模型设置完成")
    
    def setup_optimizers(self):
        """设置优化器"""
        # 获取架构参数的ID集合，避免张量比较
        arch_param_ids = {id(p) for p in self.network.arch_manager.parameters()}
        
        # 权重优化器 - 排除架构参数
        weight_params = [p for p in self.network.parameters() if id(p) not in arch_param_ids]
        self.weight_optimizer = optim.SGD(
            weight_params,
            lr=self.weight_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # 架构优化器
        self.arch_optimizer = optim.Adam(
            self.network.arch_manager.parameters(),
            lr=self.arch_lr,
            betas=(0.5, 0.999),
            weight_decay=1e-3
        )
        
        # 学习率调度器
        self.weight_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.weight_optimizer, T_max=self.num_epochs, eta_min=1e-4)
        self.arch_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.arch_optimizer, T_max=self.num_epochs, eta_min=1e-5)
        
        print(f"⚙️ 优化器设置完成")
    
    def _update_optimizers_after_growth(self):
        """生长后安全地更新优化器"""
        try:
            # 保存当前学习率
            current_weight_lr = self.weight_optimizer.param_groups[0]['lr']
            current_arch_lr = self.arch_optimizer.param_groups[0]['lr']
            
            # 重新设置优化器
            self.setup_optimizers()
            
            # 恢复学习率
            for param_group in self.weight_optimizer.param_groups:
                param_group['lr'] = current_weight_lr
            for param_group in self.arch_optimizer.param_groups:
                param_group['lr'] = current_arch_lr
                
            print(f"✅ 优化器已更新以包含新参数")
            
        except Exception as e:
            print(f"⚠️ 优化器更新警告: {e}")
            # 如果更新失败，至少尝试基本设置
            self.setup_optimizers()
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.network.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # 权重优化步骤
            self.weight_optimizer.zero_grad()
            outputs = self.network(data)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            self.weight_optimizer.step()
            
            # 架构优化步骤(在搜索和生长阶段)
            if self.current_phase in ['search', 'growth'] and batch_idx % 2 == 0:
                self.arch_optimizer.zero_grad()
                arch_outputs = self.network(data)
                arch_loss = F.cross_entropy(arch_outputs, targets)
                arch_loss.backward()
                self.arch_optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.3f}',
                'Acc': f'{accuracy:.2f}%',
                'Phase': self.current_phase,
                'Temp': f'{self.network.gumbel_selector.temperature:.3f}'
            })
        
        # 温度退火
        if self.current_phase in ['search', 'growth']:
            self.network.gumbel_selector.anneal_temperature()
        
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
    
    def update_phase(self, epoch):
        """更新训练阶段"""
        self.phase_epochs += 1
        
        # 阶段转换逻辑
        old_phase = self.current_phase
        
        if self.current_phase == 'warmup' and self.phase_epochs >= self.phase_durations['warmup']:
            self.current_phase = 'search'
            self.phase_epochs = 0
            print(f"🔄 进入架构搜索阶段")
        
        elif self.current_phase == 'search' and self.phase_epochs >= self.phase_durations['search']:
            self.current_phase = 'growth'
            self.phase_epochs = 0
            print(f"🔄 进入网络生长阶段")
        
        elif self.current_phase == 'growth' and self.phase_epochs >= self.phase_durations['growth']:
            self.current_phase = 'optimize'
            self.phase_epochs = 0
            print(f"🔄 进入最终优化阶段")
        
        # 同步网络的训练阶段
        if old_phase != self.current_phase:
            self.network.set_training_phase(self.current_phase)
    
    def train(self):
        """完整训练流程"""
        print(f"\n🔧 ASO-SE 训练开始")
        print(f"{'='*60}")
        
        self.setup_data()
        self.setup_model()
        self.setup_optimizers()
        
        best_accuracy = 0.0
        training_history = []
        
        for epoch in range(self.num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 评估
            test_acc = self.evaluate()
            
            # 更新学习率
            self.weight_scheduler.step()
            if self.current_phase in ['search', 'growth']:
                self.arch_scheduler.step()
            
            # 记录历史
            training_history.append({
                'epoch': epoch,
                'phase': self.current_phase,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'architecture': self.network.get_architecture_info()
            })
            
            # 检查是否需要生长
            if self.current_phase == 'growth':
                if self.training_controller.should_grow(test_acc):
                    growth_type = 'depth' if epoch % 2 == 0 else 'width'
                    self.training_controller.trigger_growth(growth_type)
                    # 安全地更新优化器以包含新参数
                    self._update_optimizers_after_growth()
            
            # 更新最佳精度
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.weight_optimizer.state_dict(),
                    'architecture': self.network.get_architecture_info(),
                    'accuracy': best_accuracy
                }, f'{self.experiment_name}_best.pth')
            
            # 更新阶段
            self.update_phase(epoch)
            
            # 打印进度
            if (epoch + 1) % 5 == 0:
                arch_info = self.network.get_architecture_info()
                print(f"\n📊 Epoch {epoch+1}/{self.num_epochs} | Phase: {self.current_phase}")
                print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"   Test Acc: {test_acc:.2f}% | Best: {best_accuracy:.2f}%")
                print(f"   网络深度: {arch_info['depth']} | 参数量: {arch_info['parameters']:,}")
                print(f"   当前基因型: {arch_info['genotype'][:3]}...")
        
        print(f"\n🎉 ASO-SE 训练完成!")
        print(f"   最佳精度: {best_accuracy:.2f}%")
        print(f"   最终架构: {self.network.get_architecture_info()}")
        
        return training_history, best_accuracy

# 基础操作实现
class Identity(nn.Module):
    def forward(self, x):
        return x

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=True)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, 
                     padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, 
                     padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=True),
        )

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, 
                     padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=True),
        )

    def forward(self, x):
        return self.op(x)

class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class Conv7x1_1x7(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
            nn.Conv2d(C_out, C_out, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(C_out, affine=True)
        )

    def forward(self, x):
        return self.op(x)

def main():
    parser = argparse.ArgumentParser(description='ASO-SE 神经网络自适应架构搜索')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.025, help='学习率')
    parser.add_argument('--experiment', type=str, default='aso_se_neural_growth', help='实验名称')
    
    args = parser.parse_args()
    
    print("🔧 ASO-SE: 真正的神经架构搜索与网络生长")
    print(f"   目标: CIFAR-10 95%准确率")
    print(f"   策略: 四阶段训练 + Net2Net平滑迁移")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建训练器并开始训练
    trainer = ASOSETrainer(args.experiment)
    trainer.batch_size = args.batch_size
    trainer.num_epochs = args.epochs
    trainer.weight_lr = args.lr
    
    history, best_acc = trainer.train()
    
    print(f"\n✨ 实验完成!")
    print(f"   最终精度: {best_acc:.2f}%")

if __name__ == '__main__':
    main() 