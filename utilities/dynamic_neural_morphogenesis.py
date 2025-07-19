#!/usr/bin/env python3
"""
Dynamic Neural Morphogenesis (DNM) 框架
突破ASO-SE局限的革命性神经网络自适应生长系统

核心创新：
1. 信息熵驱动的神经元动态分裂
2. 梯度引导的连接动态生长
3. 多目标进化的架构优化
4. 实时性能反馈的架构调整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
import math
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import sys
import os

# Add neuroexapt to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class InformationEntropyNeuronDivision:
    """基于信息熵的神经元动态分裂机制"""
    
    def __init__(self, entropy_threshold=0.8, split_probability=0.3, max_splits_per_layer=3):
        self.entropy_threshold = entropy_threshold
        self.split_probability = split_probability
        self.max_splits_per_layer = max_splits_per_layer
        self.activation_cache = {}
        
    def register_hooks(self, model):
        """注册hook来收集激活信息"""
        self.hooks = []
        
        def create_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activation_cache[name] = output.detach().clone()
            return hook
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(create_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def analyze_neuron_information_load(self, layer_name, activations):
        """分析每个神经元的信息承载量"""
        if len(activations.shape) == 4:  # Conv2D layer
            # 对于卷积层，计算每个通道的平均熵
            B, C, H, W = activations.shape
            channel_entropies = []
            
            for c in range(C):
                channel_data = activations[:, c, :, :].view(-1)
                entropy = self._calculate_entropy(channel_data)
                channel_entropies.append(entropy)
            
            return torch.tensor(channel_entropies)
        
        elif len(activations.shape) == 2:  # Linear layer
            # 对于全连接层，计算每个神经元的熵
            B, N = activations.shape
            neuron_entropies = []
            
            for n in range(N):
                neuron_data = activations[:, n]
                entropy = self._calculate_entropy(neuron_data)
                neuron_entropies.append(entropy)
            
            return torch.tensor(neuron_entropies)
        
        return torch.tensor([])
    
    def _calculate_entropy(self, data, bins=20):
        """计算数据的信息熵"""
        if len(data) == 0:
            return 0.0
        
        # 数据归一化和离散化
        data_min, data_max = data.min(), data.max()
        if data_max - data_min < 1e-8:
            return 0.0  # 常数数据，熵为0
        
        # 创建直方图
        hist = torch.histc(data, bins=bins, min=data_min.item(), max=data_max.item())
        
        # 计算概率分布
        prob = hist / hist.sum()
        prob = prob[prob > 0]  # 只考虑非零概率
        
        # 计算熵
        entropy = -torch.sum(prob * torch.log2(prob + 1e-8))
        return entropy.item()
    
    def decide_neuron_splits(self, model, train_loader):
        """分析所有层并决定神经元分裂策略"""
        split_decisions = {}
        
        # 注册hooks
        self.register_hooks(model)
        
        # 收集激活数据
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(train_loader):
                if batch_idx >= 5:  # 只用几个batch分析
                    break
                data = data.cuda() if torch.cuda.is_available() else data
                _ = model(data)
        
        # 分析每层的神经元熵
        for layer_name, activations in self.activation_cache.items():
            neuron_entropies = self.analyze_neuron_information_load(layer_name, activations)
            
            if len(neuron_entropies) > 0:
                # 找到高熵神经元
                high_entropy_indices = []
                for i, entropy in enumerate(neuron_entropies):
                    if entropy > self.entropy_threshold and torch.rand(1) < self.split_probability:
                        high_entropy_indices.append(i)
                
                # 限制每层的分裂数量
                if len(high_entropy_indices) > self.max_splits_per_layer:
                    # 选择熵最高的几个
                    entropy_values = [(i, neuron_entropies[i].item()) for i in high_entropy_indices]
                    entropy_values.sort(key=lambda x: x[1], reverse=True)
                    high_entropy_indices = [i for i, _ in entropy_values[:self.max_splits_per_layer]]
                
                if high_entropy_indices:
                    split_decisions[layer_name] = {
                        'split_indices': high_entropy_indices,
                        'entropies': [neuron_entropies[i].item() for i in high_entropy_indices],
                        'layer_type': type(model.get_submodule(layer_name)).__name__
                    }
        
        # 清理
        self.remove_hooks()
        self.activation_cache.clear()
        
        return split_decisions
    
    def execute_splits(self, model, split_decisions):
        """执行神经元分裂"""
        modified_layers = {}
        
        for layer_name, split_info in split_decisions.items():
            try:
                layer = model.get_submodule(layer_name)
                split_indices = split_info['split_indices']
                
                if isinstance(layer, nn.Linear):
                    new_layer = self._split_linear_layer(layer, split_indices)
                    modified_layers[layer_name] = new_layer
                elif isinstance(layer, nn.Conv2d):
                    new_layer = self._split_conv_layer(layer, split_indices)
                    modified_layers[layer_name] = new_layer
                    
            except Exception as e:
                print(f"Warning: Failed to split layer {layer_name}: {e}")
                continue
        
        # 更新模型
        for layer_name, new_layer in modified_layers.items():
            self._replace_layer_in_model(model, layer_name, new_layer)
        
        return len(modified_layers)
    
    def _split_linear_layer(self, layer, split_indices):
        """分裂线性层的神经元"""
        if not split_indices:
            return layer
        
        # 创建新层
        new_out_features = layer.out_features + len(split_indices)
        new_layer = nn.Linear(
            layer.in_features, 
            new_out_features, 
            bias=layer.bias is not None
        ).to(layer.weight.device)
        
        # 权重迁移
        with torch.no_grad():
            # 复制原始权重
            new_layer.weight[:layer.out_features] = layer.weight.data
            if layer.bias is not None:
                new_layer.bias[:layer.out_features] = layer.bias.data
            
            # 为分裂的神经元初始化权重
            for i, split_idx in enumerate(split_indices):
                new_idx = layer.out_features + i
                # 继承父神经元权重 + 小扰动
                noise_scale = 0.1
                new_layer.weight[new_idx] = (
                    layer.weight.data[split_idx] + 
                    noise_scale * torch.randn_like(layer.weight.data[split_idx])
                )
                if layer.bias is not None:
                    new_layer.bias[new_idx] = (
                        layer.bias.data[split_idx] + 
                        noise_scale * torch.randn(1).to(layer.bias.device)
                    )
        
        return new_layer
    
    def _split_conv_layer(self, layer, split_indices):
        """分裂卷积层的通道"""
        if not split_indices:
            return layer
        
        # 创建新层
        new_out_channels = layer.out_channels + len(split_indices)
        new_layer = nn.Conv2d(
            layer.in_channels,
            new_out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
            bias=layer.bias is not None
        ).to(layer.weight.device)
        
        # 权重迁移
        with torch.no_grad():
            # 复制原始权重
            new_layer.weight[:layer.out_channels] = layer.weight.data
            if layer.bias is not None:
                new_layer.bias[:layer.out_channels] = layer.bias.data
            
            # 为分裂的通道初始化权重
            for i, split_idx in enumerate(split_indices):
                new_idx = layer.out_channels + i
                noise_scale = 0.1
                new_layer.weight[new_idx] = (
                    layer.weight.data[split_idx] + 
                    noise_scale * torch.randn_like(layer.weight.data[split_idx])
                )
                if layer.bias is not None:
                    new_layer.bias[new_idx] = (
                        layer.bias.data[split_idx] + 
                        noise_scale * torch.randn(1).to(layer.bias.device)
                    )
        
        return new_layer
    
    def _replace_layer_in_model(self, model, layer_name, new_layer):
        """在模型中替换层"""
        # 获取父模块和属性名
        parts = layer_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # 替换层
        setattr(parent, parts[-1], new_layer)


class GradientGuidedConnectionGrowth:
    """基于梯度的连接动态生长机制"""
    
    def __init__(self, gradient_threshold=0.1, max_new_connections=3):
        self.gradient_threshold = gradient_threshold
        self.max_new_connections = max_new_connections
        self.gradient_cache = {}
        
    def analyze_gradient_patterns(self, model, train_loader, criterion):
        """分析梯度模式，识别有益连接"""
        self.gradient_cache.clear()
        model.train()
        
        # 收集梯度信息
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 3:  # 只用几个batch分析
                break
            
            data = data.cuda() if torch.cuda.is_available() else data
            target = target.cuda() if torch.cuda.is_available() else target
            
            # 前向传播和反向传播
            output = model(data)
            loss = criterion(output, target)
            
            model.zero_grad()
            loss.backward()
            
            # 收集梯度
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in self.gradient_cache:
                        self.gradient_cache[name] = []
                    self.gradient_cache[name].append(param.grad.clone())
        
        # 分析梯度相关性
        return self._identify_beneficial_connections()
    
    def _identify_beneficial_connections(self):
        """识别有益的跨层连接"""
        beneficial_connections = []
        
        layer_names = list(self.gradient_cache.keys())
        
        for i in range(len(layer_names)):
            for j in range(i+2, len(layer_names)):  # 跳过直接相邻层
                source_layer = layer_names[i]
                target_layer = layer_names[j]
                
                # 计算梯度相关性
                correlations = []
                for batch_idx in range(len(self.gradient_cache[source_layer])):
                    if batch_idx < len(self.gradient_cache[target_layer]):
                        corr = self._calculate_gradient_correlation(
                            self.gradient_cache[source_layer][batch_idx],
                            self.gradient_cache[target_layer][batch_idx]
                        )
                        if not math.isnan(corr):
                            correlations.append(corr)
                
                if correlations:
                    avg_correlation = np.mean(correlations)
                    if avg_correlation > self.gradient_threshold:
                        beneficial_connections.append({
                            'source': source_layer,
                            'target': target_layer,
                            'strength': avg_correlation
                        })
        
        # 按相关性排序
        beneficial_connections.sort(key=lambda x: x['strength'], reverse=True)
        return beneficial_connections[:self.max_new_connections]
    
    def _calculate_gradient_correlation(self, grad1, grad2):
        """计算两个梯度张量的相关性"""
        try:
            # 展平梯度
            flat_grad1 = grad1.view(-1).float()
            flat_grad2 = grad2.view(-1).float()
            
            # 取较小尺寸
            min_size = min(flat_grad1.size(0), flat_grad2.size(0))
            flat_grad1 = flat_grad1[:min_size]
            flat_grad2 = flat_grad2[:min_size]
            
            # 计算皮尔逊相关系数
            if min_size > 1:
                correlation = torch.corrcoef(torch.stack([flat_grad1, flat_grad2]))[0, 1]
                return correlation.abs().item() if not torch.isnan(correlation) else 0.0
            else:
                return 0.0
                
        except Exception as e:
            return 0.0


class MultiObjectiveArchitectureEvolution:
    """多目标架构进化优化"""
    
    def __init__(self, population_size=5, mutation_rate=0.3):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        
    def evaluate_fitness(self, model, train_loader, val_loader):
        """多目标适应度评估"""
        # 快速评估准确率
        accuracy = self._quick_evaluate_accuracy(model, val_loader)
        
        # 计算模型复杂度
        complexity = self._calculate_complexity(model)
        
        # 计算计算效率（简化版）
        efficiency = 1.0 / (complexity + 1e-6)
        
        # 综合适应度
        composite_fitness = 0.7 * accuracy + 0.2 * efficiency + 0.1 * (100 - complexity)
        
        return {
            'accuracy': accuracy,
            'complexity': complexity,
            'efficiency': efficiency,
            'composite': composite_fitness
        }
    
    def _quick_evaluate_accuracy(self, model, val_loader):
        """快速评估模型准确率"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx >= 3:  # 只用几个batch快速评估
                    break
                
                data = data.cuda() if torch.cuda.is_available() else data
                target = target.cuda() if torch.cuda.is_available() else target
                
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return 100.0 * correct / total if total > 0 else 0.0
    
    def _calculate_complexity(self, model):
        """计算模型复杂度（参数数量的对数）"""
        total_params = sum(p.numel() for p in model.parameters())
        return math.log10(total_params + 1)


class DNMTrainer:
    """DNM训练器 - 整合所有创新组件"""
    
    def __init__(self, model, config=None):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 默认配置
        self.config = config or {
            'entropy_threshold': 0.6,
            'split_probability': 0.3,
            'gradient_threshold': 0.1,
            'evolution_frequency': 10,
            'analysis_frequency': 5,
            'learning_rate': 0.01,
            'max_splits_per_layer': 2,
            'max_new_connections': 2
        }
        
        # 初始化DNM组件
        self.neuron_divider = InformationEntropyNeuronDivision(
            entropy_threshold=self.config['entropy_threshold'],
            split_probability=self.config['split_probability'],
            max_splits_per_layer=self.config['max_splits_per_layer']
        )
        
        self.connection_grower = GradientGuidedConnectionGrowth(
            gradient_threshold=self.config['gradient_threshold'],
            max_new_connections=self.config['max_new_connections']
        )
        
        self.evolution_optimizer = MultiObjectiveArchitectureEvolution()
        
        # 性能追踪
        self.performance_history = []
        self.architecture_changes = []
        self.current_epoch = 0
        
        # 优化器
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.config['learning_rate'], 
            momentum=0.9, 
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"🧬 DNM Trainer initialized on {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Configuration: {self.config}")
    
    def train_with_dynamic_morphogenesis(self, train_loader, val_loader, epochs):
        """使用DNM的训练流程"""
        print("\n🚀 Starting Dynamic Neural Morphogenesis Training")
        print("=" * 60)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            print(f"\n🧬 Epoch {epoch+1}/{epochs}")
            
            # 1. 标准训练
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            epoch_time = time.time() - start_time
            print(f"  📊 Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Time: {epoch_time:.1f}s")
            
            # 记录性能
            self.performance_history.append({
                'epoch': epoch,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_params': sum(p.numel() for p in self.model.parameters())
            })
            
            # 2. 动态架构分析和调整
            architecture_changed = False
            
            if epoch % self.config['analysis_frequency'] == 0 and epoch > 0:
                print("  🔄 Performing dynamic architecture analysis...")
                
                try:
                    # 分析并执行神经元分裂
                    split_decisions = self.neuron_divider.decide_neuron_splits(
                        self.model, train_loader
                    )
                    
                    if split_decisions:
                        splits_executed = self.neuron_divider.execute_splits(
                            self.model, split_decisions
                        )
                        
                        if splits_executed > 0:
                            print(f"  ✨ Executed {splits_executed} neuron splits")
                            architecture_changed = True
                            
                            # 更新优化器以包含新参数
                            self._update_optimizer()
                    
                    # 分析梯度模式和连接生长
                    beneficial_connections = self.connection_grower.analyze_gradient_patterns(
                        self.model, train_loader, self.criterion
                    )
                    
                    if beneficial_connections:
                        print(f"  🔗 Identified {len(beneficial_connections)} beneficial connections")
                        # 这里可以实现连接生长的具体逻辑
                    
                    # 记录架构变化
                    if architecture_changed or beneficial_connections:
                        self.architecture_changes.append({
                            'epoch': epoch,
                            'splits': len(split_decisions) if split_decisions else 0,
                            'connections': len(beneficial_connections),
                            'performance_before': val_acc,
                            'model_params_after': sum(p.numel() for p in self.model.parameters())
                        })
                
                except Exception as e:
                    print(f"  ⚠️ Warning: Architecture analysis failed: {e}")
            
            # 3. 输出当前状态
            current_params = sum(p.numel() for p in self.model.parameters())
            if epoch == 0:
                self.initial_params = current_params
            
            param_growth = (current_params - self.initial_params) / self.initial_params * 100
            print(f"  🧮 Model size: {current_params:,} params ({param_growth:+.1f}% from initial)")
            
            # 早期停止条件
            if val_acc > 95.0:
                print(f"  🎯 Reached target accuracy of {val_acc:.2f}%!")
                break
        
        print("\n✅ DNM Training completed")
        self._print_summary()
        
        return self.model, self.performance_history, self.architecture_changes
    
    def _train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def _update_optimizer(self):
        """更新优化器以包含新参数"""
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.config['learning_rate'], 
            momentum=0.9, 
            weight_decay=1e-4
        )
    
    def _print_summary(self):
        """打印训练总结"""
        if self.performance_history:
            best_val_acc = max(p['val_acc'] for p in self.performance_history)
            final_val_acc = self.performance_history[-1]['val_acc']
            total_changes = len(self.architecture_changes)
            
            print(f"\n📈 Training Summary:")
            print(f"   Best validation accuracy: {best_val_acc:.2f}%")
            print(f"   Final validation accuracy: {final_val_acc:.2f}%")
            print(f"   Architecture changes: {total_changes}")
            print(f"   Parameter growth: {(self.performance_history[-1]['model_params'] - self.initial_params) / self.initial_params * 100:+.1f}%")


# 示例模型定义
class EvolvableCNN(nn.Module):
    """可演化的CNN模型"""
    
    def __init__(self, num_classes=10):
        super(EvolvableCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def demo_dnm_training():
    """DNM训练演示"""
    print("🧬 Dynamic Neural Morphogenesis Demo")
    print("=" * 50)
    
    # 数据准备
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=transform
    )
    
    # 使用小数据集进行快速演示
    from torch.utils.data import Subset
    train_subset = Subset(trainset, range(1000))
    test_subset = Subset(testset, range(200))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    # 创建模型
    model = EvolvableCNN(num_classes=10)
    
    # 创建DNM训练器
    config = {
        'entropy_threshold': 0.5,
        'split_probability': 0.4,
        'gradient_threshold': 0.1,
        'analysis_frequency': 3,
        'learning_rate': 0.01,
        'max_splits_per_layer': 2,
        'max_new_connections': 2
    }
    
    trainer = DNMTrainer(model, config)
    
    # 开始DNM训练
    evolved_model, history, changes = trainer.train_with_dynamic_morphogenesis(
        train_loader, test_loader, epochs=20
    )
    
    print("\n🎉 DNM Training Complete!")
    return evolved_model, history, changes


if __name__ == "__main__":
    # 运行演示
    model, history, changes = demo_dnm_training()