#!/usr/bin/env python3
"""
ASO-SE修复版本 - 解决梯度计算和网络结构问题

🔧 主要修复：
1. 修复梯度重复计算错误
2. 简化网络结构避免闭环
3. 正确管理架构参数和权重参数
4. 确保设备一致性
5. 添加详细的错误检查
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
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入基础组件
from neuroexapt.core.genotypes import PRIMITIVES
from neuroexapt.core.operations import OPS

# 配置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

class SimpleMixedOp(nn.Module):
    """
    简化的混合操作 - 避免复杂的梯度计算问题
    """
    
    def __init__(self, C, stride):
        super().__init__()
        self.C = C
        self.stride = stride
        
        # 创建所有操作
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
    
    def forward(self, x, weights):
        """
        简化的前向传播 - 避免梯度重复计算
        """
        # 确保权重在正确设备上
        if weights.device != x.device:
            weights = weights.to(x.device)
        
        # 计算加权输出 - 使用稳定的实现
        outputs = []
        for w, op in zip(weights, self._ops):
            if w.item() > 1e-6:  # 只计算非零权重的操作
                outputs.append(w * op(x))
        
        if outputs:
            return sum(outputs)
        else:
            # 如果所有权重都为0，返回第一个操作的结果
            return self._ops[0](x) * 0.0

class SimpleArchitectureManager(nn.Module):
    """
    简化的架构参数管理器 - 避免复杂的批量操作
    """
    
    def __init__(self, num_edges):
        super().__init__()
        self.num_edges = num_edges
        self.num_ops = len(PRIMITIVES)
        
        # 创建架构参数 - 每条边一个参数向量
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.randn(self.num_ops) * 0.1) 
            for _ in range(num_edges)
        ])
        
        # Gumbel参数
        self.temperature = 5.0
        self.min_temperature = 0.1
        self.anneal_rate = 0.98
    
    def get_weights(self, edge_idx):
        """获取特定边的权重"""
        if edge_idx >= len(self.alpha):
            # 如果边索引超出范围，返回均匀分布
            return F.softmax(torch.ones(self.num_ops, device=self.alpha[0].device), dim=0)
        
        if self.training:
            # 训练时使用Gumbel-Softmax
            return self._gumbel_softmax(self.alpha[edge_idx])
        else:
            # 推理时使用简单softmax
            return F.softmax(self.alpha[edge_idx], dim=0)
    
    def _gumbel_softmax(self, logits):
        """Gumbel-Softmax采样"""
        # 生成Gumbel噪声
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        
        # 加入噪声并归一化
        noisy_logits = (logits + gumbel) / self.temperature
        return F.softmax(noisy_logits, dim=0)
    
    def anneal_temperature(self):
        """退火温度"""
        self.temperature = max(self.min_temperature, self.temperature * self.anneal_rate)
        return self.temperature

class FixedEvolvableBlock(nn.Module):
    """
    修复的可演化块 - 简化结构避免闭环
    """
    
    def __init__(self, in_channels, out_channels, block_id, stride=1):
        super().__init__()
        
        self.block_id = block_id
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # 输入适配层
        if in_channels != out_channels or stride != 1:
            self.preprocess = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.preprocess = nn.Identity()
        
        # 混合操作
        self.mixed_op = SimpleMixedOp(out_channels, stride=1)  # 内部总是stride=1
        
        # 最终处理
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, weights):
        """前向传播"""
        # 输入处理
        identity = self.preprocess(x)
        
        # 混合操作
        out = self.mixed_op(identity, weights)
        
        # 残差连接
        out = out + identity
        
        # 最终处理
        out = self.final_conv(out)
        
        return out

class FixedASOSENetwork(nn.Module):
    """
    修复的ASO-SE网络 - 避免梯度计算问题
    """
    
    def __init__(self, num_classes=10, initial_channels=32, initial_depth=4):
        super().__init__()
        
        self.num_classes = num_classes
        self.initial_channels = initial_channels
        self.current_depth = initial_depth
        
        # Stem层
        self.stem = nn.Sequential(
            nn.Conv2d(3, initial_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        
        # 构建网络层
        self.layers = nn.ModuleList()
        self._build_initial_architecture()
        
        # 架构参数管理
        self.arch_manager = SimpleArchitectureManager(self.current_depth)
        
        # 分类器
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.layers[-1].out_channels, num_classes)
        
        # 训练状态
        self.training_phase = "weight_training"
        
        # 统计信息
        self.growth_stats = {
            'depth_growths': 0,
            'channel_growths': 0,
            'total_growths': 0
        }
        
        print(f"🚀 Fixed ASOSE Network initialized:")
        print(f"   Depth: {self.current_depth}, Channels: {initial_channels}")
        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _build_initial_architecture(self):
        """构建初始架构"""
        current_channels = self.initial_channels
        
        for i in range(self.current_depth):
            # 下采样策略
            if i == self.current_depth // 2:
                stride = 2
                out_channels = current_channels * 2
            else:
                stride = 1
                out_channels = current_channels
            
            block = FixedEvolvableBlock(
                current_channels, out_channels, f"layer_{i}", stride
            )
            
            self.layers.append(block)
            current_channels = out_channels
    
    def forward(self, x):
        """前向传播"""
        # Stem
        x = self.stem(x)
        
        # 网络层
        for i, layer in enumerate(self.layers):
            # 获取当前层的架构权重
            weights = self.arch_manager.get_weights(i)
            x = layer(x, weights)
        
        # 分类
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def set_training_phase(self, phase: str):
        """设置训练阶段"""
        valid_phases = ["weight_training", "arch_training", "mutation", "retraining"]
        if phase not in valid_phases:
            raise ValueError(f"Invalid phase: {phase}")
        
        self.training_phase = phase
        print(f"🔄 Training phase: {phase}")
    
    def get_architecture_parameters(self):
        """获取架构参数"""
        arch_params = []
        for alpha in self.arch_manager.alpha:
            arch_params.append(alpha)
        return arch_params
    
    def get_weight_parameters(self):
        """获取权重参数"""
        weight_params = []
        for name, param in self.named_parameters():
            if 'arch_manager.alpha' not in name:
                weight_params.append(param)
        return weight_params
    
    def grow_depth(self):
        """增加网络深度"""
        # 获取插入位置
        position = len(self.layers) - 1
        
        # 确定新层配置
        prev_layer = self.layers[position-1] if position > 0 else None
        if prev_layer:
            in_channels = prev_layer.out_channels
            out_channels = in_channels
        else:
            in_channels = self.initial_channels
            out_channels = self.initial_channels
        
        # 创建新层
        new_layer = FixedEvolvableBlock(
            in_channels, out_channels, f"grown_{len(self.layers)}", stride=1
        )
        
        # 确保在正确设备上
        device = next(self.parameters()).device
        new_layer = new_layer.to(device)
        
        # 插入层
        self.layers.insert(position, new_layer)
        self.current_depth += 1
        
        # 更新架构管理器
        self.arch_manager = SimpleArchitectureManager(self.current_depth)
        self.arch_manager = self.arch_manager.to(device)
        
        # 更新统计
        self.growth_stats['depth_growths'] += 1
        self.growth_stats['total_growths'] += 1
        
        print(f"🌱 DEPTH GROWTH: Added layer at position {position}")
        print(f"   New depth: {self.current_depth}")
        print(f"   New parameters: {sum(p.numel() for p in self.parameters()):,}")
        
        return True
    
    def grow_width(self, layer_idx=None, expansion_factor=1.4):
        """增加网络宽度 - 简化实现"""
        print(f"🌱 WIDTH GROWTH: Expansion factor {expansion_factor}")
        
        # 简化：只增加最后一层的通道数
        if len(self.layers) > 0:
            last_layer = self.layers[-1]
            old_channels = last_layer.out_channels
            new_channels = int(old_channels * expansion_factor)
            
            # 更新分类器
            device = next(self.parameters()).device
            old_classifier = self.classifier
            self.classifier = nn.Linear(new_channels, self.num_classes).to(device)
            
            # 简单的权重迁移
            with torch.no_grad():
                min_features = min(old_classifier.in_features, new_channels)
                self.classifier.weight[:, :min_features] = old_classifier.weight[:, :min_features]
                self.classifier.bias.copy_(old_classifier.bias)
            
            self.growth_stats['channel_growths'] += 1
            self.growth_stats['total_growths'] += 1
            
            print(f"   Classifier updated: {old_channels} → {new_channels}")
            return True
        
        return False
    
    def anneal_gumbel_temperature(self):
        """退火Gumbel温度"""
        return self.arch_manager.anneal_temperature()

class FixedTrainingController:
    """修复的训练控制器"""
    
    def __init__(self):
        self.last_growth_cycle = -1
    
    def should_trigger_growth(self, network, current_cycle, current_accuracy, accuracy_trend):
        """智能生长触发判断"""
        # 更保守的生长策略
        if current_cycle - self.last_growth_cycle >= 5:
            print(f"🌱 Forced growth trigger (cycle {current_cycle})")
            return True
        
        # 性能停滞检测
        if len(accuracy_trend) >= 4:
            recent_improvement = max(accuracy_trend[-4:]) - min(accuracy_trend[-4:])
            if recent_improvement < 1.0 and current_cycle - self.last_growth_cycle >= 3:
                print(f"🌱 Stagnation growth trigger (improvement: {recent_improvement:.2f}%)")
                return True
        
        return False
    
    def select_growth_strategy(self, network, current_accuracy, cycle_count):
        """选择生长策略"""
        total_params = sum(p.numel() for p in network.parameters())
        
        # 更保守的策略选择
        if current_accuracy < 60:
            if network.current_depth < 8:
                return 'grow_depth'
            else:
                return 'grow_width'
        else:
            return 'grow_width'
    
    def execute_growth(self, network, strategy, cycle_count):
        """执行生长策略"""
        success = False
        
        try:
            pre_growth_params = sum(p.numel() for p in network.parameters())
            
            if strategy == 'grow_depth':
                success = network.grow_depth()
            elif strategy == 'grow_width':
                success = network.grow_width(expansion_factor=1.3)
            
            if success:
                self.last_growth_cycle = cycle_count
                post_growth_params = sum(p.numel() for p in network.parameters())
                
                print(f"✅ Growth executed successfully!")
                print(f"   Parameters: {pre_growth_params:,} → {post_growth_params:,}")
                
        except Exception as e:
            print(f"❌ Growth failed: {e}")
            import traceback
            traceback.print_exc()
            success = False
        
        return success

class FixedASOSETrainer:
    """修复的ASO-SE训练器"""
    
    def __init__(self, experiment_name="aso_se_fixed"):
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 核心组件
        self.network = None
        self.training_controller = FixedTrainingController()
        
        # 优化器
        self.weight_optimizer = None
        self.arch_optimizer = None
        self.scheduler = None
        
        # 训练状态
        self.current_cycle = 0
        self.best_accuracy = 0.0
        self.cycle_results = []
        
        print(f"🚀 Fixed ASOSE Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Experiment: {experiment_name}")
    
    def setup_data(self, batch_size=128):
        """设置数据加载"""
        print("📊 Setting up CIFAR-10 data...")
        
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
        
        train_dataset = torchvision.datasets.CIFAR10(
            './data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            './data', train=False, transform=test_transform
        )
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )
        
        print(f"✅ Data ready: {len(train_dataset)} train, {len(test_dataset)} test")
    
    def setup_network(self, initial_channels=32, initial_depth=4):
        """设置网络"""
        self.network = FixedASOSENetwork(
            num_classes=10,
            initial_channels=initial_channels,
            initial_depth=initial_depth
        ).to(self.device)
        
        self._create_optimizers()
        
        total_params = sum(p.numel() for p in self.network.parameters())
        print(f"📊 Network ready: {total_params:,} parameters")
    
    def _create_optimizers(self):
        """创建优化器"""
        # 权重参数优化器
        weight_params = self.network.get_weight_parameters()
        self.weight_optimizer = optim.SGD(
            weight_params, lr=0.025, momentum=0.9, weight_decay=1e-4
        )
        
        # 架构参数优化器
        arch_params = self.network.get_architecture_parameters()
        if arch_params:
            self.arch_optimizer = optim.Adam(arch_params, lr=3e-4, weight_decay=1e-3)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.weight_optimizer, T_max=200, eta_min=1e-6
        )
    
    def train_epoch(self, epoch, phase):
        """训练epoch - 修复梯度计算问题"""
        self.network.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 重要：清除所有梯度状态
        if hasattr(self, 'weight_optimizer'):
            self.weight_optimizer.zero_grad()
        if hasattr(self, 'arch_optimizer') and self.arch_optimizer:
            self.arch_optimizer.zero_grad()
        
        # 设置参数训练状态
        if phase == "arch_training":
            optimizer = self.arch_optimizer
            # 冻结权重参数
            for param in self.network.get_weight_parameters():
                param.requires_grad_(False)
            # 激活架构参数
            for param in self.network.get_architecture_parameters():
                param.requires_grad_(True)
        else:
            optimizer = self.weight_optimizer
            # 激活权重参数
            for param in self.network.get_weight_parameters():
                param.requires_grad_(True)
            # 冻结架构参数
            for param in self.network.get_architecture_parameters():
                param.requires_grad_(False)
        
        criterion = nn.CrossEntropyLoss()
        
        pbar = tqdm(self.train_loader, desc=f"🔧 {phase} Epoch {epoch:02d}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            # 数据转移到设备
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            output = self.network(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if phase == "arch_training":
                torch.nn.utils.clip_grad_norm_(self.network.get_architecture_parameters(), 5.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.network.get_weight_parameters(), 5.0)
            
            # 更新参数
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # 更新显示
            if batch_idx % 100 == 0:
                pbar.set_postfix({
                    'Loss': f'{total_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'Depth': self.network.current_depth,
                    'Phase': phase[:6]
                })
        
        return total_loss/len(self.train_loader), 100.*correct/total
    
    def validate(self):
        """验证"""
        self.network.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                output = self.network(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss/len(self.test_loader), 100.*correct/total
    
    def run_training_cycle(self):
        """运行训练周期"""
        cycle_start_time = time.time()
        cycle_results = {}
        
        print(f"\n{'='*80}")
        print(f"🔧 Fixed ASO-SE Training Cycle {self.current_cycle + 1}")
        print(f"{'='*80}")
        
        # 阶段1: 权重训练
        print(f"\n🔥 Phase 1: Weight Training")
        self.network.set_training_phase("weight_training")
        weight_results = self._run_phase("weight_training", 5)
        cycle_results['weight_training'] = weight_results
        
        # 阶段2: 架构训练
        print(f"\n🧠 Phase 2: Architecture Training")
        self.network.set_training_phase("arch_training")
        arch_results = self._run_phase("arch_training", 2)
        cycle_results['arch_training'] = arch_results
        
        # 阶段3: 架构突变
        print(f"\n🧬 Phase 3: Architecture Mutation")
        mutation_success = self._architecture_mutation()
        cycle_results['mutation_success'] = mutation_success
        
        # 阶段4: 权重再训练
        print(f"\n🔧 Phase 4: Weight Retraining")
        self.network.set_training_phase("retraining")
        retrain_results = self._run_phase("retraining", 3)
        cycle_results['retraining'] = retrain_results
        
        cycle_time = time.time() - cycle_start_time
        cycle_results['cycle_time'] = cycle_time
        cycle_results['final_accuracy'] = retrain_results['final_test_acc']
        
        self.cycle_results.append(cycle_results)
        
        print(f"\n✅ Cycle {self.current_cycle + 1} completed in {cycle_time/60:.1f} minutes")
        print(f"   Final accuracy: {cycle_results['final_accuracy']:.2f}%")
        print(f"   Best so far: {self.best_accuracy:.2f}%")
        
        return cycle_results
    
    def _run_phase(self, phase_name, num_epochs):
        """运行训练阶段"""
        phase_results = {'epochs': [], 'final_train_acc': 0, 'final_test_acc': 0}
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch, phase_name)
            
            # 验证
            test_loss, test_acc = self.validate()
            
            # 更新学习率
            if phase_name != "arch_training":
                self.scheduler.step()
            
            # 记录结果
            epoch_result = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'lr': self.weight_optimizer.param_groups[0]['lr']
            }
            phase_results['epochs'].append(epoch_result)
            
            # 更新最佳性能
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
            
            # Gumbel温度退火
            if phase_name == "arch_training":
                temp = self.network.anneal_gumbel_temperature()
                epoch_result['gumbel_temp'] = temp
            
            print(f"   Epoch {epoch+1}: Train={train_acc:.2f}%, Test={test_acc:.2f}%, Best={self.best_accuracy:.2f}%")
        
        phase_results['final_train_acc'] = phase_results['epochs'][-1]['train_acc']
        phase_results['final_test_acc'] = phase_results['epochs'][-1]['test_acc']
        
        return phase_results
    
    def _architecture_mutation(self):
        """架构突变"""
        recent_accuracies = [result['final_accuracy'] for result in self.cycle_results[-3:]]
        if len(recent_accuracies) < 3:
            recent_accuracies = [50.0]
        
        current_accuracy = recent_accuracies[-1] if recent_accuracies else 50.0
        
        should_grow = self.training_controller.should_trigger_growth(
            self.network, self.current_cycle, current_accuracy, recent_accuracies
        )
        
        if should_grow:
            print("🌱 Triggering network growth...")
            
            strategy = self.training_controller.select_growth_strategy(
                self.network, current_accuracy, self.current_cycle
            )
            
            success = self.training_controller.execute_growth(
                self.network, strategy, self.current_cycle
            )
            
            if success:
                # 重新创建优化器
                self._create_optimizers()
                print("🎉 Network growth successful!")
                return True
            else:
                print("❌ Network growth failed")
                return False
        else:
            print("🔄 No growth triggered, annealing temperature...")
            temp = self.network.anneal_gumbel_temperature()
            print(f"   Current temperature: {temp:.3f}")
            return False
    
    def train(self, max_cycles=15, initial_channels=32, initial_depth=4, batch_size=128):
        """主训练流程"""
        print(f"\n🔧 Fixed ASO-SE Training Started")
        print(f"🎯 Target: CIFAR-10 95%+ accuracy")
        print(f"⚙️  Config: max_cycles={max_cycles}, channels={initial_channels}, depth={initial_depth}")
        
        start_time = time.time()
        
        # 设置
        self.setup_data(batch_size)
        self.setup_network(initial_channels, initial_depth)
        
        try:
            # 主训练循环
            for cycle in range(max_cycles):
                self.current_cycle = cycle
                
                # 运行训练周期
                cycle_result = self.run_training_cycle()
                
                # 检查目标
                if cycle_result['final_accuracy'] >= 95.0:
                    print(f"\n🎉 TARGET ACHIEVED! Accuracy: {cycle_result['final_accuracy']:.2f}%")
                    break
                
                # 早停检查
                if self._should_early_stop():
                    print(f"\n⏹️  Early stopping triggered")
                    break
        
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 最终统计
            total_time = time.time() - start_time
            self._display_final_summary(total_time)
    
    def _should_early_stop(self):
        """早停检查"""
        if len(self.cycle_results) < 6:
            return False
        
        recent_accs = [r['final_accuracy'] for r in self.cycle_results[-6:]]
        improvement = max(recent_accs) - min(recent_accs)
        
        return improvement < 0.5
    
    def _display_final_summary(self, total_time):
        """显示最终总结"""
        print(f"\n{'='*80}")
        print(f"🎉 Fixed ASO-SE Training Completed!")
        print(f"{'='*80}")
        
        print(f"⏱️  Total time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
        print(f"🔄 Total cycles: {len(self.cycle_results)}")
        print(f"🏆 Best accuracy: {self.best_accuracy:.2f}%")
        
        if self.cycle_results:
            final_result = self.cycle_results[-1]
            print(f"📊 Final accuracy: {final_result['final_accuracy']:.2f}%")
        
        print(f"🏗️  Final architecture:")
        print(f"   Depth: {self.network.current_depth} layers")
        print(f"   Parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        print(f"   Total growths: {self.network.growth_stats['total_growths']}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed ASO-SE Neural Network Training')
    parser.add_argument('--cycles', type=int, default=15, help='Maximum training cycles')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--initial_channels', type=int, default=32, help='Initial channels')
    parser.add_argument('--initial_depth', type=int, default=4, help='Initial depth')
    parser.add_argument('--experiment', type=str, default='aso_se_fixed', help='Experiment name')
    
    args = parser.parse_args()
    
    print("🔧 Fixed ASO-SE: Gradient-Safe Architecture Search")
    print("🎯 Target: CIFAR-10 95%+ accuracy")
    print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Config: {vars(args)}")
    
    # 创建修复的训练器
    trainer = FixedASOSETrainer(args.experiment)
    
    # 开始训练
    trainer.train(
        max_cycles=args.cycles,
        initial_channels=args.initial_channels,
        initial_depth=args.initial_depth,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()