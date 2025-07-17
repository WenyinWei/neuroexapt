#!/usr/bin/env python3
"""
重构版ASO-SE神经架构搜索训练脚本
使用全新设计的稳定架构搜索框架
解决架构搜索阶段性能崩溃问题
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

# 添加neuroexapt到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入重构后的组件
try:
    from neuroexapt.core.aso_se_operators import StableMixedOp, PRIMITIVES, create_operation
    from neuroexapt.core.aso_se_architecture import (
        StableGumbelSampler, 
        ArchitectureParameterManager, 
        ProgressiveArchitectureNetwork
    )
    from neuroexapt.core.aso_se_trainer import StableASO_SETrainer
    print("✅ 使用重构后的ASO-SE框架")
except ImportError as e:
    print(f"⚠️ 重构框架导入失败: {e}")
    print("🔄 使用内联重构实现")
    
    # 内联重构实现
    class StableOp(nn.Module):
        """稳定的基础操作类"""
        
        def __init__(self, C, stride, affine=True):
            super().__init__()
            self.C = C
            self.stride = stride
            self.affine = affine
        
        def forward(self, x):
            raise NotImplementedError
    
    class Identity(StableOp):
        """恒等映射"""
        
        def forward(self, x):
            if self.stride == 1:
                return x
            else:
                return x[:, :, ::self.stride, ::self.stride]
    
    class Zero(StableOp):
        """零操作"""
        
        def forward(self, x):
            if self.stride == 1:
                return torch.zeros_like(x)
            else:
                shape = list(x.shape)
                shape[2] = (shape[2] + self.stride - 1) // self.stride
                shape[3] = (shape[3] + self.stride - 1) // self.stride
                return torch.zeros(shape, dtype=x.dtype, device=x.device)
    
    class ReLUConvBN(StableOp):
        """ReLU + Conv + BatchNorm"""
        
        def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
            super().__init__(C_out, stride, affine)
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(C_out, affine=affine)
            )
        
        def forward(self, x):
            return self.op(x)
    
    class SepConv(StableOp):
        """深度可分离卷积"""
        
        def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
            super().__init__(C_out, stride, affine)
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, 
                         padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_in, affine=affine),
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, 
                         padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )
        
        def forward(self, x):
            return self.op(x)
    
    class FactorizedReduce(StableOp):
        """因式化降维"""
        
        def __init__(self, C_in, C_out, affine=True):
            super().__init__(C_out, 2, affine)
            assert C_out % 2 == 0
            self.relu = nn.ReLU(inplace=False)
            self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
            self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
            self.bn = nn.BatchNorm2d(C_out, affine=affine)
        
        def forward(self, x):
            x = self.relu(x)
            out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
            out = self.bn(out)
            return out
    
    # 定义操作符映射
    PRIMITIVES = [
        'none',
        'max_pool_3x3',
        'avg_pool_3x3', 
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'conv_1x1',
        'conv_3x3',
    ]
    
    def create_operation(primitive, C_in, C_out, stride, affine=True):
        """创建操作实例"""
        if primitive == 'none':
            return Zero(C_out, stride, affine)
        elif primitive == 'max_pool_3x3':
            if C_in == C_out:
                return nn.Sequential(
                    nn.MaxPool2d(3, stride=stride, padding=1),
                    nn.BatchNorm2d(C_in, affine=affine)
                )
            else:
                return nn.Sequential(
                    nn.MaxPool2d(3, stride=stride, padding=1),
                    nn.Conv2d(C_in, C_out, 1, bias=False),
                    nn.BatchNorm2d(C_out, affine=affine)
                )
        elif primitive == 'avg_pool_3x3':
            if C_in == C_out:
                return nn.Sequential(
                    nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
                    nn.BatchNorm2d(C_in, affine=affine)
                )
            else:
                return nn.Sequential(
                    nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
                    nn.Conv2d(C_in, C_out, 1, bias=False),
                    nn.BatchNorm2d(C_out, affine=affine)
                )
        elif primitive == 'skip_connect':
            if stride == 1 and C_in == C_out:
                return Identity(C_out, stride, affine)
            else:
                return FactorizedReduce(C_in, C_out, affine)
        elif primitive == 'sep_conv_3x3':
            return SepConv(C_in, C_out, 3, stride, 1, affine)
        elif primitive == 'sep_conv_5x5':
            return SepConv(C_in, C_out, 5, stride, 2, affine)
        elif primitive == 'conv_1x1':
            return ReLUConvBN(C_in, C_out, 1, stride, 0, affine)
        elif primitive == 'conv_3x3':
            return ReLUConvBN(C_in, C_out, 3, stride, 1, affine)
        else:
            raise ValueError(f"Unknown primitive: {primitive}")
    
    class StableMixedOp(nn.Module):
        """稳定的混合操作"""
        
        def __init__(self, C_in, C_out, stride, primitives=None):
            super().__init__()
            self.C_in = C_in
            self.C_out = C_out
            self.stride = stride
            
            if primitives is None:
                primitives = PRIMITIVES
            
            self.primitives = primitives
            self.operations = nn.ModuleList()
            
            # 创建所有候选操作
            for primitive in primitives:
                op = create_operation(primitive, C_in, C_out, stride)
                self.operations.append(op)
        
        def forward(self, x, weights):
            """前向传播，使用稳定的加权求和"""
            # 输入验证
            if torch.isnan(weights).any() or torch.isinf(weights).any():
                # 安全回退：使用skip_connect
                skip_idx = 3 if len(self.operations) > 3 else 0
                return self.operations[skip_idx](x)
            
            # 确保权重和为1
            weights = F.softmax(weights, dim=0)
            
            # 智能操作选择策略
            max_weight_idx = torch.argmax(weights).item()
            max_weight = weights[max_weight_idx].item()
            
            # 如果有明显的主导操作 (>0.8)，主要使用该操作
            if max_weight > 0.8:
                try:
                    return self.operations[max_weight_idx](x)
                except Exception:
                    # 回退到skip连接
                    skip_idx = 3 if len(self.operations) > 3 else 0
                    return self.operations[skip_idx](x)
            
            # 计算加权和，但只使用权重>0.02的操作
            result = 0.0
            total_weight = 0.0
            
            for i, op in enumerate(self.operations):
                weight = weights[i]
                if weight > 0.02:  # 忽略权重太小的操作
                    try:
                        op_result = op(x)
                        result += weight * op_result
                        total_weight += weight
                    except Exception:
                        continue
            
            if total_weight < 0.1:
                # 如果所有操作都失败，使用skip连接
                skip_idx = 3 if len(self.operations) > 3 else 0
                return self.operations[skip_idx](x)
            
            return result / total_weight if total_weight > 0 else result
    
    class StableGumbelSampler(nn.Module):
        """稳定的Gumbel采样器"""
        
        def __init__(self, tau_max=1.2, tau_min=0.2, anneal_rate=0.999):
            super().__init__()
            self.tau_max = tau_max
            self.tau_min = tau_min
            self.tau = tau_max
            self.anneal_rate = anneal_rate
            
        def forward(self, logits, hard=True):
            """Gumbel Softmax采样"""
            if not self.training:
                # 推理时使用argmax
                y_hard = torch.zeros_like(logits)
                y_hard.scatter_(-1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
                return y_hard
            
            # 训练时使用更稳定的Gumbel-Softmax
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            y_soft = F.softmax((logits + gumbel_noise) / self.tau, dim=-1)
            
            if hard:
                # 更稳定的straight-through estimator
                y_hard = torch.zeros_like(y_soft)
                y_hard.scatter_(-1, torch.argmax(y_soft, dim=-1, keepdim=True), 1.0)
                return y_hard - y_soft.detach() + y_soft
            else:
                return y_soft
        
        def anneal_temperature(self):
            """温度退火"""
            self.tau = max(self.tau_min, self.tau * self.anneal_rate)
    
    class ArchitectureParameterManager(nn.Module):
        """架构参数管理器"""
        
        def __init__(self, num_nodes, primitives=None):
            super().__init__()
            if primitives is None:
                primitives = PRIMITIVES
            
            self.num_nodes = num_nodes
            self.primitives = primitives
            self.num_ops = len(primitives)
            
            # 架构参数 - 使用更保守的初始化
            self.alpha = nn.ParameterList()
            for i in range(num_nodes):
                alpha = nn.Parameter(torch.randn(self.num_ops) * 0.05)  # 更小的初始化
                with torch.no_grad():
                    # 给skip_connect更高的初始权重
                    skip_idx = 3 if 'skip_connect' in primitives else 0
                    alpha[skip_idx] += 0.5
                    # 降低none操作的权重
                    if 'none' in primitives:
                        none_idx = 0
                        alpha[none_idx] -= 0.5
                self.alpha.append(alpha)
            
            # Gumbel采样器
            self.sampler = StableGumbelSampler()
            
            # 训练阶段控制
            self.training_phase = 'warmup'
        
        def get_architecture_weights(self, node_idx, mode='gumbel'):
            """获取指定节点的架构权重"""
            if node_idx >= len(self.alpha):
                raise IndexError(f"Node index {node_idx} out of range")
            
            logits = self.alpha[node_idx]
            
            if self.training_phase == 'warmup':
                # warmup阶段使用固定的skip_connect
                weights = torch.zeros_like(logits)
                skip_idx = 3 if 'skip_connect' in self.primitives else 0
                weights[skip_idx] = 1.0
                return weights.detach()
            
            elif self.training_phase in ['search', 'growth']:
                # 搜索阶段使用更保守的Gumbel采样
                return self.sampler(logits.unsqueeze(0), hard=True).squeeze(0)
            
            elif self.training_phase == 'optimize':
                # 优化阶段使用确定性选择
                weights = torch.zeros_like(logits)
                best_idx = torch.argmax(logits).item()
                weights[best_idx] = 1.0
                return weights.detach()
            
            else:
                # 默认回退
                weights = torch.zeros_like(logits)
                skip_idx = 3 if 'skip_connect' in self.primitives else 0
                weights[skip_idx] = 1.0
                return weights.detach()
        
        def set_training_phase(self, phase):
            """设置训练阶段"""
            print(f"🔄 设置训练阶段: {phase}")
            self.training_phase = phase
            if phase == 'search':
                self.sampler.tau = 1.0  # 重置温度
        
        def get_current_architecture(self):
            """获取当前架构"""
            architecture = []
            for alpha in self.alpha:
                best_op_idx = torch.argmax(alpha).item()
                best_op_name = self.primitives[best_op_idx]
                architecture.append(best_op_name)
            return architecture
        
        def get_architecture_entropy(self):
            """计算架构熵"""
            total_entropy = 0.0
            for alpha in self.alpha:
                probs = F.softmax(alpha, dim=0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                total_entropy += entropy.item()
            return total_entropy / len(self.alpha)
        
        def anneal_temperature(self):
            """温度退火"""
            self.sampler.anneal_temperature()


class StableASO_SECell(nn.Module):
    """稳定的ASO-SE单元"""
    
    def __init__(self, C_in, C_out, stride, node_id, arch_manager):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.node_id = node_id
        self.arch_manager = arch_manager
        
        # 预处理层（确保输入输出通道匹配）
        if C_in != C_out or stride != 1:
            self.preprocess = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(C_out)
            )
        else:
            self.preprocess = None
        
        # 混合操作
        self.mixed_op = StableMixedOp(C_out, C_out, 1, PRIMITIVES)
        
    def forward(self, x):
        """前向传播"""
        # 预处理
        if self.preprocess is not None:
            x = self.preprocess(x)
        
        # 获取架构权重
        try:
            weights = self.arch_manager.get_architecture_weights(self.node_id)
        except Exception as e:
            print(f"⚠️ 节点 {self.node_id} 权重获取失败: {e}")
            # 回退到skip连接
            return x
        
        # 混合操作
        return self.mixed_op(x, weights)


class StableProgressiveNetwork(nn.Module):
    """稳定的渐进式架构网络"""
    
    def __init__(self, input_channels=3, init_channels=32, num_classes=10, init_depth=4):
        super().__init__()
        self.input_channels = input_channels
        self.init_channels = init_channels
        self.num_classes = num_classes
        self.current_depth = init_depth
        
        # 主干网络
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, init_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=False)
        )
        
        # 架构管理器
        self.arch_manager = ArchitectureParameterManager(init_depth)
        
        # 搜索单元
        self.cells = nn.ModuleList()
        current_channels = init_channels
        
        for i in range(init_depth):
            # 每隔几层增加通道数并降采样
            if i > 0 and i % 2 == 0:
                stride = 2
                out_channels = min(current_channels * 2, 256)  # 限制最大通道数
            else:
                stride = 1
                out_channels = current_channels
            
            cell = StableASO_SECell(current_channels, out_channels, stride, i, self.arch_manager)
            self.cells.append(cell)
            current_channels = out_channels
        
        # 分类头
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, num_classes)
        
        print(f"🚀 稳定网络初始化:")
        print(f"   深度: {self.current_depth} 层")
        print(f"   初始通道: {init_channels}")
        print(f"   当前通道: {current_channels}")
        print(f"   参数量: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
    def forward(self, x):
        """前向传播"""
        x = self.stem(x)
        
        for cell in self.cells:
            x = cell(x)
        
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def set_training_phase(self, phase):
        """设置训练阶段"""
        self.arch_manager.set_training_phase(phase)
    
    def get_architecture_info(self):
        """获取架构信息"""
        architecture = self.arch_manager.get_current_architecture()
        entropy = self.arch_manager.get_architecture_entropy()
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'depth': self.current_depth,
            'architecture': architecture,
            'entropy': entropy,
            'parameters': total_params,
            'temperature': self.arch_manager.sampler.tau
        }


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_data(batch_size=128):
    """设置数据加载器"""
    # 数据增强
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
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"📊 CIFAR-10数据加载完成: 训练集 {len(train_dataset)}, 测试集 {len(test_dataset)}")
    return train_loader, test_loader


def main():
    print("🔧 重构版ASO-SE: 稳定的神经架构搜索")
    print("   目标: CIFAR-10 高准确率")
    print("   策略: 四阶段稳定训练")
    print("   框架: 全新重构架构")
    
    # 设置环境
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   设备: {device}")
    
    # 配置
    config = {
        'batch_size': 128,
        'num_epochs': 60,  # 减少epoch数以便快速测试
        'warmup_epochs': 10,
        'search_epochs': 20,
        'growth_epochs': 20,
        'optimize_epochs': 10,
        'weight_lr': 0.025,
        'arch_lr': 3e-4,
        'arch_update_freq': 8,  # 更低的架构更新频率
    }
    
    # 数据和模型
    train_loader, test_loader = setup_data(config['batch_size'])
    network = StableProgressiveNetwork(
        input_channels=3,
        init_channels=32,
        num_classes=10,
        init_depth=4
    ).to(device)
    
    # 优化器
    weight_params = []
    arch_params = []
    
    for name, param in network.named_parameters():
        if 'arch_manager.alpha' in name:
            arch_params.append(param)
        else:
            weight_params.append(param)
    
    weight_optimizer = optim.SGD(
        weight_params, lr=config['weight_lr'], 
        momentum=0.9, weight_decay=3e-4
    )
    arch_optimizer = optim.Adam(
        arch_params, lr=config['arch_lr'], 
        betas=(0.5, 0.999), weight_decay=1e-3
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        weight_optimizer, T_max=config['num_epochs']
    )
    
    print(f"⚙️ 优化器设置:")
    print(f"   权重参数: {len(weight_params)}")
    print(f"   架构参数: {len(arch_params)}")
    
    # 训练循环
    best_accuracy = 0.0
    current_phase = 'warmup'
    phase_epochs = 0
    
    print(f"\n🔧 开始稳定ASO-SE训练")
    print(f"{'='*60}")
    
    for epoch in range(config['num_epochs']):
        # 更新阶段
        phase_epochs += 1
        old_phase = current_phase
        
        if current_phase == 'warmup' and phase_epochs >= config['warmup_epochs']:
            current_phase = 'search'
            phase_epochs = 0
            network.set_training_phase('search')
            print(f"🔄 进入架构搜索阶段")
        
        elif current_phase == 'search' and phase_epochs >= config['search_epochs']:
            current_phase = 'optimize'  # 跳过growth阶段进行测试
            phase_epochs = 0
            network.set_training_phase('optimize')
            print(f"🔄 进入优化阶段")
        
        # 训练
        network.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)
            
            # 根据阶段选择优化策略
            if current_phase == 'warmup' or current_phase == 'optimize':
                # 只优化权重
                weight_optimizer.zero_grad()
                outputs = network(data)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                weight_optimizer.step()
                
            elif current_phase == 'search':
                # 交替优化，更低频率的架构更新
                if batch_idx % config['arch_update_freq'] == 0:
                    # 架构优化
                    arch_optimizer.zero_grad()
                    outputs = network(data)
                    loss = F.cross_entropy(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(arch_params, 5.0)
                    arch_optimizer.step()
                    network.arch_manager.anneal_temperature()
                else:
                    # 权重优化
                    weight_optimizer.zero_grad()
                    outputs = network(data)
                    loss = F.cross_entropy(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(weight_params, 5.0)
                    weight_optimizer.step()
            
            # 统计
            with torch.no_grad():
                if 'outputs' not in locals() or outputs is None:
                    outputs = network(data)
                total_loss += F.cross_entropy(outputs, targets).item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            accuracy = 100. * correct / total
            arch_info = network.get_architecture_info()
            
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.3f}',
                'Acc': f'{accuracy:.2f}%',
                'Phase': current_phase,
                'Temp': f'{arch_info["temperature"]:.3f}',
            })
        
        # 评估
        network.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = network(data)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * test_correct / test_total
        train_acc = 100. * correct / total
        train_loss = total_loss / len(train_loader)
        
        # 更新学习率
        scheduler.step()
        
        # 更新最佳精度
        if test_acc > best_accuracy:
            best_accuracy = test_acc
        
        # 定期汇报
        if (epoch + 1) % 3 == 0:
            arch_info = network.get_architecture_info()
            print(f"\n📊 Epoch {epoch+1}/{config['num_epochs']} | 阶段: {current_phase}")
            print(f"   训练损失: {train_loss:.4f} | 训练精度: {train_acc:.2f}%")
            print(f"   测试精度: {test_acc:.2f}% | 最佳: {best_accuracy:.2f}%")
            print(f"   架构熵: {arch_info['entropy']:.3f} | 温度: {arch_info['temperature']:.3f}")
            
            if current_phase == 'search':
                print(f"   当前架构: {arch_info['architecture']}")
    
    # 训练完成
    print(f"\n🎉 训练完成!")
    print(f"   最佳精度: {best_accuracy:.2f}%")
    print(f"   最终架构: {network.get_architecture_info()['architecture']}")


if __name__ == '__main__':
    main() 