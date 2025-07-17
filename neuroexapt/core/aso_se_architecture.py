"""
ASO-SE 架构管理框架
重新设计的稳定架构搜索管理器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .aso_se_operators import StableMixedOp, PRIMITIVES


class StableGumbelSampler(nn.Module):
    """稳定的Gumbel采样器"""
    
    def __init__(self, tau_max=2.0, tau_min=0.1, anneal_rate=0.998):
        super().__init__()
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.tau = tau_max
        self.anneal_rate = anneal_rate
        self.training_steps = 0
        
    def forward(self, logits, hard=True):
        """Gumbel Softmax采样"""
        if not self.training:
            # 推理时使用argmax
            y_hard = torch.zeros_like(logits)
            y_hard.scatter_(-1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
            return y_hard
        
        # 训练时使用Gumbel Softmax
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        y_soft = F.softmax((logits + gumbel_noise) / self.tau, dim=-1)
        
        if hard:
            # Straight-through estimator
            y_hard = torch.zeros_like(y_soft)
            y_hard.scatter_(-1, torch.argmax(y_soft, dim=-1, keepdim=True), 1.0)
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_soft
    
    def anneal_temperature(self):
        """温度退火"""
        self.training_steps += 1
        self.tau = max(self.tau_min, self.tau * self.anneal_rate)
    
    def reset_temperature(self, tau=None):
        """重置温度"""
        if tau is None:
            tau = self.tau_max
        self.tau = tau
        self.training_steps = 0


class ArchitectureParameterManager(nn.Module):
    """架构参数管理器"""
    
    def __init__(self, num_nodes, primitives=None, init_strategy='uniform'):
        super().__init__()
        if primitives is None:
            primitives = PRIMITIVES
        
        self.num_nodes = num_nodes
        self.primitives = primitives
        self.num_ops = len(primitives)
        self.init_strategy = init_strategy
        
        # 架构参数
        self.alpha = nn.ParameterList()
        for i in range(num_nodes):
            self.alpha.append(self._create_alpha())
        
        # Gumbel采样器
        self.sampler = StableGumbelSampler()
        
        # 训练阶段控制
        self.training_phase = 'warmup'
        self.fixed_architecture = None
        
    def _create_alpha(self):
        """创建单个节点的架构参数"""
        if self.init_strategy == 'uniform':
            alpha = nn.Parameter(torch.randn(self.num_ops) * 0.1)
        elif self.init_strategy == 'skip_biased':
            alpha = nn.Parameter(torch.randn(self.num_ops) * 0.1)
            with torch.no_grad():
                # 给skip_connect更高的初始权重
                skip_idx = self.primitives.index('skip_connect') if 'skip_connect' in self.primitives else 0
                alpha[skip_idx] += 1.0
                # 降低none操作的权重
                if 'none' in self.primitives:
                    none_idx = self.primitives.index('none')
                    alpha[none_idx] -= 1.0
        else:
            alpha = nn.Parameter(torch.zeros(self.num_ops))
        
        return alpha
    
    def get_architecture_weights(self, node_idx, mode='gumbel'):
        """获取指定节点的架构权重"""
        if node_idx >= len(self.alpha):
            raise IndexError(f"Node index {node_idx} out of range")
        
        logits = self.alpha[node_idx]
        
        if self.training_phase == 'warmup':
            # warmup阶段使用固定架构
            if self.fixed_architecture is None:
                # 使用skip_connect
                weights = torch.zeros_like(logits)
                skip_idx = self.primitives.index('skip_connect') if 'skip_connect' in self.primitives else 0
                weights[skip_idx] = 1.0
                return weights.detach()
            else:
                # 使用预设的固定架构
                weights = torch.zeros_like(logits)
                op_name = self.fixed_architecture[node_idx]
                op_idx = self.primitives.index(op_name)
                weights[op_idx] = 1.0
                return weights.detach()
        
        elif self.training_phase in ['search', 'growth']:
            # 搜索阶段使用Gumbel采样
            if mode == 'gumbel':
                return self.sampler(logits.unsqueeze(0), hard=True).squeeze(0)
            elif mode == 'softmax':
                return F.softmax(logits, dim=0)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        elif self.training_phase == 'optimize':
            # 优化阶段使用确定性选择
            weights = torch.zeros_like(logits)
            best_idx = torch.argmax(logits).item()
            weights[best_idx] = 1.0
            return weights.detach()
        
        else:
            raise ValueError(f"Unknown training phase: {self.training_phase}")
    
    def set_training_phase(self, phase):
        """设置训练阶段"""
        self.training_phase = phase
        if phase == 'search':
            self.sampler.reset_temperature(tau=1.5)  # 搜索阶段重置温度
        elif phase == 'optimize':
            self.sampler.reset_temperature(tau=0.1)  # 优化阶段使用低温度
    
    def get_current_architecture(self):
        """获取当前架构"""
        architecture = []
        for i in range(len(self.alpha)):
            logits = self.alpha[i]
            best_op_idx = torch.argmax(logits).item()
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
    
    def get_architecture_confidence(self):
        """计算架构置信度（最大权重的平均值）"""
        total_confidence = 0.0
        for alpha in self.alpha:
            probs = F.softmax(alpha, dim=0)
            max_prob = torch.max(probs).item()
            total_confidence += max_prob
        return total_confidence / len(self.alpha)
    
    def anneal_temperature(self):
        """温度退火"""
        self.sampler.anneal_temperature()
    
    def add_node(self):
        """添加新节点（网络生长时使用）"""
        new_alpha = self._create_alpha()
        if len(self.alpha) > 0:
            # 确保设备一致性
            device = self.alpha[0].device
            new_alpha = new_alpha.to(device)
        self.alpha.append(new_alpha)
    
    def smooth_transition_to_search(self):
        """平滑过渡到搜索阶段"""
        with torch.no_grad():
            for alpha in self.alpha:
                # 添加小量噪声，鼓励探索
                noise = torch.randn_like(alpha) * 0.02
                alpha.data += noise
    
    def freeze_architecture(self, architecture=None):
        """冻结架构"""
        if architecture is None:
            architecture = self.get_current_architecture()
        self.fixed_architecture = architecture
    
    def unfreeze_architecture(self):
        """解冻架构"""
        self.fixed_architecture = None


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
            print(f"Warning: Failed to get architecture weights for node {self.node_id}: {e}")
            # 回退到skip连接
            return x
        
        # 混合操作
        return self.mixed_op(x, weights)


class ProgressiveArchitectureNetwork(nn.Module):
    """渐进式架构网络"""
    
    def __init__(self, input_channels=3, init_channels=32, num_classes=10, 
                 init_depth=3, max_depth=8):
        super().__init__()
        self.input_channels = input_channels
        self.init_channels = init_channels
        self.num_classes = num_classes
        self.init_depth = init_depth
        self.max_depth = max_depth
        self.current_depth = init_depth
        
        # 干干网络
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
                out_channels = current_channels * 2
            else:
                stride = 1
                out_channels = current_channels
            
            cell = StableASO_SECell(current_channels, out_channels, stride, i, self.arch_manager)
            self.cells.append(cell)
            current_channels = out_channels
        
        # 分类头
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, num_classes)
        
        # 训练控制
        self.training_phase = 'warmup'
        
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
        self.training_phase = phase
        self.arch_manager.set_training_phase(phase)
    
    def grow_depth(self, num_layers=1):
        """增加网络深度"""
        if self.current_depth + num_layers > self.max_depth:
            print(f"Cannot grow beyond max depth {self.max_depth}")
            return
        
        for _ in range(num_layers):
            # 添加新的架构参数
            self.arch_manager.add_node()
            
            # 添加新的cell
            in_channels = self.cells[-1].C_out if self.cells else self.init_channels
            out_channels = in_channels
            
            new_cell = StableASO_SECell(
                in_channels, out_channels, 1, 
                self.current_depth, self.arch_manager
            )
            
            # 设置设备
            if self.cells:
                device = next(self.cells[0].parameters()).device
                new_cell = new_cell.to(device)
            
            self.cells.append(new_cell)
            self.current_depth += 1
        
        print(f"网络深度增长到: {self.current_depth}")
    
    def get_architecture_info(self):
        """获取架构信息"""
        architecture = self.arch_manager.get_current_architecture()
        entropy = self.arch_manager.get_architecture_entropy()
        confidence = self.arch_manager.get_architecture_confidence()
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'depth': self.current_depth,
            'architecture': architecture,
            'entropy': entropy,
            'confidence': confidence,
            'parameters': total_params,
            'temperature': self.arch_manager.sampler.tau
        }