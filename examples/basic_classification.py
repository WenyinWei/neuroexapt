#!/usr/bin/env python3
"""
NeuroExapt Basic Classification Example

This example demonstrates various optimization strategies for neural architecture search:
1. Standard DARTS-style architecture search
2. Fixed architecture training (bypass MixedOp)
3. Optimized architecture search with caching
4. Performance comparison and benchmarking

Use command line arguments to control different optimization strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import sys
import os
import time
import argparse
from typing import Optional
from tqdm import tqdm
from pathlib import Path

# New high-performance utilities
from neuroexapt.utils.async_dataloader import build_cifar10_pipeline
from neuroexapt.utils.fuse_utils import fuse_model
from neuroexapt.utils.compile_utils import compile_submodules

# Add the parent directory to the path to import neuroexapt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import NeuroExapt core components
from neuroexapt.core.model import Network
from neuroexapt.core.architect import Architect
from neuroexapt.core.operations import SepConv, DilConv, Identity, Zero, FactorizedReduce, ReLUConvBN
from neuroexapt.utils.minimal_monitor import MinimalMonitor
from neuroexapt.utils.utils import AvgrageMeter, accuracy

class FixedCell(nn.Module):
    """
    Fixed architecture cell that bypasses MixedOp computation
    """
    def __init__(self, steps, block_multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(FixedCell, self).__init__()
        self.reduction = reduction
        self.block_multiplier = block_multiplier
        self.steps = steps

        # Preprocessing layers
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        # Fixed operations (high-performance genotype)
        self.ops = nn.ModuleList()
        
        # Create operations for each connection
        for i in range(self.steps):
            for j in range(2 + i):
                # Choose operation type
                if reduction:
                    if j < 2:
                        op_name = 'max_pool_3x3'
                    else:
                        op_name = 'sep_conv_3x3'
                else:
                    if j == 0:
                        op_name = 'sep_conv_3x3'
                    elif j == 1:
                        op_name = 'sep_conv_5x5'
                    else:
                        op_name = 'sep_conv_3x3'
                
                stride = 2 if reduction and j < 2 else 1
                op = self._create_operation(op_name, C, stride)
                self.ops.append(op)

    def _create_operation(self, op_name, C, stride):
        """Create a single operation"""
        if op_name == 'none':
            return Zero(stride)
        elif op_name == 'avg_pool_3x3':
            return nn.Sequential(
                nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
                nn.BatchNorm2d(C, affine=False)
            )
        elif op_name == 'max_pool_3x3':
            return nn.Sequential(
                nn.MaxPool2d(3, stride=stride, padding=1),
                nn.BatchNorm2d(C, affine=False)
            )
        elif op_name == 'skip_connect':
            return Identity() if stride == 1 else FactorizedReduce(C, C, affine=False)
        elif op_name == 'sep_conv_3x3':
            return SepConv(C, C, 3, stride, 1, affine=False)
        elif op_name == 'sep_conv_5x5':
            return SepConv(C, C, 5, stride, 2, affine=False)
        else:
            return Identity()

    def forward(self, s0, s1):
        """Forward pass without architecture parameters"""
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        
        for i in range(self.steps):
            s = 0
            for j in range(2 + i):
                op = self.ops[offset + j]
                h = op(states[j])
                s = s + h
            
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self.block_multiplier:], dim=1)

class FixedNetwork(nn.Module):
    """
    Fixed architecture network that bypasses architecture search
    """
    def __init__(self, C, num_classes, layers, steps=4, block_multiplier=4):
        super(FixedNetwork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._block_multiplier = block_multiplier

        C_curr = self._block_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            
            cell = FixedCell(steps, block_multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, self._block_multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        """Fast forward pass"""
        s0 = s1 = self.stem(input)
        
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

class CachedNetwork:
    """Wrapper for Network class to provide caching functionality"""
    def __init__(self, model):
        self.model = model
        self.cached_normal_weights = None
        self.cached_reduce_weights = None
        self.cache_valid = False
    
    def update_weight_cache(self):
        """Update weight cache"""
        with torch.no_grad():
            self.cached_normal_weights = torch.softmax(self.model.alphas_normal, dim=-1)
            self.cached_reduce_weights = torch.softmax(self.model.alphas_reduce, dim=-1)
            self.cache_valid = True
    
    def forward(self, input):
        """Forward pass with cached weights"""
        if not self.cache_valid:
            self.update_weight_cache()
        
        # Use cached weights directly in Network's forward logic
        import torch.utils.checkpoint as cp
        from neuroexapt.core.model import GatedCell
        
        s0 = s1 = self.model.stem(input)
        
        for i, cell in enumerate(self.model.cells):
            # Determine which set of precomputed weights to use
            if isinstance(cell, GatedCell):
                if cell.cell.reduction:
                    weights = self.cached_reduce_weights
                else:
                    weights = self.cached_normal_weights
            else:
                weights = self.cached_reduce_weights if cell.reduction else self.cached_normal_weights

            if self.model.use_checkpoint and self.model.training:
                # Wrap cell forward in checkpoint to save memory
                def _cell_forward(a, b):
                    return cell(a, b, weights)

                s1_new = cp.checkpoint(_cell_forward, s0, s1)
            else:
                s1_new = cell(s0, s1, weights)

            s0, s1 = s1, s1_new
        
        out = self.model.global_pooling(s1)
        logits = self.model.classifier(out.view(out.size(0), -1))
        return logits
    
    def __call__(self, input):
        return self.forward(input)
    
    def __getattr__(self, name):
        # 🔧 修复：防止__getattr__无穷递归
        # 添加黑名单和递归深度检查
        if name.startswith('_') or name in {'model', 'cached_normal_weights', 'cached_reduce_weights', 'cache_valid'}:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # 防止递归：检查是否正在查找model属性
        if not hasattr(self, '_getattr_in_progress'):
            self._getattr_in_progress = set()
        
        if name in self._getattr_in_progress:
            raise AttributeError(f"递归调用检测：'{name}' 属性查找循环")
        
        try:
            self._getattr_in_progress.add(name)
            # 安全地获取model属性
            if hasattr(self, 'model') and hasattr(self.model, name):
                return getattr(self.model, name)
            else:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        finally:
            self._getattr_in_progress.discard(name)

def setup_args():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description='NeuroExapt Basic Classification')
    
    # Training parameters
    parser.add_argument('--data', type=str, default='./data', help='dataset location')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='training data portion')
    parser.add_argument('--report_freq', type=int, default=20, help='report frequency')
    
    # Architecture parameters
    parser.add_argument('--init_channels', type=int, default=16, help='initial channels')
    parser.add_argument('--layers', type=int, default=8, help='number of layers')
    parser.add_argument('--potential_layers', type=int, default=4, help='potential layers')
    
    # Architecture evolution parameters (renamed from search)
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='arch learning rate')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='arch weight decay')
    parser.add_argument('--arch_update_freq', type=int, default=50, help='arch update frequency')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs')
    parser.add_argument('--use_first_order', action='store_true', help='use first order approximation')
    
    # Optimization strategy - renamed search to exapt
    parser.add_argument('--mode', type=str, default='exapt', 
                       choices=['exapt', 'fixed', 'cached', 'benchmark'],
                       help='optimization mode: exapt (intelligent evolution), fixed (no evolution), cached (optimized), benchmark (compare all)')
    
    # Performance options
    parser.add_argument('--num_workers', type=int, default=4, help='data loader workers')
    parser.add_argument('--async_loader', action='store_true', help='use async DataPipe loader')
    parser.add_argument('--compile_cell', action='store_true', help='torch.compile Cell submodules')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='pin memory')
    parser.add_argument('--disable_cudnn_benchmark', action='store_true', help='disable cudnn benchmark')
    parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint for memory optimization')
    parser.add_argument('--use_model_compile', action='store_true', help='use torch.compile for entire model')
    parser.add_argument('--use_optimized_ops', action='store_true', help='use optimized MixedOp operations')
    parser.add_argument('--progress_tracking', action='store_true', default=True, help='enable detailed progress tracking')
    
    # New Exapt-specific optimizations
    parser.add_argument('--lazy_computation', action='store_true', default=True, help='enable lazy computation for operations')
    parser.add_argument('--smart_caching', action='store_true', default=True, help='enable intelligent result caching')
    parser.add_argument('--early_convergence', action='store_true', default=True, help='enable early convergence detection')
    parser.add_argument('--memory_pool', action='store_true', default=True, help='use pre-allocated memory pools')
    parser.add_argument('--operation_pruning', action='store_true', default=True, help='prune low-weight operations during evolution')
    parser.add_argument('--adaptive_update_freq', action='store_true', default=True, help='adaptively adjust architecture update frequency')
    
    # Advanced backward pass optimizations
    parser.add_argument('--gradient_optimized', action='store_true', default=True, help='use gradient-optimized MixedOp for faster backward pass')
    parser.add_argument('--memory_efficient', action='store_true', help='use memory-efficient MixedOp to reduce GPU memory usage')
    parser.add_argument('--gradient_threshold', type=float, default=0.01, help='threshold for selective gradient computation')
    parser.add_argument('--disable_progress_spam', action='store_true', help='reduce progress output to speed up training')
    parser.add_argument('--quiet', action='store_true', help='minimal output for maximum speed')
    
    # Debug options
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save_dir', type=str, default='./results', help='save directory')
    parser.add_argument('--exp_name', type=str, default='basic_classification', help='experiment name')
    
    return parser.parse_args()

def setup_environment(args):
    """Setup training environment"""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # CUDA optimizations
    if torch.cuda.is_available() and not args.disable_cudnn_benchmark:
        cudnn.benchmark = True
        cudnn.enabled = True
        cudnn.deterministic = False
        
        # Enable TensorFloat-32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    if not getattr(args, 'quiet', False):
        print(f"✅ 环境设置完成")
        print(f"   设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"   模式: {args.mode}")
        print(f"   随机种子: {args.seed}")

def check_cifar10_data(data_path: str) -> bool:
    """检查CIFAR-10数据是否已存在且完整"""
    import os
    from pathlib import Path
    
    data_dir = Path(data_path)
    cifar_dir = data_dir / "cifar-10-batches-py"
    
    # 检查目录是否存在
    if not cifar_dir.exists():
        return False
    
    # 检查关键文件是否存在
    required_files = [
        "data_batch_1",
        "data_batch_2", 
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        "test_batch",
        "batches.meta"
    ]
    
    for file in required_files:
        if not (cifar_dir / file).exists():
            return False
    
    # 检查文件大小（简单验证）
    total_size = sum((cifar_dir / file).stat().st_size for file in required_files)
    if total_size < 100 * 1024 * 1024:  # 小于100MB说明数据不完整
        return False
    
    return True

def download_cifar10_with_xunlei(data_path: str) -> bool:
    """使用迅雷下载CIFAR-10数据集"""
    try:
        from neuroexapt.utils.xunlei_downloader import XunleiDatasetDownloader
        
        print("🚀 检测到迅雷下载器，尝试使用迅雷下载CIFAR-10...")
        downloader = XunleiDatasetDownloader(data_dir=data_path)
        
        if not downloader.xunlei_downloader.is_available:
            print("⚠️ 迅雷未检测到，将使用标准下载方式")
            print("💡 如需使用迅雷加速，请先安装迅雷: https://www.xunlei.com/")
            return False
        
        print("✅ 迅雷可用，启动下载...")
        print("=" * 60)
        print("📋 CIFAR-10数据集信息:")
        print("   文件: cifar-10-python.tar.gz")
        print("   大小: ~162MB")
        print("   来源: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        print("=" * 60)
        
        success = downloader.download_dataset('cifar10', wait_for_completion=False)
        
        if success:
            print("✅ 迅雷下载任务已启动！")
            print("=" * 60)
            print("📋 迅雷下载说明:")
            print("1. 📁 目标保存路径已自动复制到剪贴板")
            print("2. 🚀 请检查迅雷下载窗口")
            print("3. 📋 在下载窗口中按 Ctrl+V 粘贴保存路径")
            print("4. ✅ 确认文件名为: cifar-10-python.tar.gz")
            print("5. 🎯 点击'立即下载'开始下载")
            print("=" * 60)
            print("💡 下载完成后，数据会自动解压到指定目录")
            print("💡 请等待下载完成后重新运行此程序")
            return True
        else:
            print("⚠️ 迅雷下载启动失败，将使用标准下载方式")
            return False
            
    except ImportError:
        print("⚠️ 迅雷下载器模块未找到，将使用标准下载方式")
        print("💡 要使用迅雷加速，请确保 neuroexapt.utils.xunlei_downloader 模块可用")
        return False
    except Exception as e:
        print(f"⚠️ 迅雷下载出错: {e}")
        print("💡 将回退到标准下载方式")
        return False

def create_data_loaders(args):
    """Create data loaders with intelligent data checking and 迅雷 support"""
    if not getattr(args, 'quiet', False):
        print("📊 Creating data loaders...")
    
    # 检查数据是否存在
    data_dir = Path(args.data)
    
    if check_cifar10_data(args.data):
        if not getattr(args, 'quiet', False):
            print("✅ CIFAR-10数据已存在，跳过下载")
        download_flag = False
    else:
        if not getattr(args, 'quiet', False):
            print("❌ CIFAR-10数据不存在")
        
        # 创建数据目录
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # 尝试使用迅雷下载
        if not getattr(args, 'quiet', False):
            print("🚀 尝试使用迅雷加速下载...")
        if download_cifar10_with_xunlei(args.data):
            # 迅雷下载已启动，询问用户选择
            if not getattr(args, 'quiet', False):
                print("\n" + "="*60)
                print("⚡ 迅雷下载已启动！您有以下选择:")
                print("1. [推荐] 等待迅雷下载完成后重新运行程序")
                print("2. 继续使用PyTorch标准下载（较慢但自动）")
                print("3. 退出程序")
                print("="*60)
                
                while True:
                    choice = input("请选择 (1/2/3): ").strip()
                    if choice == "1":
                        print("👋 请等待迅雷下载完成后重新运行程序")
                        sys.exit(0)
                    elif choice == "2":
                        print("📥 继续使用PyTorch标准下载...")
                        download_flag = True
                        break
                    elif choice == "3":
                        print("👋 程序退出")
                        sys.exit(0)
                    else:
                        print("⚠️ 请输入 1、2 或 3")
            else:
                # Quiet mode - automatically use standard download
                download_flag = True
        else:
            if not getattr(args, 'quiet', False):
                print("📥 使用PyTorch标准下载方式获取CIFAR-10数据...")
            download_flag = True
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    if download_flag and not getattr(args, 'quiet', False):
        print("📦 加载CIFAR-10数据集...")
        print("   📥 开始下载 (首次下载可能需要几分钟)...")
    
    try:
        train_data = dset.CIFAR10(root=args.data, train=True, download=download_flag, transform=train_transform)
        if download_flag and not getattr(args, 'quiet', False):
            print("✅ CIFAR-10数据集下载完成！")
                
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        if not getattr(args, 'quiet', False):
            print("💡 建议解决方案:")
            print("   1. 检查网络连接")
            print("   2. 使用迅雷下载器加速下载")
            print("   3. 手动下载数据集到指定目录")
        raise e
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    
    if not getattr(args, 'quiet', False):
        print(f"   📊 训练样本: {num_train}, 分割: {split}/{num_train - split}")
    
    # Create data loaders
    if args.async_loader:
        if not getattr(args, 'quiet', False):
            print("   ⚡ 使用异步数据加载器...")
        train_queue = build_cifar10_pipeline(batch_size=args.batch_size, num_workers=args.num_workers, prefetch=8)
        valid_queue = build_cifar10_pipeline(batch_size=args.batch_size, num_workers=args.num_workers, prefetch=4)
    else:
        train_queue = data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=data.SubsetRandomSampler(indices[:split]),
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=True
        )
        
        valid_queue = data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=data.SubsetRandomSampler(indices[split:num_train]),
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=True
        )
    
    if not getattr(args, 'quiet', False):
        print(f"   ✅ 数据加载器创建完成: {len(train_queue)}训练批次, {len(valid_queue)}验证批次")
    
    return train_queue, valid_queue

def create_model(args, mode='exapt'):
    """Create model based on mode"""
    if not getattr(args, 'quiet', False):
        print(f"🧠 Creating model (mode: {mode})...")
    
    if mode == 'fixed':
        # Fixed architecture - no evolution
        model = FixedNetwork(
            C=args.init_channels,
            num_classes=10,
            layers=args.layers
        )
    elif mode == 'cached':
        # For now, use fixed architecture to avoid hanging issues
        if not getattr(args, 'quiet', False):
            print("  ⚠️  Using fixed architecture for cached mode")
        model = FixedNetwork(
            C=args.init_channels,
            num_classes=10,
            layers=args.layers
        )
    elif mode == 'exapt':
        # Exapt mode with intelligent neural architecture evolution
        if not getattr(args, 'quiet', False):
            optimizations = []
            if getattr(args, 'lazy_computation', False):
                optimizations.append("懒计算")
            if getattr(args, 'gradient_optimized', True):
                optimizations.append("梯度优化")
            if getattr(args, 'memory_efficient', False):
                optimizations.append("内存优化")
            if optimizations:
                print(f"  🧬 智能演化模式: {', '.join(optimizations)}")
        
        model = Network(
            C=args.init_channels,
            num_classes=10,
            layers=args.layers,
            potential_layers=args.potential_layers,
            use_checkpoint=getattr(args, 'use_checkpoint', False),
            use_compile=getattr(args, 'use_model_compile', False),
            use_optimized_ops=getattr(args, 'use_optimized_ops', False),
            use_lazy_ops=getattr(args, 'lazy_computation', False),
            use_gradient_optimized=getattr(args, 'gradient_optimized', True),
            use_memory_efficient=getattr(args, 'memory_efficient', False),
            progress_tracking=not getattr(args, 'disable_progress_spam', False),
            quiet=getattr(args, 'quiet', False)
        )
    else:
        # Legacy search mode (fallback)
        model = Network(
            C=args.init_channels,
            num_classes=10,
            layers=args.layers,
            potential_layers=args.potential_layers,
            use_checkpoint=getattr(args, 'use_checkpoint', False),
            use_compile=getattr(args, 'use_model_compile', False),
            use_optimized_ops=getattr(args, 'use_optimized_ops', False),
            progress_tracking=getattr(args, 'progress_tracking', True),
            quiet=getattr(args, 'quiet', False)
        )
    
    if torch.cuda.is_available():
        if hasattr(model, 'cuda'):
            model = model.cuda()
        elif hasattr(model, 'model'):
            model.model = model.model.cuda()  # type: ignore[attr-defined]

    # Optional compile of Cell submodules for extra speed
    if args.compile_cell and hasattr(torch, 'compile'):
        target = model.model if hasattr(model, 'model') else model
        from neuroexapt.core.model import Cell
        compile_submodules(target, predicate=lambda m: isinstance(m, Cell))  # type: ignore[arg-type]
    
    # Get parameters correctly based on model type
    if hasattr(model, 'model'):  # CachedNetwork
        params = sum(p.numel() for p in model.model.parameters())  # type: ignore[attr-defined]
    else:  # Regular model
        params = sum(p.numel() for p in model.parameters())
    
    if not getattr(args, 'quiet', False):
        print(f"   参数量: {params:,}")
    
    return model

def create_architect(args, model):
    """Create architect based on configuration"""
    if args.mode == 'fixed' or args.mode == 'cached':
        return None  # No architecture evolution needed
    
    if hasattr(model, 'model'):  # CachedNetwork
        architect = Architect(model.model, args)
    else:
        architect = Architect(model, args)
    
    architect.criterion = nn.CrossEntropyLoss().cuda()
    
    return architect

def train_epoch(train_queue, valid_queue, model, architect, criterion, optimizer, epoch, args, monitor=None):
    """Optimized train epoch with minimal output overhead"""
    model.train()
    
    # 重置模型的epoch计数器
    if hasattr(model, 'reset_epoch_counters'):
        model.reset_epoch_counters()
    
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    
    # Performance tracking
    batch_times = []
    
    # For architecture search modes
    if architect is not None:
        architect.set_epoch(epoch)
        valid_iter = iter(valid_queue)
        arch_updates = 0
    
    start_time = time.time()
    
    # 极简进度条
    total_batches = len(train_queue)
    quiet_mode = getattr(args, 'quiet', False)
    
    if quiet_mode:
        # 最小化的进度条
        pbar = tqdm(enumerate(train_queue), total=total_batches, 
                    desc=f"E{epoch}", 
                    bar_format='{desc}: {percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt} {rate_fmt}',
                    mininterval=2.0)  # 减少更新频率
    else:
        # 正常模式也简化
        pbar = tqdm(enumerate(train_queue), total=total_batches, 
                    desc=f"Epoch {epoch}", 
                    bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt}',
                    mininterval=1.0)
    
    for step, (input, target) in pbar:
        batch_start_time = time.perf_counter()
        
        n = input.size(0)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # Architecture evolution step - 简化逻辑
        if architect is not None:
            should_update = (step + 1) % getattr(args, 'arch_update_freq', 50) == 0 and epoch >= getattr(args, 'warmup_epochs', 5)
            
            if should_update:
                try:
                    input_search, target_search = next(valid_iter)
                except StopIteration:
                    valid_iter = iter(valid_queue)
                    input_search, target_search = next(valid_iter)
                
                input_search = input_search.cuda(non_blocking=True)
                target_search = target_search.cuda(non_blocking=True)
                
                # Invalidate cache if using cached mode
                if hasattr(model, 'cache_valid'):
                    model.cache_valid = False
                
                architect.step(input, target, input_search, target_search,
                              optimizer.param_groups[0]['lr'], optimizer, args.use_first_order)
                arch_updates += 1
        
        # Weight update step
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        # Statistics
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        
        batch_time = time.perf_counter() - batch_start_time
        batch_times.append(batch_time)
        
        # 极简的进度条更新（移除所有复杂逻辑）
        if step % 100 == 0:  # 减少更新频率
            pbar.set_postfix({
                'Loss': f'{objs.avg:.3f}',
                'Acc': f'{top1.avg:.1f}%'
            })
        
        # GPU缓存清理（减少频率）
        if step % 500 == 0:
            torch.cuda.empty_cache()
    
    pbar.close()
    epoch_time = time.time() - start_time
    
    # 极简epoch总结
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    
    if quiet_mode:
        print(f"E{epoch}: {objs.avg:.3f} {top1.avg:.1f}% {epoch_time:.0f}s")
    else:
        print(f"✅ Epoch {epoch}: Loss={objs.avg:.3f} Acc={top1.avg:.1f}% Time={epoch_time:.0f}s")
        if architect and arch_updates > 0:
            print(f"   🧬 架构更新: {arch_updates}次")
    
    return top1.avg, objs.avg, epoch_time

def validate_epoch(valid_queue, model, criterion):
    """Validate one epoch with minimal output"""
    # Clear cache before validation to avoid memory issues
    torch.cuda.empty_cache()
    
    # Switch to evaluation mode
    model.eval()
    
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    
    with torch.no_grad():
        # 简化的进度条，仅显示百分比
        pbar = tqdm(valid_queue, desc="验证", 
                   bar_format='{desc}: {percentage:3.0f}%|{bar:20}|{n_fmt}/{total_fmt}',
                   leave=False)  # 验证完成后清除进度条
        
        for input, target in pbar:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            logits = model(input)
            loss = criterion(logits, target)
            
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
        
        pbar.close()
    
    return top1.avg, objs.avg

def run_single_mode(args, mode):
    """Run training in a single mode with simplified output"""
    print(f"\n🚀 Running {mode} mode")
    print("=" * 50)
    
    # Create model and components
    model = create_model(args, mode)
    architect = create_architect(args, model)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Create data loaders
    train_queue, valid_queue = create_data_loaders(args)
    
    # Training loop
    best_acc = 0.0
    total_time = 0
    
    for epoch in range(args.epochs):
        if not getattr(args, 'quiet', False):
            print(f"\n{'='*50}")
            print(f"📊 EPOCH {epoch}/{args.epochs-1} - {mode}")
            print(f"{'='*50}")
        else:
            print(f"\nEpoch {epoch}/{args.epochs-1}")
        
        # Train
        train_acc, train_loss, train_time = train_epoch(
            train_queue, valid_queue, model, architect, criterion, optimizer, epoch, args
        )
        
        # Validate
        val_acc, val_loss = validate_epoch(valid_queue, model, criterion)
        
        if val_acc > best_acc:
            best_acc = val_acc
            if not getattr(args, 'quiet', False):
                print(f"🌟 新最佳准确率: {best_acc:.2f}%")
        
        total_time += train_time
        
        # 简化epoch总结
        if not getattr(args, 'quiet', False):
            print(f"📈 训练={train_acc:.2f}% 验证={val_acc:.2f}% 最佳={best_acc:.2f}% 时间={train_time:.0f}s")
        else:
            print(f"训练{train_acc:.1f}% 验证{val_acc:.1f}% 最佳{best_acc:.1f}%")
    
    avg_time = total_time / args.epochs
    
    if not getattr(args, 'quiet', False):
        print(f"\n📊 {mode}模式结果:")
        print(f"   最佳准确率: {best_acc:.2f}%")
        print(f"   平均每轮时间: {avg_time:.1f}s")
        print(f"   总时间: {total_time:.1f}s")
    else:
        print(f"{mode}: {best_acc:.2f}% {avg_time:.1f}s/epoch")
    
    # Get final genotype if available
    genotype = None
    if hasattr(model, 'genotype') and callable(model.genotype):
        genotype = model.genotype()
    elif hasattr(model, 'model') and hasattr(model.model, 'genotype'):
        genotype = model.model.genotype()  # type: ignore[attr-defined]
    
    # Show final architecture for exapt mode (only if not quiet)
    if mode == 'exapt' and genotype is not None and not getattr(args, 'quiet', False):
        print(f"\n🏗️  最终发现的架构:")
        print(f"   Normal Cell: {genotype.normal}")
        print(f"   Reduce Cell: {genotype.reduce}")
        
        # Show the most frequent operations
        normal_ops = [op for op, _ in genotype.normal]
        reduce_ops = [op for op, _ in genotype.reduce]
        
        from collections import Counter
        normal_counts = Counter(normal_ops)
        reduce_counts = Counter(reduce_ops)
        
        print(f"\n📈 操作统计:")
        print(f"   Normal cell: {normal_counts.most_common(3)}")
        print(f"   Reduce cell: {reduce_counts.most_common(3)}")
    
    # Fuse kernels for inference-ready model
    target_final = model.model if hasattr(model, 'model') else model
    fuse_model(target_final)  # type: ignore[arg-type]

    return {
        'mode': mode,
        'best_acc': best_acc,
        'avg_time': avg_time,
        'total_time': total_time,
        'genotype': genotype
    }

def benchmark_all_modes(args):
    """Benchmark all optimization modes"""
    print("🏁 Benchmarking all modes")
    print("=" * 70)
    
    modes = ['fixed', 'cached', 'exapt']
    results = []
    
    for mode in modes:
        # Update args for this mode
        args.mode = mode
        result = run_single_mode(args, mode)
        results.append(result)
        
        # Clean up
        torch.cuda.empty_cache()
    
    # Compare results
    print("\n📈 Benchmark Results:")
    print("=" * 70)
    print(f"{'Mode':<10} {'Accuracy':<12} {'Time/Epoch':<12} {'Speedup':<10}")
    print("-" * 50)
    
    baseline_time = results[0]['avg_time']  # Fixed mode as baseline
    
    for result in results:
        speedup = baseline_time / result['avg_time']
        print(f"{result['mode']:<10} {result['best_acc']:<12.2f} {result['avg_time']:<12.1f}s {speedup:<10.1f}x")
    
    # Best mode
    best_mode = max(results, key=lambda x: x['best_acc'])
    fastest_mode = min(results, key=lambda x: x['avg_time'])
    
    print(f"\n🏆 Best accuracy: {best_mode['mode']} ({best_mode['best_acc']:.2f}%)")
    print(f"⚡ Fastest: {fastest_mode['mode']} ({fastest_mode['avg_time']:.1f}s/epoch)")
    
    return results

def main():
    """Main function with automated batch size optimization"""
    args = setup_args()
    setup_environment(args)
    
    # 🧠 智能Batch Size自动优化 (仅在用户未指定batch_size时)
    import sys  # 确保sys可用
    user_specified_batch_size = '--batch_size' in sys.argv
    
    if not getattr(args, 'quiet', False):
        print(f"🚀 NeuroExapt Basic Classification")
        if not user_specified_batch_size:
            print("🔍 正在自动检测最优batch size...")
            print()
    
    if not user_specified_batch_size:
        # 导入并运行智能batch size优化器
        try:
            # 添加项目根目录到路径以导入intelligent_batch_optimizer
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from intelligent_batch_optimizer import find_optimal_batch_size
            
            # 自动找到最优batch size
            optimal_batch_size = find_optimal_batch_size(quiet=getattr(args, 'quiet', False))
            args.batch_size = optimal_batch_size
            
            if not getattr(args, 'quiet', False):
                print(f"\n✅ 已自动设置最优batch size: {optimal_batch_size}")
                print("🚀 开始训练...")
                print()
            
        except Exception as e:
            if not getattr(args, 'quiet', False):
                print(f"⚠️ 智能batch size优化失败: {e}")
                print(f"🔄 使用默认batch size: {args.batch_size}")
                print()
    else:
        if not getattr(args, 'quiet', False):
            print(f"📋 使用用户指定的batch size: {args.batch_size}")
            print()
    
    if not getattr(args, 'quiet', False):
        print(f"📊 训练配置:")
        print(f"   Mode: {args.mode}")
        print(f"   Epochs: {args.epochs}")
        batch_size_label = "智能优化" if not user_specified_batch_size else "用户指定"
        print(f"   Batch size: {args.batch_size} ({batch_size_label})")
        print(f"   Layers: {args.layers}")
        
        # 检查迅雷下载器状态（简化）
        try:
            from neuroexapt.utils.xunlei_downloader import XunleiDatasetDownloader
            downloader = XunleiDatasetDownloader()
            if downloader.xunlei_downloader.is_available:
                print(f"✅ 迅雷下载器: 可用")
            else:
                print(f"⚠️ 迅雷下载器: 不可用")
        except:
            print(f"⚠️ 迅雷下载器: 模块未找到")
        
        print()  # 空行分隔
    
    if args.mode == 'benchmark':
        results = benchmark_all_modes(args)
    else:
        result = run_single_mode(args, args.mode)
        results = [result]
    
    if not getattr(args, 'quiet', False):
        print("\n✅ 训练完成!")
        
        # 显示最终统计（简化）
        optimizations = []
        if hasattr(args, 'gradient_optimized') and args.gradient_optimized:
            optimizations.append("梯度优化")
        if hasattr(args, 'lazy_computation') and args.lazy_computation:
            optimizations.append("懒计算")
        if hasattr(args, 'memory_efficient') and args.memory_efficient:
            optimizations.append("内存优化")
        
        if optimizations:
            print(f"🔧 已启用优化: {', '.join(optimizations)}")
    else:
        print("完成!")

if __name__ == "__main__":
    main() 