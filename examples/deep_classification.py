#!/usr/bin/env python3
"""
NeuroExapt Deep Classification Example

This example demonstrates advanced optimization strategies for deeper networks:
1. Standard DARTS-style architecture search on ResNet-like models
2. Fixed architecture training with deep residual connections
3. Optimized architecture search with reduced complexity
4. Performance benchmarking across different strategies

Use command line arguments to control optimization strategies and network depth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from typing import List, Optional

# performance utilities
from neuroexapt.utils.async_dataloader import build_cifar10_pipeline
from neuroexapt.utils.fuse_utils import fuse_model
from neuroexapt.utils.compile_utils import compile_submodules

# Add the parent directory to the path to import neuroexapt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import NeuroExapt core components
from neuroexapt.core.model import Network
from neuroexapt.core.architect import Architect
from neuroexapt.core.simple_architect import SimpleArchitect
from neuroexapt.core.operations import SepConv, DilConv, Identity, Zero, FactorizedReduce, ReLUConvBN
from neuroexapt.utils.utils import AvgrageMeter, accuracy

class BasicBlock(nn.Module):
    """Standard ResNet Basic Block"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    """ResNet Bottleneck Block"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FixedResNet(nn.Module):
    """Fixed ResNet architecture for fast training"""
    def __init__(self, block, num_blocks, num_classes=10, base_width=64):
        super(FixedResNet, self).__init__()
        self.in_planes = base_width

        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.layer1 = self._make_layer(block, base_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_width*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_width*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_width*8, num_blocks[3], stride=2)
        self.linear = nn.Linear(base_width*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes=10, base_width=64):
    return FixedResNet(BasicBlock, [2, 2, 2, 2], num_classes, base_width)

def ResNet34(num_classes=10, base_width=64):
    return FixedResNet(BasicBlock, [3, 4, 6, 3], num_classes, base_width)

def ResNet50(num_classes=10, base_width=64):
    return FixedResNet(Bottleneck, [3, 4, 6, 3], num_classes, base_width)

class AdaptiveResNet(nn.Module):
    """Adaptive ResNet that can be used with NeuroExapt"""
    def __init__(self, num_classes=10, base_width=64, depth=18):
        super(AdaptiveResNet, self).__init__()
        
        if depth == 18:
            self.model = ResNet18(num_classes, base_width)
        elif depth == 34:
            self.model = ResNet34(num_classes, base_width)
        elif depth == 50:
            self.model = ResNet50(num_classes, base_width)
        else:
            # Default to ResNet18
            self.model = ResNet18(num_classes, base_width)
        
        self.num_classes = num_classes
        self.base_width = base_width
        self.depth = depth
    
    def forward(self, x):
        return self.model(x)
    
    def parameters(self):
        return self.model.parameters()
    
    def named_parameters(self):
        return self.model.named_parameters()
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)

def setup_args():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description='NeuroExapt Deep Classification')
    
    # Training parameters
    parser.add_argument('--data', type=str, default='./data', help='dataset location')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.8, help='training data portion')
    parser.add_argument('--report_freq', type=int, default=100, help='report frequency')
    
    # Architecture parameters
    parser.add_argument('--depth', type=int, default=18, choices=[18, 34, 50], help='ResNet depth')
    parser.add_argument('--base_width', type=int, default=64, help='base width for ResNet')
    parser.add_argument('--init_channels', type=int, default=16, help='initial channels for DARTS')
    parser.add_argument('--layers', type=int, default=8, help='number of layers for DARTS')
    parser.add_argument('--potential_layers', type=int, default=4, help='potential layers for DARTS')
    
    # Architecture search parameters
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='arch learning rate')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='arch weight decay')
    parser.add_argument('--arch_update_freq', type=int, default=50, help='arch update frequency')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs')
    parser.add_argument('--use_first_order', action='store_true', help='use first order approximation')
    
    # Optimization strategy
    parser.add_argument('--mode', type=str, default='fixed', 
                       choices=['fixed', 'search', 'benchmark'],
                       help='optimization mode: fixed (ResNet), search (DARTS), benchmark (compare)')
    
    # Performance options
    parser.add_argument('--num_workers', type=int, default=4, help='data loader workers')
    parser.add_argument('--async_loader', action='store_true', help='use async DataPipe loader')
    parser.add_argument('--compile_cell', action='store_true', help='torch.compile Cell submodules')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='pin memory')
    parser.add_argument('--disable_cudnn_benchmark', action='store_true', help='disable cudnn benchmark')
    
    # Debug options
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save_dir', type=str, default='./results', help='save directory')
    parser.add_argument('--exp_name', type=str, default='deep_classification', help='experiment name')
    
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
    
    print(f"✅ Environment setup complete")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"   Mode: {args.mode}")
    print(f"   Seed: {args.seed}")

def create_data_loaders(args):
    """Create data loaders for CIFAR-10"""
    print("📊 Creating data loaders...")
    
    # Enhanced data augmentation
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
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    
    if args.async_loader:
        train_queue = build_cifar10_pipeline(batch_size=args.batch_size, num_workers=args.num_workers, prefetch=8)
        valid_queue = build_cifar10_pipeline(batch_size=args.batch_size, num_workers=args.num_workers, prefetch=4)
        test_queue  = build_cifar10_pipeline(batch_size=args.batch_size, num_workers=args.num_workers, prefetch=4, train=False)  # type: ignore[arg-type]
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

        test_queue = data.DataLoader(
            test_data, batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
    
    print(f"   Train: {len(train_queue)} batches, Valid: {len(valid_queue)} batches, Test: {len(test_queue)} batches")
    
    return train_queue, valid_queue, test_queue

def create_model(args, mode='fixed'):
    """Create model based on mode"""
    print(f"🧠 Creating model (mode: {mode}, depth: {args.depth})...")
    
    if mode == 'fixed':
        # Fixed ResNet architecture
        if args.depth == 18:
            model = ResNet18(num_classes=10, base_width=args.base_width)
        elif args.depth == 34:
            model = ResNet34(num_classes=10, base_width=args.base_width)
        elif args.depth == 50:
            model = ResNet50(num_classes=10, base_width=args.base_width)
        else:
            model = ResNet18(num_classes=10, base_width=args.base_width)
    else:
        # DARTS architecture search
        model = Network(
            C=args.init_channels,
            num_classes=10,
            layers=args.layers,
            potential_layers=args.potential_layers
        )
    
    if torch.cuda.is_available():
        model = model.cuda()

    # optional compile MixedOp cells when in search mode
    if args.compile_cell and args.mode == 'search' and hasattr(torch, 'compile'):
        from neuroexapt.core.model import Cell
        compile_submodules(model, predicate=lambda m: isinstance(m, Cell))  # type: ignore[arg-type]
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {params:,}")
    
    return model

def create_architect(args, model):
    """Create architect for architecture search"""
    if args.mode == 'fixed':
        return None  # No architecture search needed
    
    architect = SimpleArchitect(model, args)
    architect.criterion = nn.CrossEntropyLoss().cuda()
    
    return architect

def create_optimizer(args, model):
    """Create optimizer based on mode"""
    if args.mode == 'fixed':
        # Standard SGD with momentum for fixed architectures
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )
    else:
        # DARTS-style optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    
    return optimizer

def train_epoch(train_queue, valid_queue, model, architect, criterion, optimizer, epoch, args):
    """Train one epoch"""
    model.train()
    
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    
    # For architecture search modes
    if architect is not None:
        architect.set_epoch(epoch)
        valid_iter = iter(valid_queue)
        arch_updates = 0
    
    start_time = time.time()
    
    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # Architecture search step
        if architect is not None and architect.should_update_arch():
            try:
                input_search, target_search = next(valid_iter)
            except StopIteration:
                valid_iter = iter(valid_queue)
                input_search, target_search = next(valid_iter)
            
            input_search = input_search.cuda(non_blocking=True)
            target_search = target_search.cuda(non_blocking=True)
            
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
        
        # Progress report
        if step % args.report_freq == 0:
            arch_info = f"Arch: {arch_updates}" if architect else "Fixed"
            print(f"  Step {step:3d}: Loss={objs.avg:.4f} Top1={top1.avg:.2f}% {arch_info}")
    
    epoch_time = time.time() - start_time
    print(f"  Epoch {epoch}: Loss={objs.avg:.4f} Top1={top1.avg:.2f}% Time={epoch_time:.1f}s")
    
    return top1.avg, objs.avg, epoch_time

def validate_epoch(valid_queue, model, criterion):
    """Validate one epoch"""
    model.eval()
    
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    
    with torch.no_grad():
        for input, target in valid_queue:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            logits = model(input)
            loss = criterion(logits, target)
            
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
    
    return top1.avg, objs.avg

def run_single_mode(args, mode):
    """Run training in a single mode"""
    print(f"\n🚀 Running {mode} mode")
    print("=" * 60)
    
    # Create model and components
    model = create_model(args, mode)
    architect = create_architect(args, model)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = create_optimizer(args, model)
    
    # Learning rate scheduler
    if args.mode == 'fixed':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.001)
    
    # Create data loaders
    train_queue, valid_queue, test_queue = create_data_loaders(args)
    
    # Training loop
    best_acc = 0.0
    total_time = 0
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch} ---")
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.6f}")
        
        # Train
        train_acc, train_loss, train_time = train_epoch(
            train_queue, valid_queue, model, architect, criterion, optimizer, epoch, args
        )
        
        # Validate
        val_acc, val_loss = validate_epoch(valid_queue, model, criterion)
        
        if val_acc > best_acc:
            best_acc = val_acc
            
            # Test on best validation accuracy
            test_acc, test_loss = validate_epoch(test_queue, model, criterion)
            print(f"  New best: Val={val_acc:.2f}% Test={test_acc:.2f}%")
        
        total_time += train_time
        
        print(f"Epoch {epoch}: Train={train_acc:.2f}% Valid={val_acc:.2f}% Best={best_acc:.2f}%")
    
    # Final test
    final_test_acc, final_test_loss = validate_epoch(test_queue, model, criterion)
    avg_time = total_time / args.epochs
    
    print(f"\n📊 Results for {mode} mode:")
    print(f"   Best validation accuracy: {best_acc:.2f}%")
    print(f"   Final test accuracy: {final_test_acc:.2f}%")
    print(f"   Average time per epoch: {avg_time:.1f}s")
    print(f"   Total training time: {total_time:.1f}s")
    
    # Get final genotype if available
    genotype = None
    if hasattr(model, 'genotype') and callable(model.genotype):
        genotype = model.genotype()
    
    # fuse kernels for inference
    fuse_model(model)  # type: ignore[arg-type]
    
    return {
        'mode': mode,
        'best_val_acc': best_acc,
        'final_test_acc': final_test_acc,
        'avg_time': avg_time,
        'total_time': total_time,
        'genotype': genotype
    }

def benchmark_modes(args):
    """Benchmark different modes"""
    print("🏁 Benchmarking modes")
    print("=" * 70)
    
    modes = ['fixed', 'search']
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
    print(f"{'Mode':<10} {'Val Acc':<10} {'Test Acc':<10} {'Time/Epoch':<12} {'Speedup':<10}")
    print("-" * 60)
    
    baseline_time = results[0]['avg_time']  # Fixed mode as baseline
    
    for result in results:
        speedup = baseline_time / result['avg_time']
        print(f"{result['mode']:<10} {result['best_val_acc']:<10.2f} {result['final_test_acc']:<10.2f} {result['avg_time']:<12.1f}s {speedup:<10.1f}x")
    
    # Best mode
    best_mode = max(results, key=lambda x: x['final_test_acc'])
    fastest_mode = min(results, key=lambda x: x['avg_time'])
    
    print(f"\n🏆 Best test accuracy: {best_mode['mode']} ({best_mode['final_test_acc']:.2f}%)")
    print(f"⚡ Fastest training: {fastest_mode['mode']} ({fastest_mode['avg_time']:.1f}s/epoch)")
    
    return results

def main():
    """Main function with automated batch size optimization"""
    args = setup_args()
    setup_environment(args)
    
    print(f"🚀 NeuroExapt Deep Classification")
    
    # 🧠 智能Batch Size自动优化 (仅在用户未指定batch_size时)
    import sys  # 确保sys可用
    user_specified_batch_size = '--batch_size' in sys.argv
    
    if not user_specified_batch_size:
        print("🔍 正在自动检测最优batch size...")
        print()
        
        try:
            # 添加项目根目录到路径以导入intelligent_batch_optimizer
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from intelligent_batch_optimizer import find_optimal_batch_size
            
            # 自动找到最优batch size
            optimal_batch_size = find_optimal_batch_size(quiet=False)
            args.batch_size = optimal_batch_size
            
            print(f"\n✅ 已自动设置最优batch size: {optimal_batch_size}")
            print("🚀 开始训练...")
            print()
            
        except Exception as e:
            print(f"⚠️ 智能batch size优化失败: {e}")
            print(f"🔄 使用默认batch size: {args.batch_size}")
            print()
    else:
        print(f"📋 使用用户指定的batch size: {args.batch_size}")
        print()
    
    print(f"📊 训练配置:")
    print(f"   Mode: {args.mode}")
    print(f"   Depth: {args.depth}")
    print(f"   Epochs: {args.epochs}")
    batch_size_label = "智能优化" if not user_specified_batch_size else "用户指定"
    print(f"   Batch size: {args.batch_size} ({batch_size_label})")
    print(f"   Base width: {args.base_width}")
    print()
    
    if args.mode == 'benchmark':
        results = benchmark_modes(args)
    else:
        result = run_single_mode(args, args.mode)
        results = [result]
    
    print("\n✅ Deep classification completed successfully!")

if __name__ == "__main__":
    main()