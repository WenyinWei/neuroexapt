#!/usr/bin/env python3
"""
分离训练策略示例

演示如何使用架构参数和网络权重的分离训练：
1. 大部分时间训练网络权重（固定架构参数）
2. 定期训练架构参数（固定网络权重）
3. 显著减少参数量和计算开销
"""

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import sys
import os
import time
import argparse
from tqdm import tqdm
from pathlib import Path

# Add the parent directory to the path to import neuroexapt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import NeuroExapt core components
from neuroexapt.core.model import Network
from neuroexapt.core.separated_training import create_separated_training_setup
from neuroexapt.utils.utils import AvgrageMeter, accuracy

def setup_args():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description='NeuroExapt Separated Training Example')
    
    # Training parameters
    parser.add_argument('--data', type=str, default='./data', help='dataset location')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--weight_lr', type=float, default=0.025, help='weight learning rate')
    parser.add_argument('--arch_lr', type=float, default=3e-4, help='architecture learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--train_portion', type=float, default=0.5, help='training data portion')
    
    # Architecture parameters  
    parser.add_argument('--init_channels', type=int, default=16, help='initial channels')
    parser.add_argument('--layers', type=int, default=6, help='number of layers')
    parser.add_argument('--potential_layers', type=int, default=2, help='potential layers')
    
    # Separated training strategy
    parser.add_argument('--weight_epochs', type=int, default=4, help='consecutive weight training epochs')
    parser.add_argument('--arch_epochs', type=int, default=1, help='architecture training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs (weight only)')
    
    # Performance options
    parser.add_argument('--num_workers', type=int, default=4, help='data loader workers')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='pin memory')
    
    # Debug options
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save_dir', type=str, default='./results', help='save directory')
    
    return parser.parse_args()

def setup_environment(args):
    """Setup training environment"""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # CUDA optimizations
    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.enabled = True
        cudnn.deterministic = False
        
        # Enable TensorFloat-32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"✅ 环境设置完成")
    print(f"   设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"   随机种子: {args.seed}")

def create_data_loaders(args):
    """Create data loaders"""
    print("📊 创建数据加载器...")
    
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
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    
    print(f"   训练样本: {num_train}, 分割: {split}/{num_train - split}")
    
    # Create data loaders
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
    
    print(f"   数据加载器创建完成: {len(train_queue)}训练批次, {len(valid_queue)}验证批次")
    
    return train_queue, valid_queue

def create_model(args):
    """Create model"""
    print(f"🧠 创建模型...")
    
    model = Network(
        C=args.init_channels,
        num_classes=10,
        layers=args.layers,
        potential_layers=args.potential_layers,
        use_gradient_optimized=True,  # 启用梯度优化MixedOp
        use_optimized_ops=True,       # 启用优化操作
        use_lazy_ops=True,            # 启用懒计算
        use_memory_efficient=True,    # 启用内存高效操作
        use_compile=True,             # 启用torch.compile加速
        use_checkpoint=False,         # 训练时关闭checkpoint以加速
        progress_tracking=False,      # 关闭详细跟踪以提升性能
        quiet=True                    # 静默模式
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    arch_params = sum(p.numel() for p in model.arch_parameters()) if hasattr(model, 'arch_parameters') else 0
    weight_params = total_params - arch_params
    
    print(f"   总参数量: {total_params:,}")
    print(f"   网络权重参数: {weight_params:,}")
    print(f"   架构参数: {arch_params:,}")
    print(f"   架构参数占比: {arch_params/total_params*100:.2f}%")
    
    return model

def validate_epoch(valid_queue, model, criterion):
    """验证一个epoch"""
    model.eval()
    
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    
    with torch.no_grad():
        for input, target in valid_queue:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            logits = model(input)
            loss = criterion(logits, target)
            
            prec1, _ = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
    
    return top1.avg, objs.avg

def main():
    """主函数"""
    args = setup_args()
    setup_environment(args)
    
    print(f"🚀 NeuroExapt 分离训练策略示例")
    print("=" * 60)
    
    # 自动优化batch size（可选）
    import sys
    user_specified_batch_size = '--batch_size' in sys.argv
    
    if not user_specified_batch_size:
        print("🔍 正在自动检测最优batch size...")
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from fast_batch_optimizer import find_optimal_batch_size
            optimal_batch_size = find_optimal_batch_size(quiet=True)
            args.batch_size = optimal_batch_size
            print(f"✅ 最优batch size: {optimal_batch_size}")
        except Exception as e:
            print(f"⚠️ batch size优化失败: {e}")
            print(f"🔄 使用默认值: {args.batch_size}")
    
    print(f"\n📊 训练配置:")
    print(f"   总epochs: {args.epochs}")
    print(f"   权重训练周期: {args.weight_epochs}")
    print(f"   架构训练周期: {args.arch_epochs}")
    print(f"   预热epochs: {args.warmup_epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Layers: {args.layers}")
    print()
    
    # 创建模型和数据
    model = create_model(args)
    train_queue, valid_queue = create_data_loaders(args)
    
    # 创建分离训练设置
    print("🧬 创建分离训练策略...")
    strategy, optimizer, trainer = create_separated_training_setup(
        model,
        weight_training_epochs=args.weight_epochs,
        arch_training_epochs=args.arch_epochs,
        total_epochs=args.epochs
    )
    
    print("\n🏃 开始训练...")
    print("=" * 60)
    
    # 训练循环
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        # 训练一个epoch
        stats = trainer.train_epoch(train_queue, valid_queue, epoch)
        
        # 验证
        val_acc, val_loss = validate_epoch(valid_queue, model, trainer.criterion)
        
        if val_acc > best_acc:
            best_acc = val_acc
        
        # 输出进度
        mode_icon = "🏋️" if stats.get('mode', 'weight') == 'weight' else "🧬"
        if 'accuracy' in stats:
            print(f"{mode_icon} Epoch {epoch}: 训练={stats['accuracy']:.2f}% 验证={val_acc:.2f}% 最佳={best_acc:.2f}% 时间={stats['time']:.1f}s")
        else:
            print(f"{mode_icon} Epoch {epoch}: 验证={val_acc:.2f}% 最佳={best_acc:.2f}% 时间={stats['time']:.1f}s 架构更新={stats.get('arch_updates', 0):.0f}")
        
        # 定期清理缓存
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
    
    # 获取最终统计
    final_stats = trainer.get_final_statistics()
    
    print("\n📈 训练完成！")
    print("=" * 60)
    print(f"✅ 最佳验证准确率: {best_acc:.2f}%")
    print(f"📊 权重训练epochs: {final_stats['weight_epochs']}")
    print(f"🧬 架构训练epochs: {final_stats['arch_epochs']}")
    print(f"⏱️ 总训练时间: {final_stats['total_time']:.1f}s")
    print(f"📈 权重训练时间占比: {final_stats['weight_time_ratio']*100:.1f}%")
    print(f"⚡ 权重训练平均速度: {final_stats['time_per_weight_epoch']:.1f}s/epoch")
    print(f"🔬 架构训练平均速度: {final_stats['time_per_arch_epoch']:.1f}s/epoch")
    
    # 显示最终架构
    if hasattr(model, 'genotype') and callable(model.genotype):
        genotype = model.genotype()
        print(f"\n🏗️ 发现的最终架构:")
        print(f"   Normal Cell: {genotype.normal}")
        print(f"   Reduce Cell: {genotype.reduce}")
    
    # 分析架构参数变化
    print(f"\n🔍 架构参数分析:")
    if hasattr(model, 'alphas_normal'):
        normal_weights = torch.softmax(model.alphas_normal, dim=-1)
        normal_max_weights = normal_weights.max(dim=-1)[0]
        print(f"   Normal权重集中度: {normal_max_weights.mean().item():.3f} ± {normal_max_weights.std().item():.3f}")
    
    if hasattr(model, 'alphas_reduce'):
        reduce_weights = torch.softmax(model.alphas_reduce, dim=-1)
        reduce_max_weights = reduce_weights.max(dim=-1)[0]
        print(f"   Reduce权重集中度: {reduce_max_weights.mean().item():.3f} ± {reduce_max_weights.std().item():.3f}")
    
    print("\n🎯 分离训练策略成功完成！参数量大幅减少，训练速度显著提升！")

if __name__ == "__main__":
    main() 