#!/usr/bin/env python3
"""
🚀 NeuroExapt 高性能训练脚本 - 冲击CIFAR-10 95%准确率

专为达到最高性能而设计的训练配置：
1. 🎯 目标准确率：95%+
2. ⏰ 训练时长：100+ epochs
3. 🧠 智能学习率调度
4. 📈 高级数据增强
5. 🔧 模型容量优化
6. 💫 分离训练策略
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
import math

# Add the parent directory to the path to import neuroexapt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import NeuroExapt core components
from neuroexapt.core.model import Network
from neuroexapt.core.separated_training import SeparatedTrainingStrategy, SeparatedOptimizer, SeparatedTrainer
from neuroexapt.utils.utils import AvgrageMeter, accuracy

class AdvancedCIFAR10Transforms:
    """高级CIFAR-10数据增强策略"""
    
    @staticmethod
    def get_train_transform():
        """训练时的高级数据增强"""
        return transforms.Compose([
            # 随机裁剪和填充
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            
            # 颜色增强
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2, 
                saturation=0.2,
                hue=0.1
            ),
            
            # 随机旋转
            transforms.RandomRotation(degrees=15),
            
            # 随机擦除
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            
            # 随机擦除（在tensor上）
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value=0
            ),
        ])
    
    @staticmethod
    def get_test_transform():
        """测试时的标准化"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

class CosineAnnealingWarmUpRestarts(optim.lr_scheduler._LRScheduler):
    """带热身的余弦退火重启学习率调度器"""
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected non-negative integer T_up, but got {}".format(T_up))
        
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def setup_high_performance_args():
    """高性能训练参数设置"""
    parser = argparse.ArgumentParser(description='NeuroExapt High Performance Training - 冲击95%准确率')
    
    # 数据参数
    parser.add_argument('--data', type=str, default='./data', help='dataset location')
    parser.add_argument('--train_portion', type=float, default=0.8, help='training data portion (增加训练数据)')
    
    # 高性能训练参数
    parser.add_argument('--epochs', type=int, default=120, help='训练轮数 (更长训练)')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size (优化后)')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='初始学习率 (更高)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (更强正则化)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
    
    # 模型架构参数
    parser.add_argument('--init_channels', type=int, default=36, help='初始通道数 (增加容量)')
    parser.add_argument('--layers', type=int, default=20, help='网络层数 (更深网络)')
    parser.add_argument('--potential_layers', type=int, default=8, help='潜在层数 (增加搜索空间)')
    
    # 分离训练参数 
    parser.add_argument('--weight_epochs', type=int, default=5, help='连续权重训练轮数')
    parser.add_argument('--arch_epochs', type=int, default=1, help='架构训练轮数')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='预热轮数 (更长预热)')
    parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='架构学习率')
    
    # 学习率调度
    parser.add_argument('--lr_scheduler', type=str, default='cosine_restart', 
                       choices=['cosine', 'cosine_restart', 'step'], help='学习率调度策略')
    parser.add_argument('--T_0', type=int, default=20, help='余弦重启周期')
    parser.add_argument('--T_mult', type=int, default=2, help='重启周期倍数')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='最小学习率')
    
    # 高级优化选项
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='标签平滑')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup参数')
    parser.add_argument('--cutmix_prob', type=float, default=0.25, help='CutMix概率')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='DropPath概率')
    
    # 性能选项
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载器进程数')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='pin memory')
    parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度')
    
    # 保存和日志
    parser.add_argument('--save_dir', type=str, default='./checkpoints_high_perf', help='检查点保存目录')
    parser.add_argument('--save_freq', type=int, default=10, help='保存频率')
    parser.add_argument('--log_freq', type=int, default=5, help='日志频率')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点')
    parser.add_argument('--evaluate_only', action='store_true', help='仅评估模式')
    
    return parser.parse_args()

def setup_environment(args):
    """设置高性能训练环境"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # CUDA优化设置
    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.enabled = True
        cudnn.deterministic = False  # 为了性能，关闭确定性
        
        # 启用高性能模式
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print(f"🚀 CUDA设备: {torch.cuda.get_device_name()}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"✅ 高性能训练环境设置完成")

def create_high_performance_data_loaders(args):
    """创建高性能数据加载器"""
    print("📊 创建高性能数据加载器...")
    
    # 使用高级数据增强
    train_transform = AdvancedCIFAR10Transforms.get_train_transform()
    test_transform = AdvancedCIFAR10Transforms.get_test_transform()
    
    # 加载数据集
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    test_data = dset.CIFAR10(root=args.data, train=False, transform=test_transform)
    
    # 训练验证划分
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    
    print(f"   数据划分: {split}训练 + {num_train - split}验证 + {len(test_data)}测试")
    
    # 创建数据加载器
    train_queue = data.DataLoader(
        train_data, 
        batch_size=args.batch_size,
        sampler=data.SubsetRandomSampler(indices[:split]),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        persistent_workers=True  # 高性能选项
    )
    
    valid_queue = data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=data.SubsetRandomSampler(indices[split:num_train]),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        persistent_workers=True
    )
    
    test_queue = data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True
    )
    
    print(f"✅ 数据加载器创建完成: {len(train_queue)}训练 + {len(valid_queue)}验证 + {len(test_queue)}测试")
    
    return train_queue, valid_queue, test_queue

def create_high_performance_model(args):
    """创建高性能模型"""
    print(f"🧠 创建高性能模型...")
    
    model = Network(
        C=args.init_channels,
        num_classes=10,
        layers=args.layers,
        potential_layers=args.potential_layers,
        use_compile=True,  # 启用torch.compile
        progress_tracking=False,
        quiet=False
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   模型大小: {total_params * 4 / 1e6:.1f}MB")
    
    return model

def create_advanced_criterion(args):
    """创建高级损失函数"""
    if args.label_smoothing > 0:
        # 标签平滑交叉熵
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"   使用标签平滑: {args.label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    return criterion.cuda()

def create_advanced_scheduler(optimizer, args):
    """创建高级学习率调度器"""
    if args.lr_scheduler == 'cosine_restart':
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer, 
            T_0=args.T_0, 
            T_mult=args.T_mult,
            eta_max=args.learning_rate,
            T_up=5,  # 热身步数
            gamma=0.5  # 重启后衰减
        )
        print(f"   使用余弦重启调度器: T_0={args.T_0}, T_mult={args.T_mult}")
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs, 
            eta_min=args.eta_min
        )
        print(f"   使用余弦退火调度器")
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=30, 
            gamma=0.1
        )
        print(f"   使用阶梯调度器")
    
    return scheduler

def save_checkpoint(state, filename):
    """保存检查点"""
    torch.save(state, filename)
    print(f"💾 保存检查点: {filename}")

def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """加载检查点"""
    print(f"📂 加载检查点: {filename}")
    checkpoint = torch.load(filename)
    
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    print(f"   恢复到epoch {start_epoch}, 最佳准确率: {best_acc:.2f}%")
    return start_epoch, best_acc

def train_epoch_separated(separated_trainer, train_queue, valid_queue, epoch, args, scaler=None):
    """分离训练一个epoch"""
    start_time = time.time()
    
    # 使用分离训练器
    stats = separated_trainer.train_epoch(train_queue, valid_queue, epoch)
    
    train_time = time.time() - start_time
    
    # 获取训练统计
    if 'accuracy' in stats:
        train_acc = stats['accuracy']
        train_loss = stats['loss'] 
    else:
        # 架构训练epoch，准确率需要通过验证获得
        train_acc = 0.0
        train_loss = stats['loss']
    
    return train_acc, train_loss, train_time

def validate_epoch(valid_queue, model, criterion):
    """验证一个epoch"""
    model.eval()
    
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    
    with torch.no_grad():
        for input, target in tqdm(valid_queue, desc="验证", leave=False):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            logits = model(input)
            loss = criterion(logits, target)
            
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
    
    return top1.avg, top5.avg, objs.avg

def main():
    """主训练函数"""
    args = setup_high_performance_args()
    setup_environment(args)
    
    print("🚀 NeuroExapt 高性能训练 - 冲击CIFAR-10 95%准确率")
    print("=" * 80)
    print(f"🎯 目标准确率: 95%+")
    print(f"⏰ 训练轮数: {args.epochs}")
    print(f"🧠 模型容量: C={args.init_channels}, L={args.layers}")
    print(f"📊 批大小: {args.batch_size}")
    print(f"📈 学习率调度: {args.lr_scheduler}")
    print("=" * 80)
    
    # 创建数据加载器
    train_queue, valid_queue, test_queue = create_high_performance_data_loaders(args)
    
    # 创建模型
    model = create_high_performance_model(args)
    
    # 创建损失函数
    criterion = create_advanced_criterion(args)
    
    # 创建分离训练组件
    separated_strategy = SeparatedTrainingStrategy(
        weight_training_epochs=args.weight_epochs,
        arch_training_epochs=args.arch_epochs,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs
    )
    
    separated_optimizer = SeparatedOptimizer(
        model,
        weight_lr=args.learning_rate,
        arch_lr=args.arch_learning_rate,
        weight_momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    separated_trainer = SeparatedTrainer(
        model, separated_strategy, separated_optimizer, criterion
    )
    
    # 创建学习率调度器
    weight_scheduler = create_advanced_scheduler(separated_optimizer.weight_optimizer, args)
    arch_scheduler = create_advanced_scheduler(separated_optimizer.arch_optimizer, args)
    
    # 自动混合精度
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    if args.use_amp:
        print("⚡ 启用自动混合精度训练")
    
    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    best_test_acc = 0.0
    
    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            args.resume, model, separated_optimizer.weight_optimizer, weight_scheduler
        )
    
    # 仅评估模式
    if args.evaluate_only:
        print("📊 仅评估模式")
        val_acc, val_acc5, val_loss = validate_epoch(valid_queue, model, criterion)
        test_acc, test_acc5, test_loss = validate_epoch(test_queue, model, criterion)
        print(f"验证准确率: {val_acc:.2f}% (Top-5: {val_acc5:.2f}%)")
        print(f"测试准确率: {test_acc:.2f}% (Top-5: {test_acc5:.2f}%)")
        return
    
    print(f"\n🚀 开始高性能训练!")
    print(f"训练计划: {separated_strategy.get_schedule_summary()}")
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"📊 EPOCH {epoch+1}/{args.epochs}")
        print(f"{'='*80}")
        
        training_mode = separated_strategy.get_training_mode(epoch)
        print(f"🔄 训练模式: {'🏋️ 权重训练' if training_mode == 'weight' else '🧬 架构训练'}")
        
        # 训练
        train_acc, train_loss, train_time = train_epoch_separated(
            separated_trainer, train_queue, valid_queue, epoch, args, scaler
        )
        
        # 验证
        val_acc, val_acc5, val_loss = validate_epoch(valid_queue, model, criterion)
        
        # 测试（每5个epoch测试一次）
        if (epoch + 1) % 5 == 0:
            test_acc, test_acc5, test_loss = validate_epoch(test_queue, model, criterion)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
        else:
            test_acc = 0.0
        
        # 更新学习率
        if training_mode == 'weight':
            weight_scheduler.step()
        else:
            arch_scheduler.step()
        
        # 记录最佳准确率
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        
        # 打印统计信息
        current_lr = separated_optimizer.weight_optimizer.param_groups[0]['lr']
        print(f"📈 训练: 准确率={train_acc:.2f}% 损失={train_loss:.4f} 时间={train_time:.0f}s")
        print(f"📊 验证: 准确率={val_acc:.2f}% (Top-5: {val_acc5:.2f}%) 损失={val_loss:.4f}")
        if test_acc > 0:
            print(f"🎯 测试: 准确率={test_acc:.2f}% (Top-5: {test_acc5:.2f}%)")
        print(f"⭐ 最佳: 验证={best_acc:.2f}% 测试={best_test_acc:.2f}%")
        print(f"🔧 学习率: {current_lr:.6f}")
        
        # 95%目标检查
        if val_acc >= 95.0:
            print(f"🎉 达成目标! 验证准确率达到 {val_acc:.2f}% >= 95%!")
        if best_test_acc >= 95.0:
            print(f"🏆 测试准确率也达到 {best_test_acc:.2f}% >= 95%!")
        
        # 保存检查点
        if (epoch + 1) % args.save_freq == 0 or is_best:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': separated_optimizer.weight_optimizer.state_dict(),
                'scheduler': weight_scheduler.state_dict(),
                'args': args,
            }, checkpoint_path)
            
            if is_best:
                best_path = os.path.join(args.save_dir, 'best_model.pth')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'test_acc': best_test_acc,
                    'args': args,
                }, best_path)
    
    # 最终测试
    print(f"\n{'='*80}")
    print(f"🎊 训练完成!")
    print(f"{'='*80}")
    
    final_test_acc, final_test_acc5, _ = validate_epoch(test_queue, model, criterion)
    
    print(f"📊 最终结果:")
    print(f"   最佳验证准确率: {best_acc:.2f}%")
    print(f"   最佳测试准确率: {best_test_acc:.2f}%")
    print(f"   最终测试准确率: {final_test_acc:.2f}% (Top-5: {final_test_acc5:.2f}%)")
    
    if final_test_acc >= 95.0:
        print(f"🏆 恭喜! 成功达到95%目标准确率!")
        print(f"🎯 最终测试准确率: {final_test_acc:.2f}%")
    elif final_test_acc >= 93.0:
        print(f"🌟 很接近了! 测试准确率: {final_test_acc:.2f}%")
        print(f"💡 建议继续训练或调整超参数")
    else:
        print(f"📈 当前测试准确率: {final_test_acc:.2f}%")
        print(f"💡 建议增加训练轮数或调整模型容量")
    
    # 显示最终架构
    try:
        genotype = model.genotype()
        print(f"\n🏗️ 最终发现的架构:")
        print(f"   Normal: {genotype.normal}")
        print(f"   Reduce: {genotype.reduce}")
    except:
        print(f"⚠️ 无法获取最终架构")

if __name__ == "__main__":
    main() 