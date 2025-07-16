#!/usr/bin/env python3
"""
强化版ASO-SE分类训练脚本 - 真刀真枪版

这个脚本专门设计来展示ASO-SE框架相比固定架构的准确率突破能力，包含：

🔥 核心特性：
1. 强大的断点续练系统 - 像下载软件的断点续传
2. 智能错误恢复 - 从任何失败中重新站起来
3. 设备一致性管理 - 充分利用GPU计算潜力
4. 架构演进追踪 - 记录每次突变和改进
5. 性能监控 - 实时跟踪准确率突破

🎯 目标：在CIFAR-10上突破传统固定架构的准确率瓶颈！

使用方法：
python examples/aso_se_classification_robust.py --epochs 100 --resume_from latest
"""

import argparse
import os
import sys
import time
import logging
from datetime import datetime
import json
import signal
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.core import (
    Network, ASOSEFramework, ASOSEConfig, StabilityMonitor,
    DeviceManager, get_device_manager, CheckpointManager, get_checkpoint_manager
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'robust_aso_se_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class RobustASOSETrainer:
    """
    强化版ASO-SE训练器
    
    具备强大的错误恢复和断点续练能力
    """
    
    def __init__(self, config: ASOSEConfig, experiment_name: str = "robust_aso_se_cifar10"):
        """
        Args:
            config: ASO-SE配置
            experiment_name: 实验名称
        """
        self.config = config
        self.experiment_name = experiment_name
        
        # 初始化核心组件
        self.device_manager = get_device_manager(config.device)
        self.checkpoint_manager = get_checkpoint_manager(
            checkpoint_dir="./robust_checkpoints",
            experiment_name=experiment_name
        )
        
        # 模型和框架（延迟初始化）
        self.search_model = None
        self.framework = None
        self.stability_monitor = None
        self.weight_optimizer = None
        self.arch_optimizer = None
        self.weight_scheduler = None
        self.arch_scheduler = None
        
        # 训练状态
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_stats = {
            "epoch_stats": [],
            "architecture_evolution": [],
            "performance_breakthroughs": [],
            "error_recoveries": 0,
            "mutation_count": 0
        }
        
        # 错误恢复机制
        self.max_recovery_attempts = 3
        self.recovery_count = 0
        
        # 信号处理（用于优雅中断）
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"🚀 Robust ASO-SE Trainer initialized")
        logger.info(f"🔧 Device: {self.device_manager.device}")
        logger.info(f"📁 Checkpoint Manager: {experiment_name}")
    
    def _signal_handler(self, signum, frame):
        """处理中断信号，优雅保存检查点"""
        logger.warning(f"⚠️ Received signal {signum}, saving checkpoint...")
        if self.current_epoch > 0:
            self._emergency_save_checkpoint()
        logger.info("💾 Emergency checkpoint saved, exiting...")
        sys.exit(0)
    
    def setup_data_loaders(self, batch_size: int = 128, num_workers: int = 2) -> tuple:
        """设置CIFAR-10数据加载器"""
        logger.info("📊 Setting up CIFAR-10 dataset...")
        
        # 数据增强
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # 加载数据集
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=False, transform=transform_test
        )
        
        # 分割训练集和验证集
        train_size = int(0.9 * len(train_dataset))  # 90%用于训练
        valid_size = len(train_dataset) - train_size
        train_subset, valid_subset = random_split(
            train_dataset, [train_size, valid_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        valid_loader = DataLoader(
            valid_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        # 包装为设备自动转移的加载器
        train_loader = self.device_manager.create_data_loader_wrapper(train_loader)
        valid_loader = self.device_manager.create_data_loader_wrapper(valid_loader)
        test_loader = self.device_manager.create_data_loader_wrapper(test_loader)
        
        logger.info(f"✅ Data loaded: {len(train_subset)} train, {len(valid_subset)} valid, {len(test_dataset)} test")
        
        return train_loader, valid_loader, test_loader
    
    def initialize_model_and_framework(self, init_channels: int = 16, layers: int = 8):
        """初始化模型和ASO-SE框架"""
        logger.info("🏗️ Initializing model and ASO-SE framework...")
        
        try:
            # 创建搜索模型
            self.search_model = self.device_manager.safe_model_creation(
                Network, 
                C=init_channels,
                num_classes=10,
                layers=layers,
                quiet=False
            )
            
            # 注册模型到设备管理器
            self.search_model = self.device_manager.register_model("search_model", self.search_model)
            
            # 创建ASO-SE框架
            self.framework = ASOSEFramework(self.search_model, self.config)
            self.framework.initialize_optimizers()
            
            # 获取优化器引用
            self.weight_optimizer = self.framework.weight_optimizer
            self.arch_optimizer = self.framework.arch_optimizer
            
            # 创建学习率调度器
            if self.weight_optimizer:
                self.weight_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.weight_optimizer, T_max=200, eta_min=0.001
                )
            
            if self.arch_optimizer:
                self.arch_scheduler = optim.lr_scheduler.ExponentialLR(
                    self.arch_optimizer, gamma=0.99
                )
            
            # 创建稳定性监控器
            self.stability_monitor = StabilityMonitor({
                'oscillation_window': 10,
                'oscillation_threshold': 0.1,
                'convergence_patience': 15,
                'degradation_threshold': 0.02
            })
            
            logger.info("✅ Model and framework initialized successfully")
            
            # 记录模型信息
            total_params = sum(p.numel() for p in self.search_model.parameters())
            arch_params = sum(p.numel() for p in self.search_model.arch_parameters())
            logger.info(f"📊 Model info: {total_params:,} total params, {arch_params:,} arch params")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            raise e
    
    def resume_or_start_training(self, resume_from: str = "latest") -> int:
        """恢复训练或开始新训练"""
        if resume_from == "none":
            logger.info("🆕 Starting fresh training (no resume)")
            return 0
        
        try:
            # 准备优化器字典
            optimizers = {}
            if self.weight_optimizer:
                optimizers['weight'] = self.weight_optimizer
            if self.arch_optimizer:
                optimizers['arch'] = self.arch_optimizer
            
            # 准备调度器字典
            schedulers = {}
            if self.weight_scheduler:
                schedulers['weight'] = self.weight_scheduler
            if self.arch_scheduler:
                schedulers['arch'] = self.arch_scheduler
            
            # 尝试恢复训练
            resume_epoch, training_stats = self.checkpoint_manager.resume_training(
                model=self.search_model,
                optimizers=optimizers,
                schedulers=schedulers,
                framework=self.framework,
                preferred_checkpoint=resume_from
            )
            
            if resume_epoch > 0:
                self.current_epoch = resume_epoch
                self.training_stats.update(training_stats)
                logger.info(f"🎯 Successfully resumed from epoch {resume_epoch}")
            
            return resume_epoch
            
        except Exception as e:
            logger.warning(f"⚠️ Resume failed: {e}, starting fresh training")
            return 0
    
    def train_single_epoch(self, epoch: int, train_loader, valid_loader, criterion) -> dict:
        """训练单个epoch"""
        epoch_start_time = time.time()
        
        try:
            # 执行ASO-SE训练周期
            stats = self.framework.train_cycle(train_loader, valid_loader, criterion, epoch)
            
            # 添加epoch信息
            stats['epoch'] = epoch
            stats['epoch_time'] = time.time() - epoch_start_time
            
            # 更新稳定性监控
            if self.stability_monitor:
                stability_metrics = {
                    'loss': stats.get('valid_loss', 0.0),
                    'accuracy': stats.get('valid_accuracy', 0.0)
                }
                self.stability_monitor.update(stability_metrics, epoch)
            
            # 检查性能突破
            current_acc = stats.get('valid_accuracy', 0.0)
            if isinstance(current_acc, (int, float)) and current_acc > self.best_accuracy + 1.0:  # 1%以上的提升才算突破
                breakthrough = {
                    'epoch': epoch,
                    'previous_best': self.best_accuracy,
                    'new_best': current_acc,
                    'improvement': current_acc - self.best_accuracy,
                    'phase': stats.get('phase', 'unknown')
                }
                self.training_stats['performance_breakthroughs'].append(breakthrough)
                logger.info(f"🚀 Performance breakthrough! {self.best_accuracy:.2f}% → {current_acc:.2f}%")
            
            # 更新最佳性能
            if isinstance(current_acc, (int, float)) and current_acc > self.best_accuracy:
                self.best_accuracy = current_acc
            
            # 记录架构演进
            if stats.get('phase') == 'mutation':
                mutation_info = {
                    'epoch': epoch,
                    'genotype': str(self.framework.current_genotype) if hasattr(self.framework, 'current_genotype') else None,
                    'performance_before': stats.get('pre_mutation_accuracy', 0.0),
                    'performance_after': stats.get('post_mutation_accuracy', 0.0)
                }
                self.training_stats['architecture_evolution'].append(mutation_info)
                self.training_stats['mutation_count'] += 1
                logger.info(f"🧬 Architecture evolution #{self.training_stats['mutation_count']}")
            
            # 更新学习率调度器
            if self.weight_scheduler and stats.get('phase') in ['warmup', 'weight_retraining']:
                self.weight_scheduler.step()
            
            if self.arch_scheduler and stats.get('phase') == 'arch_training':
                self.arch_scheduler.step()
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Training epoch {epoch} failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # 尝试错误恢复
            if self.recovery_count < self.max_recovery_attempts:
                self.recovery_count += 1
                self.training_stats['error_recoveries'] += 1
                logger.warning(f"🔧 Attempting error recovery {self.recovery_count}/{self.max_recovery_attempts}")
                
                # 内存清理
                self.device_manager.optimize_memory()
                
                # 降低批大小（如果需要）
                # 这里可以实现动态批大小调整
                
                return {
                    'epoch': epoch,
                    'error': str(e),
                    'phase': 'recovery',
                    'valid_accuracy': 0.0,
                    'train_accuracy': 0.0
                }
            else:
                logger.error(f"❌ Maximum recovery attempts reached, stopping training")
                raise e
    
    def save_checkpoint_if_needed(self, epoch: int, stats: dict):
        """根据需要保存检查点"""
        should_save = (
            self.checkpoint_manager.should_auto_save(epoch) or
            epoch % 10 == 0 or  # 每10个epoch强制保存
            stats.get('phase') == 'mutation' or  # 突变后保存
            stats.get('valid_accuracy', 0.0) > self.best_accuracy  # 性能提升时保存
        )
        
        if should_save:
            try:
                # 准备保存数据
                model_state = self.search_model.state_dict()
                
                optimizer_states = {}
                if self.weight_optimizer:
                    optimizer_states['weight'] = self.weight_optimizer.state_dict()
                if self.arch_optimizer:
                    optimizer_states['arch'] = self.arch_optimizer.state_dict()
                
                scheduler_states = {}
                if self.weight_scheduler:
                    scheduler_states['weight'] = self.weight_scheduler.state_dict()
                if self.arch_scheduler:
                    scheduler_states['arch'] = self.arch_scheduler.state_dict()
                
                framework_state = {
                    'current_cycle': getattr(self.framework, 'current_cycle', 0),
                    'current_phase': getattr(self.framework, 'current_phase', 'warmup'),
                    'current_genotype': str(getattr(self.framework, 'current_genotype', None)),
                    'best_performance': self.best_accuracy,
                    'training_history': getattr(self.framework, 'training_history', {})
                }
                
                architecture_info = {
                    'mutation_count': self.training_stats['mutation_count'],
                    'current_genotype': str(getattr(self.framework, 'current_genotype', None)),
                    'phase': stats.get('phase', 'unknown')
                }
                
                # 保存检查点
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model_state=model_state,
                    optimizer_states=optimizer_states,
                    scheduler_states=scheduler_states,
                    training_stats=self.training_stats,
                    framework_state=framework_state,
                    performance_metric=stats.get('valid_accuracy', 0.0),
                    architecture_info=architecture_info
                )
                
                logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to save checkpoint: {e}")
    
    def _emergency_save_checkpoint(self):
        """紧急保存检查点"""
        try:
            if self.search_model and self.current_epoch > 0:
                emergency_stats = {
                    'valid_accuracy': self.best_accuracy,
                    'emergency_save': True,
                    'timestamp': datetime.now().isoformat()
                }
                self.save_checkpoint_if_needed(self.current_epoch, emergency_stats)
        except Exception as e:
            logger.error(f"❌ Emergency checkpoint save failed: {e}")
    
    def evaluate_on_test(self, test_loader, criterion) -> float:
        """在测试集上评估最终性能"""
        logger.info("🧪 Evaluating on test set...")
        
        model = self.framework.evolvable_model if (
            hasattr(self.framework, 'evolvable_model') and self.framework.evolvable_model
        ) else self.search_model
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                # 数据已通过设备管理器自动转移
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_accuracy = 100.0 * correct / total
        logger.info(f"🎯 Final test accuracy: {test_accuracy:.2f}%")
        
        return test_accuracy
    
    def train(self, epochs: int = 100, batch_size: int = 128, 
              init_channels: int = 16, layers: int = 8, resume_from: str = "latest"):
        """执行完整的训练流程"""
        logger.info(f"🚀 Starting robust ASO-SE training for {epochs} epochs")
        logger.info(f"📊 Configuration: epochs={epochs}, batch_size={batch_size}, channels={init_channels}, layers={layers}")
        
        start_time = time.time()
        
        try:
            # 设置数据加载器
            train_loader, valid_loader, test_loader = self.setup_data_loaders(batch_size)
            
            # 初始化模型和框架
            self.initialize_model_and_framework(init_channels, layers)
            
            # 恢复或开始训练
            start_epoch = self.resume_or_start_training(resume_from)
            
            # 损失函数
            criterion = nn.CrossEntropyLoss().to(self.device_manager.device)
            
            logger.info(f"🎬 Training started from epoch {start_epoch}")
            
            # 训练主循环
            for epoch in range(start_epoch, epochs):
                self.current_epoch = epoch
                
                logger.info(f"\n{'='*80}")
                logger.info(f"Epoch {epoch}/{epochs-1} - Phase: {getattr(self.framework, 'current_phase', 'unknown')}")
                logger.info(f"{'='*80}")
                
                # 训练单个epoch
                stats = self.train_single_epoch(epoch, train_loader, valid_loader, criterion)
                
                # 记录统计信息
                self.training_stats['epoch_stats'].append(stats)
                
                # 输出进度
                self._log_epoch_progress(epoch, stats)
                
                # 保存检查点
                self.save_checkpoint_if_needed(epoch, stats)
                
                # 内存管理
                if epoch % 5 == 0:
                    self.device_manager.optimize_memory()
                
                # 重置恢复计数器（成功完成epoch后）
                if 'error' not in stats:
                    self.recovery_count = 0
                
                # 检查早停
                if hasattr(self.framework, 'should_early_stop') and self.framework.should_early_stop():
                    logger.info(f"⏹️ Early stopping at epoch {epoch}")
                    break
            
            # 训练完成，最终评估
            total_time = time.time() - start_time
            test_accuracy = self.evaluate_on_test(test_loader, criterion)
            
            # 生成最终报告
            self._generate_final_report(total_time, test_accuracy)
            
            logger.info("🎉 Training completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("⏹️ Training interrupted by user")
            self._emergency_save_checkpoint()
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._emergency_save_checkpoint()
            raise e
    
    def _log_epoch_progress(self, epoch: int, stats: dict):
        """记录epoch进度"""
        phase = stats.get('phase', 'unknown')
        train_acc = stats.get('train_accuracy', 0.0)
        valid_acc = stats.get('valid_accuracy', 0.0)
        train_loss = stats.get('train_loss', 0.0)
        valid_loss = stats.get('valid_loss', 0.0)
        epoch_time = stats.get('epoch_time', 0.0)
        
        # 内存信息
        memory_stats = self.device_manager.get_memory_stats()
        memory_info = ""
        if memory_stats.get('allocated_mb'):
            memory_info = f", GPU: {memory_stats['allocated_mb']:.0f}MB"
        
        # 性能提升信息
        improvement_info = ""
        if valid_acc > self.best_accuracy:
            improvement_info = f" 🚀(+{valid_acc - self.best_accuracy:.2f}%)"
        
        logger.info(f"Epoch {epoch:3d} | {phase:15s} | "
                   f"Train: {train_loss:.4f}/{train_acc:.2f}% | "
                   f"Valid: {valid_loss:.4f}/{valid_acc:.2f}%{improvement_info} | "
                   f"Time: {epoch_time:.1f}s{memory_info}")
        
        # 架构信息
        if phase == 'mutation':
            mutation_count = self.training_stats['mutation_count']
            logger.info(f"🧬 Architecture mutation #{mutation_count} completed")
        
        # 稳定性警告
        if self.stability_monitor:
            try:
                report = self.stability_monitor.get_comprehensive_report()
                health_score = report.get('current_health_score', 1.0)
                if health_score < 0.7:
                    logger.warning(f"⚠️ Training instability detected (health: {health_score:.3f})")
            except:
                pass
    
    def _generate_final_report(self, total_time: float, test_accuracy: float):
        """生成最终训练报告"""
        # 创建训练报告
        report_path = self.checkpoint_manager.create_training_report()
        
        # 汇总信息
        breakthrough_count = len(self.training_stats['performance_breakthroughs'])
        mutation_count = self.training_stats['mutation_count']
        recovery_count = self.training_stats['error_recoveries']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 FINAL RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"   Best Validation Accuracy: {self.best_accuracy:.2f}%")
        logger.info(f"   Final Test Accuracy: {test_accuracy:.2f}%")
        logger.info(f"   Total Training Time: {total_time/3600:.2f} hours")
        logger.info(f"   Performance Breakthroughs: {breakthrough_count}")
        logger.info(f"   Architecture Mutations: {mutation_count}")
        logger.info(f"   Error Recoveries: {recovery_count}")
        logger.info(f"   Training Report: {report_path}")
        logger.info(f"{'='*80}")
        
        # 保存详细统计
        final_stats = {
            'experiment_name': self.experiment_name,
            'best_validation_accuracy': self.best_accuracy,
            'final_test_accuracy': test_accuracy,
            'total_training_time_hours': total_time / 3600,
            'performance_breakthroughs': self.training_stats['performance_breakthroughs'],
            'architecture_evolution': self.training_stats['architecture_evolution'],
            'mutation_count': mutation_count,
            'error_recoveries': recovery_count,
            'final_genotype': str(getattr(self.framework, 'current_genotype', None)),
            'device_info': self.device_manager.get_device_report()
        }
        
        stats_path = f"./robust_checkpoints/{self.experiment_name}/final_results.json"
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(final_stats, f, indent=2, default=str)
        
        logger.info(f"📊 Final statistics saved: {stats_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Robust ASO-SE Classification Training')
    
    # 基本参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    parser.add_argument('--init_channels', type=int, default=16, help='初始通道数')
    parser.add_argument('--layers', type=int, default=8, help='网络层数')
    
    # ASO-SE参数
    parser.add_argument('--warmup_epochs', type=int, default=10, help='预热轮数')
    parser.add_argument('--arch_training_epochs', type=int, default=3, help='架构训练轮数')
    parser.add_argument('--weight_training_epochs', type=int, default=8, help='权重训练轮数')
    parser.add_argument('--mutation_strength', type=float, default=0.3, help='突变强度')
    
    # 续练参数
    parser.add_argument('--resume_from', type=str, default='latest', 
                       choices=['latest', 'best', 'none'], help='从哪个检查点恢复')
    parser.add_argument('--experiment_name', type=str, default='robust_aso_se_cifar10',
                       help='实验名称')
    
    # 设备参数
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载进程数')
    
    args = parser.parse_args()
    
    # 创建ASO-SE配置
    config = ASOSEConfig(
        warmup_epochs=args.warmup_epochs,
        arch_training_epochs=args.arch_training_epochs,
        weight_training_epochs=args.weight_training_epochs,
        mutation_strength=args.mutation_strength,
        device=args.device,
        save_checkpoints=True,
        checkpoint_frequency=5
    )
    
    # 创建训练器
    trainer = RobustASOSETrainer(config, args.experiment_name)
    
    # 开始训练
    logger.info("🚀 Starting Robust ASO-SE Training - Ready to breakthrough accuracy limits!")
    
    try:
        trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            init_channels=args.init_channels,
            layers=args.layers,
            resume_from=args.resume_from
        )
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 