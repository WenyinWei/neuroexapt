#!/usr/bin/env python3
"""
ASO-SE框架完整演示

展示交替式稳定优化与随机探索(ASO-SE)框架的完整使用流程，包括：
1. 四阶段训练流程
2. 函数保持初始化
3. Gumbel-Softmax引导式探索
4. 架构突变与稳定
5. 渐进式架构生长

这个演示展示了重构后的ASO-SE框架如何解决可微架构搜索中的核心矛盾。
"""

import os
import sys
import argparse
import logging
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neuroexapt.core.aso_se_trainer import ASOSETrainer, create_aso_se_trainer
from neuroexapt.core.aso_se_framework import ASOSEConfig
from neuroexapt.core.model import Network as SearchNetwork
from neuroexapt.utils.train_utils import count_parameters_in_MB

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('aso_se_demo.log')
    ]
)
logger = logging.getLogger(__name__)

def get_data_loaders(args):
    """获取数据加载器"""
    logger.info(f"📊 Loading {args.dataset} dataset...")
    
    if args.dataset == 'CIFAR10':
        # CIFAR-10数据变换
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # 加载数据集
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        num_classes = 10
        
    elif args.dataset == 'CIFAR100':
        # CIFAR-100数据变换
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # 分割训练集为训练和验证
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    
    np.random.shuffle(indices)
    
    train_idx, valid_idx = indices[:split], indices[split:]
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
    
    # 创建数据加载器
    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    
    valid_loader = DataLoader(
        trainset, batch_size=args.batch_size, sampler=valid_sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    logger.info(f"✅ Dataset loaded: {len(trainset)} total, "
               f"{len(train_idx)} train, {len(valid_idx)} valid, {len(testset)} test")
    
    return train_loader, valid_loader, test_loader, num_classes

def create_demo_trainer(args, num_classes):
    """创建ASO-SE演示训练器"""
    logger.info("🏗️ Creating ASO-SE trainer with enhanced configuration...")
    
    # 搜索模型参数
    search_model_args = {
        'C': args.init_channels,
        'num_classes': num_classes,
        'layers': args.layers,
        'steps': 4,
        'block_multiplier': 4,
        'stem_multiplier': 3
    }
    
    # 可进化模型参数
    model_args = {
        'C': args.init_channels,
        'num_classes': num_classes,
        'layers': args.layers
    }
    
    # 训练参数（展示ASO-SE框架的完整配置）
    training_args = {
        # 四阶段训练配置
        'warmup_epochs': args.warmup_epochs,
        'arch_epochs': args.arch_epochs,
        'weight_epochs': args.weight_epochs,
        'total_cycles': args.total_cycles,
        
        # Gumbel-Softmax探索配置
        'initial_temp': args.initial_temp,
        'min_temp': args.min_temp,
        'temp_annealing_rate': args.anneal_rate,
        'exploration_factor': 1.2,  # 增强探索
        
        # 架构突变配置
        'mutation_strength': 0.3,
        'mutation_frequency': 2,  # 每2个周期突变一次
        
        # 优化器配置
        'learning_rate': args.learning_rate,
        'arch_learning_rate': 3e-4,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        
        # 早停和监控
        'early_stopping_patience': 15,
        'performance_threshold': 0.01
    }
    
    # 创建训练器
    trainer = create_aso_se_trainer(search_model_args, model_args, training_args)
    
    # 打印模型信息
    logger.info(f"📊 Search model parameters: {count_parameters_in_MB(trainer.get_search_model()):.2f}MB")
    
    return trainer

def demonstrate_aso_se_features(trainer):
    """演示ASO-SE框架的特色功能"""
    logger.info("=" * 60)
    logger.info("🔬 Demonstrating ASO-SE Framework Features")
    logger.info("=" * 60)
    
    # 1. 展示Gumbel-Softmax探索
    logger.info("1️⃣ Gumbel-Softmax Guided Exploration:")
    explorer = trainer.framework.explorer
    logger.info(f"   🌡️ Initial Temperature: {explorer.initial_temp}")
    logger.info(f"   🎯 Minimum Temperature: {explorer.min_temp}")
    logger.info(f"   📈 Annealing Rate: {explorer.anneal_rate}")
    
    # 2. 展示函数保持初始化
    logger.info("2️⃣ Function-Preserving Initialization:")
    initializer = trainer.framework.initializer
    logger.info(f"   🛡️ Preserve Ratio: {initializer.preserve_ratio}")
    logger.info(f"   🔊 Noise Scale: {initializer.noise_scale}")
    
    # 3. 展示架构突变器
    logger.info("3️⃣ Architecture Mutator:")
    mutator = trainer.framework.mutator
    logger.info(f"   🧬 Mutation Strength: {mutator.mutation_strength}")
    logger.info(f"   ⚙️ Function Preservation: {mutator.preserve_function}")
    
    # 4. 展示训练配置
    logger.info("4️⃣ Four-Stage Training Configuration:")
    config = trainer.framework.config
    logger.info(f"   🔥 Warmup Epochs: {config.warmup_epochs}")
    logger.info(f"   🔍 Architecture Training Epochs: {config.arch_training_epochs}")
    logger.info(f"   🔧 Weight Retraining Epochs: {config.weight_training_epochs}")
    logger.info(f"   🔄 Total Cycles: {config.total_cycles}")
    
    logger.info("=" * 60)

def train_with_monitoring(trainer, train_loader, valid_loader, args):
    """带监控的训练过程"""
    logger.info("🚀 Starting ASO-SE training with comprehensive monitoring...")
    
    # 记录开始时间
    start_time = time.time()
    best_accuracy = 0.0
    
    # 训练循环
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        try:
            # 执行一个训练周期
            stats = trainer.train_epoch(train_loader, valid_loader, epoch)
            
            # 提取关键指标
            phase = stats.get('phase', 'unknown')
            train_acc = stats.get('train_accuracy', 0.0)
            valid_acc = stats.get('valid_accuracy', 0.0)
            
            # 更新最佳准确率
            if valid_acc > best_accuracy:
                best_accuracy = valid_acc
                # 保存最佳模型
                if args.save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
                    trainer.save_checkpoint(checkpoint_path)
                    logger.info(f"💾 Best model saved: {valid_acc:.2f}%")
            
            # 计算epoch耗时
            epoch_time = time.time() - epoch_start_time
            
            # 详细日志
            logger.info(f"📈 Epoch {epoch:3d}/{args.epochs} [{phase:>15s}] "
                       f"Train: {train_acc:5.2f}% Valid: {valid_acc:5.2f}% "
                       f"Best: {best_accuracy:5.2f}% Time: {epoch_time:.1f}s")
            
            # 阶段特定的监控
            if phase == 'mutation':
                logger.info(f"🧬 Architecture mutation completed at epoch {epoch}")
                current_genotype = trainer.get_current_architecture()
                if current_genotype:
                    logger.info(f"   New architecture: {len(current_genotype.normal)} normal ops")
            
            # 定期报告探索状态
            if epoch % 10 == 0:
                exploration_report = trainer.framework.explorer.get_exploration_report()
                if 'current_temperature' in exploration_report:
                    temp = exploration_report['current_temperature']
                    logger.info(f"🌡️ Current exploration temperature: {temp:.3f}")
            
            # 早停检查
            if trainer.framework.should_early_stop():
                logger.info(f"🛑 Early stopping triggered at epoch {epoch}")
                break
                
        except Exception as e:
            logger.error(f"❌ Training error at epoch {epoch}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            break
    
    # 训练完成统计
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("🎉 ASO-SE Training Completed!")
    logger.info(f"📊 Total Time: {total_time/60:.1f} minutes")
    logger.info(f"📈 Best Accuracy: {best_accuracy:.2f}%")
    
    # 框架报告
    framework_report = trainer.get_framework_report()
    logger.info(f"🔬 Total Cycles: {framework_report['current_cycle']}")
    logger.info(f"🧬 Total Mutations: {framework_report['total_mutations']}")
    
    exploration_report = framework_report.get('exploration_report', {})
    if 'average_entropy' in exploration_report:
        logger.info(f"🔍 Average Exploration Entropy: {exploration_report['average_entropy']:.3f}")
    
    logger.info("=" * 60)
    
    return best_accuracy, framework_report

def evaluate_final_architecture(trainer, test_loader, args):
    """评估最终架构"""
    logger.info("📋 Evaluating final evolved architecture...")
    
    # 获取最终架构
    final_genotype = trainer.get_current_architecture()
    if final_genotype is None:
        logger.warning("⚠️ No final architecture available")
        return
    
    logger.info(f"🏗️ Final Architecture:")
    logger.info(f"   Normal: {final_genotype.normal}")
    logger.info(f"   Reduce: {final_genotype.reduce}")
    
    # 获取可进化模型
    evolvable_model = trainer.get_evolvable_model()
    if evolvable_model is None:
        logger.warning("⚠️ No evolvable model available for evaluation")
        return
    
    # 评估模式
    evolvable_model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            
            output = evolvable_model(data)
            test_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_accuracy = 100. * correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    logger.info(f"🎯 Final Test Results:")
    logger.info(f"   Accuracy: {test_accuracy:.2f}%")
    logger.info(f"   Loss: {avg_test_loss:.4f}")
    logger.info(f"   Parameters: {count_parameters_in_MB(evolvable_model):.2f}MB")
    
    return test_accuracy

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ASO-SE Framework Demo')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--train_portion', type=float, default=0.8, help='训练集比例')
    
    # 模型参数
    parser.add_argument('--init_channels', type=int, default=16, help='初始通道数')
    parser.add_argument('--layers', type=int, default=8, help='网络层数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='总训练轮数')
    parser.add_argument('--warmup_epochs', type=int, default=8, help='预热轮数')
    parser.add_argument('--arch_epochs', type=int, default=3, help='架构训练轮数')
    parser.add_argument('--weight_epochs', type=int, default=6, help='权重训练轮数')
    parser.add_argument('--total_cycles', type=int, default=3, help='总循环数')
    
    # 优化器参数
    parser.add_argument('--learning_rate', type=float, default=0.025, help='学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='权重衰减')
    
    # ASO-SE特有参数
    parser.add_argument('--initial_temp', type=float, default=5.0, help='初始温度')
    parser.add_argument('--min_temp', type=float, default=0.1, help='最小温度')
    parser.add_argument('--anneal_rate', type=float, default=0.98, help='退火率')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints_aso_se', help='保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        logger.info(f"🔧 Using GPU {args.gpu}: {torch.cuda.get_device_name()}")
    else:
        logger.info("🔧 Using CPU")
    
    logger.info("🚀 ASO-SE Framework Demo Starting...")
    logger.info(f"📊 Configuration: {args}")
    
    try:
        # 1. 加载数据
        train_loader, valid_loader, test_loader, num_classes = get_data_loaders(args)
        
        # 2. 创建训练器
        trainer = create_demo_trainer(args, num_classes)
        
        # 3. 演示框架特性
        demonstrate_aso_se_features(trainer)
        
        # 4. 执行训练
        best_accuracy, framework_report = train_with_monitoring(trainer, train_loader, valid_loader, args)
        
        # 5. 评估最终架构
        test_accuracy = evaluate_final_architecture(trainer, test_loader, args)
        
        # 6. 保存最终结果
        if args.save_dir:
            results = {
                'args': vars(args),
                'best_validation_accuracy': best_accuracy,
                'test_accuracy': test_accuracy,
                'framework_report': framework_report
            }
            
            results_path = os.path.join(args.save_dir, 'final_results.pth')
            torch.save(results, results_path)
            logger.info(f"💾 Results saved to {results_path}")
        
        logger.info("✅ ASO-SE Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 