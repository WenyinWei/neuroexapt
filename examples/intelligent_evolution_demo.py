#!/usr/bin/env python3
"""
智能架构进化演示 - 基于理论框架的重构版本
Intelligent Architecture Evolution Demo - Theoretical Framework Refactored Version

🎯 核心目标：
1. 在CIFAR-10上达到95%+的准确率
2. 使用无参数结构评估技术
3. 实现多变异类型收益期望建模
4. 应用贝叶斯决策框架进行智能变异
5. 通过轻量级抽样验证校准收益期望

🧬 核心理论实现：
基于用户提供的理论框架：
- 有效信息(EI): EI(S) = max_{p(x)} [I(X; Y) - I(X; Y|S)]
- 积分信息(Φ): Φ ≈ Σ_{i,j} MI(H_i; H_j) - Σ_i MI(H_i; H_i)
- 结构冗余度(SR): SR = rank(1/N Σ_n W_n^T W_n)
- 变异优先级: Score(S, M) = α·ΔI + β·Φ(S) - γ·SR(S) - δ·Cost(M)
- 期望效用: E[U(ΔI)] = E[1 - exp(-λ·ΔI)]

🔧 技术栈：
- Enhanced ResNet with advanced architectures
- Parameter-free structural evaluation
- Multi-mutation type benefit modeling
- Lightweight sampling validation
- Bayesian decision framework

📋 使用方式：
1. 基础演示：python examples/intelligent_evolution_demo.py
2. 高级演示：python examples/intelligent_evolution_demo.py --enhanced
3. 基准对比：python examples/intelligent_evolution_demo.py --baseline
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入新的理论框架组件
from neuroexapt.core import (
    UnifiedIntelligentEvolutionEngine,
    NewEvolutionConfig as EvolutionConfig,
    ParameterFreeStructuralEvaluator,
    MultiMutationTypeEvaluator,
    LightweightSamplingValidator
)

# 导入模型组件
try:
    from neuroexapt.models import create_enhanced_model
except ImportError:
    # 如果没有增强模型，使用基础ResNet
    from torchvision.models import resnet18, resnet34
    def create_enhanced_model(model_type='resnet18', num_classes=10, **kwargs):
        if model_type == 'resnet18':
            model = resnet18(num_classes=num_classes)
        elif model_type == 'resnet34':
            model = resnet34(num_classes=num_classes)
        else:
            model = resnet18(num_classes=num_classes)
        
        # CIFAR-10适配
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model

logger = logging.getLogger(__name__)


class CIFAR10DataManager:
    """CIFAR-10数据管理器"""
    
    def __init__(self, data_root='./data', batch_size=128, num_workers=4):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def get_data_loaders(self, enhanced_augmentation=True):
        """获取数据加载器"""
        if enhanced_augmentation:
            # 增强数据增广
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.RandomErasing(p=0.1)
            ])
        else:
            # 基础数据增广
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
        
        # 加载数据集
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_root, train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_root, train=False, download=True, transform=test_transform
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        
        return train_loader, test_loader


class AdvancedTrainer:
    """高级训练器"""
    
    def __init__(self, model, device, criterion=None):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        
    def train_model(self, train_loader, test_loader, epochs=15, 
                   learning_rate=0.1, weight_decay=5e-4):
        """训练模型"""
        # 使用学习率调度
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)
            
            # 测试阶段
            test_accuracy = self.evaluate_model(test_loader)
            train_accuracy = 100.0 * train_correct / train_total
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
            
            scheduler.step()
            
            # 打印进度
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Acc={train_accuracy:.2f}%, "
                      f"Test Acc={test_accuracy:.2f}%, "
                      f"Best={best_accuracy:.2f}%")
        
        return best_accuracy
    
    def evaluate_model(self, test_loader):
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / total
        return accuracy


def setup_logging(verbose=True):
    """设置日志"""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='智能架构进化演示 - 理论框架版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s                    # 运行标准演示
  %(prog)s --enhanced         # 运行增强版演示 (95%目标)
  %(prog)s --baseline         # 运行基准对比
  %(prog)s --quick            # 快速演示 (减少训练轮数)
  %(prog)s --target 90        # 设置目标准确率为90%
        """
    )
    
    parser.add_argument('--enhanced', action='store_true',
                       help='使用增强版架构和训练技术')
    parser.add_argument('--baseline', action='store_true', 
                       help='运行基准对比 (不使用进化)')
    parser.add_argument('--quick', action='store_true',
                       help='快速演示模式 (减少训练轮数)')
    parser.add_argument('--target', type=float, default=95.0,
                       help='目标准确率 (默认: 95.0)')
    parser.add_argument('--epochs', type=int, default=15,
                       help='初始训练轮数 (默认: 15)')
    parser.add_argument('--evolution-rounds', type=int, default=3,
                       help='进化轮数 (默认: 3)')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                       help='计算设备 (默认: auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='详细输出模式')
    parser.add_argument('--quiet', action='store_true',
                       help='安静模式 (覆盖verbose)')
    
    return parser.parse_args()


def setup_device(device_arg='auto'):
    """设置计算设备"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    if device.type == 'cuda':
        print(f"🚀 使用GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("🖥️ 使用CPU")
    
    return device


def run_baseline_demo(args):
    """运行基准演示（无进化）"""
    print("="*60)
    print("📊 基准对比演示 - 无架构进化")
    print(f"🎯 目标：CIFAR-10上{args.target}%准确率")
    print("="*60)
    
    # 设置环境
    device = setup_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 数据管理
    data_manager = CIFAR10DataManager()
    train_loader, test_loader = data_manager.get_data_loaders(enhanced_augmentation=args.enhanced)
    
    # 创建基础模型
    model_type = 'resnet34' if args.enhanced else 'resnet18'
    model = create_enhanced_model(model_type=model_type, num_classes=10)
    
    print(f"基准模型: {model_type}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    trainer = AdvancedTrainer(model, device)
    epochs = args.epochs if not args.quick else max(5, args.epochs // 3)
    
    print(f"\n📚 开始基准训练 ({epochs} epochs)...")
    final_accuracy = trainer.train_model(train_loader, test_loader, epochs=epochs)
    
    # 结果摘要
    print(f"\n🏁 基准演示完成")
    print(f"最终准确率: {final_accuracy:.2f}%")
    
    if final_accuracy >= args.target:
        print(f"🎉 基准模型达到{args.target}%准确率目标！")
    else:
        print(f"📊 基准模型距离{args.target}%目标还差: {args.target - final_accuracy:.2f}%")
    
    return {
        'baseline_accuracy': final_accuracy,
        'target_reached': final_accuracy >= args.target,
        'model_type': model_type
    }


def run_intelligent_evolution_demo(args):
    """运行智能进化演示"""
    print("="*60)
    print("🧬 智能架构进化演示 - 理论框架版本")
    print(f"🎯 目标：CIFAR-10上{args.target}%准确率")
    print("="*60)
    
    # 设置环境
    device = setup_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 数据管理
    data_manager = CIFAR10DataManager()
    train_loader, test_loader = data_manager.get_data_loaders(enhanced_augmentation=args.enhanced)
    
    # 创建初始模型
    model_type = 'resnet34' if args.enhanced else 'resnet18'
    initial_model = create_enhanced_model(model_type=model_type, num_classes=10)
    
    print(f"初始模型: {model_type}")
    print(f"参数量: {sum(p.numel() for p in initial_model.parameters()):,}")
    
    # 初始训练
    trainer = AdvancedTrainer(initial_model, device)
    epochs = args.epochs if not args.quick else max(5, args.epochs // 3)
    
    print(f"\n📚 初始训练 ({epochs} epochs)...")
    initial_accuracy = trainer.train_model(train_loader, test_loader, epochs=epochs)
    print(f"初始准确率: {initial_accuracy:.2f}%")
    
    # 配置进化引擎
    evolution_config = EvolutionConfig(
        max_evolution_rounds=args.evolution_rounds if not args.quick else 2,
        target_accuracy=args.target,
        max_mutations_per_round=2 if args.quick else 3,
        enable_sampling_validation=not args.quick,  # 快速模式禁用抽样验证
        validation_sample_ratio=0.05 if args.quick else 0.1,
        quick_validation_epochs=2 if args.quick else 3
    )
    
    # 创建进化引擎
    evolution_engine = UnifiedIntelligentEvolutionEngine(
        config=evolution_config,
        device=device
    )
    
    print(f"\n🧬 开始智能架构进化...")
    print(f"进化配置: {evolution_config.max_evolution_rounds}轮, "
          f"目标{evolution_config.target_accuracy}%")
    
    # 执行进化
    start_time = time.time()
    
    def optimizer_factory(params):
        return optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    evolved_model, evolution_state = evolution_engine.evolve_architecture(
        model=initial_model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer_factory=optimizer_factory
    )
    
    evolution_time = time.time() - start_time
    
    # 获取进化摘要
    summary = evolution_engine.get_evolution_summary()
    
    # 最终评估
    final_trainer = AdvancedTrainer(evolved_model, device)
    final_accuracy = final_trainer.evaluate_model(test_loader)
    
    # 结果展示
    print(f"\n🎊 智能进化完成！")
    print(f"进化时间: {evolution_time:.1f} 秒")
    print(f"初始准确率: {initial_accuracy:.2f}%")
    print(f"最终准确率: {final_accuracy:.2f}%")
    print(f"总体改进: {final_accuracy - initial_accuracy:.2f}%")
    print(f"进化轮数: {summary['rounds_completed']}")
    print(f"成功变异: {summary['successful_mutations']}")
    print(f"失败变异: {summary['failed_mutations']}")
    
    if final_accuracy >= args.target:
        print(f"🎉 成功达到{args.target}%准确率目标！")
    else:
        print(f"📈 距离{args.target}%目标还差: {args.target - final_accuracy:.2f}%")
    
    return {
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'total_improvement': final_accuracy - initial_accuracy,
        'target_reached': final_accuracy >= args.target,
        'evolution_summary': summary,
        'evolution_time': evolution_time
    }


def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_arguments()
        
        # 设置日志
        setup_logging(args.verbose and not args.quiet)
        
        print("🚀 智能架构进化演示启动")
        print(f"PyTorch版本: {torch.__version__}")
        
        # 运行对应的演示
        if args.baseline:
            results = run_baseline_demo(args)
        else:
            results = run_intelligent_evolution_demo(args)
        
        # 显示最终结果摘要
        print("\n" + "="*60)
        print("📈 演示结果摘要")
        print("="*60)
        
        if 'baseline_accuracy' in results:
            print(f"基准准确率: {results['baseline_accuracy']:.2f}%")
        else:
            print(f"初始准确率: {results['initial_accuracy']:.2f}%")
            print(f"最终准确率: {results['final_accuracy']:.2f}%")
            print(f"总体改进: {results['total_improvement']:.2f}%")
            print(f"进化时间: {results['evolution_time']:.1f}秒")
        
        print(f"目标达成: {'✅ 是' if results['target_reached'] else '❌ 否'}")
        print("="*60)
        
        # 特别庆祝95%成就
        if (args.enhanced and results.get('final_accuracy', 
            results.get('baseline_accuracy', 0)) >= 95.0):
            print("\n🎊🎊🎊 恭喜！成功达成95%准确率挑战！🎊🎊🎊")
            print("🏆 理论框架验证成功！")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 演示被用户中断")
        logger.info("演示被用户中断")
        
    except Exception as e:
        print(f"\n\n❌ 演示过程中遇到错误: {e}")
        logger.error(f"演示过程中遇到错误: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())