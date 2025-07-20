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

# 导入统一演示工具
from demo_utils import (
    DemoConfiguration,
    DemoLogger,
    DeviceManager,
    CIFAR10DataManager,
    ModelManager,
    AdvancedTrainer,
    ResultFormatter
)


# 删除重复的类定义，使用demo_utils中的统一实现


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


# 移除重复的设备设置函数，使用DeviceManager


def run_baseline_demo(args):
    """运行基准演示（无进化）"""
    # 初始化日志器和配置
    logger = DemoLogger('baseline_demo', level='INFO', verbose=args.verbose and not args.quiet)
    
    logger.info("="*60)
    logger.info("📊 基准对比演示 - 无架构进化")
    logger.info(f"🎯 目标：CIFAR-10上{args.target}%准确率")
    logger.info("="*60)
    
    # 创建配置
    config = DemoConfiguration(
        device_type=args.device,
        seed=args.seed,
        enhanced_augmentation=args.enhanced,
        model_type='enhanced_resnet34' if args.enhanced else 'enhanced_resnet18',
        verbose=args.verbose and not args.quiet
    )
    
    # 设置环境
    device = DeviceManager.setup_environment(args.seed)
    device_info = DeviceManager.get_device_info(device)
    logger.info(f"设备信息:\n{ResultFormatter.format_device_info(device_info)}")
    
    # 数据管理
    data_manager = CIFAR10DataManager(config)
    train_loader, test_loader = data_manager.create_data_loaders()
    
    # 创建基础模型
    model = ModelManager.create_model(config)
    model_info = ModelManager.get_model_info(model)
    logger.info(f"模型信息:\n{ResultFormatter.format_model_info(model_info)}")
    
    # 训练模型
    trainer = AdvancedTrainer(model, device, config, logger)
    epochs = args.epochs if not args.quick else max(5, args.epochs // 3)
    
    logger.progress(f"开始基准训练 ({epochs} epochs)")
    final_accuracy = trainer.train_model(train_loader, test_loader, epochs=epochs)
    
    # 结果摘要
    logger.info(f"\n🏁 基准演示完成")
    logger.info(f"最终准确率: {final_accuracy:.2f}%")
    
    if final_accuracy >= args.target:
        logger.success(f"基准模型达到{args.target}%准确率目标！")
    else:
        logger.warning(f"基准模型距离{args.target}%目标还差: {args.target - final_accuracy:.2f}%")
    
    return {
        'baseline_accuracy': final_accuracy,
        'target_reached': final_accuracy >= args.target,
        'model_type': config.model_type
    }


def run_intelligent_evolution_demo(args):
    """运行智能进化演示"""
    # 初始化日志器和配置
    logger = DemoLogger('evolution_demo', level='INFO', verbose=args.verbose and not args.quiet)
    
    logger.info("="*60)
    logger.info("🧬 智能架构进化演示 - 理论框架版本")
    logger.info(f"🎯 目标：CIFAR-10上{args.target}%准确率")
    logger.info("="*60)
    
    # 创建配置
    config = DemoConfiguration(
        device_type=args.device,
        seed=args.seed,
        enhanced_augmentation=args.enhanced,
        model_type='enhanced_resnet34' if args.enhanced else 'enhanced_resnet18',
        verbose=args.verbose and not args.quiet
    )
    
    # 设置环境
    device = DeviceManager.setup_environment(args.seed)
    device_info = DeviceManager.get_device_info(device)
    logger.info(f"设备信息:\n{ResultFormatter.format_device_info(device_info)}")
    
    # 数据管理
    data_manager = CIFAR10DataManager(config)
    train_loader, test_loader = data_manager.create_data_loaders()
    
    # 创建初始模型
    initial_model = ModelManager.create_model(config)
    model_info = ModelManager.get_model_info(initial_model)
    logger.info(f"初始模型信息:\n{ResultFormatter.format_model_info(model_info)}")
    
    # 初始训练
    trainer = AdvancedTrainer(initial_model, device, config, logger)
    epochs = args.epochs if not args.quick else max(5, args.epochs // 3)
    
    logger.progress(f"初始训练 ({epochs} epochs)")
    initial_accuracy = trainer.train_model(train_loader, test_loader, epochs=epochs)
    logger.info(f"初始准确率: {initial_accuracy:.2f}%")
    
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
    
    logger.progress(f"开始智能架构进化")
    logger.info(f"进化配置: {evolution_config.max_evolution_rounds}轮, "
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
    final_trainer = AdvancedTrainer(evolved_model, device, config, logger)
    final_accuracy = final_trainer.evaluate_model(test_loader)
    
    # 结果展示
    logger.success(f"智能进化完成！(用时: {evolution_time:.1f}s)")
    logger.info(f"进化结果摘要:\n{ResultFormatter.format_evolution_summary(summary)}")
    
    if final_accuracy >= args.target:
        logger.success(f"成功达到{args.target}%准确率目标！")
    else:
        logger.warning(f"距离{args.target}%目标还差: {args.target - final_accuracy:.2f}%")
    
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
        
        # 初始化主日志器
        logger = DemoLogger('main', level='INFO', verbose=args.verbose and not args.quiet)
        
        logger.info("🚀 智能架构进化演示启动")
        logger.info(f"PyTorch版本: {torch.__version__}")
        
        # 运行对应的演示
        if args.baseline:
            results = run_baseline_demo(args)
        else:
            results = run_intelligent_evolution_demo(args)
        
        # 显示最终结果摘要
        logger.info("\n" + "="*60)
        logger.info("📈 演示结果摘要")
        logger.info("="*60)
        
        if 'baseline_accuracy' in results:
            logger.info(f"基准准确率: {results['baseline_accuracy']:.2f}%")
        else:
            logger.info(f"初始准确率: {results['initial_accuracy']:.2f}%")
            logger.info(f"最终准确率: {results['final_accuracy']:.2f}%")
            logger.info(f"总体改进: {results['total_improvement']:.2f}%")
            logger.info(f"进化时间: {results['evolution_time']:.1f}秒")
        
        status = "✅ 是" if results['target_reached'] else "❌ 否"
        logger.info(f"目标达成: {status}")
        logger.info("="*60)
        
        # 特别庆祝95%成就
        final_acc = results.get('final_accuracy', results.get('baseline_accuracy', 0))
        if args.enhanced and final_acc >= 95.0:
            logger.success("\n🎊🎊🎊 恭喜！成功达成95%准确率挑战！🎊🎊🎊")
            logger.success("🏆 理论框架验证成功！")
            
    except KeyboardInterrupt:
        logger.warning("\n⏹️ 演示被用户中断")
        
    except Exception as e:
        logger.error(f"\n❌ 演示过程中遇到错误: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())