#!/usr/bin/env python3
"""
智能架构进化演示 - 统一版本
Intelligent Architecture Evolution Demo - Unified Version

🎯 核心目标：
1. 在CIFAR-10上达到95%+的准确率
2. 使用Monte Carlo Dropout进行不确定性量化
3. 实现贝叶斯决策框架来判断变异价值
4. 提供模块化、可配置的演示环境

🧬 核心理论实现：
ΔI = α·ΔI_MI + β·ΔI_cond + γ·ΔI_uncert - δ·ΔI_cost

基于期望效用最大化的变异决策：
E[U(ΔI)] = E[1 - exp(-λ·ΔI)]

🔧 技术栈：
- Enhanced ResNet with SE-attention
- Monte Carlo Dropout uncertainty estimation  
- Bayesian mutation decision framework
- Shared utilities for reduced code duplication

📋 使用方式：
1. 基础演示：python examples/intelligent_evolution_demo.py
2. 高级演示：python examples/intelligent_evolution_demo.py --enhanced
3. 基准对比：python examples/intelligent_evolution_demo.py --baseline
"""

import argparse
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_demo_utils import (
    DemoConfig, 
    run_complete_demo, 
    setup_demo_environment,
    create_model_from_config,
    SharedDataManager,
    SharedTrainer
)
import logging

logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='智能架构进化演示',
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
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                       help='计算设备 (默认: auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='详细输出模式')
    parser.add_argument('--quiet', action='store_true',
                       help='安静模式 (覆盖verbose)')
    
    return parser.parse_args()


def create_config_from_args(args) -> DemoConfig:
    """从命令行参数创建配置"""
    config = DemoConfig()
    
    # 基础设置
    config.device = args.device
    config.seed = args.seed
    config.target_accuracy = args.target
    config.initial_epochs = args.epochs
    config.verbose = args.verbose and not args.quiet
    
    # 模式设置
    if args.enhanced:
        # 增强模式 - 95%准确率目标
        config.use_enhanced_features = True
        config.use_monte_carlo_uncertainty = True
        config.use_bayesian_decision = True
        config.model_type = 'enhanced_resnet34'
        config.target_accuracy = max(args.target, 95.0)
        config.log_level = 'INFO'
        print("🚀 启动增强版演示 - 目标95%准确率")
        
    elif args.baseline:
        # 基准模式 - 无进化
        config.use_enhanced_features = False
        config.use_monte_carlo_uncertainty = False
        config.use_bayesian_decision = False
        config.evolution_rounds = 0  # 不进行进化
        config.model_type = 'resnet18'
        config.target_accuracy = args.target
        print("📊 启动基准对比演示 - 无架构进化")
        
    else:
        # 标准模式
        config.use_enhanced_features = True
        config.use_monte_carlo_uncertainty = True
        config.use_bayesian_decision = True
        config.model_type = 'enhanced_resnet18'  # 使用较小模型作为默认
        config.target_accuracy = args.target
        print("🧬 启动标准智能进化演示")
    
    # 快速模式调整
    if args.quick:
        config.initial_epochs = max(5, config.initial_epochs // 3)
        config.evolution_rounds = min(2, config.evolution_rounds)
        config.additional_epochs_per_round = max(3, config.additional_epochs_per_round // 2)
        print("⚡ 快速模式已启用")
        
    return config


def run_baseline_demo(config: DemoConfig):
    """运行基准演示（无进化）"""
    print("="*60)
    print("📊 基准对比演示 - 无架构进化")
    print(f"🎯 目标：CIFAR-10上{config.target_accuracy}%准确率")
    print("="*60)
    
    # 设置环境
    device = setup_demo_environment(config)
    logger.info(f"使用设备: {device}")
    
    # 设置数据
    data_manager = SharedDataManager(config)
    train_loader, test_loader = data_manager.setup_data_loaders()
    
    # 创建基础模型
    from torchvision.models import resnet18
    import torch.nn as nn
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    logger.info(f"基准模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = SharedTrainer(model, device, train_loader, test_loader, config)
    
    # 训练模型
    print("\n📚 开始基准训练...")
    total_epochs = config.initial_epochs + config.evolution_rounds * config.additional_epochs_per_round
    final_acc = trainer.train_epochs(total_epochs)
    
    print(f"\n🏁 基准演示完成，最终准确率: {final_acc:.2f}%")
    
    if final_acc >= config.target_accuracy:
        print(f"\n🎉 基准模型达到{config.target_accuracy}%准确率目标！")
    else:
        print(f"\n📊 基准模型距离{config.target_accuracy}%目标还差: {config.target_accuracy - final_acc:.2f}%")
        
    return {
        'baseline_accuracy': final_acc,
        'target_reached': final_acc >= config.target_accuracy,
        'model_type': 'resnet18_baseline'
    }


def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_arguments()
        config = create_config_from_args(args)
        
        # 运行对应的演示
        if args.baseline:
            results = run_baseline_demo(config)
        else:
            results = run_complete_demo(config)
            
        # 显示结果摘要
        print("\n" + "="*60)
        print("📈 演示结果摘要")
        print("="*60)
        
        if 'baseline_accuracy' in results:
            print(f"基准准确率: {results['baseline_accuracy']:.2f}%")
        else:
            print(f"初始准确率: {results['initial_accuracy']:.2f}%")
            print(f"最终准确率: {results['final_accuracy']:.2f}%")
            print(f"进化轮数: {len(results['evolution_rounds'])}")
            
            if results['evolution_rounds']:
                total_improvement = sum(r['improvement'] for r in results['evolution_rounds'])
                print(f"总提升: {total_improvement:.2f}%")
                
        print(f"目标达成: {'✅' if results['target_reached'] else '❌'}")
        print("="*60)
        
        # 如果是增强模式且达到95%，特别庆祝
        if (args.enhanced and 
            results.get('final_accuracy', results.get('baseline_accuracy', 0)) >= 95.0):
            print("\n🎊🎊🎊 恭喜！成功达成95%准确率挑战！🎊🎊🎊")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 演示被用户中断")
        logger.info("演示被用户中断")
        
    except Exception as e:
        print(f"\n\n❌ 演示过程中遇到错误: {e}")
        logger.error(f"演示过程中遇到错误: {e}")
        import traceback
        if config.verbose:
            traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())