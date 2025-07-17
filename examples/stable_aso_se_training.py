#!/usr/bin/env python3
"""
重构后的ASO-SE神经架构搜索训练脚本
使用全新设计的稳定架构搜索框架
"""

import os
import sys
import logging
import argparse
import random
import numpy as np
import torch

# 添加neuroexapt到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.core.aso_se_trainer import StableASO_SETrainer


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger()


def create_config():
    """创建训练配置"""
    config = {
        # 数据集配置
        'dataset': 'CIFAR-10',
        'batch_size': 128,
        
        # 训练配置
        'num_epochs': 80,  # 减少总epoch数进行快速测试
        'init_channels': 32,
        'init_depth': 4,
        'max_depth': 7,
        
        # 优化器配置
        'weight_lr': 0.025,
        'arch_lr': 3e-4,
        'momentum': 0.9,
        'weight_decay': 3e-4,
        
        # 阶段配置 - 调整为更合理的比例
        'warmup_epochs': 12,   # 权重预热
        'search_epochs': 25,   # 架构搜索
        'growth_epochs': 28,   # 网络生长
        'optimize_epochs': 15, # 最终优化
        
        # 搜索控制
        'arch_update_freq': 5,     # 每5个batch更新架构
        'growth_patience': 6,      # 6个epoch无改善后生长
        'growth_threshold': 0.015, # 1.5%改善阈值
    }
    
    return config


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='稳定ASO-SE训练')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='检查点保存目录')
    
    args = parser.parse_args()
    
    # 设置环境
    set_seed(args.seed)
    logger = setup_logging()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("🔧 重构版ASO-SE: 稳定的神经架构搜索")
    print("   目标: CIFAR-10 高准确率")
    print("   策略: 四阶段渐进式训练")
    print("   框架: 全新重构的稳定架构")
    
    # 创建配置
    config = create_config()
    
    # 创建训练器
    trainer = StableASO_SETrainer(config)
    
    # 如果有检查点，加载它
    if args.resume and os.path.exists(args.resume):
        print(f"📂 恢复训练: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    try:
        # 开始训练
        training_history, best_accuracy = trainer.train()
        
        # 保存最终结果
        final_checkpoint = os.path.join(args.save_dir, 'final_model.pth')
        trainer._save_checkpoint('final')
        
        print(f"\n🎉 训练完成!")
        print(f"   最佳测试精度: {best_accuracy:.2f}%")
        print(f"   模型保存到: {args.save_dir}")
        
        # 打印最终架构
        final_info = trainer.network.get_architecture_info()
        print(f"\n📋 最终架构:")
        print(f"   深度: {final_info['depth']}")
        print(f"   参数量: {final_info['parameters']:,}")
        print(f"   架构: {final_info['architecture']}")
        
        # 简单的性能分析
        print(f"\n📊 训练总结:")
        phases = ['warmup', 'search', 'growth', 'optimize']
        for phase in phases:
            phase_history = [h for h in training_history if h['phase'] == phase]
            if phase_history:
                best_in_phase = max(phase_history, key=lambda x: x['test_acc'])
                print(f"   {phase:8s}: 最佳 {best_in_phase['test_acc']:.2f}% (Epoch {best_in_phase['epoch']+1})")
        
        return best_accuracy
        
    except KeyboardInterrupt:
        print(f"\n⏸️ 训练被用户中断")
        # 保存中断时的状态
        interrupt_checkpoint = os.path.join(args.save_dir, 'interrupted.pth')
        trainer._save_checkpoint('interrupted')
        print(f"   状态已保存到: {interrupt_checkpoint}")
        return trainer.best_accuracy
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


if __name__ == '__main__':
    main()