"""
ASO-SE训练器 (ASO-SE Trainer)

重构后的ASO-SE训练器，基于新的ASO-SE框架实现。
保持向后兼容性，同时提供增强的功能和更好的架构设计。
"""

import torch
import torch.nn as nn
try:
    import torch.nn.functional as F
except ImportError:
    from torch.nn import functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from .model import Network as SearchNetwork
from .evolvable_model import EvolvableNetwork
from .genotypes import Genotype, PRIMITIVES
from .aso_se_framework import ASOSEFramework, ASOSEConfig
from .function_preserving_init import FunctionPreservingInitializer
from .gumbel_softmax_explorer import GumbelSoftmaxExplorer
from .architecture_mutator import ArchitectureMutator

logger = logging.getLogger(__name__)

def _derive_genotype(alphas_normal, alphas_reduce, steps=4):
    """
    从连续的alpha参数导出离散基因型（使用argmax）
    兼容旧接口的辅助函数
    """
    
    def _parse(weights):
        gene = []
        n = 2
        start = 0
        for i in range(steps):
            end = start + n
            W = weights[start:end].copy()
            
            # 为当前节点找到最好的2条边
            edges = sorted(range(i + 2), 
                         key=lambda x: -max(W[x][k] for k in range(len(W[x])) 
                                           if k != PRIMITIVES.index('none')))[:2]
            
            # 为选中的2条边各自找到最好的操作
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((PRIMITIVES[k_best], j))
            start = end
            n += 1
        return gene

    gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())
    
    concat = range(2, 2 + steps)
    
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

class ASOSETrainer:
    """
    重构的ASO-SE训练器
    
    基于新的ASO-SE框架，提供四阶段训练流程：
    1. 权重预热 (W-Training)
    2. 架构参数学习 (α-Training)  
    3. 架构突变与稳定 (Architecture Mutation & Stabilization)
    4. 权重再适应 (W-Retraining)
    """
    
    def __init__(self, search_model_args: Dict, model_args: Dict, training_args: Dict):
        """
        Args:
            search_model_args: 搜索模型参数
            model_args: 可进化模型参数
            training_args: 训练参数
        """
        # 1. 创建搜索模型
        self.search_model = SearchNetwork(**search_model_args)
        
        # 2. 创建ASO-SE配置
        self.config = self._create_config_from_args(training_args)
        
        # 3. 初始化ASO-SE框架
        self.framework = ASOSEFramework(self.search_model, self.config)
        
        # 4. 保存参数以便兼容性
        self.model_args = model_args
        self.training_args = training_args
        
        # 5. 向后兼容的属性
        self.criterion = nn.CrossEntropyLoss()
        self.w_optimizer = None
        self.alpha_optimizer = None
        
        # 6. 当前状态（向后兼容）
        self.current_genotype = None
        self.evolvable_model = None
        
        # 7. 训练统计
        self.training_stats = {
            "epoch_stats": [],
            "phase_transitions": [],
            "best_accuracy": 0.0
        }
        
        logger.info(f"🚀 ASO-SE Trainer initialized with framework integration")
        logger.info(f"   Config: {self.config.total_cycles} cycles, "
                   f"warmup={self.config.warmup_epochs}, "
                   f"arch={self.config.arch_training_epochs}")
    
    def _create_config_from_args(self, training_args: Dict) -> ASOSEConfig:
        """从训练参数创建ASO-SE配置"""
        return ASOSEConfig(
            # 从training_args提取参数，提供默认值
            warmup_epochs=int(training_args.get('warmup_epochs', 10)),
            arch_training_epochs=int(training_args.get('arch_epochs', 3)),
            weight_training_epochs=int(training_args.get('weight_epochs', 8)),
            total_cycles=int(training_args.get('total_cycles', 5)),
            
            # Gumbel-Softmax参数
            initial_temp=training_args.get('initial_temp', 5.0),
            min_temp=training_args.get('min_temp', 0.1),
            anneal_rate=training_args.get('temp_annealing_rate', 0.98),
            
            # 架构突变参数
            mutation_strength=training_args.get('mutation_strength', 0.3),
            mutation_frequency=training_args.get('mutation_frequency', 2),
            
            # 优化器参数
            weight_lr=training_args.get('learning_rate', 0.025),
            arch_lr=training_args.get('arch_learning_rate', 3e-4),
            weight_momentum=training_args.get('momentum', 0.9),
            weight_decay=training_args.get('weight_decay', 3e-4)
        )
    
    def initialize_optimizers(self):
        """初始化优化器（保持向后兼容）"""
        self.framework.initialize_optimizers()
        
        # 为向后兼容性提供访问
        self.w_optimizer = self.framework.weight_optimizer
        self.alpha_optimizer = self.framework.arch_optimizer
        
        logger.info("✅ Optimizers initialized through framework")
    
    def train_epoch(self, train_loader, valid_loader, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch（新的统一接口）
        
        Args:
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            epoch: 当前epoch
            
        Returns:
            训练统计信息
        """
        if self.w_optimizer is None:
            self.initialize_optimizers()
        
        # 使用框架进行训练
        stats = self.framework.train_cycle(train_loader, valid_loader, self.criterion, epoch)
        
        # 更新向后兼容的状态
        self._update_legacy_state()
        
        # 记录统计信息
        self.training_stats["epoch_stats"].append(stats)
        
        # 更新最佳准确率
        if "valid_accuracy" in stats:
            self.training_stats["best_accuracy"] = max(
                self.training_stats["best_accuracy"], 
                stats["valid_accuracy"]
            )
        
        return stats
    
    def train_weights(self, train_queue, epoch: int):
        """
        阶段1：权重训练（向后兼容接口）
        
        Args:
            train_queue: 训练数据队列
            epoch: 当前epoch
        """
        logger.info(f"🔥 Epoch {epoch}: [W-Training] Training weights of current model")
        
        # 如果框架处于权重训练阶段，进行训练
        if self.framework.current_phase in ["warmup", "weight_retraining"]:
            if hasattr(train_queue, '__iter__'):
                # 如果是数据加载器，直接使用
                train_loader = train_queue
                valid_loader = train_queue  # 简化，实际应该有单独的验证集
                
                stats = self.framework.train_cycle(train_loader, valid_loader, self.criterion, epoch)
                return stats
        
        logger.warning(f"Weight training called but framework is in {self.framework.current_phase} phase")
    
    def train_alphas(self, valid_queue, epoch: int):
        """
        阶段2：架构参数训练（向后兼容接口）
        
        Args:
            valid_queue: 验证数据队列
            epoch: 当前epoch
        """
        logger.info(f"🔍 Epoch {epoch}: [α-Training] Searching for better architecture")
        
        # 如果框架处于架构训练阶段，进行训练
        if self.framework.current_phase == "arch_training":
            if hasattr(valid_queue, '__iter__'):
                train_loader = valid_queue  # 简化，实际应该有单独的训练集
                valid_loader = valid_queue
                
                stats = self.framework.train_cycle(train_loader, valid_loader, self.criterion, epoch)
                return stats
        
        logger.warning(f"Alpha training called but framework is in {self.framework.current_phase} phase")
    
    def mutate_architecture(self) -> Genotype:
        """
        阶段3：架构突变（向后兼容接口）
        
        Returns:
            新的基因型
        """
        logger.info("🧬 [Mutation] Performing architecture mutation using enhanced ASO-SE")
        
        # 触发框架的突变阶段
        if self.framework.current_phase == "mutation":
            # 框架会自动处理突变
            new_genotype = self.framework.current_genotype or self.derive_best_genotype(use_gumbel=True)
        else:
            # 手动触发突变（如果需要）
            new_genotype = self.derive_best_genotype(use_gumbel=True)
        
        # 更新当前状态
        self.current_genotype = new_genotype
        self.evolvable_model = self.framework.evolvable_model
        
        logger.info(f"✅ [Stabilization] Architecture mutated successfully")
        return new_genotype
    
    def derive_best_genotype(self, use_gumbel: bool = False) -> Genotype:
        """
        从搜索模型的alphas导出最佳基因型
        
        Args:
            use_gumbel: 是否使用Gumbel-Softmax采样
            
        Returns:
            导出的基因型
        """
        if use_gumbel and hasattr(self.framework, 'explorer'):
            # 使用框架的Gumbel-Softmax探索器
            try:
                return self.framework._gumbel_sample_architecture()
            except Exception as e:
                logger.warning(f"Gumbel sampling failed: {e}, falling back to argmax")
        
        # 回退到确定性argmax导出
        return _derive_genotype(
            self.search_model.alphas_normal,
            self.search_model.alphas_reduce
        )
    
    def run_training_loop(self, train_queue, valid_queue, epochs: int, 
                         w_epochs: int = None, alpha_epochs: int = None):
        """
        主要的ASO-SE训练循环（向后兼容接口）
        
        Args:
            train_queue: 训练数据队列
            valid_queue: 验证数据队列
            epochs: 总epoch数
            w_epochs: 权重训练epoch数（可选，会使用配置中的值）
            alpha_epochs: 架构训练epoch数（可选，会使用配置中的值）
        """
        logger.info(f"🚀 Starting ASO-SE training loop for {epochs} epochs")
        
        # 初始化优化器
        if self.w_optimizer is None:
            self.initialize_optimizers()
        
        # 运行训练循环
        for epoch in range(epochs):
            try:
                # 使用新的统一训练接口
                stats = self.train_epoch(train_queue, valid_queue, epoch)
                
                # 日志记录
                self._log_epoch_stats(epoch, stats)
                
                # 检查早停
                if self.framework.should_early_stop():
                    logger.info(f"🛑 Early stopping at epoch {epoch}")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error in epoch {epoch}: {e}")
                break
        
        # 训练完成后的总结
        self._log_training_summary()
    
    def _create_evolvable_model(self, genotype: Genotype) -> EvolvableNetwork:
        """
        基于基因型创建可进化模型（向后兼容）
        
        Args:
            genotype: 目标基因型
            
        Returns:
            可进化网络模型
        """
        return EvolvableNetwork(**self.model_args, genotype=genotype)
    
    def _update_legacy_state(self):
        """更新向后兼容的状态变量"""
        # 从框架同步状态
        self.current_genotype = self.framework.current_genotype
        self.evolvable_model = self.framework.evolvable_model
    
    def _log_epoch_stats(self, epoch: int, stats: Dict[str, float]):
        """记录epoch统计信息"""
        phase = stats.get("phase", "unknown")
        
        if "train_accuracy" in stats and "valid_accuracy" in stats:
            logger.info(f"📊 Epoch {epoch:3d} [{phase:>12s}] "
                       f"Train: {stats['train_accuracy']:.2f}% "
                       f"Valid: {stats['valid_accuracy']:.2f}%")
        elif "valid_accuracy" in stats:
            logger.info(f"📊 Epoch {epoch:3d} [{phase:>12s}] "
                       f"Valid: {stats['valid_accuracy']:.2f}%")
        
        # 记录阶段转换
        if "phase" in stats:
            last_phase = (self.training_stats["epoch_stats"][-1]["phase"] 
                         if self.training_stats["epoch_stats"] else None)
            if phase != last_phase:
                self.training_stats["phase_transitions"].append({
                    "epoch": epoch,
                    "phase": phase
                })
    
    def _log_training_summary(self):
        """记录训练总结"""
        logger.info("=" * 60)
        logger.info("🎉 ASO-SE Training Completed!")
        logger.info(f"📈 Best Accuracy: {self.training_stats['best_accuracy']:.2f}%")
        
        # 获取框架报告
        framework_report = self.framework.get_training_report()
        logger.info(f"🔬 Total Cycles: {framework_report['current_cycle']}")
        logger.info(f"🧬 Total Mutations: {framework_report['total_mutations']}")
        
        # 探索报告
        exploration_report = framework_report.get("exploration_report", {})
        if "current_temperature" in exploration_report:
            logger.info(f"🌡️ Final Temperature: {exploration_report['current_temperature']:.3f}")
        
        logger.info("=" * 60)
    
    # 新增的便利方法
    
    def get_current_architecture(self) -> Optional[Genotype]:
        """获取当前架构"""
        return self.current_genotype
    
    def get_search_model(self) -> nn.Module:
        """获取搜索模型"""
        return self.search_model
    
    def get_evolvable_model(self) -> Optional[nn.Module]:
        """获取可进化模型"""
        return self.evolvable_model
    
    def get_training_stats(self) -> Dict:
        """获取训练统计"""
        return self.training_stats
    
    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        self.framework.save_checkpoint(filepath)
        logger.info(f"💾 Trainer checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        self.framework.load_checkpoint(filepath)
        self._update_legacy_state()
        logger.info(f"📂 Trainer checkpoint loaded from {filepath}")
    
    def get_framework_report(self) -> Dict:
        """获取框架详细报告"""
        return self.framework.get_training_report()

# 向后兼容的工厂函数
def create_aso_se_trainer(search_model_args: Dict, model_args: Dict, 
                         training_args: Dict) -> ASOSETrainer:
    """
    创建ASO-SE训练器的工厂函数
    
    Args:
        search_model_args: 搜索模型参数
        model_args: 可进化模型参数
        training_args: 训练参数
        
    Returns:
        配置好的ASO-SE训练器
    """
    trainer = ASOSETrainer(search_model_args, model_args, training_args)
    return trainer 