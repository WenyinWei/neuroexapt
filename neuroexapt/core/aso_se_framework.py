"""
ASO-SE框架 (Alternating Stable Optimization with Stochastic Exploration)

交替式稳定优化与随机探索框架的完整实现，包含：

四阶段训练流程：
1. 阶段一：网络权重预热 (W-Training) - 稳定化基础权重
2. 阶段二：架构参数学习 (α-Training) - 搜索最优架构配置  
3. 阶段三：架构突变与稳定 (Architecture Mutation & Stabilization) - 函数保持突变
4. 阶段四：权重再适应 (W-Retraining) - 在新架构上继续优化

核心特性：
- 函数保持初始化确保架构变化时的平滑过渡
- Gumbel-Softmax引导式探索避免局部最优
- 自适应温度控制平衡探索与利用
- 渐进式架构生长避免剧烈变化
- 设备一致性管理和内存优化
- 完整的检查点保存和恢复机制
"""

import torch
import torch.nn as nn
try:
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    from torch.nn import functional as F
    from torch import optim
import numpy as np
import logging
import time
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass

from .function_preserving_init import FunctionPreservingInitializer
from .gumbel_softmax_explorer import GumbelSoftmaxExplorer, create_annealing_schedule
from .architecture_mutator import ArchitectureMutator, GradualArchitectureGrowth
from .genotypes import Genotype, PRIMITIVES
from .device_manager import DeviceManager, get_device_manager

logger = logging.getLogger(__name__)

@dataclass
class ASOSEConfig:
    """ASO-SE框架配置"""
    # 训练阶段配置
    warmup_epochs: int = 10
    arch_training_epochs: int = 3
    weight_training_epochs: int = 8
    total_cycles: int = 5
    
    # Gumbel-Softmax探索配置
    initial_temp: float = 5.0
    min_temp: float = 0.1
    anneal_rate: float = 0.98
    exploration_factor: float = 1.0
    
    # 架构突变配置
    mutation_strength: float = 0.3
    mutation_frequency: int = 2  # 每几个cycle进行一次突变
    preserve_function: bool = True
    
    # 优化器配置
    weight_lr: float = 0.025
    weight_momentum: float = 0.9
    weight_decay: float = 3e-4
    arch_lr: float = 3e-4
    
    # 生长策略配置
    enable_gradual_growth: bool = True
    growth_schedule: Optional[List[Dict]] = None
    
    # 监控配置
    early_stopping_patience: int = 10
    performance_threshold: float = 0.01
    
    # 设备和内存配置
    device: Optional[str] = None
    memory_fraction: float = 0.9
    
    # 检查点配置
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5  # 每几个epoch保存一次
    max_checkpoints: int = 3  # 最多保留几个检查点

class ASOSEFramework:
    """
    ASO-SE框架主类
    
    统一管理四阶段训练流程和各个组件，包含设备管理和内存优化
    """
    
    def __init__(self, search_model: nn.Module, config: ASOSEConfig):
        """
        Args:
            search_model: 搜索模型（带架构参数的网络）
            config: ASO-SE配置
        """
        self.config = config
        
        # 初始化设备管理器
        self.device_manager = get_device_manager(config.device)
        
        # 注册搜索模型到设备管理器
        self.search_model = self.device_manager.register_model("search_model", search_model)
        
        # 核心组件初始化
        self.initializer = FunctionPreservingInitializer()
        self.explorer = GumbelSoftmaxExplorer(
            initial_temp=config.initial_temp,
            min_temp=config.min_temp,
            anneal_rate=config.anneal_rate,
            exploration_factor=config.exploration_factor
        )
        self.mutator = ArchitectureMutator(
            preserve_function=config.preserve_function,
            mutation_strength=config.mutation_strength
        )
        
        # 当前状态
        self.current_cycle = 0
        self.current_phase = "warmup"  # warmup, arch_training, mutation, weight_retraining
        self.current_genotype = None
        self.evolvable_model = None
        
        # 训练历史
        self.training_history = {
            "loss_history": [],
            "accuracy_history": [],
            "architecture_history": [],
            "mutation_history": [],
            "phase_transitions": [],
            "device_stats": []
        }
        
        # 优化器（延迟初始化）
        self.weight_optimizer = None
        self.arch_optimizer = None
        
        # 性能监控
        self.best_performance = 0.0
        self.patience_counter = 0
        
        # 渐进式生长
        if config.enable_gradual_growth:
            self.growth_manager = GradualArchitectureGrowth(
                self.mutator, config.growth_schedule
            )
        else:
            self.growth_manager = None
        
        # 检查点管理
        self.checkpoint_dir = None
        self.checkpoint_counter = 0
        
        logger.info(f"🚀 ASO-SE Framework initialized: "
                   f"{config.total_cycles} cycles, "
                   f"warmup={config.warmup_epochs}, "
                   f"arch={config.arch_training_epochs}, "
                   f"weight={config.weight_training_epochs}")
        
        # 记录设备信息
        device_report = self.device_manager.get_device_report()
        logger.info(f"🔧 Device: {device_report['device']}")
        
        # 内存监控
        self._log_memory_usage("initialization")
    
    def setup_checkpoint_dir(self, checkpoint_dir: str):
        """设置检查点目录"""
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"💾 Checkpoint directory: {checkpoint_dir}")
    
    def initialize_optimizers(self):
        """初始化优化器"""
        # 分离参数
        arch_params = list(self.search_model.arch_parameters()) if hasattr(self.search_model, 'arch_parameters') else []
        weight_params = [p for p in self.search_model.parameters() 
                        if not any(p is ap for ap in arch_params)]
        
        self.weight_optimizer = optim.SGD(
            weight_params,
            lr=self.config.weight_lr,
            momentum=self.config.weight_momentum,
            weight_decay=self.config.weight_decay
        )
        
        if arch_params:
            self.arch_optimizer = optim.Adam(
                arch_params,
                lr=self.config.arch_lr,
                weight_decay=1e-3
            )
        
        logger.info(f"📊 Optimizers initialized: "
                   f"weight_params={len(weight_params)}, "
                   f"arch_params={len(arch_params)}")
    
    def train_cycle(self, train_loader, valid_loader, 
                   criterion: nn.Module, epoch: int) -> Dict[str, float]:
        """
        执行一个完整的ASO-SE训练周期
        
        Args:
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器  
            criterion: 损失函数
            epoch: 当前epoch
            
        Returns:
            训练统计信息
        """
        # 包装数据加载器以自动转移设备
        train_loader = self.device_manager.create_data_loader_wrapper(train_loader)
        valid_loader = self.device_manager.create_data_loader_wrapper(valid_loader)
        
        # 确保损失函数在正确设备上
        criterion = criterion.to(self.device_manager.device)
        
        cycle_stats = {}
        
        # 确定当前阶段
        phase = self._determine_current_phase(epoch)
        
        if phase != self.current_phase:
            self._transition_to_phase(phase, epoch)
        
        # 根据阶段执行对应的训练
        try:
            if phase == "warmup":
                stats = self._warmup_phase(train_loader, valid_loader, criterion, epoch)
            elif phase == "arch_training":
                stats = self._arch_training_phase(train_loader, valid_loader, criterion, epoch)
            elif phase == "mutation":
                stats = self._mutation_phase(train_loader, valid_loader, criterion, epoch)
            elif phase == "weight_retraining":
                stats = self._weight_retraining_phase(train_loader, valid_loader, criterion, epoch)
            else:
                raise ValueError(f"Unknown phase: {phase}")
            
            # 记录设备统计
            device_stats = self.device_manager.get_memory_stats()
            # 将设备统计转换为数值格式
            if isinstance(device_stats, dict) and 'device' in device_stats:
                # CPU情况下记录设备类型为数值（0表示CPU）
                stats['device_type'] = 0.0
            else:
                # GPU情况下记录内存使用情况
                allocated = device_stats.get('allocated_mb', 0.0)
                utilization = device_stats.get('utilization', 0.0)
                stats['memory_allocated_mb'] = float(allocated) if isinstance(allocated, (int, float)) else 0.0
                stats['memory_utilization'] = float(utilization) if isinstance(utilization, (int, float)) else 0.0
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"🚨 OOM error in epoch {epoch}, attempting recovery...")
                self.device_manager.optimize_memory()
                
                # 重试一次
                try:
                    if phase == "warmup":
                        stats = self._warmup_phase(train_loader, valid_loader, criterion, epoch)
                    elif phase == "arch_training":
                        stats = self._arch_training_phase(train_loader, valid_loader, criterion, epoch)
                    elif phase == "mutation":
                        stats = self._mutation_phase(train_loader, valid_loader, criterion, epoch)
                    elif phase == "weight_retraining":
                        stats = self._weight_retraining_phase(train_loader, valid_loader, criterion, epoch)
                except RuntimeError as e2:
                    logger.error(f"❌ Memory recovery failed: {e2}")
                    stats = {"error": "out_of_memory", "phase": phase}
            else:
                logger.error(f"❌ Training error in epoch {epoch}: {e}")
                stats = {"error": str(e), "phase": phase}
        
        # 更新历史记录
        self._update_training_history(stats, epoch, phase)
        
        # 检查是否需要进行渐进式生长
        if self.growth_manager and self.growth_manager.should_grow(epoch):
            self._perform_gradual_growth(epoch)
        
        # 更新探索温度
        self._update_exploration_temperature(stats.get("valid_accuracy", 0.0))
        
        # 定期内存优化
        if epoch % 10 == 0:
            self.device_manager.optimize_memory()
            self._log_memory_usage(f"epoch_{epoch}")
        
        # 保存检查点
        if (self.config.save_checkpoints and 
            epoch % self.config.checkpoint_frequency == 0 and 
            self.checkpoint_dir):
            self._save_checkpoint(epoch, stats)
        
        return stats
    
    def _determine_current_phase(self, epoch: int) -> str:
        """确定当前训练阶段"""
        cycle_length = (self.config.warmup_epochs + 
                       self.config.arch_training_epochs + 
                       1 +  # mutation phase
                       self.config.weight_training_epochs)
        
        position_in_cycle = epoch % cycle_length
        
        if position_in_cycle < self.config.warmup_epochs:
            return "warmup"
        elif position_in_cycle < (self.config.warmup_epochs + self.config.arch_training_epochs):
            return "arch_training"
        elif position_in_cycle == (self.config.warmup_epochs + self.config.arch_training_epochs):
            return "mutation"
        else:
            return "weight_retraining"
    
    def _transition_to_phase(self, new_phase: str, epoch: int):
        """阶段转换处理"""
        logger.info(f"🔄 Phase transition: {self.current_phase} → {new_phase} at epoch {epoch}")
        
        # 保存阶段转换前的检查点
        if (self.config.save_checkpoints and self.checkpoint_dir and 
            new_phase == "mutation"):
            self._save_pre_mutation_checkpoint(epoch)
        
        self.training_history["phase_transitions"].append({
            "epoch": epoch,
            "from_phase": self.current_phase,
            "to_phase": new_phase
        })
        
        self.current_phase = new_phase
        
        # 阶段特定的处理
        if new_phase == "arch_training":
            self._prepare_arch_training()
        elif new_phase == "mutation":
            self._prepare_mutation()
        elif new_phase == "weight_retraining":
            self._prepare_weight_retraining()
    
    def _warmup_phase(self, train_loader, valid_loader, criterion, epoch: int) -> Dict[str, float]:
        """阶段一：网络权重预热"""
        logger.debug(f"🔥 Warmup phase - Epoch {epoch}")
        
        # 冻结架构参数
        if self.arch_optimizer:
            for param_group in self.arch_optimizer.param_groups:
                for param in param_group['params']:
                    param.requires_grad = False
        
        train_loss, train_acc = self._train_weights(train_loader, criterion)
        valid_loss, valid_acc = self._validate(valid_loader, criterion)
        
        # 恢复架构参数的梯度
        if self.arch_optimizer:
            for param_group in self.arch_optimizer.param_groups:
                for param in param_group['params']:
                    param.requires_grad = True
        
        return {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_acc,
            "phase": "warmup"
        }
    
    def _arch_training_phase(self, train_loader, valid_loader, criterion, epoch: int) -> Dict[str, float]:
        """阶段二：架构参数学习"""
        logger.debug(f"🔍 Architecture training phase - Epoch {epoch}")
        
        # 冻结权重参数
        for param_group in self.weight_optimizer.param_groups:
            for param in param_group['params']:
                param.requires_grad = False
        
        arch_loss, arch_acc = self._train_architecture(valid_loader, criterion)
        valid_loss, valid_acc = self._validate(valid_loader, criterion)
        
        # 恢复权重参数的梯度
        for param_group in self.weight_optimizer.param_groups:
            for param in param_group['params']:
                param.requires_grad = True
        
        return {
            "arch_loss": arch_loss,
            "arch_accuracy": arch_acc,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_acc,
            "phase": "arch_training"
        }
    
    def _mutation_phase(self, train_loader, valid_loader, criterion, epoch: int) -> Dict[str, float]:
        """阶段三：架构突变与稳定"""
        logger.info(f"🧬 Architecture mutation phase - Epoch {epoch}")
        
        # 从当前架构参数导出基因型
        if hasattr(self.search_model, 'genotype'):
            current_genotype = self.search_model.genotype()
        else:
            # 使用Gumbel-Softmax采样生成基因型
            current_genotype = self._sample_genotype_from_search_model()
        
        # 记录突变前的性能
        pre_mutation_loss, pre_mutation_acc = self._validate(valid_loader, criterion)
        
        # 使用Gumbel-Softmax采样新架构（而不是贪婪选择）
        new_genotype = self._gumbel_sample_architecture()
        
        # 如果配置允许，执行基因型突变
        if self.current_cycle % self.config.mutation_frequency == 0:
            new_genotype = self.mutator.mutate_genotype(new_genotype, "conservative")
            logger.info("🧬 Applied genotype mutation")
        
        # 创建新的可进化模型（使用设备管理器）
        old_evolvable_model = self.evolvable_model
        try:
            self.evolvable_model = self._create_evolvable_model(new_genotype)
            
            # 使用设备管理器进行模型切换
            if old_evolvable_model is not None:
                self.evolvable_model = self.device_manager.context_switch_model(
                    old_evolvable_model, self.evolvable_model
                )
            else:
                self.evolvable_model = self.device_manager.register_model(
                    "evolvable_model", self.evolvable_model
                )
            
            # 函数保持参数传递
            if self.config.preserve_function:
                self._transfer_parameters_with_function_preservation()
            
        except Exception as e:
            logger.error(f"❌ Failed to create evolvable model: {e}")
            # 保持旧模型
            self.evolvable_model = old_evolvable_model
            new_genotype = current_genotype
        
        self.current_genotype = new_genotype
        
        # 记录突变后的性能（应该与突变前基本一致）
        post_mutation_loss, post_mutation_acc = self._validate(
            valid_loader, criterion, use_evolvable=True
        )
        
        # 记录突变历史
        mutation_record = {
            "epoch": epoch,
            "cycle": self.current_cycle,
            "pre_mutation_acc": pre_mutation_acc,
            "post_mutation_acc": post_mutation_acc,
            "genotype": str(new_genotype),
            "performance_preservation": abs(pre_mutation_acc - post_mutation_acc) < 0.05
        }
        self.training_history["mutation_history"].append(mutation_record)
        
        return {
            "pre_mutation_loss": pre_mutation_loss,
            "pre_mutation_accuracy": pre_mutation_acc,
            "post_mutation_loss": post_mutation_loss,
            "post_mutation_accuracy": post_mutation_acc,
            "genotype": str(new_genotype),
            "phase": "mutation"
        }
    
    def _weight_retraining_phase(self, train_loader, valid_loader, criterion, epoch: int) -> Dict[str, float]:
        """阶段四：权重再适应"""
        logger.debug(f"🔧 Weight retraining phase - Epoch {epoch}")
        
        if self.evolvable_model is None:
            logger.warning("No evolvable model available, skipping weight retraining")
            return {"phase": "weight_retraining", "error": "no_evolvable_model"}
        
        # 使用可进化模型进行权重训练
        train_loss, train_acc = self._train_evolvable_weights(train_loader, criterion)
        valid_loss, valid_acc = self._validate(valid_loader, criterion, use_evolvable=True)
        
        return {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_acc,
            "phase": "weight_retraining"
        }
    
    def _gumbel_sample_architecture(self) -> Genotype:
        """使用Gumbel-Softmax采样架构"""
        if not hasattr(self.search_model, 'arch_parameters'):
            raise ValueError("Search model must have arch_parameters method")
        
        arch_params = self.search_model.arch_parameters()
        
        # 简化：使用第一个架构参数进行采样
        if len(arch_params) >= 2:
            normal_weights = arch_params[0]  # 正常边的权重
            reduce_weights = arch_params[1]  # 减少边的权重
            
            # Gumbel-Softmax采样
            normal_samples, _ = self.explorer.sample_architecture(normal_weights, hard=True)
            reduce_samples, _ = self.explorer.sample_architecture(reduce_weights, hard=True)
            
            # 转换为基因型格式
            normal_genotype = self._samples_to_genotype_edges(normal_samples)
            reduce_genotype = self._samples_to_genotype_edges(reduce_samples)
            
            return Genotype(
                normal=normal_genotype,
                normal_concat=list(range(2, 6)),  # 假设4个中间节点
                reduce=reduce_genotype,
                reduce_concat=list(range(2, 6))
            )
        else:
            logger.warning("Insufficient architecture parameters for Gumbel sampling")
            return self._default_genotype()
    
    def _samples_to_genotype_edges(self, samples: torch.Tensor) -> List[Tuple[str, int]]:
        """将采样结果转换为基因型边格式"""
        edges = []
        edge_idx = 0
        
        # 假设每个节点从前面所有节点选择两条边
        for node in range(4):  # 4个中间节点
            node_edges = []
            for _ in range(2):  # 每个节点2条边
                if edge_idx < len(samples):
                    op_idx = torch.argmax(samples[edge_idx]).item()
                    op_name = PRIMITIVES[op_idx] if op_idx < len(PRIMITIVES) else 'none'
                    predecessor = edge_idx % (node + 2)  # 连接到的前驱节点
                    node_edges.append((op_name, predecessor))
                    edge_idx += 1
            edges.extend(node_edges)
        
        return edges
    
    def _train_weights(self, train_loader, criterion) -> Tuple[float, float]:
        """训练权重参数"""
        self.search_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 数据已经通过设备管理器自动转移到正确设备
            
            self.weight_optimizer.zero_grad()
            output = self.search_model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.weight_optimizer.param_groups[0]['params'], 5.0)
            
            self.weight_optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def _train_architecture(self, valid_loader, criterion) -> Tuple[float, float]:
        """训练架构参数"""
        self.search_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(valid_loader):
            # 数据已经通过设备管理器自动转移到正确设备
            
            if self.arch_optimizer:
                self.arch_optimizer.zero_grad()
                output = self.search_model(data)
                loss = criterion(output, target)
                loss.backward()
                self.arch_optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(valid_loader) if len(valid_loader) > 0 else 0.0
        accuracy = 100. * correct / total if total > 0 else 0.0
        return avg_loss, accuracy
    
    def _validate(self, valid_loader, criterion, use_evolvable: bool = False) -> Tuple[float, float]:
        """验证性能"""
        model = self.evolvable_model if use_evolvable and self.evolvable_model else self.search_model
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in valid_loader:
                # 数据已经通过设备管理器自动转移到正确设备
                
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(valid_loader) if len(valid_loader) > 0 else 0.0
        accuracy = 100. * correct / total if total > 0 else 0.0
        return avg_loss, accuracy
    
    def _train_evolvable_weights(self, train_loader, criterion) -> Tuple[float, float]:
        """训练可进化模型的权重"""
        if self.evolvable_model is None:
            return 0.0, 0.0
        
        # 为可进化模型创建优化器
        evolvable_optimizer = optim.SGD(
            self.evolvable_model.parameters(),
            lr=self.config.weight_lr,
            momentum=self.config.weight_momentum,
            weight_decay=self.config.weight_decay
        )
        
        self.evolvable_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 数据已经通过设备管理器自动转移到正确设备
            
            evolvable_optimizer.zero_grad()
            output = self.evolvable_model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.evolvable_model.parameters(), 5.0)
            
            evolvable_optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def _create_evolvable_model(self, genotype: Genotype) -> nn.Module:
        """创建可进化模型"""
        # 这里需要根据具体的网络结构创建
        # 简化版本：返回一个包装类
        from .evolvable_model import EvolvableNetwork
        
        # 从搜索模型获取基本参数
        model_args = {
            'C': getattr(self.search_model, '_C', 16),
            'num_classes': getattr(self.search_model, '_num_classes', 10),
            'layers': getattr(self.search_model, '_layers', 8)
        }
        
        # 使用设备管理器安全创建模型
        return self.device_manager.safe_model_creation(
            EvolvableNetwork, **model_args, genotype=genotype
        )
    
    def _transfer_parameters_with_function_preservation(self):
        """使用函数保持的参数传递"""
        if self.evolvable_model is None:
            return
        
        try:
            # 使用设备管理器进行参数传递
            self.evolvable_model = self.device_manager.transfer_model_state(
                self.search_model, self.evolvable_model
            )
            logger.info("✅ Parameters transferred with function preservation")
        except Exception as e:
            logger.warning(f"⚠️ Parameter transfer failed: {e}")
    
    def _update_training_history(self, stats: Dict[str, Any], epoch: int, phase: str):
        """更新训练历史"""
        stats['epoch'] = epoch
        stats['phase'] = phase
        
        for key, value in stats.items():
            if key not in self.training_history:
                self.training_history[key] = []
            self.training_history[key].append(value)
    
    def _update_exploration_temperature(self, current_performance: float):
        """更新探索温度"""
        performance_gain = current_performance - self.best_performance
        self.explorer.update_temperature(performance_gain)
        
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.patience_counter = 0
        else:
            self.patience_counter += 1
    
    def _perform_gradual_growth(self, epoch: int):
        """执行渐进式生长"""
        if self.growth_manager:
            grown_model = self.growth_manager.perform_growth(self.search_model, epoch)
            if grown_model is not None:
                # 使用设备管理器进行模型更新
                self.search_model = self.device_manager.context_switch_model(
                    self.search_model, grown_model
                )
                self.device_manager.registered_models["search_model"] = self.search_model
                logger.info(f"🌱 Performed gradual growth at epoch {epoch}")
    
    def _default_genotype(self) -> Genotype:
        """默认基因型"""
        return Genotype(
            normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
            normal_concat=[2, 3, 4, 5],
            reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)],
            reduce_concat=[2, 3, 4, 5]
        )
    
    def _sample_genotype_from_search_model(self) -> Genotype:
        """从搜索模型采样基因型"""
        if hasattr(self.search_model, 'genotype'):
            return self.search_model.genotype()
        else:
            return self._default_genotype()
    
    def _prepare_arch_training(self):
        """准备架构训练阶段"""
        logger.debug("🔍 Preparing architecture training phase")
    
    def _prepare_mutation(self):
        """准备突变阶段"""
        logger.debug("🧬 Preparing mutation phase")
        self.current_cycle += 1
    
    def _prepare_weight_retraining(self):
        """准备权重再训练阶段"""
        logger.debug("🔧 Preparing weight retraining phase")
    
    def _log_memory_usage(self, context: str):
        """记录内存使用情况"""
        memory_stats = self.device_manager.get_memory_stats()
        logger.debug(f"💾 Memory usage ({context}): {memory_stats}")
        
        # 记录到历史
        self.training_history["device_stats"].append({
            "context": context,
            "stats": memory_stats
        })
    
    def _save_checkpoint(self, epoch: int, stats: Dict[str, float]):
        """保存训练检查点"""
        try:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"aso_se_checkpoint_epoch_{epoch}.pth"
            )
            
            checkpoint = {
                "epoch": epoch,
                "search_model_state": self.search_model.state_dict(),
                "evolvable_model_state": self.evolvable_model.state_dict() if self.evolvable_model else None,
                "weight_optimizer_state": self.weight_optimizer.state_dict() if self.weight_optimizer else None,
                "arch_optimizer_state": self.arch_optimizer.state_dict() if self.arch_optimizer else None,
                "current_genotype": self.current_genotype,
                "current_cycle": self.current_cycle,
                "current_phase": self.current_phase,
                "best_performance": self.best_performance,
                "explorer_state": {
                    "current_temp": self.explorer.current_temp,
                    "step_count": self.explorer.step_count
                },
                "training_history": self.training_history,
                "config": self.config,
                "stats": stats
            }
            
            torch.save(checkpoint, checkpoint_path)
            self.checkpoint_counter += 1
            
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # 清理旧检查点
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def _save_pre_mutation_checkpoint(self, epoch: int):
        """保存突变前的特殊检查点"""
        if not self.checkpoint_dir:
            return
            
        try:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"pre_mutation_checkpoint_epoch_{epoch}.pth"
            )
            
            checkpoint = {
                "epoch": epoch,
                "search_model_state": self.search_model.state_dict(),
                "weight_optimizer_state": self.weight_optimizer.state_dict() if self.weight_optimizer else None,
                "arch_optimizer_state": self.arch_optimizer.state_dict() if self.arch_optimizer else None,
                "current_genotype": self.current_genotype,
                "best_performance": self.best_performance,
                "pre_mutation": True
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"💾 Pre-mutation checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save pre-mutation checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点文件"""
        if not self.checkpoint_dir or self.config.max_checkpoints <= 0:
            return
        
        try:
            # 获取所有检查点文件
            checkpoint_files = []
            for filename in os.listdir(self.checkpoint_dir):
                if filename.startswith("aso_se_checkpoint_epoch_") and filename.endswith(".pth"):
                    filepath = os.path.join(self.checkpoint_dir, filename)
                    checkpoint_files.append((filepath, os.path.getctime(filepath)))
            
            # 按创建时间排序
            checkpoint_files.sort(key=lambda x: x[1], reverse=True)
            
            # 删除超出限制的文件
            for filepath, _ in checkpoint_files[self.config.max_checkpoints:]:
                os.remove(filepath)
                logger.debug(f"🗑️ Removed old checkpoint: {filepath}")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup checkpoints: {e}")
    
    def should_early_stop(self) -> bool:
        """检查是否应该早停"""
        return self.patience_counter >= self.config.early_stopping_patience
    
    def get_training_report(self) -> Dict:
        """获取训练报告"""
        return {
            "current_cycle": self.current_cycle,
            "current_phase": self.current_phase,
            "best_performance": self.best_performance,
            "total_mutations": len(self.training_history["mutation_history"]),
            "exploration_report": self.explorer.get_exploration_report(),
            "mutation_report": self.mutator.get_mutation_report(),
            "device_report": self.device_manager.get_device_report(),
            "training_history": self.training_history
        }
    
    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        checkpoint = {
            "search_model_state": self.search_model.state_dict(),
            "evolvable_model_state": self.evolvable_model.state_dict() if self.evolvable_model else None,
            "current_genotype": self.current_genotype,
            "current_cycle": self.current_cycle,
            "current_phase": self.current_phase,
            "training_history": self.training_history,
            "config": self.config
        }
        torch.save(checkpoint, filepath)
        logger.info(f"💾 Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device_manager.device)
        
        self.search_model.load_state_dict(checkpoint["search_model_state"])
        
        if checkpoint["evolvable_model_state"] and self.evolvable_model:
            self.evolvable_model.load_state_dict(checkpoint["evolvable_model_state"])
        
        self.current_genotype = checkpoint["current_genotype"]
        self.current_cycle = checkpoint["current_cycle"]
        self.current_phase = checkpoint["current_phase"]
        self.training_history = checkpoint["training_history"]
        
        logger.info(f"📂 Checkpoint loaded from {filepath}")

def test_aso_se_framework():
    """测试ASO-SE框架功能"""
    print("🧪 Testing ASO-SE Framework...")
    
    # 创建配置
    config = ASOSEConfig(
        warmup_epochs=2,
        arch_training_epochs=1,
        weight_training_epochs=2,
        total_cycles=2,
        save_checkpoints=False  # 测试时不保存检查点
    )
    
    # 创建简单的搜索模型
    class SimpleSearchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc = nn.Linear(32 * 8 * 8, 10)
            
            # 模拟架构参数
            self.alphas_normal = nn.Parameter(torch.randn(4, 8))
            self.alphas_reduce = nn.Parameter(torch.randn(4, 8))
            
            self._C = 16
            self._num_classes = 10
            self._layers = 8
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        
        def arch_parameters(self):
            return [self.alphas_normal, self.alphas_reduce]
        
        def genotype(self):
            from .genotypes import Genotype
            return Genotype(
                normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
                normal_concat=[2],
                reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
                reduce_concat=[2]
            )
    
    model = SimpleSearchModel()
    framework = ASOSEFramework(model, config)
    framework.initialize_optimizers()
    
    print(f"✅ Framework initialized with {framework.config.total_cycles} cycles")
    print(f"✅ Device: {framework.device_manager.device}")
    
    # 测试阶段确定
    phase = framework._determine_current_phase(0)
    print(f"✅ Phase determination: epoch 0 -> {phase}")
    
    phase = framework._determine_current_phase(3)
    print(f"✅ Phase determination: epoch 3 -> {phase}")
    
    # 测试内存管理
    memory_stats = framework.device_manager.get_memory_stats()
    print(f"✅ Memory stats: {memory_stats}")
    
    # 测试报告生成
    report = framework.get_training_report()
    print(f"✅ Training report generated: {report['current_cycle']} cycles completed")
    
    print("🎉 ASO-SE Framework tests passed!")

if __name__ == "__main__":
    test_aso_se_framework() 