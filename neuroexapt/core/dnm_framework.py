#!/usr/bin/env python3
"""
Dynamic Neural Morphogenesis (DNM) 框架主控制器

整合三大创新模块：
1. 信息熵驱动的神经元分裂 (DNMNeuronDivision)
2. 梯度引导的连接生长 (DNMConnectionGrowth)  
3. 多目标进化优化 (DNMMultiObjectiveOptimization)

提供统一的训练接口，实现真正的神经网络自适应生长
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import copy
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict, deque
import logging

# 导入DNM核心模块
from .dnm_neuron_division import DNMNeuronDivision
from .dnm_connection_growth import DNMConnectionGrowth
from ..math.pareto_optimization import DNMMultiObjectiveOptimization

logger = logging.getLogger(__name__)


class DNMFramework:
    """
    Dynamic Neural Morphogenesis 主框架
    
    集成所有DNM组件，提供完整的自适应神经网络生长功能
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化DNM核心组件
        self.neuron_division = DNMNeuronDivision(self.config.get('neuron_division'))
        self.connection_growth = DNMConnectionGrowth(self.config.get('connection_growth'))
        self.multi_objective = DNMMultiObjectiveOptimization(self.config.get('multi_objective'))
        
        # 训练状态
        self.training_active = False
        self.current_epoch = 0
        self.model_population = []
        
        # 性能和演化追踪
        self.performance_history = []
        self.morphogenesis_events = []
        self.architecture_snapshots = []
        
        # 统计信息
        self.statistics = {
            'total_neuron_splits': 0,
            'total_connections_grown': 0,
            'total_optimizations': 0,
            'performance_improvements': 0,
            'architecture_complexity_growth': 0.0
        }
        
        logger.info(f"🧬 DNM Framework initialized on {self.device}")
        logger.info(f"   Configuration: {self.config}")
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'neuron_division': {
                'splitter': {
                    'entropy_threshold': 0.7,
                    'overload_threshold': 0.6,
                    'split_probability': 0.4,
                    'max_splits_per_layer': 3,
                    'inheritance_noise': 0.1
                },
                'monitoring': {
                    'target_layers': ['conv', 'linear'],
                    'analysis_frequency': 5,
                    'min_epoch_before_split': 10
                }
            },
            'connection_growth': {
                'analyzer': {
                    'correlation_threshold': 0.15,
                    'history_length': 8
                },
                'growth': {
                    'max_new_connections': 3,
                    'min_correlation_threshold': 0.1,
                    'growth_frequency': 8,
                    'connection_types': ['skip_connection', 'attention_connection']
                },
                'filtering': {
                    'min_layer_distance': 2,
                    'max_layer_distance': 6,
                    'avoid_redundant_connections': True
                }
            },
            'multi_objective': {
                'evolution': {
                    'population_size': 12,
                    'max_generations': 15,
                    'mutation_rate': 0.3,
                    'crossover_rate': 0.7,
                    'elitism_ratio': 0.15
                },
                'optimization': {
                    'trigger_frequency': 20,
                    'performance_plateau_threshold': 0.01,
                    'min_improvement_epochs': 5
                }
            },
            'framework': {
                'morphogenesis_frequency': 5,  # 每5个epoch检查一次形态发生
                'performance_tracking_window': 10,  # 性能追踪窗口
                'early_stopping_patience': 15,
                'target_accuracy_threshold': 95.0,
                'enable_architecture_snapshots': True,
                'adaptive_morphogenesis': True  # 自适应形态发生
            }
        }
    
    def train_with_morphogenesis(self, 
                                model: nn.Module,
                                train_loader: DataLoader,
                                val_loader: DataLoader,
                                epochs: int,
                                optimizer: Optional[torch.optim.Optimizer] = None,
                                criterion: Optional[nn.Module] = None,
                                callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """
        使用DNM进行训练
        
        Args:
            model: 初始模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            optimizer: 优化器（可选）
            criterion: 损失函数（可选）
            callbacks: 回调函数列表（可选）
            
        Returns:
            训练结果字典
        """
        logger.info(f"🚀 Starting DNM training for {epochs} epochs")
        logger.info(f"   Initial model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 初始化
        self.training_active = True
        self.current_epoch = 0
        model = model.to(self.device)
        
        # 默认优化器和损失函数
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        # 注册神经元分裂监控
        self.neuron_division.register_model_hooks(model)
        
        # 初始化模型种群（用于多目标优化）
        self.model_population = [copy.deepcopy(model) for _ in range(3)]
        
        # 记录初始架构快照
        if self.config['framework']['enable_architecture_snapshots']:
            self._take_architecture_snapshot(model, epoch=0)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        try:
            for epoch in range(epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                logger.info(f"\n🧬 Epoch {epoch+1}/{epochs} - Dynamic Morphogenesis")
                
                # 1. 标准训练步骤
                train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
                
                # 2. 收集梯度用于连接生长分析
                self.connection_growth.collect_and_analyze_gradients(model)
                
                # 3. 记录性能
                epoch_record = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'model_params': sum(p.numel() for p in model.parameters()),
                    'epoch_time': time.time() - epoch_start_time
                }
                self.performance_history.append(epoch_record)
                
                # 4. 形态发生检查和执行
                morphogenesis_triggered = False
                
                if self._should_trigger_morphogenesis(epoch, val_acc):
                    logger.info("  🔄 Triggering morphogenesis analysis...")
                    
                    # 神经元分裂
                    neuron_result = self.neuron_division.analyze_and_split(model, epoch)
                    
                    # 连接生长
                    connection_result = self.connection_growth.analyze_and_grow_connections(model, epoch)
                    
                    # 多目标优化
                    optimization_result = self.multi_objective.optimize_architecture_population(
                        self.model_population, train_loader, val_loader, epoch
                    )
                    
                    # 记录形态发生事件
                    if (neuron_result.get('splits_executed', 0) > 0 or 
                        connection_result.get('connections_grown', 0) > 0 or
                        optimization_result.get('optimized', False)):
                        
                        morphogenesis_event = {
                            'epoch': epoch,
                            'neuron_splits': neuron_result.get('splits_executed', 0),
                            'connections_grown': connection_result.get('connections_grown', 0),
                            'optimization_triggered': optimization_result.get('optimized', False),
                            'performance_before': val_acc,
                            'details': {
                                'neuron_division': neuron_result,
                                'connection_growth': connection_result,
                                'multi_objective': optimization_result
                            }
                        }
                        self.morphogenesis_events.append(morphogenesis_event)
                        morphogenesis_triggered = True
                        
                        # 更新统计
                        self.statistics['total_neuron_splits'] += neuron_result.get('splits_executed', 0)
                        self.statistics['total_connections_grown'] += connection_result.get('connections_grown', 0)
                        if optimization_result.get('optimized', False):
                            self.statistics['total_optimizations'] += 1
                        
                        # 更新优化器以包含新参数
                        self._update_optimizer(optimizer, model)
                        
                        # 更新模型种群
                        if optimization_result.get('optimized', False) and optimization_result.get('best_models'):
                            self.model_population = optimization_result['best_models'][:3]
                            # 选择最好的模型继续训练
                            model = self._select_best_model_for_training(optimization_result['best_models'], 
                                                                        optimization_result['best_fitness'])
                            self._update_optimizer(optimizer, model)
                
                # 5. 输出状态
                current_params = sum(p.numel() for p in model.parameters())
                if epoch == 0:
                    initial_params = current_params
                else:
                    param_growth = (current_params - initial_params) / initial_params * 100
                
                logger.info(f"  📊 Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
                           f"Params: {current_params:,} | "
                           f"{'🧬 Morphogenesis' if morphogenesis_triggered else ''}")
                
                # 6. 早期停止和性能追踪
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    self.statistics['performance_improvements'] += 1
                else:
                    patience_counter += 1
                
                if val_acc >= self.config['framework']['target_accuracy_threshold']:
                    logger.info(f"  🎯 Reached target accuracy: {val_acc:.2f}%!")
                    break
                
                if patience_counter >= self.config['framework']['early_stopping_patience']:
                    logger.info(f"  🛑 Early stopping triggered (patience: {patience_counter})")
                    break
                
                # 7. 执行回调
                if callbacks:
                    for callback in callbacks:
                        callback(self, model, epoch_record)
                
                # 8. 架构快照
                if (self.config['framework']['enable_architecture_snapshots'] and 
                    epoch % 10 == 0 and morphogenesis_triggered):
                    self._take_architecture_snapshot(model, epoch)
        
        except KeyboardInterrupt:
            logger.info("  ⏹️ Training interrupted by user")
        
        finally:
            # 清理
            self.training_active = False
            self.neuron_division.remove_model_hooks(model)
        
        # 生成训练总结
        training_summary = self._generate_training_summary(model, best_val_acc)
        
        logger.info("✅ DNM training completed")
        logger.info(f"   Final accuracy: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")
        logger.info(f"   Morphogenesis events: {len(self.morphogenesis_events)}")
        
        return {
            'model': model,
            'best_val_accuracy': best_val_acc,
            'final_val_accuracy': val_acc,
            'performance_history': self.performance_history,
            'morphogenesis_events': self.morphogenesis_events,
            'architecture_snapshots': self.architecture_snapshots,
            'statistics': self.statistics,
            'training_summary': training_summary
        }
    
    def _should_trigger_morphogenesis(self, epoch: int, current_performance: float) -> bool:
        """判断是否应该触发形态发生"""
        
        # 基础频率检查
        if epoch % self.config['framework']['morphogenesis_frequency'] != 0:
            return False
        
        # 最小epoch检查
        if epoch < 10:
            return False
        
        # 自适应形态发生
        if self.config['framework']['adaptive_morphogenesis']:
            # 检查最近的性能平台期
            recent_history = self.performance_history[-self.config['framework']['performance_tracking_window']:]
            if len(recent_history) >= 5:
                recent_accs = [h['val_acc'] for h in recent_history]
                performance_std = np.std(recent_accs)
                
                # 如果性能变化很小，更可能触发形态发生
                if performance_std < self.config['multi_objective']['optimization']['performance_plateau_threshold']:
                    return True
        
        return True
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, float]:
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def _update_optimizer(self, optimizer: torch.optim.Optimizer, model: nn.Module) -> None:
        """更新优化器参数组以包含新参数"""
        # 简单重建优化器
        state_dict = optimizer.state_dict()
        lr = optimizer.param_groups[0]['lr']
        
        # 创建新优化器
        new_optimizer = type(optimizer)(model.parameters(), lr=lr)
        
        # 尝试恢复状态（对于新参数，状态将为空）
        try:
            new_optimizer.load_state_dict(state_dict)
        except Exception as e:
            logger.warning(f"Could not restore optimizer state: {e}")
        
        # 更新原优化器的引用
        optimizer.param_groups = new_optimizer.param_groups
        optimizer.state = new_optimizer.state
    
    def _select_best_model_for_training(self, models: List[nn.Module], fitness_list: List) -> nn.Module:
        """从候选模型中选择最佳的用于继续训练"""
        if not models or not fitness_list:
            return models[0] if models else None
        
        # 选择准确率最高的模型
        best_idx = 0
        best_accuracy = fitness_list[0].accuracy
        
        for i, fitness in enumerate(fitness_list):
            if fitness.accuracy > best_accuracy:
                best_accuracy = fitness.accuracy
                best_idx = i
        
        logger.info(f"  🎯 Selected model {best_idx} with accuracy {best_accuracy:.2f}% for continued training")
        return copy.deepcopy(models[best_idx])
    
    def _take_architecture_snapshot(self, model: nn.Module, epoch: int) -> None:
        """拍摄架构快照"""
        snapshot = {
            'epoch': epoch,
            'model_structure': str(model),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'layer_info': []
        }
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_info = {
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                }
                
                if isinstance(module, nn.Conv2d):
                    layer_info.update({
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size
                    })
                elif isinstance(module, nn.Linear):
                    layer_info.update({
                        'in_features': module.in_features,
                        'out_features': module.out_features
                    })
                
                snapshot['layer_info'].append(layer_info)
        
        self.architecture_snapshots.append(snapshot)
        logger.debug(f"Architecture snapshot taken at epoch {epoch}")
    
    def _generate_training_summary(self, model: nn.Module, best_val_acc: float) -> Dict[str, Any]:
        """生成训练总结"""
        if not self.performance_history:
            return {}
        
        initial_params = self.performance_history[0]['model_params']
        final_params = self.performance_history[-1]['model_params']
        param_growth = (final_params - initial_params) / initial_params * 100
        
        summary = {
            'training_epochs': len(self.performance_history),
            'best_validation_accuracy': best_val_acc,
            'final_validation_accuracy': self.performance_history[-1]['val_acc'],
            'accuracy_improvement': self.performance_history[-1]['val_acc'] - self.performance_history[0]['val_acc'],
            'parameter_growth': param_growth,
            'initial_parameters': initial_params,
            'final_parameters': final_params,
            'morphogenesis_events': len(self.morphogenesis_events),
            'total_neuron_splits': self.statistics['total_neuron_splits'],
            'total_connections_grown': self.statistics['total_connections_grown'],
            'total_optimizations': self.statistics['total_optimizations'],
            'avg_epoch_time': np.mean([h['epoch_time'] for h in self.performance_history]),
            'total_training_time': sum(h['epoch_time'] for h in self.performance_history)
        }
        
        return summary
    
    def get_morphogenesis_summary(self) -> Dict[str, Any]:
        """获取形态发生总结"""
        return {
            'framework_statistics': self.statistics,
            'neuron_division_summary': self.neuron_division.get_split_summary(),
            'connection_growth_summary': self.connection_growth.get_growth_summary(),
            'multi_objective_summary': self.multi_objective.get_optimization_summary(),
            'morphogenesis_events': self.morphogenesis_events,
            'architecture_snapshots': self.architecture_snapshots[-5:] if self.architecture_snapshots else []
        }
    
    def export_evolved_model(self, filepath: str, model: nn.Module) -> None:
        """导出演化后的模型"""
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_structure': str(model),
            'morphogenesis_summary': self.get_morphogenesis_summary(),
            'performance_history': self.performance_history,
            'config': self.config
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"Evolved model exported to {filepath}")


# 便捷函数
def train_with_dnm(model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   epochs: int = 100,
                   config: Optional[Dict] = None,
                   **kwargs) -> Dict[str, Any]:
    """
    便捷的DNM训练函数
    
    Args:
        model: 模型
        train_loader: 训练数据
        val_loader: 验证数据
        epochs: 训练轮数
        config: DNM配置
        **kwargs: 其他参数
        
    Returns:
        训练结果
    """
    dnm = DNMFramework(config)
    return dnm.train_with_morphogenesis(model, train_loader, val_loader, epochs, **kwargs)


# 测试函数
def test_dnm_framework():
    """测试DNM框架"""
    print("🧬 Testing Complete DNM Framework")
    
    # 创建测试模型
    class TestEvolvableCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 16, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    # 创建虚拟数据集
    from torch.utils.data import TensorDataset
    
    train_data = torch.randn(500, 3, 32, 32)
    train_labels = torch.randint(0, 10, (500,))
    val_data = torch.randn(100, 3, 32, 32)
    val_labels = torch.randint(0, 10, (100,))
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = TestEvolvableCNN()
    
    # 使用DNM训练
    result = train_with_dnm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30
    )
    
    print(f"Training completed: {result['training_summary']}")
    print(f"Morphogenesis events: {len(result['morphogenesis_events'])}")
    
    return result


if __name__ == "__main__":
    test_dnm_framework()