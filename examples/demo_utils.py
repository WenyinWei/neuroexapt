"""
演示工具模块
Demo Utilities Module

提供统一的演示辅助功能，减少代码重复，提高可读性：
1. 设备配置和环境设置
2. 数据加载和预处理
3. 训练器创建和配置
4. 日志系统配置
5. 结果展示和格式化
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
import sys
import os
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.models import create_enhanced_model


@dataclass
class DemoConfiguration:
    """演示配置"""
    # 设备配置
    device_type: str = 'auto'
    seed: int = 42
    
    # 数据配置
    data_root: str = './data'
    batch_size: int = 128
    num_workers: int = 4
    enhanced_augmentation: bool = True
    
    # 模型配置
    model_type: str = 'enhanced_resnet34'
    num_classes: int = 10
    
    # 训练配置
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    
    # 日志配置
    log_level: str = 'INFO'
    verbose: bool = True


class DemoLogger:
    """统一的演示日志系统"""
    
    def __init__(self, name: str = 'demo', level: str = 'INFO', verbose: bool = True):
        """
        初始化日志系统
        
        Args:
            name: 日志器名称
            level: 日志级别
            verbose: 是否详细输出
        """
        self.logger = logging.getLogger(name)
        self.verbose = verbose
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 设置日志级别
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        self.logger.setLevel(level_map.get(level.upper(), logging.INFO))
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.logger.level)
        
        # 设置格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.propagate = False
    
    def info(self, message: str):
        """信息日志"""
        if self.verbose:
            print(f"📋 {message}")
        self.logger.info(message)
    
    def success(self, message: str):
        """成功日志"""
        if self.verbose:
            print(f"✅ {message}")
        self.logger.info(f"SUCCESS: {message}")
    
    def warning(self, message: str):
        """警告日志"""
        if self.verbose:
            print(f"⚠️ {message}")
        self.logger.warning(message)
    
    def error(self, message: str):
        """错误日志"""
        if self.verbose:
            print(f"❌ {message}")
        self.logger.error(message)
    
    def progress(self, message: str):
        """进度日志"""
        if self.verbose:
            print(f"🔄 {message}")
        self.logger.info(f"PROGRESS: {message}")


class DeviceManager:
    """设备管理器"""
    
    @staticmethod
    def setup_device(device_type: str = 'auto') -> torch.device:
        """
        设置计算设备
        
        Args:
            device_type: 设备类型 ('auto', 'cuda', 'cpu')
            
        Returns:
            torch.device: 配置好的设备
        """
        if device_type == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_type)
        
        return device
    
    @staticmethod
    def setup_environment(seed: int = 42, device: torch.device = None) -> torch.device:
        """
        设置环境和随机种子
        
        Args:
            seed: 随机种子
            device: 计算设备
            
        Returns:
            torch.device: 配置好的设备
        """
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if device is None:
            device = DeviceManager.setup_device()
        
        # CUDA相关设置
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        return device
    
    @staticmethod
    def get_device_info(device: torch.device) -> Dict[str, Any]:
        """
        获取设备信息
        
        Args:
            device: 设备对象
            
        Returns:
            Dict: 设备信息
        """
        info = {
            'device_type': device.type,
            'device_name': str(device)
        }
        
        if device.type == 'cuda':
            info.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9,
                'gpu_memory_cached': torch.cuda.memory_reserved() / 1e9
            })
        
        return info


class CIFAR10DataManager:
    """CIFAR-10数据管理器"""
    
    def __init__(self, config: DemoConfiguration):
        """
        初始化数据管理器
        
        Args:
            config: 演示配置
        """
        self.config = config
        
    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        获取数据变换
        
        Returns:
            Tuple: (训练变换, 测试变换)
        """
        if self.config.enhanced_augmentation:
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
        
        return train_transform, test_transform
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        创建数据加载器
        
        Returns:
            Tuple: (训练加载器, 测试加载器)
        """
        train_transform, test_transform = self.get_transforms()
        
        # 加载数据集
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.config.data_root, train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.config.data_root, train=False, download=True, transform=test_transform
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=self.config.num_workers, 
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=self.config.num_workers, 
            pin_memory=True
        )
        
        return train_loader, test_loader


class ModelManager:
    """模型管理器"""
    
    @staticmethod
    def create_model(config: DemoConfiguration) -> nn.Module:
        """
        创建模型
        
        Args:
            config: 演示配置
            
        Returns:
            nn.Module: 创建的模型
        """
        try:
            model = create_enhanced_model(
                model_type=config.model_type,
                num_classes=config.num_classes
            )
        except Exception:
            # 如果增强模型创建失败，使用基础模型
            from torchvision.models import resnet18, resnet34
            
            if 'resnet34' in config.model_type.lower():
                model = resnet34(num_classes=config.num_classes)
            else:
                model = resnet18(num_classes=config.num_classes)
            
            # CIFAR-10适配
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        
        return model
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model: 模型对象
            
        Returns:
            Dict: 模型信息
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 计算模型大小（MB）
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        return {
            'model_name': type(model).__name__,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
            'param_size_mb': param_size / (1024 ** 2),
            'buffer_size_mb': buffer_size / (1024 ** 2)
        }


class AdvancedTrainer:
    """高级训练器"""
    
    def __init__(self, model: nn.Module, device: torch.device, config: DemoConfiguration, 
                 logger: DemoLogger = None):
        """
        初始化训练器
        
        Args:
            model: 模型
            device: 设备
            config: 配置
            logger: 日志器
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.logger = logger or DemoLogger()
        self.criterion = nn.CrossEntropyLoss()
        
    def create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        return optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
    
    def create_scheduler(self, optimizer: optim.Optimizer, epochs: int) -> optim.lr_scheduler._LRScheduler:
        """创建学习率调度器"""
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    def train_model(self, train_loader: DataLoader, test_loader: DataLoader, 
                   epochs: int = 15) -> float:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            epochs: 训练轮数
            
        Returns:
            float: 最佳准确率
        """
        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(optimizer, epochs)
        
        best_accuracy = 0.0
        start_time = time.time()
        
        self.logger.progress(f"开始训练 ({epochs} epochs)")
        
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
                elapsed_time = time.time() - start_time
                self.logger.info(f"Epoch {epoch+1}/{epochs}: "
                               f"Train={train_accuracy:.2f}%, Test={test_accuracy:.2f}%, "
                               f"Best={best_accuracy:.2f}%, Time={elapsed_time:.1f}s")
        
        total_time = time.time() - start_time
        self.logger.success(f"训练完成! 最佳准确率: {best_accuracy:.2f}%, 用时: {total_time:.1f}s")
        
        return best_accuracy
    
    def evaluate_model(self, test_loader: DataLoader) -> float:
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            float: 准确率
        """
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


class ResultFormatter:
    """结果格式化器"""
    
    @staticmethod
    def format_device_info(device_info: Dict[str, Any]) -> str:
        """格式化设备信息"""
        lines = [f"设备类型: {device_info['device_type'].upper()}"]
        
        if device_info['device_type'] == 'cuda':
            lines.extend([
                f"GPU名称: {device_info['gpu_name']}",
                f"GPU内存: {device_info['gpu_memory_total']:.1f} GB",
                f"已分配: {device_info['gpu_memory_allocated']:.2f} GB",
                f"已缓存: {device_info['gpu_memory_cached']:.2f} GB"
            ])
        
        return "\n".join(lines)
    
    @staticmethod
    def format_model_info(model_info: Dict[str, Any]) -> str:
        """格式化模型信息"""
        return (f"模型: {model_info['model_name']}\n"
                f"参数量: {model_info['total_params']:,}\n"
                f"可训练参数: {model_info['trainable_params']:,}\n"
                f"模型大小: {model_info['model_size_mb']:.2f} MB")
    
    @staticmethod
    def format_evolution_summary(summary: Dict[str, Any]) -> str:
        """格式化进化摘要"""
        lines = [
            f"进化轮数: {summary['rounds_completed']}",
            f"初始准确率: {summary['initial_accuracy']:.2f}%",
            f"最终准确率: {summary['final_accuracy']:.2f}%",
            f"总体改进: {summary['total_improvement']:.2f}%",
            f"成功变异: {summary['successful_mutations']}",
            f"失败变异: {summary['failed_mutations']}",
            f"目标达成: {'✅' if summary['target_reached'] else '❌'}"
        ]
        
        if 'total_parameter_increase' in summary:
            lines.extend([
                f"参数增长: {summary['total_parameter_increase']:.3f}",
                f"计算增长: {summary['total_computation_increase']:.3f}"
            ])
        
        return "\n".join(lines)


# 导出主要类和函数
__all__ = [
    'DemoConfiguration',
    'DemoLogger', 
    'DeviceManager',
    'CIFAR10DataManager',
    'ModelManager',
    'AdvancedTrainer',
    'ResultFormatter'
]