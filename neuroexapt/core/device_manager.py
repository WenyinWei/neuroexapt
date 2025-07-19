"""
@defgroup group_device_manager Device Manager
@ingroup core
Device Manager module for NeuroExapt framework.


设备管理器 (Device Manager)

统一管理ASO-SE框架中所有模型和数据的设备分配，确保：
1. 所有模型在同一设备上
2. 数据自动转移到正确设备
3. 内存高效利用
4. 多GPU支持（未来扩展）
"""

import torch
import torch.nn as nn
import logging
import gc
from typing import Dict, List, Optional, Union, Any
import warnings

logger = logging.getLogger(__name__)

class DeviceManager:
    """
    设备管理器
    
    统一管理设备分配和内存优化
    """
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None, 
                 memory_fraction: float = 0.9):
        """
        Args:
            device: 目标设备，None表示自动选择
            memory_fraction: GPU内存使用比例
        """
        # 设备选择
        if device is None:
            if torch.cuda.is_available():
                # 选择显存最多的GPU
                gpu_count = torch.cuda.device_count()
                if gpu_count > 1:
                    best_gpu = self._select_best_gpu()
                    self.device = torch.device(f'cuda:{best_gpu}')
                else:
                    self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
                warnings.warn("CUDA not available, using CPU. Performance will be significantly slower.")
        else:
            self.device = torch.device(device)
        
        self.memory_fraction = memory_fraction
        
        # 设备信息
        self._log_device_info()
        
        # 内存监控
        self.peak_memory = 0.0
        self.memory_history = []
        
        # 模型注册表
        self.registered_models = {}
        
        logger.info(f"🔧 Device Manager initialized on {self.device}")
    
    def _select_best_gpu(self) -> int:
        """选择最优GPU"""
        best_gpu = 0
        max_memory = 0
        
        for i in range(torch.cuda.device_count()):
            memory = torch.cuda.get_device_properties(i).total_memory
            if memory > max_memory:
                max_memory = memory
                best_gpu = i
        
        logger.info(f"🎯 Selected GPU {best_gpu} with {max_memory / 1024**3:.1f}GB memory")
        return best_gpu
    
    def _log_device_info(self):
        """记录设备信息"""
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            total_memory = props.total_memory / 1024**3
            logger.info(f"🔧 Using GPU: {props.name}")
            logger.info(f"   Total Memory: {total_memory:.1f}GB")
            logger.info(f"   Compute Capability: {props.major}.{props.minor}")
            
            # 设置内存分配策略
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction, self.device.index)
        else:
            logger.info("🔧 Using CPU")
    
    def register_model(self, name: str, model: nn.Module) -> nn.Module:
        """
        注册并转移模型到目标设备
        
        Args:
            name: 模型名称
            model: 模型实例
            
        Returns:
            转移到目标设备的模型
        """
        logger.info(f"📋 Registering model '{name}' to {self.device}")
        
        # 转移到目标设备
        model = model.to(self.device)
        
        # 注册模型
        self.registered_models[name] = model
        
        # 记录模型参数量
        param_count = sum(p.numel() for p in model.parameters())
        param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        
        logger.info(f"   Parameters: {param_count:,} ({param_size_mb:.1f}MB)")
        
        return model
    
    def to_device(self, data: Any) -> Any:
        """
        将数据转移到目标设备
        
        Args:
            data: 要转移的数据
            
        Returns:
            转移到目标设备的数据
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_device(item) for item in data)
        elif isinstance(data, dict):
            return {key: self.to_device(value) for key, value in data.items()}
        else:
            return data
    
    def create_tensor(self, *args, **kwargs) -> torch.Tensor:
        """在正确设备上创建张量"""
        if 'device' not in kwargs:
            kwargs['device'] = self.device
        return torch.tensor(*args, **kwargs)
    
    def zeros_like(self, tensor: torch.Tensor) -> torch.Tensor:
        """创建同设备的零张量"""
        return torch.zeros_like(tensor, device=self.device)
    
    def ones_like(self, tensor: torch.Tensor) -> torch.Tensor:
        """创建同设备的单位张量"""
        return torch.ones_like(tensor, device=self.device)
    
    def transfer_model_state(self, source_model: nn.Module, 
                           target_model: nn.Module) -> nn.Module:
        """
        在同一设备上转移模型状态
        
        Args:
            source_model: 源模型
            target_model: 目标模型
            
        Returns:
            加载了状态的目标模型
        """
        # 确保两个模型在同一设备上
        source_model = source_model.to(self.device)
        target_model = target_model.to(self.device)
        
        source_dict = source_model.state_dict()
        target_dict = target_model.state_dict()
        
        # 逐层转移兼容的参数
        transferred = 0
        for key in target_dict.keys():
            if key in source_dict:
                source_param = source_dict[key]
                target_param = target_dict[key]
                
                if source_param.shape == target_param.shape:
                    target_dict[key] = source_param.clone()
                    transferred += 1
                else:
                    logger.warning(f"Shape mismatch for {key}: "
                                 f"{source_param.shape} -> {target_param.shape}")
        
        target_model.load_state_dict(target_dict)
        logger.info(f"✅ Transferred {transferred} parameters between models")
        
        return target_model
    
    def optimize_memory(self):
        """优化内存使用"""
        if self.device.type == 'cuda':
            # 清空GPU缓存
            torch.cuda.empty_cache()
            
            # 触发垃圾回收
            gc.collect()
            
            # 记录内存使用
            current_memory = torch.cuda.memory_allocated(self.device) / 1024**2
            self.memory_history.append(current_memory)
            
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            
            logger.debug(f"💾 Memory: {current_memory:.1f}MB (Peak: {self.peak_memory:.1f}MB)")
    
    def get_memory_stats(self) -> Dict[str, Union[float, str]]:
        """获取内存统计信息"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            cached = torch.cuda.memory_reserved(self.device) / 1024**2
            total = torch.cuda.get_device_properties(self.device).total_memory / 1024**2
            
            return {
                'allocated_mb': allocated,
                'cached_mb': cached,
                'total_mb': total,
                'peak_mb': self.peak_memory,
                'utilization': allocated / total
            }
        else:
            return {'device': 'cpu'}
    
    def create_data_loader_wrapper(self, data_loader):
        """创建自动转移数据的DataLoader包装器"""
        class DeviceDataLoader:
            def __init__(self, loader, device_manager):
                self.loader = loader
                self.device_manager = device_manager
            
            def __iter__(self):
                for batch in self.loader:
                    yield self.device_manager.to_device(batch)
            
            def __len__(self):
                return len(self.loader)
            
            def __getattr__(self, name):
                return getattr(self.loader, name)
        
        return DeviceDataLoader(data_loader, self)
    
    def safe_model_creation(self, model_class, *args, **kwargs) -> nn.Module:
        """安全的模型创建，自动处理设备和内存"""
        try:
            # 直接在目标设备上创建模型
            model = model_class(*args, **kwargs)
            
            # 确保在正确设备上
            model = model.to(self.device)
            
            # 优化内存
            self.optimize_memory()
            
            return model
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("🚨 GPU memory insufficient for model creation")
                self.optimize_memory()
                
                # 尝试在CPU上创建
                logger.warning("🔄 Falling back to CPU")
                original_device = self.device
                self.device = torch.device('cpu')
                try:
                    model = model_class(*args, **kwargs)
                    return model
                except Exception as e2:
                    # 恢复原设备设置
                    self.device = original_device
                    raise e2
            else:
                raise e
    
    def checkpoint_memory_state(self) -> Dict[str, Any]:
        """检查点内存状态"""
        if self.device.type == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated(self.device),
                'cached': torch.cuda.memory_reserved(self.device),
                'peak': self.peak_memory
            }
        return {}
    
    def restore_memory_state(self, state: Dict[str, Any]):
        """恢复内存状态"""
        if self.device.type == 'cuda' and state:
            # 清理到目标状态
            target_allocated = state.get('allocated', 0)
            current_allocated = torch.cuda.memory_allocated(self.device)
            
            if current_allocated > target_allocated * 1.2:  # 20%容忍度
                logger.info("🧹 Cleaning up excess memory")
                self.optimize_memory()
    
    def context_switch_model(self, old_model: nn.Module, new_model: nn.Module) -> nn.Module:
        """
        上下文切换模型，优化内存使用
        
        Args:
            old_model: 旧模型
            new_model: 新模型
            
        Returns:
            优化后的新模型
        """
        # 保存内存状态
        memory_state = self.checkpoint_memory_state()
        
        # 释放旧模型
        if old_model is not None:
            old_model.cpu()  # 先移到CPU
            del old_model
        
        # 优化内存
        self.optimize_memory()
        
        # 确保新模型在正确设备上
        new_model = new_model.to(self.device)
        
        logger.info("🔄 Model context switch completed")
        return new_model
    
    def get_device_report(self) -> Dict[str, Any]:
        """获取设备使用报告"""
        report = {
            'device': str(self.device),
            'device_type': self.device.type,
            'registered_models': list(self.registered_models.keys())
        }
        
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            memory_stats = self.get_memory_stats()
            
            report.update({
                'gpu_name': props.name,
                'gpu_memory_total_gb': props.total_memory / 1024**3,
                'memory_stats': memory_stats,
                'memory_fraction': self.memory_fraction
            })
        
        return report

# 全局设备管理器实例
_global_device_manager = None

def get_device_manager(device: Optional[Union[str, torch.device]] = None) -> DeviceManager:
    """获取全局设备管理器实例"""
    global _global_device_manager
    
    if _global_device_manager is None:
        _global_device_manager = DeviceManager(device)
    
    return _global_device_manager

def reset_device_manager():
    """重置全局设备管理器"""
    global _global_device_manager
    _global_device_manager = None

def auto_device(data: Any) -> Any:
    """自动转移数据到管理的设备"""
    dm = get_device_manager()
    return dm.to_device(data)

def register_model(name: str, model: nn.Module) -> nn.Module:
    """注册模型到设备管理器"""
    dm = get_device_manager()
    return dm.register_model(name, model)

def optimize_memory():
    """优化内存使用"""
    dm = get_device_manager()
    dm.optimize_memory()

def test_device_manager():
    """测试设备管理器功能"""
    print("🧪 Testing Device Manager...")
    
    # 创建设备管理器
    dm = DeviceManager()
    
    # 测试模型注册
    test_model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3)
    )
    
    registered_model = dm.register_model("test_model", test_model)
    print(f"✅ Model registered on {next(registered_model.parameters()).device}")
    
    # 测试数据转移
    test_data = torch.randn(4, 3, 32, 32)
    device_data = dm.to_device(test_data)
    print(f"✅ Data transferred to {device_data.device}")
    
    # 测试内存优化
    dm.optimize_memory()
    memory_stats = dm.get_memory_stats()
    print(f"✅ Memory stats: {memory_stats}")
    
    # 测试报告
    report = dm.get_device_report()
    print(f"✅ Device report: {report['device']}")
    
    print("🎉 Device Manager tests passed!")

if __name__ == "__main__":
    test_device_manager() 