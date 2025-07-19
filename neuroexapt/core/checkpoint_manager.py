"""
@defgroup group_checkpoint_manager Checkpoint Manager
@ingroup core
Checkpoint Manager module for NeuroExapt framework.

断点续练管理器 (Checkpoint Manager)

像下载软件的断点续传一样，支持训练的断点续练，从任何中断点无缝恢复训练。

核心功能：
1. 自动检测训练中断点
2. 智能恢复训练状态
3. 架构演进历史追踪
4. 多版本检查点管理
5. 训练进度可视化
"""

import os
import json
import time
import torch
import logging
import shutil
import glob
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    断点续练管理器
    
    管理训练的完整生命周期，支持任意时刻的中断和恢复
    """
    
    def __init__(self, checkpoint_dir: str, experiment_name: str = "aso_se_training",
                 max_checkpoints: int = 5, auto_save_interval: int = 5):
        """
        Args:
            checkpoint_dir: 检查点保存目录
            experiment_name: 实验名称
            max_checkpoints: 最大保留检查点数量
            auto_save_interval: 自动保存间隔（epochs）
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.max_checkpoints = max_checkpoints
        self.auto_save_interval = auto_save_interval
        
        # 创建目录结构
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = self.checkpoint_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # 子目录
        self.models_dir = self.experiment_dir / "models"
        self.logs_dir = self.experiment_dir / "logs"
        self.metadata_dir = self.experiment_dir / "metadata"
        self.best_dir = self.experiment_dir / "best"
        
        for dir_path in [self.models_dir, self.logs_dir, self.metadata_dir, self.best_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 训练状态
        self.training_session_id = self._generate_session_id()
        self.current_checkpoint = None
        self.checkpoint_history = []
        self.best_performance = 0.0
        self.best_checkpoint_path: Optional[str] = None
        
        # 元数据文件
        self.metadata_file = self.metadata_dir / "training_metadata.json"
        self.session_file = self.logs_dir / f"session_{self.training_session_id}.log"
        
        # 加载历史元数据
        self._load_metadata()
        
        logger.info(f"📁 Checkpoint Manager initialized")
        logger.info(f"   Experiment: {experiment_name}")
        logger.info(f"   Directory: {self.experiment_dir}")
        logger.info(f"   Session ID: {self.training_session_id}")
        
    def _generate_session_id(self) -> str:
        """生成训练会话ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{os.getpid()}"
    
    def _load_metadata(self):
        """加载训练元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                self.checkpoint_history = metadata.get('checkpoint_history', [])
                self.best_performance = metadata.get('best_performance', 0.0)
                self.best_checkpoint_path = metadata.get('best_checkpoint_path', None)
                
                logger.info(f"📂 Loaded metadata: {len(self.checkpoint_history)} checkpoints")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load metadata: {e}")
                self.checkpoint_history = []
    
    def _save_metadata(self):
        """保存训练元数据"""
        metadata = {
            'experiment_name': self.experiment_name,
            'training_session_id': self.training_session_id,
            'checkpoint_history': self.checkpoint_history,
            'best_performance': self.best_performance,
            'best_checkpoint_path': self.best_checkpoint_path,
            'last_updated': datetime.now().isoformat(),
            'total_checkpoints': len(self.checkpoint_history),
            'total_sessions': len(set(cp.get('session_id', '') for cp in self.checkpoint_history))
        }
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save metadata: {e}")
    
    def save_checkpoint(self, epoch: int, model_state: Dict, optimizer_states: Dict,
                       scheduler_states: Optional[Dict], training_stats: Dict,
                       framework_state: Dict, performance_metric: float,
                       architecture_info: Optional[Dict] = None) -> str:
        """
        保存训练检查点
        
        Args:
            epoch: 当前epoch
            model_state: 模型状态字典
            optimizer_states: 优化器状态字典
            scheduler_states: 学习率调度器状态
            training_stats: 训练统计信息
            framework_state: ASO-SE框架状态
            performance_metric: 性能指标（用于判断最佳模型）
            architecture_info: 架构信息
            
        Returns:
            保存的检查点路径
        """
        checkpoint_filename = f"checkpoint_epoch_{epoch:04d}.pth"
        checkpoint_path = self.models_dir / checkpoint_filename
        
        # 完整的检查点数据
        checkpoint_data = {
            # 基本信息
            'epoch': epoch,
            'session_id': self.training_session_id,
            'timestamp': datetime.now().isoformat(),
            'performance_metric': performance_metric,
            
            # 模型状态
            'model_state': model_state,
            'optimizer_states': optimizer_states,
            'scheduler_states': scheduler_states,
            
            # 框架状态
            'framework_state': framework_state,
            'training_stats': training_stats,
            'architecture_info': architecture_info,
            
            # 元信息
            'checkpoint_version': '2.0',
            'pytorch_version': torch.__version__,
            'device_info': self._get_device_info()
        }
        
        try:
            # 保存检查点
            torch.save(checkpoint_data, checkpoint_path)
            
            # 更新历史记录
            checkpoint_record = {
                'epoch': epoch,
                'path': str(checkpoint_path),
                'performance_metric': performance_metric,
                'timestamp': checkpoint_data['timestamp'],
                'session_id': self.training_session_id,
                'size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
                'architecture_info': architecture_info
            }
            
            self.checkpoint_history.append(checkpoint_record)
            self.current_checkpoint = checkpoint_record
            
            # 检查是否是最佳性能
            if performance_metric > self.best_performance:
                self.best_performance = performance_metric
                self.best_checkpoint_path = str(checkpoint_path)
                self._save_best_model(checkpoint_data)
                logger.info(f"🏆 New best performance: {performance_metric:.4f}")
            
            # 清理旧检查点
            self._cleanup_old_checkpoints()
            
            # 保存元数据
            self._save_metadata()
            
            logger.info(f"💾 Checkpoint saved: epoch {epoch}, performance {performance_metric:.4f}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            raise e
    
    def _save_best_model(self, checkpoint_data: Dict):
        """保存最佳模型副本"""
        best_path = self.best_dir / "best_model.pth"
        try:
            torch.save(checkpoint_data, best_path)
            
            # 保存轻量级最佳模型（只包含模型权重）
            lightweight_best = {
                'model_state': checkpoint_data['model_state'],
                'epoch': checkpoint_data['epoch'],
                'performance_metric': checkpoint_data['performance_metric'],
                'architecture_info': checkpoint_data.get('architecture_info'),
                'timestamp': checkpoint_data['timestamp']
            }
            
            lightweight_path = self.best_dir / "best_model_weights_only.pth"
            torch.save(lightweight_best, lightweight_path)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save best model: {e}")
    
    def load_latest_checkpoint(self) -> Optional[Dict]:
        """加载最新的检查点"""
        if not self.checkpoint_history:
            logger.info("📂 No checkpoint history found")
            return None
        
        # 按epoch排序，获取最新的
        latest_record = max(self.checkpoint_history, key=lambda x: x['epoch'])
        return self.load_checkpoint(latest_record['path'])
    
    def load_best_checkpoint(self) -> Optional[Dict]:
        """加载最佳性能的检查点"""
        if not self.best_checkpoint_path:
            logger.info("🏆 No best checkpoint found")
            return None
        
        return self.load_checkpoint(self.best_checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict]:
        """加载指定的检查点"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            # PyTorch 2.6+ requires weights_only=False for complex checkpoints
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            logger.info(f"📂 Loaded checkpoint from epoch {checkpoint_data['epoch']}")
            logger.info(f"   Performance: {checkpoint_data['performance_metric']:.4f}")
            logger.info(f"   Timestamp: {checkpoint_data['timestamp']}")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return None
    
    def resume_training(self, model, optimizers: Dict, schedulers: Optional[Dict],
                       framework, preferred_checkpoint: str = "latest") -> Tuple[int, Dict]:
        """
        恢复训练状态
        
        Args:
            model: 要恢复的模型
            optimizers: 优化器字典 {'weight': optimizer1, 'arch': optimizer2}
            schedulers: 学习率调度器字典
            framework: ASO-SE框架实例
            preferred_checkpoint: 'latest', 'best', 或具体路径
            
        Returns:
            (resume_epoch, training_stats)
        """
        logger.info(f"🔄 Attempting to resume training from {preferred_checkpoint} checkpoint...")
        
        # 选择检查点
        if preferred_checkpoint == "latest":
            checkpoint_data = self.load_latest_checkpoint()
        elif preferred_checkpoint == "best":
            checkpoint_data = self.load_best_checkpoint()
        else:
            checkpoint_data = self.load_checkpoint(preferred_checkpoint)
        
        if checkpoint_data is None:
            logger.info("🆕 No checkpoint found, starting fresh training")
            return 0, {}
        
        try:
            # 恢复模型状态
            model.load_state_dict(checkpoint_data['model_state'])
            logger.info("✅ Model state restored")
            
            # 恢复优化器状态
            optimizer_states = checkpoint_data.get('optimizer_states', {})
            for name, optimizer in optimizers.items():
                if name in optimizer_states:
                    optimizer.load_state_dict(optimizer_states[name])
                    logger.info(f"✅ {name} optimizer state restored")
            
            # 恢复学习率调度器状态
            if schedulers and checkpoint_data.get('scheduler_states'):
                scheduler_states = checkpoint_data['scheduler_states']
                for name, scheduler in schedulers.items():
                    if name in scheduler_states:
                        scheduler.load_state_dict(scheduler_states[name])
                        logger.info(f"✅ {name} scheduler state restored")
            
            # 恢复ASO-SE框架状态
            framework_state = checkpoint_data.get('framework_state', {})
            if hasattr(framework, 'load_state'):
                framework.load_state(framework_state)
                logger.info("✅ ASO-SE framework state restored")
            
            # 恢复训练统计
            training_stats = checkpoint_data.get('training_stats', {})
            resume_epoch = checkpoint_data['epoch'] + 1
            
            logger.info(f"🎯 Training resumed from epoch {resume_epoch}")
            logger.info(f"   Previous performance: {checkpoint_data['performance_metric']:.4f}")
            
            return resume_epoch, training_stats
            
        except Exception as e:
            logger.error(f"❌ Failed to resume training: {e}")
            logger.info("🆕 Starting fresh training due to resume failure")
            return 0, {}
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点文件"""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # 按性能排序，保留最佳的checkpoints
        sorted_checkpoints = sorted(
            self.checkpoint_history, 
            key=lambda x: x['performance_metric'], 
            reverse=True
        )
        
        # 保留最佳的检查点
        to_keep = sorted_checkpoints[:self.max_checkpoints]
        to_remove = [cp for cp in self.checkpoint_history if cp not in to_keep]
        
        for checkpoint in to_remove:
            try:
                checkpoint_path = Path(checkpoint['path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.debug(f"🗑️ Removed old checkpoint: {checkpoint_path.name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove checkpoint: {e}")
        
        # 更新历史记录
        self.checkpoint_history = to_keep
    
    def _get_device_info(self) -> Dict:
        """获取设备信息"""
        device_info = {'cpu_count': os.cpu_count()}
        
        if torch.cuda.is_available():
            device_info.update({
                'cuda_available': True,
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_current_device': torch.cuda.current_device(),
                'cuda_device_name': torch.cuda.get_device_name(),
                'cuda_memory_total': torch.cuda.get_device_properties(0).total_memory
            })
        else:
            device_info['cuda_available'] = False
        
        return device_info
    
    def get_training_progress(self) -> Dict:
        """获取训练进度信息"""
        if not self.checkpoint_history:
            return {
                'total_epochs': 0,
                'best_performance': 0.0,
                'total_checkpoints': 0,
                'latest_epoch': 0,
                'progress_trend': []
            }
        
        # 性能趋势
        progress_trend = [
            {
                'epoch': cp['epoch'],
                'performance': cp['performance_metric'],
                'timestamp': cp['timestamp']
            }
            for cp in sorted(self.checkpoint_history, key=lambda x: x['epoch'])
        ]
        
        latest_checkpoint = max(self.checkpoint_history, key=lambda x: x['epoch'])
        
        return {
            'total_epochs': latest_checkpoint['epoch'],
            'best_performance': self.best_performance,
            'total_checkpoints': len(self.checkpoint_history),
            'latest_epoch': latest_checkpoint['epoch'],
            'latest_performance': latest_checkpoint['performance_metric'],
            'progress_trend': progress_trend,
            'experiment_name': self.experiment_name,
            'checkpoint_directory': str(self.experiment_dir)
        }
    
    def create_training_report(self) -> str:
        """创建训练报告"""
        progress = self.get_training_progress()
        
        report = f"""
# ASO-SE Training Report

## Experiment Information
- **Experiment Name**: {self.experiment_name}
- **Session ID**: {self.training_session_id}
- **Checkpoint Directory**: {self.experiment_dir}

## Training Progress
- **Total Epochs**: {progress['total_epochs']}
- **Total Checkpoints**: {progress['total_checkpoints']}
- **Best Performance**: {progress['best_performance']:.4f}
- **Latest Performance**: {progress.get('latest_performance', 0.0):.4f}

## Architecture Evolution
"""
        
        # 添加架构演进信息
        arch_evolution = []
        for cp in sorted(self.checkpoint_history, key=lambda x: x['epoch']):
            if cp.get('architecture_info'):
                arch_info = cp['architecture_info']
                arch_evolution.append(f"- **Epoch {cp['epoch']}**: {arch_info}")
        
        if arch_evolution:
            report += "\n".join(arch_evolution)
        else:
            report += "- No architecture evolution recorded"
        
        report += f"""

## Performance Trend
"""
        
        # 添加性能趋势
        for trend in progress['progress_trend'][-10:]:  # 最近10个检查点
            report += f"- **Epoch {trend['epoch']}**: {trend['performance']:.4f}\n"
        
        # 保存报告
        report_path = self.logs_dir / f"training_report_{self.training_session_id}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📊 Training report saved: {report_path}")
        return str(report_path)
    
    def should_auto_save(self, epoch: int) -> bool:
        """判断是否应该自动保存"""
        return epoch % self.auto_save_interval == 0

# 全局检查点管理器实例
_global_checkpoint_manager = None

def get_checkpoint_manager(checkpoint_dir: str = "./checkpoints", 
                          experiment_name: str = "aso_se_training") -> CheckpointManager:
    """获取全局检查点管理器"""
    global _global_checkpoint_manager
    
    if _global_checkpoint_manager is None:
        _global_checkpoint_manager = CheckpointManager(checkpoint_dir, experiment_name)
    
    return _global_checkpoint_manager

def reset_checkpoint_manager():
    """重置全局检查点管理器"""
    global _global_checkpoint_manager
    _global_checkpoint_manager = None

def test_checkpoint_manager():
    """测试检查点管理器功能"""
    print("🧪 Testing Checkpoint Manager...")
    
    # 创建临时测试目录
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        cm = CheckpointManager(temp_dir, "test_experiment")
        
        # 模拟模型和优化器
        import torch.nn as nn
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # 测试保存检查点
        model_state = model.state_dict()
        optimizer_states = {'weight': optimizer.state_dict()}
        training_stats = {'loss': 0.5, 'accuracy': 85.0}
        framework_state = {'current_cycle': 1, 'current_phase': 'warmup'}
        
        checkpoint_path = cm.save_checkpoint(
            epoch=10,
            model_state=model_state,
            optimizer_states=optimizer_states,
            scheduler_states=None,
            training_stats=training_stats,
            framework_state=framework_state,
            performance_metric=85.0,
            architecture_info={'mutation_count': 2}
        )
        
        print(f"✅ Checkpoint saved: {checkpoint_path}")
        
        # 测试加载检查点
        loaded_data = cm.load_checkpoint(checkpoint_path)
        assert loaded_data is not None
        assert loaded_data['epoch'] == 10
        assert loaded_data['performance_metric'] == 85.0
        
        print(f"✅ Checkpoint loaded successfully")
        
        # 测试进度报告
        progress = cm.get_training_progress()
        assert progress['total_epochs'] == 10
        assert progress['best_performance'] == 85.0
        
        print(f"✅ Progress tracking works")
        
        # 测试训练报告
        report_path = cm.create_training_report()
        assert os.path.exists(report_path)
        
        print(f"✅ Training report created: {report_path}")
    
    print("🎉 Checkpoint Manager tests passed!")

if __name__ == "__main__":
    test_checkpoint_manager() 