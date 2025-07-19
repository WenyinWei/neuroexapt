"""
"""
\defgroup group_evolution_checkpoint Evolution Checkpoint
\ingroup core
Evolution Checkpoint module for NeuroExapt framework.
"""


架构进化检查点管理器

负责管理神经网络架构进化过程中的checkpoints：
- 架构变异前自动保存
- 支持恢复训练
- 支持架构回撤
- 维护进化谱系
"""

import torch
import torch.nn as nn
import os
import json
import time
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class EvolutionCheckpoint:
    """单个进化检查点"""
    
    def __init__(self, 
                 checkpoint_id: str,
                 epoch: int,
                 network_state: Dict,
                 optimizer_state: Dict,
                 scheduler_state: Optional[Dict],
                 training_stats: Dict,
                 architecture_summary: Dict,
                 parent_id: Optional[str] = None,
                 growth_type: Optional[str] = None):
        
        self.checkpoint_id = checkpoint_id
        self.epoch = epoch
        self.network_state = network_state
        self.optimizer_state = optimizer_state  
        self.scheduler_state = scheduler_state
        self.training_stats = training_stats
        self.architecture_summary = architecture_summary
        self.parent_id = parent_id
        self.growth_type = growth_type
        self.created_time = time.time()
        self.created_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'checkpoint_id': self.checkpoint_id,
            'epoch': self.epoch,
            'training_stats': self.training_stats,
            'architecture_summary': self.architecture_summary,
            'parent_id': self.parent_id,
            'growth_type': self.growth_type,
            'created_time': self.created_time,
            'created_datetime': self.created_datetime
        }

class EvolutionCheckpointManager:
    """架构进化检查点管理器"""
    
    def __init__(self, experiment_name: str, base_dir: str = "./evolution_checkpoints"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.checkpoint_dir = self.base_dir / experiment_name
        
        # 创建目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 进化谱系
        self.evolution_tree: Dict[str, EvolutionCheckpoint] = {}
        self.checkpoint_order: List[str] = []
        
        # 当前活跃的checkpoint
        self.current_checkpoint_id: Optional[str] = None
        
        # 元数据文件
        self.metadata_file = self.checkpoint_dir / "evolution_metadata.json"
        
        # 加载已有的进化历史
        self._load_evolution_history()
        
        logger.info(f"Evolution checkpoint manager initialized: {self.checkpoint_dir}")
    
    def _generate_checkpoint_id(self, epoch: int, growth_type: Optional[str] = None) -> str:
        """生成checkpoint ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if growth_type:
            return f"epoch_{epoch:03d}_{growth_type}_{timestamp}"
        else:
            return f"epoch_{epoch:03d}_baseline_{timestamp}"
    
    def save_checkpoint(self, 
                       network: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int,
                       training_stats: Dict,
                       growth_type: Optional[str] = None,
                       parent_id: Optional[str] = None) -> str:
        """
        保存架构进化检查点
        
        Args:
            network: 神经网络模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            training_stats: 训练统计信息
            growth_type: 生长类型 ('grow_depth', 'grow_width', 'grow_branches')
            parent_id: 父节点checkpoint ID
            
        Returns:
            checkpoint_id: 新创建的checkpoint ID
        """
        
        checkpoint_id = self._generate_checkpoint_id(epoch, growth_type)
        
        # 获取架构摘要
        if hasattr(network, 'get_architecture_summary'):
            arch_summary = network.get_architecture_summary()
        else:
            arch_summary = {
                'total_parameters': sum(p.numel() for p in network.parameters()),
                'epoch': epoch
            }
        
        # 创建检查点
        checkpoint = EvolutionCheckpoint(
            checkpoint_id=checkpoint_id,
            epoch=epoch,
            network_state=network.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler else None,
            training_stats=training_stats.copy(),
            architecture_summary=arch_summary,
            parent_id=parent_id or self.current_checkpoint_id,
            growth_type=growth_type
        )
        
        # 保存到文件
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        torch.save({
            'network_state': checkpoint.network_state,
            'optimizer_state': checkpoint.optimizer_state,
            'scheduler_state': checkpoint.scheduler_state,
            'metadata': checkpoint.to_dict()
        }, checkpoint_path)
        
        # 更新进化树
        self.evolution_tree[checkpoint_id] = checkpoint
        self.checkpoint_order.append(checkpoint_id)
        self.current_checkpoint_id = checkpoint_id
        
        # 保存元数据
        self._save_evolution_history()
        
        logger.info(f"✅ Checkpoint saved: {checkpoint_id}")
        logger.info(f"   Epoch: {epoch}, Growth: {growth_type or 'baseline'}")
        logger.info(f"   Parameters: {arch_summary.get('total_parameters', 0):,}")
        logger.info(f"   Best accuracy: {training_stats.get('best_accuracy', 0):.2f}%")
        
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        加载检查点
        
        Returns:
            (network_state, optimizer_state, scheduler_state, metadata)
        """
        
        if checkpoint_id not in self.evolution_tree:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        logger.info(f"📥 Checkpoint loaded: {checkpoint_id}")
        
        return (
            checkpoint_data['network_state'],
            checkpoint_data['optimizer_state'], 
            checkpoint_data['scheduler_state'],
            checkpoint_data['metadata']
        )
    
    def get_best_checkpoint(self) -> Optional[str]:
        """获取最佳性能的checkpoint"""
        if not self.evolution_tree:
            return None
        
        best_checkpoint_id = None
        best_accuracy = -1.0
        
        for checkpoint_id, checkpoint in self.evolution_tree.items():
            accuracy = checkpoint.training_stats.get('best_accuracy', 0.0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_checkpoint_id = checkpoint_id
        
        return best_checkpoint_id
    
    def get_recent_checkpoints(self, n: int = 5) -> List[str]:
        """获取最近的n个checkpoints"""
        return self.checkpoint_order[-n:] if self.checkpoint_order else []
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> None:
        """回滚到指定checkpoint（谱系管理）"""
        if checkpoint_id not in self.evolution_tree:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        # 找到所有在此checkpoint之后的衍生checkpoints
        descendants = self._find_descendants(checkpoint_id)
        
        logger.info(f"🔄 Rolling back to checkpoint: {checkpoint_id}")
        if descendants:
            logger.info(f"   Will remove {len(descendants)} descendant checkpoints")
            
            # 删除衍生checkpoints（可选，或者保留作为分支）
            for desc_id in descendants:
                if desc_id in self.evolution_tree:
                    checkpoint_path = self.checkpoint_dir / f"{desc_id}.pt"
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()  # 删除文件
                    del self.evolution_tree[desc_id]
                    if desc_id in self.checkpoint_order:
                        self.checkpoint_order.remove(desc_id)
        
        # 更新当前checkpoint
        self.current_checkpoint_id = checkpoint_id
        
        # 保存更新后的元数据
        self._save_evolution_history()
        
        logger.info(f"✅ Rollback completed to {checkpoint_id}")
    
    def _find_descendants(self, parent_id: str) -> List[str]:
        """找到指定checkpoint的所有后代"""
        descendants = []
        
        def _find_children(parent: str):
            for checkpoint_id, checkpoint in self.evolution_tree.items():
                if checkpoint.parent_id == parent:
                    descendants.append(checkpoint_id)
                    _find_children(checkpoint_id)  # 递归查找
        
        _find_children(parent_id)
        return descendants
    
    def get_evolution_lineage(self) -> List[Dict]:
        """获取进化谱系"""
        lineage = []
        
        for checkpoint_id in self.checkpoint_order:
            if checkpoint_id in self.evolution_tree:
                checkpoint = self.evolution_tree[checkpoint_id]
                lineage.append({
                    'id': checkpoint_id,
                    'epoch': checkpoint.epoch,
                    'growth_type': checkpoint.growth_type,
                    'parent_id': checkpoint.parent_id,
                    'accuracy': checkpoint.training_stats.get('best_accuracy', 0.0),
                    'parameters': checkpoint.architecture_summary.get('total_parameters', 0),
                    'depth': checkpoint.architecture_summary.get('depth', 0),
                    'created_time': checkpoint.created_datetime
                })
        
        return lineage
    
    def display_evolution_tree(self):
        """显示进化树"""
        logger.info(f"\n🌳 EVOLUTION TREE for {self.experiment_name}")
        logger.info(f"{'='*80}")
        
        lineage = self.get_evolution_lineage()
        
        for i, checkpoint in enumerate(lineage):
            prefix = "├── " if i < len(lineage) - 1 else "└── "
            
            growth_emoji = {
                'grow_depth': '🌱',
                'grow_width': '🌿', 
                'grow_branches': '🌲',
                None: '🏗️'
            }.get(checkpoint['growth_type'], '🔸')
            
            logger.info(f"{prefix}{growth_emoji} {checkpoint['id'][:20]}...")
            logger.info(f"    Epoch: {checkpoint['epoch']:3d} | "
                       f"Acc: {checkpoint['accuracy']:5.2f}% | "
                       f"Params: {checkpoint['parameters']:,} | "
                       f"Depth: {checkpoint['depth']}")
            logger.info(f"    Growth: {checkpoint['growth_type'] or 'baseline'} | "
                       f"Time: {checkpoint['created_time']}")
        
        # 显示统计
        total_checkpoints = len(lineage)
        if total_checkpoints > 0:
            best_acc = max(cp['accuracy'] for cp in lineage)
            total_params = lineage[-1]['parameters'] if lineage else 0
            
            logger.info(f"\n📊 Evolution Statistics:")
            logger.info(f"   Total checkpoints: {total_checkpoints}")
            logger.info(f"   Best accuracy: {best_acc:.2f}%")
            logger.info(f"   Final parameters: {total_params:,}")
            logger.info(f"   Current checkpoint: {self.current_checkpoint_id}")
    
    def _save_evolution_history(self):
        """保存进化历史元数据"""
        metadata = {
            'experiment_name': self.experiment_name,
            'evolution_tree': {cid: cp.to_dict() for cid, cp in self.evolution_tree.items()},
            'checkpoint_order': self.checkpoint_order,
            'current_checkpoint_id': self.current_checkpoint_id,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_evolution_history(self):
        """加载进化历史元数据"""
        if not self.metadata_file.exists():
            return
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # 重建进化树
            for checkpoint_id, cp_data in metadata.get('evolution_tree', {}).items():
                # 注意：这里只加载元数据，不加载实际的模型状态
                checkpoint = EvolutionCheckpoint(
                    checkpoint_id=cp_data['checkpoint_id'],
                    epoch=cp_data['epoch'],
                    network_state={},  # 实际状态在需要时从文件加载
                    optimizer_state={},
                    scheduler_state={},
                    training_stats=cp_data['training_stats'],
                    architecture_summary=cp_data['architecture_summary'],
                    parent_id=cp_data.get('parent_id'),
                    growth_type=cp_data.get('growth_type')
                )
                checkpoint.created_time = cp_data['created_time']
                checkpoint.created_datetime = cp_data['created_datetime']
                
                self.evolution_tree[checkpoint_id] = checkpoint
            
            self.checkpoint_order = metadata.get('checkpoint_order', [])
            self.current_checkpoint_id = metadata.get('current_checkpoint_id')
            
            logger.info(f"📚 Loaded evolution history: {len(self.evolution_tree)} checkpoints")
            
        except Exception as e:
            logger.warning(f"Failed to load evolution history: {e}")
            self.evolution_tree = {}
            self.checkpoint_order = []
            self.current_checkpoint_id = None
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 10, keep_best: bool = True):
        """清理旧的checkpoints，保留最近的N个和最佳的"""
        if len(self.checkpoint_order) <= keep_last_n:
            return
        
        # 确定要保留的checkpoints
        keep_ids = set()
        
        # 保留最近的N个
        keep_ids.update(self.checkpoint_order[-keep_last_n:])
        
        # 保留最佳的
        if keep_best:
            best_id = self.get_best_checkpoint()
            if best_id:
                keep_ids.add(best_id)
        
        # 删除其他的
        deleted_count = 0
        for checkpoint_id in list(self.checkpoint_order):
            if checkpoint_id not in keep_ids:
                # 删除文件
                checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                
                # 从内存中删除
                if checkpoint_id in self.evolution_tree:
                    del self.evolution_tree[checkpoint_id]
                self.checkpoint_order.remove(checkpoint_id)
                
                deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"🧹 Cleaned up {deleted_count} old checkpoints")
            self._save_evolution_history()