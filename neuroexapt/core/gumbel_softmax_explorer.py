"""
Gumbel-Softmax引导式探索 (Gumbel-Softmax Guided Exploration)

ASO-SE框架的核心机制之二：通过Gumbel-Softmax技巧进行引导式架构探索，
避免贪婪选择导致的局部最优，支持温度调节和探索/利用平衡。

核心功能：
1. Gumbel-Softmax采样 - 可微的随机架构选择
2. 温度退火策略 - 从探索到利用的平滑过渡
3. 多样性保持 - 避免过早收敛到次优架构
"""

import torch
import torch.nn as nn
try:
    import torch.nn.functional as F
except ImportError:
    # Fallback for linter issues
    from torch.nn import functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
import math

logger = logging.getLogger(__name__)

class GumbelSoftmaxExplorer:
    """
    Gumbel-Softmax引导式探索器
    
    通过可控的随机性实现架构空间的有效探索
    """
    
    def __init__(self, initial_temp: float = 5.0, min_temp: float = 0.1, 
                 anneal_rate: float = 0.98, exploration_factor: float = 1.0):
        """
        Args:
            initial_temp: 初始温度，控制随机性强度
            min_temp: 最小温度，防止完全确定性选择
            anneal_rate: 温度退火率，每轮衰减比例
            exploration_factor: 探索因子，调节探索强度
        """
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.anneal_rate = anneal_rate
        self.exploration_factor = exploration_factor
        
        # 探索历史统计
        self.exploration_history = []
        self.diversity_scores = []
        self.step_count = 0
        
        logger.info(f"🌡️ Gumbel-Softmax Explorer initialized: "
                   f"temp={initial_temp:.2f}→{min_temp:.2f}, "
                   f"anneal_rate={anneal_rate:.3f}")
    
    def sample_architecture(self, alpha_weights: torch.Tensor, 
                          hard: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        使用Gumbel-Softmax采样架构
        
        Args:
            alpha_weights: 架构参数 [num_edges, num_ops]
            hard: 是否使用硬采样（straight-through）
            
        Returns:
            采样结果和统计信息
        """
        self.step_count += 1
        
        # 生成Gumbel噪声
        gumbel_noise = self._sample_gumbel(alpha_weights.shape)
        
        # 加入温度和噪声的logits
        logits = (alpha_weights + gumbel_noise) / self.current_temp
        
        # Softmax采样
        soft_samples = F.softmax(logits, dim=-1)
        
        if hard:
            # 硬采样：选择最大值，但保持梯度
            hard_samples = self._straight_through_softmax(soft_samples)
            samples = hard_samples
        else:
            samples = soft_samples
        
        # 计算探索统计
        stats = self._compute_exploration_stats(alpha_weights, samples, logits)
        
        # 记录探索历史
        self.exploration_history.append({
            'step': self.step_count,
            'temperature': self.current_temp,
            'entropy': stats['entropy'],
            'diversity': stats['diversity']
        })
        
        return samples, stats
    
    def _sample_gumbel(self, shape: Tuple[int, ...], eps: float = 1e-20) -> torch.Tensor:
        """采样Gumbel分布噪声"""
        uniform = torch.rand(shape, device=torch.cuda.current_device() if torch.cuda.is_available() else None)
        gumbel = -torch.log(-torch.log(uniform + eps) + eps)
        return gumbel * self.exploration_factor
    
    def _straight_through_softmax(self, soft_samples: torch.Tensor) -> torch.Tensor:
        """Straight-through硬采样，保持梯度流"""
        # 硬采样：选择最大概率的操作
        hard_samples = torch.zeros_like(soft_samples)
        max_indices = torch.argmax(soft_samples, dim=-1, keepdim=True)
        hard_samples.scatter_(-1, max_indices, 1.0)
        
        # Straight-through梯度：前向使用硬采样，反向使用软采样
        return hard_samples.detach() + soft_samples - soft_samples.detach()
    
    def _compute_exploration_stats(self, alpha_weights: torch.Tensor, 
                                 samples: torch.Tensor, 
                                 logits: torch.Tensor) -> Dict:
        """计算探索相关统计信息"""
        with torch.no_grad():
            # 熵计算 - 衡量探索程度
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
            
            # 多样性分数 - 衡量选择的多样性
            diversity = self._compute_diversity_score(samples)
            
            # 与贪婪选择的差异
            greedy_selection = torch.zeros_like(alpha_weights)
            greedy_indices = torch.argmax(alpha_weights, dim=-1, keepdim=True)
            greedy_selection.scatter_(-1, greedy_indices, 1.0)
            
            difference_from_greedy = torch.abs(samples - greedy_selection).sum().item()
            
            return {
                'entropy': entropy,
                'diversity': diversity,
                'temperature': self.current_temp,
                'difference_from_greedy': difference_from_greedy,
                'exploration_strength': entropy / math.log(alpha_weights.size(-1))  # 归一化熵
            }
    
    def _compute_diversity_score(self, samples: torch.Tensor) -> float:
        """计算架构选择的多样性分数"""
        # 计算每个操作被选择的频率
        op_frequencies = samples.sum(dim=0)
        total_selections = op_frequencies.sum()
        
        if total_selections > 0:
            op_probs = op_frequencies / total_selections
            # 使用归一化熵作为多样性指标
            diversity = -(op_probs * torch.log(op_probs + 1e-8)).sum().item()
            max_diversity = math.log(len(op_frequencies))
            return diversity / max_diversity if max_diversity > 0 else 0.0
        return 0.0
    
    def update_temperature(self, performance_gain: Optional[float] = None, 
                          force_anneal: bool = False) -> float:
        """
        更新温度参数
        
        Args:
            performance_gain: 性能提升，用于自适应温度调节
            force_anneal: 强制退火
            
        Returns:
            新的温度值
        """
        if force_anneal or self.current_temp > self.min_temp:
            if performance_gain is not None:
                # 自适应温度调节：性能提升时降温，性能下降时升温
                if performance_gain > 0:
                    # 性能提升，可以适当降低探索
                    self.current_temp *= self.anneal_rate
                else:
                    # 性能下降，增加探索
                    self.current_temp = min(self.current_temp * 1.05, self.initial_temp)
            else:
                # 标准线性退火
                self.current_temp = max(self.current_temp * self.anneal_rate, self.min_temp)
        
        logger.debug(f"Temperature updated to {self.current_temp:.3f}")
        return self.current_temp
    
    def get_exploration_schedule(self, total_steps: int) -> List[float]:
        """
        生成完整的探索温度计划
        
        Args:
            total_steps: 总步数
            
        Returns:
            温度计划列表
        """
        schedule = []
        temp = self.initial_temp
        
        for step in range(total_steps):
            schedule.append(temp)
            temp = max(temp * self.anneal_rate, self.min_temp)
        
        return schedule
    
    def adaptive_temperature_control(self, validation_loss_history: List[float], 
                                   window_size: int = 5) -> float:
        """
        基于验证损失历史的自适应温度控制
        
        Args:
            validation_loss_history: 验证损失历史
            window_size: 滑动窗口大小
            
        Returns:
            调整后的温度
        """
        if len(validation_loss_history) < window_size:
            return self.current_temp
        
        # 计算最近窗口的损失趋势
        recent_losses = validation_loss_history[-window_size:]
        trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        if trend > 0:  # 损失上升，增加探索
            self.current_temp = min(self.current_temp * 1.1, self.initial_temp)
            logger.info(f"🔥 Increasing exploration: temp={self.current_temp:.3f} (loss trend: +{trend:.4f})")
        elif trend < -0.001:  # 损失显著下降，减少探索
            self.current_temp = max(self.current_temp * 0.95, self.min_temp)
            logger.info(f"❄️ Reducing exploration: temp={self.current_temp:.3f} (loss trend: {trend:.4f})")
        
        return self.current_temp
    
    def get_exploration_report(self) -> Dict:
        """获取探索过程的详细报告"""
        if not self.exploration_history:
            return {"message": "No exploration history available"}
        
        history = self.exploration_history
        
        return {
            "total_steps": len(history),
            "current_temperature": self.current_temp,
            "average_entropy": np.mean([h['entropy'] for h in history]),
            "average_diversity": np.mean([h['diversity'] for h in history]),
            "temperature_range": {
                "initial": history[0]['temperature'],
                "current": history[-1]['temperature'],
                "min": min(h['temperature'] for h in history)
            },
            "exploration_trend": {
                "entropy_trend": np.polyfit(range(len(history)), 
                                          [h['entropy'] for h in history], 1)[0],
                "diversity_trend": np.polyfit(range(len(history)), 
                                            [h['diversity'] for h in history], 1)[0]
            }
        }


class MultiObjectiveExplorer(GumbelSoftmaxExplorer):
    """
    多目标Gumbel-Softmax探索器
    
    同时考虑准确率、延迟、参数量等多个目标的架构探索
    """
    
    def __init__(self, objectives: List[str] = ["accuracy", "latency", "params"], 
                 objective_weights: Optional[List[float]] = None, **kwargs):
        super().__init__(**kwargs)
        
        self.objectives = objectives
        self.objective_weights = objective_weights or [1.0] * len(objectives)
        self.objective_history = {obj: [] for obj in objectives}
        
        logger.info(f"🎯 Multi-objective explorer: {objectives} with weights {objective_weights}")
    
    def sample_with_objectives(self, alpha_weights: torch.Tensor, 
                             objective_scores: Dict[str, float],
                             hard: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        基于多目标的架构采样
        
        Args:
            alpha_weights: 架构参数
            objective_scores: 各目标的当前分数
            hard: 是否硬采样
            
        Returns:
            采样结果和统计信息
        """
        # 更新目标历史
        for obj, score in objective_scores.items():
            if obj in self.objective_history:
                self.objective_history[obj].append(score)
        
        # 计算多目标权重调整
        objective_adjustment = self._compute_objective_adjustment(objective_scores)
        
        # 调整架构参数权重
        adjusted_alpha = alpha_weights + objective_adjustment
        
        return self.sample_architecture(adjusted_alpha, hard)
    
    def _compute_objective_adjustment(self, objective_scores: Dict[str, float]) -> torch.Tensor:
        """基于多目标分数计算架构参数调整"""
        # 这里可以实现复杂的多目标优化策略
        # 简化版本：根据目标达成情况调整温度
        
        weighted_score = sum(score * weight 
                           for score, weight in zip(objective_scores.values(), self.objective_weights))
        
        # 根据综合分数调整探索强度
        if weighted_score > 0.8:  # 高分时减少探索
            self.current_temp *= 0.98
        elif weighted_score < 0.5:  # 低分时增加探索
            self.current_temp *= 1.02
        
        # 返回零调整（简化版本，实际应用中可扩展为更复杂的调整策略）
        return torch.zeros(1, 1) * 0.1


def create_annealing_schedule(initial_temp: float, min_temp: float, 
                            total_epochs: int, schedule_type: str = "cosine") -> List[float]:
    """
    创建温度退火计划
    
    Args:
        initial_temp: 初始温度
        min_temp: 最小温度
        total_epochs: 总轮数
        schedule_type: 计划类型 ("linear", "cosine", "exponential")
        
    Returns:
        温度计划列表
    """
    temperatures = []
    
    for epoch in range(total_epochs):
        if schedule_type == "linear":
            progress = epoch / (total_epochs - 1)
            temp = initial_temp * (1 - progress) + min_temp * progress
            
        elif schedule_type == "cosine":
            progress = epoch / (total_epochs - 1)
            temp = min_temp + (initial_temp - min_temp) * 0.5 * (1 + math.cos(math.pi * progress))
            
        elif schedule_type == "exponential":
            decay_rate = (min_temp / initial_temp) ** (1 / (total_epochs - 1))
            temp = initial_temp * (decay_rate ** epoch)
            
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        temperatures.append(max(temp, min_temp))
    
    return temperatures


def test_gumbel_softmax_explorer():
    """测试Gumbel-Softmax探索器的功能"""
    print("🧪 Testing Gumbel-Softmax Explorer...")
    
    # 创建探索器
    explorer = GumbelSoftmaxExplorer(initial_temp=2.0, min_temp=0.1)
    
    # 模拟架构参数
    alpha_weights = torch.randn(10, 8)  # 10条边，8种操作
    alpha_weights = F.softmax(alpha_weights, dim=-1)
    
    print(f"📊 Original alpha shape: {alpha_weights.shape}")
    
    # 测试软采样
    soft_samples, soft_stats = explorer.sample_architecture(alpha_weights, hard=False)
    print(f"✅ Soft sampling completed, entropy: {soft_stats['entropy']:.3f}")
    
    # 测试硬采样
    hard_samples, hard_stats = explorer.sample_architecture(alpha_weights, hard=True)
    print(f"✅ Hard sampling completed, entropy: {hard_stats['entropy']:.3f}")
    
    # 测试温度退火
    original_temp = explorer.current_temp
    explorer.update_temperature()
    print(f"✅ Temperature annealed: {original_temp:.3f} → {explorer.current_temp:.3f}")
    
    # 测试多目标探索器
    multi_explorer = MultiObjectiveExplorer(
        objectives=["accuracy", "latency"], 
        objective_weights=[0.7, 0.3]
    )
    
    objective_scores = {"accuracy": 0.85, "latency": 0.6}
    multi_samples, multi_stats = multi_explorer.sample_with_objectives(
        alpha_weights, objective_scores, hard=True
    )
    print(f"✅ Multi-objective sampling completed")
    
    # 测试退火计划
    schedule = create_annealing_schedule(5.0, 0.1, 20, "cosine")
    print(f"✅ Created annealing schedule: {len(schedule)} steps")
    print(f"   Start: {schedule[0]:.2f}, Mid: {schedule[10]:.2f}, End: {schedule[-1]:.2f}")
    
    print("🎉 Gumbel-Softmax Explorer tests passed!")


if __name__ == "__main__":
    test_gumbel_softmax_explorer() 