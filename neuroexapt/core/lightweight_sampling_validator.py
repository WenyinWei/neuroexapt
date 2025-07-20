"""
轻量级抽样验证器
Lightweight Sampling Validator

通过短时间训练快速验证变异收益期望，提供：
1. 多个随机初始化实例的快速训练
2. 小批量数据的抽样验证
3. 收益期望的贝叶斯校准
4. 变异成功概率的经验估计

核心思想：用少量计算获得变异效果的可靠估计
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
import time
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

from .multi_mutation_type_evaluator import (
    MutationConfig, MutationBenefitExpectation, MutationEvidence, MutationType
)

logger = logging.getLogger(__name__)


@dataclass
class SamplingValidationConfig:
    """抽样验证配置"""
    # 抽样参数
    num_samples: int = 5              # 随机初始化样本数
    sample_epochs: int = 3            # 每个样本的训练轮数
    sample_data_ratio: float = 0.1    # 抽样数据比例
    
    # 训练参数
    learning_rate: float = 0.01
    batch_size: int = 128
    weight_decay: float = 1e-4
    
    # 验证参数
    confidence_level: float = 0.95    # 置信水平
    min_improvement_threshold: float = 0.005  # 最小改进阈值
    
    # 并行参数
    max_workers: int = 2              # 最大并行工作数
    timeout_seconds: int = 300        # 超时时间


@dataclass
class SamplingResult:
    """抽样结果"""
    mean_improvement: float = 0.0     # 平均改进
    std_improvement: float = 0.0      # 改进标准差
    success_rate: float = 0.0         # 成功率
    confidence_interval: Tuple[float, float] = (0.0, 0.0)  # 置信区间
    sample_improvements: List[float] = None  # 各样本改进
    validation_time: float = 0.0      # 验证时间
    
    def __post_init__(self):
        if self.sample_improvements is None:
            self.sample_improvements = []


class LightweightSamplingValidator:
    """
    轻量级抽样验证器
    
    核心功能：
    1. 快速验证变异的实际效果
    2. 通过多次随机初始化获得可靠统计
    3. 校准理论收益期望
    4. 提供变异决策的经验支持
    """
    
    def __init__(self, 
                 config: SamplingValidationConfig = None,
                 device: torch.device = None):
        """
        初始化验证器
        
        Args:
            config: 验证配置
            device: 计算设备
        """
        self.config = config or SamplingValidationConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"初始化轻量级抽样验证器，设备: {self.device}")
    
    def validate_mutation_benefit(self,
                                mutation_config: MutationConfig,
                                base_model: nn.Module,
                                train_loader: DataLoader,
                                test_loader: DataLoader,
                                criterion: nn.Module = None) -> SamplingResult:
        """
        验证变异收益
        
        Args:
            mutation_config: 变异配置
            base_model: 基础模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            criterion: 损失函数
            
        Returns:
            SamplingResult: 抽样验证结果
        """
        start_time = time.time()
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        # 创建抽样数据集
        sample_train_loader = self._create_sample_dataloader(train_loader)
        sample_test_loader = self._create_sample_dataloader(test_loader)
        
        # 生成变异模型
        try:
            mutated_model = self._apply_mutation(base_model, mutation_config)
        except Exception as e:
            logger.error(f"应用变异失败: {e}")
            return SamplingResult(validation_time=time.time() - start_time)
        
        # 并行验证多个样本
        sample_improvements = self._parallel_sample_validation(
            base_model, mutated_model, sample_train_loader, sample_test_loader, criterion
        )
        
        # 计算统计结果
        result = self._compute_sampling_statistics(sample_improvements)
        result.validation_time = time.time() - start_time
        
        logger.info(f"变异验证完成: 平均改进={result.mean_improvement:.4f}, "
                   f"成功率={result.success_rate:.2f}, 时间={result.validation_time:.1f}s")
        
        return result
    
    def _create_sample_dataloader(self, original_loader: DataLoader) -> DataLoader:
        """创建抽样数据加载器"""
        dataset = original_loader.dataset
        total_size = len(dataset)
        sample_size = int(total_size * self.config.sample_data_ratio)
        
        # 随机抽样
        indices = torch.randperm(total_size)[:sample_size]
        sample_dataset = Subset(dataset, indices.tolist())
        
        return DataLoader(
            sample_dataset,
            batch_size=min(self.config.batch_size, len(sample_dataset)),
            shuffle=True,
            num_workers=0,  # 避免多进程问题
            pin_memory=False
        )
    
    def _apply_mutation(self, base_model: nn.Module, mutation_config: MutationConfig) -> nn.Module:
        """应用变异到模型"""
        mutated_model = copy.deepcopy(base_model)
        
        if mutation_config.mutation_type == MutationType.SERIAL_SPLIT:
            return self._apply_serial_split(mutated_model, mutation_config)
        elif mutation_config.mutation_type == MutationType.PARALLEL_SPLIT:
            return self._apply_parallel_split(mutated_model, mutation_config)
        elif mutation_config.mutation_type == MutationType.CHANNEL_WIDENING:
            return self._apply_channel_widening(mutated_model, mutation_config)
        elif mutation_config.mutation_type == MutationType.LAYER_REPLACEMENT:
            return self._apply_layer_replacement(mutated_model, mutation_config)
        else:
            raise ValueError(f"不支持的变异类型: {mutation_config.mutation_type}")
    
    def _apply_serial_split(self, model: nn.Module, config: MutationConfig) -> nn.Module:
        """应用串行分裂"""
        # 简化实现：在目标层后添加一个新层
        target_layer = config.target_layer
        
        if isinstance(target_layer, nn.Conv2d):
            # 在卷积层后添加新的卷积层
            new_layer = nn.Conv2d(
                in_channels=target_layer.out_channels,
                out_channels=target_layer.out_channels,
                kernel_size=3,
                padding=1,
                bias=target_layer.bias is not None
            )
        elif isinstance(target_layer, nn.Linear):
            # 在线性层后添加新的线性层
            new_layer = nn.Linear(
                in_features=target_layer.out_features,
                out_features=target_layer.out_features,
                bias=target_layer.bias is not None
            )
        else:
            raise ValueError(f"不支持的层类型进行串行分裂: {type(target_layer)}")
        
        # 简化：直接在模型末尾添加层（实际应该精确插入位置）
        model.add_module(f"serial_split_{id(new_layer)}", new_layer)
        return model
    
    def _apply_parallel_split(self, model: nn.Module, config: MutationConfig) -> nn.Module:
        """应用并行分裂"""
        # 简化实现：添加一个并行分支
        target_layer = config.target_layer
        
        if isinstance(target_layer, nn.Conv2d) and config.parallel_layer_types:
            # 创建并行层
            for i, layer_type in enumerate(config.parallel_layer_types):
                if layer_type == nn.Conv2d:
                    parallel_layer = nn.Conv2d(
                        in_channels=target_layer.in_channels,
                        out_channels=target_layer.out_channels // 2,  # 减少通道避免参数爆炸
                        kernel_size=3,
                        padding=1
                    )
                elif layer_type == nn.MaxPool2d:
                    parallel_layer = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                else:
                    continue
                
                model.add_module(f"parallel_split_{i}_{id(parallel_layer)}", parallel_layer)
        
        return model
    
    def _apply_channel_widening(self, model: nn.Module, config: MutationConfig) -> nn.Module:
        """应用通道展宽"""
        target_layer = config.target_layer
        
        if isinstance(target_layer, nn.Conv2d):
            # 增加输出通道数
            new_out_channels = int(target_layer.out_channels * config.widening_factor)
            
            new_layer = nn.Conv2d(
                in_channels=target_layer.in_channels,
                out_channels=new_out_channels,
                kernel_size=target_layer.kernel_size,
                stride=target_layer.stride,
                padding=target_layer.padding,
                bias=target_layer.bias is not None
            )
            
            # 初始化新权重（保持原有权重+随机初始化新权重）
            with torch.no_grad():
                # 复制原有权重
                new_layer.weight[:target_layer.out_channels] = target_layer.weight
                # 随机初始化新权重
                nn.init.kaiming_normal_(new_layer.weight[target_layer.out_channels:])
                
                if new_layer.bias is not None and target_layer.bias is not None:
                    new_layer.bias[:target_layer.out_channels] = target_layer.bias
                    nn.init.zeros_(new_layer.bias[target_layer.out_channels:])
            
            # 替换原层（简化实现）
            setattr(model, config.target_layer_name.split('.')[-1], new_layer)
        
        return model
    
    def _apply_layer_replacement(self, model: nn.Module, config: MutationConfig) -> nn.Module:
        """应用层替换"""
        if config.replacement_layer_type:
            # 创建新层
            if config.replacement_layer_type == nn.GELU:
                new_layer = nn.GELU()
            elif config.replacement_layer_type == nn.ReLU:
                new_layer = nn.ReLU()
            elif config.replacement_layer_type == nn.Tanh:
                new_layer = nn.Tanh()
            else:
                new_layer = config.replacement_layer_type()
            
            # 替换层（简化实现）
            layer_name = config.target_layer_name.split('.')[-1]
            if hasattr(model, layer_name):
                setattr(model, layer_name, new_layer)
        
        return model
    
    def _parallel_sample_validation(self,
                                  base_model: nn.Module,
                                  mutated_model: nn.Module,
                                  train_loader: DataLoader,
                                  test_loader: DataLoader,
                                  criterion: nn.Module) -> List[float]:
        """并行验证多个样本"""
        improvements = []
        
        # 准备样本任务
        sample_tasks = []
        for i in range(self.config.num_samples):
            sample_tasks.append({
                'sample_id': i,
                'base_model': copy.deepcopy(base_model),
                'mutated_model': copy.deepcopy(mutated_model),
                'train_loader': train_loader,
                'test_loader': test_loader,
                'criterion': criterion
            })
        
        # 并行执行
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_sample = {
                executor.submit(self._validate_single_sample, task): task['sample_id']
                for task in sample_tasks
            }
            
            for future in as_completed(future_to_sample, timeout=self.config.timeout_seconds):
                sample_id = future_to_sample[future]
                try:
                    improvement = future.result()
                    improvements.append(improvement)
                    logger.debug(f"样本 {sample_id} 验证完成，改进: {improvement:.4f}")
                except Exception as e:
                    logger.warning(f"样本 {sample_id} 验证失败: {e}")
                    improvements.append(0.0)  # 失败时记录为0改进
        
        return improvements
    
    def _validate_single_sample(self, task: Dict[str, Any]) -> float:
        """验证单个样本"""
        base_model = task['base_model'].to(self.device)
        mutated_model = task['mutated_model'].to(self.device)
        train_loader = task['train_loader']
        test_loader = task['test_loader']
        criterion = task['criterion']
        
        # 随机初始化（保持相同的随机种子以便比较）
        seed = np.random.randint(0, 10000)
        
        # 训练基础模型
        torch.manual_seed(seed)
        base_accuracy = self._quick_train_and_evaluate(
            base_model, train_loader, test_loader, criterion
        )
        
        # 训练变异模型
        torch.manual_seed(seed)
        mutated_accuracy = self._quick_train_and_evaluate(
            mutated_model, train_loader, test_loader, criterion
        )
        
        # 计算改进
        improvement = mutated_accuracy - base_accuracy
        return improvement
    
    def _quick_train_and_evaluate(self,
                                model: nn.Module,
                                train_loader: DataLoader,
                                test_loader: DataLoader,
                                criterion: nn.Module) -> float:
        """快速训练并评估模型"""
        model.train()
        
        # 优化器
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=0.9,
            weight_decay=self.config.weight_decay
        )
        
        # 快速训练
        for epoch in range(self.config.sample_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # 限制训练步数以控制时间
                if batch_idx >= 10:  # 最多10个batch
                    break
        
        # 评估
        return self._evaluate_model(model, test_loader)
    
    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> float:
        """评估模型准确率"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return accuracy
    
    def _compute_sampling_statistics(self, improvements: List[float]) -> SamplingResult:
        """计算抽样统计结果"""
        if not improvements:
            return SamplingResult()
        
        improvements = np.array(improvements)
        
        # 基础统计
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements, ddof=1) if len(improvements) > 1 else 0.0
        
        # 成功率（改进超过阈值的比例）
        success_count = np.sum(improvements > self.config.min_improvement_threshold)
        success_rate = success_count / len(improvements)
        
        # 置信区间
        if len(improvements) > 1:
            confidence_interval = self._compute_confidence_interval(
                improvements, self.config.confidence_level
            )
        else:
            confidence_interval = (mean_improvement, mean_improvement)
        
        return SamplingResult(
            mean_improvement=mean_improvement,
            std_improvement=std_improvement,
            success_rate=success_rate,
            confidence_interval=confidence_interval,
            sample_improvements=improvements.tolist()
        )
    
    def _compute_confidence_interval(self, 
                                   data: np.ndarray, 
                                   confidence_level: float) -> Tuple[float, float]:
        """计算置信区间"""
        from scipy import stats as scipy_stats
        
        n = len(data)
        mean = np.mean(data)
        sem = scipy_stats.sem(data)  # 标准误差
        
        # t分布置信区间
        confidence_interval = scipy_stats.t.interval(
            confidence_level, n-1, loc=mean, scale=sem
        )
        
        return confidence_interval
    
    def calibrate_benefit_expectation(self,
                                    theoretical_expectation: MutationBenefitExpectation,
                                    sampling_result: SamplingResult) -> MutationBenefitExpectation:
        """
        使用抽样结果校准理论收益期望
        
        Args:
            theoretical_expectation: 理论期望
            sampling_result: 抽样验证结果
            
        Returns:
            MutationBenefitExpectation: 校准后的期望
        """
        # 贝叶斯更新：结合理论预测和经验观测
        
        # 理论预测作为先验
        prior_mean = theoretical_expectation.expected_benefit
        prior_var = theoretical_expectation.benefit_variance
        
        # 抽样结果作为似然
        observed_mean = sampling_result.mean_improvement / 100.0  # 转换为准确率改进
        observed_var = (sampling_result.std_improvement / 100.0) ** 2
        
        # 贝叶斯更新（正态-正态共轭）
        if observed_var > 0:
            prior_precision = 1.0 / prior_var if prior_var > 0 else 1.0
            likelihood_precision = 1.0 / observed_var
            
            posterior_precision = prior_precision + likelihood_precision
            posterior_mean = (
                prior_precision * prior_mean + 
                likelihood_precision * observed_mean
            ) / posterior_precision
            posterior_var = 1.0 / posterior_precision
        else:
            # 观测方差为0时，完全相信观测结果
            posterior_mean = observed_mean
            posterior_var = prior_var * 0.1  # 大幅降低不确定性
        
        # 更新成功概率
        calibrated_success_prob = (
            theoretical_expectation.success_probability * 0.5 + 
            sampling_result.success_rate * 0.5
        )
        
        # 创建校准后的期望
        calibrated_expectation = MutationBenefitExpectation(
            expected_benefit=posterior_mean,
            benefit_variance=posterior_var,
            success_probability=calibrated_success_prob,
            risk_adjusted_benefit=posterior_mean * calibrated_success_prob,
            confidence_interval=(
                posterior_mean - 1.96 * np.sqrt(posterior_var),
                posterior_mean + 1.96 * np.sqrt(posterior_var)
            ),
            information_gain_benefit=theoretical_expectation.information_gain_benefit,
            diversity_benefit=theoretical_expectation.diversity_benefit,
            flow_improvement_benefit=theoretical_expectation.flow_improvement_benefit,
            cost_penalty=theoretical_expectation.cost_penalty,
            parameter_increase=theoretical_expectation.parameter_increase,
            computation_increase=theoretical_expectation.computation_increase
        )
        
        logger.info(f"收益期望校准: {prior_mean:.4f} -> {posterior_mean:.4f}, "
                   f"成功概率: {theoretical_expectation.success_probability:.2f} -> "
                   f"{calibrated_success_prob:.2f}")
        
        return calibrated_expectation