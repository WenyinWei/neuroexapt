#!/usr/bin/env python3
"""
"""
\defgroup group_pareto_optimization Pareto Optimization
\ingroup core
Pareto Optimization module for NeuroExapt framework.
"""


Dynamic Neural Morphogenesis - 多目标进化优化模块

基于帕累托最优的多目标架构演化：
1. 同时优化准确率、计算效率、模型复杂度
2. 使用遗传算法进行全局架构搜索
3. 实现帕累托前沿分析和非支配排序
4. 支持自适应种群管理和多样性保持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
import random
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FitnessMetrics:
    """适应度指标"""
    accuracy: float          # 准确率 (最大化)
    efficiency: float        # 计算效率 (GFLOPS/s, 最大化)
    complexity: float        # 模型复杂度 (参数量, 最小化)
    memory_usage: float      # 内存使用 (MB, 最小化)
    training_speed: float    # 训练速度 (samples/s, 最大化)
    energy_consumption: float # 能耗 (J, 最小化)
    
    def __post_init__(self):
        """后处理：确保所有指标都是有效的数值"""
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if math.isnan(value) or math.isinf(value):
                setattr(self, field_name, 0.0)


class ModelFitnessEvaluator:
    """模型适应度评估器"""
    
    def __init__(self, device='cuda', evaluation_samples=500):
        self.device = device
        self.evaluation_samples = evaluation_samples
        self.baseline_metrics = None
        
    def evaluate_comprehensive_fitness(self, 
                                     model: nn.Module, 
                                     train_loader, 
                                     val_loader,
                                     optimization_objectives: List[str] = None) -> FitnessMetrics:
        """
        综合评估模型适应度
        
        Args:
            model: 待评估模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimization_objectives: 优化目标列表
            
        Returns:
            FitnessMetrics对象包含所有适应度指标
        """
        if optimization_objectives is None:
            optimization_objectives = ['accuracy', 'efficiency', 'complexity']
        
        model = model.to(self.device)
        
        # 1. 准确率评估
        accuracy = self._evaluate_accuracy(model, val_loader)
        
        # 2. 计算效率评估
        efficiency = self._evaluate_efficiency(model)
        
        # 3. 模型复杂度评估
        complexity = self._evaluate_complexity(model)
        
        # 4. 内存使用评估
        memory_usage = self._evaluate_memory_usage(model)
        
        # 5. 训练速度评估
        training_speed = self._evaluate_training_speed(model, train_loader)
        
        # 6. 能耗评估（近似）
        energy_consumption = self._estimate_energy_consumption(model, complexity, efficiency)
        
        return FitnessMetrics(
            accuracy=accuracy,
            efficiency=efficiency,
            complexity=complexity,
            memory_usage=memory_usage,
            training_speed=training_speed,
            energy_consumption=energy_consumption
        )
    
    def _evaluate_accuracy(self, model: nn.Module, val_loader) -> float:
        """评估模型准确率"""
        model.eval()
        correct = 0
        total = 0
        samples_processed = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                if samples_processed >= self.evaluation_samples:
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                samples_processed += target.size(0)
        
        return 100.0 * correct / total if total > 0 else 0.0
    
    def _evaluate_efficiency(self, model: nn.Module) -> float:
        """评估计算效率 (GFLOPS)"""
        try:
            # 使用torch.profiler评估FLOPS
            model.eval()
            
            # 创建随机输入
            if hasattr(model, 'input_size'):
                input_size = model.input_size
            else:
                # 假设是CIFAR-10输入
                input_size = (1, 3, 32, 32)
            
            dummy_input = torch.randn(input_size).to(self.device)
            
            # 计算FLOPS（简化估算）
            flops = self._estimate_flops(model, dummy_input)
            
            # 测量实际推理时间
            inference_time = self._measure_inference_time(model, dummy_input)
            
            # 计算GFLOPS/s
            if inference_time > 0:
                efficiency = flops / (inference_time * 1e9)  # GFLOPS/s
            else:
                efficiency = 0.0
                
            return efficiency
            
        except Exception as e:
            logger.warning(f"Efficiency evaluation failed: {e}")
            return 0.0
    
    def _estimate_flops(self, model: nn.Module, input_tensor: torch.Tensor) -> float:
        """估算模型的FLOPS"""
        total_flops = 0
        
        def flop_count_hook(module, input, output):
            nonlocal total_flops
            
            if isinstance(module, nn.Conv2d):
                # 卷积层FLOPS
                batch_size = output.shape[0]
                output_height, output_width = output.shape[2], output.shape[3]
                kernel_flops = module.kernel_size[0] * module.kernel_size[1]
                output_elements = batch_size * output_height * output_width
                filters = module.out_channels
                flops = output_elements * kernel_flops * filters * module.in_channels
                total_flops += flops
                
            elif isinstance(module, nn.Linear):
                # 全连接层FLOPS
                batch_size = input[0].shape[0]
                flops = batch_size * module.in_features * module.out_features
                total_flops += flops
        
        # 注册hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(flop_count_hook)
                hooks.append(hook)
        
        # 前向传播
        with torch.no_grad():
            _ = model(input_tensor)
        
        # 清理hooks
        for hook in hooks:
            hook.remove()
        
        return total_flops
    
    def _measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor, 
                              warmup_runs=10, measure_runs=100) -> float:
        """测量推理时间"""
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # 测量
        import time
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(measure_runs):
                _ = model(input_tensor)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / measure_runs
        return avg_time
    
    def _evaluate_complexity(self, model: nn.Module) -> float:
        """评估模型复杂度（参数量）"""
        total_params = sum(p.numel() for p in model.parameters())
        return float(total_params)
    
    def _evaluate_memory_usage(self, model: nn.Module) -> float:
        """评估内存使用（MB）"""
        try:
            # 计算模型参数占用的内存
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            
            # 计算缓冲区占用的内存
            buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
            
            total_memory = param_memory + buffer_memory
            memory_mb = total_memory / (1024 * 1024)  # 转换为MB
            
            return memory_mb
            
        except Exception as e:
            logger.warning(f"Memory evaluation failed: {e}")
            return 0.0
    
    def _evaluate_training_speed(self, model: nn.Module, train_loader) -> float:
        """评估训练速度（samples/s）"""
        try:
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            total_samples = 0
            import time
            start_time = time.time()
            
            batch_count = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 3:  # 只测试几个batch
                    break
                    
                try:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    total_samples += data.size(0)
                    batch_count += 1
                    
                except Exception as e:
                    logger.debug(f"Batch {batch_idx} failed in speed evaluation: {e}")
                    continue
            
            elapsed_time = time.time() - start_time
            speed = total_samples / elapsed_time if elapsed_time > 0 and total_samples > 0 else 1.0
            
            return max(speed, 1.0)  # 确保返回正值
            
        except Exception as e:
            logger.warning(f"Training speed evaluation failed: {e}")
            return 1.0  # 返回默认值而不是0
    
    def _estimate_energy_consumption(self, model: nn.Module, complexity: float, efficiency: float) -> float:
        """估算能耗（焦耳）"""
        # 简化的能耗模型：基于参数量和计算量
        base_energy = complexity * 1e-9  # 基础能耗
        compute_energy = (1.0 / (efficiency + 1e-6)) * 1e-3  # 计算能耗
        
        total_energy = base_energy + compute_energy
        return total_energy


class ParetoOptimizer:
    """帕累托优化器"""
    
    def __init__(self, objectives: List[str], maximize_objectives: List[str] = None):
        """
        Args:
            objectives: 优化目标列表
            maximize_objectives: 需要最大化的目标（其余为最小化）
        """
        self.objectives = objectives
        self.maximize_objectives = maximize_objectives or ['accuracy', 'efficiency', 'training_speed']
        self.minimize_objectives = [obj for obj in objectives if obj not in self.maximize_objectives]
        
    def is_dominated(self, fitness1: FitnessMetrics, fitness2: FitnessMetrics) -> bool:
        """
        判断fitness1是否被fitness2支配
        
        如果fitness2在所有目标上都不差于fitness1，且至少在一个目标上更好，
        则fitness1被fitness2支配
        """
        better_in_any = False
        
        for obj in self.objectives:
            val1 = getattr(fitness1, obj)
            val2 = getattr(fitness2, obj)
            
            if obj in self.maximize_objectives:
                # 最大化目标
                if val2 < val1:
                    return False  # fitness2在此目标上更差
                elif val2 > val1:
                    better_in_any = True
            else:
                # 最小化目标
                if val2 > val1:
                    return False  # fitness2在此目标上更差
                elif val2 < val1:
                    better_in_any = True
        
        return better_in_any
    
    def non_dominated_sort(self, population_fitness: List[FitnessMetrics]) -> List[List[int]]:
        """
        非支配排序
        
        Returns:
            每个前沿的个体索引列表
        """
        n = len(population_fitness)
        domination_count = [0] * n  # 每个个体被支配的次数
        dominated_solutions = [[] for _ in range(n)]  # 每个个体支配的解集合
        fronts = [[]]
        
        # 计算支配关系
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.is_dominated(population_fitness[j], population_fitness[i]):
                        # i支配j
                        dominated_solutions[i].append(j)
                    elif self.is_dominated(population_fitness[i], population_fitness[j]):
                        # j支配i
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # 构建后续前沿
        front_idx = 0
        while len(fronts[front_idx]) > 0:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
            front_idx += 1
        
        return fronts[:-1] if not fronts[-1] else fronts
    
    def calculate_crowding_distance(self, front: List[int], 
                                  population_fitness: List[FitnessMetrics]) -> List[float]:
        """
        计算拥挤距离
        
        Args:
            front: 前沿中个体的索引
            population_fitness: 种群适应度
            
        Returns:
            每个个体的拥挤距离
        """
        distances = [0.0] * len(front)
        
        if len(front) <= 2:
            return [float('inf')] * len(front)
        
        for obj in self.objectives:
            # 按目标值排序
            front_sorted = sorted(front, key=lambda x: getattr(population_fitness[x], obj))
            
            # 边界个体设置为无穷大
            idx_min = front.index(front_sorted[0])
            idx_max = front.index(front_sorted[-1])
            distances[idx_min] = float('inf')
            distances[idx_max] = float('inf')
            
            # 计算目标值范围
            obj_min = getattr(population_fitness[front_sorted[0]], obj)
            obj_max = getattr(population_fitness[front_sorted[-1]], obj)
            obj_range = obj_max - obj_min
            
            if obj_range > 0:
                # 计算中间个体的拥挤距离
                for i in range(1, len(front_sorted) - 1):
                    idx = front.index(front_sorted[i])
                    obj_prev = getattr(population_fitness[front_sorted[i-1]], obj)
                    obj_next = getattr(population_fitness[front_sorted[i+1]], obj)
                    distances[idx] += (obj_next - obj_prev) / obj_range
        
        return distances


class MultiObjectiveEvolution:
    """多目标进化算法"""
    
    def __init__(self, 
                 population_size: int = 20,
                 max_generations: int = 50,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7,
                 elitism_ratio: float = 0.1):
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        
        self.optimizer = ParetoOptimizer(['accuracy', 'efficiency', 'complexity'])
        self.evaluator = ModelFitnessEvaluator()
        
        # 进化历史
        self.evolution_history = []
        self.pareto_fronts_history = []
        
    def evolve_population(self, 
                         initial_models: List[nn.Module],
                         train_loader,
                         val_loader,
                         target_generation: int = None) -> Dict[str, Any]:
        """
        进化种群
        
        Args:
            initial_models: 初始模型种群
            train_loader: 训练数据
            val_loader: 验证数据
            target_generation: 目标代数
            
        Returns:
            进化结果
        """
        if target_generation is None:
            target_generation = self.max_generations
        
        # 确保种群大小
        population = self._ensure_population_size(initial_models)
        
        logger.info(f"Starting multi-objective evolution with {len(population)} individuals")
        
        for generation in range(target_generation):
            logger.info(f"Generation {generation + 1}/{target_generation}")
            
            # 评估种群适应度
            population_fitness = self._evaluate_population(population, train_loader, val_loader)
            
            # 非支配排序
            fronts = self.optimizer.non_dominated_sort(population_fitness)
            
            # 记录帕累托前沿
            pareto_front = [population[i] for i in fronts[0]]
            pareto_fitness = [population_fitness[i] for i in fronts[0]]
            
            self.pareto_fronts_history.append({
                'generation': generation,
                'front_size': len(fronts[0]),
                'fitness_metrics': pareto_fitness
            })
            
            # 选择和繁殖
            if generation < target_generation - 1:
                population = self._selection_and_reproduction(population, population_fitness, fronts)
            
            # 记录进化历史
            generation_stats = self._calculate_generation_statistics(population_fitness)
            self.evolution_history.append({
                'generation': generation,
                'stats': generation_stats,
                'pareto_front_size': len(fronts[0])
            })
            
            logger.info(f"Generation {generation + 1} completed: "
                       f"Pareto front size: {len(fronts[0])}, "
                       f"Best accuracy: {generation_stats['best_accuracy']:.2f}%")
        
        # 返回最终结果
        final_fitness = self._evaluate_population(population, train_loader, val_loader)
        final_fronts = self.optimizer.non_dominated_sort(final_fitness)
        
        best_models = [population[i] for i in final_fronts[0]]
        best_fitness = [final_fitness[i] for i in final_fronts[0]]
        
        return {
            'best_models': best_models,
            'best_fitness': best_fitness,
            'evolution_history': self.evolution_history,
            'pareto_fronts_history': self.pareto_fronts_history,
            'final_population': population,
            'final_fitness': final_fitness
        }
    
    def _ensure_population_size(self, models: List[nn.Module]) -> List[nn.Module]:
        """确保种群大小"""
        if len(models) == self.population_size:
            return models
        elif len(models) > self.population_size:
            return models[:self.population_size]
        else:
            # 通过复制和变异扩展种群
            population = models.copy()
            while len(population) < self.population_size:
                base_model = random.choice(models)
                mutated_model = self._mutate_model(copy.deepcopy(base_model))
                population.append(mutated_model)
            return population
    
    def _evaluate_population(self, 
                           population: List[nn.Module], 
                           train_loader, 
                           val_loader) -> List[FitnessMetrics]:
        """评估种群适应度"""
        fitness_list = []
        
        for i, model in enumerate(population):
            try:
                fitness = self.evaluator.evaluate_comprehensive_fitness(
                    model, train_loader, val_loader
                )
                fitness_list.append(fitness)
                
                logger.debug(f"Model {i}: Accuracy={fitness.accuracy:.2f}%, "
                           f"Efficiency={fitness.efficiency:.2f}, "
                           f"Complexity={fitness.complexity:.0f}")
                           
            except Exception as e:
                logger.warning(f"Failed to evaluate model {i}: {e}")
                # 使用默认的差适应度
                fitness_list.append(FitnessMetrics(0.0, 0.0, 1e9, 1e9, 0.0, 1e9))
        
        return fitness_list
    
    def _selection_and_reproduction(self, 
                                  population: List[nn.Module],
                                  fitness: List[FitnessMetrics],
                                  fronts: List[List[int]]) -> List[nn.Module]:
        """选择和繁殖"""
        new_population = []
        
        # 精英保留
        elite_count = max(1, int(self.population_size * self.elitism_ratio))
        
        # 从帕累托前沿选择精英
        elite_indices = fronts[0][:elite_count]
        for idx in elite_indices:
            new_population.append(copy.deepcopy(population[idx]))
        
        # 繁殖剩余个体
        while len(new_population) < self.population_size:
            # 锦标赛选择父母
            parent1 = self._tournament_selection(population, fitness, fronts)
            parent2 = self._tournament_selection(population, fitness, fronts)
            
            # 交叉
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)
            
            # 变异
            if random.random() < self.mutation_rate:
                child = self._mutate_model(child)
            
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, 
                            population: List[nn.Module],
                            fitness: List[FitnessMetrics],
                            fronts: List[List[int]],
                            tournament_size: int = 3) -> nn.Module:
        """锦标赛选择"""
        candidates = random.sample(range(len(population)), min(tournament_size, len(population)))
        
        # 找到最好的候选者（基于前沿排名和拥挤距离）
        best_candidate = candidates[0]
        best_front_rank = float('inf')
        
        for front_idx, front in enumerate(fronts):
            for candidate in candidates:
                if candidate in front:
                    if front_idx < best_front_rank:
                        best_candidate = candidate
                        best_front_rank = front_idx
                    elif front_idx == best_front_rank:
                        # 同一前沿，比较拥挤距离
                        distances = self.optimizer.calculate_crowding_distance(front, fitness)
                        candidate_distance = distances[front.index(candidate)]
                        best_distance = distances[front.index(best_candidate)]
                        if candidate_distance > best_distance:
                            best_candidate = candidate
        
        return copy.deepcopy(population[best_candidate])
    
    def _crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        """模型交叉"""
        # 简化的交叉：随机选择每层的参数来源
        child = copy.deepcopy(parent1)
        
        for (name1, param1), (name2, param2) in zip(parent1.named_parameters(), parent2.named_parameters()):
            if name1 == name2 and param1.shape == param2.shape:
                if random.random() < 0.5:
                    # 使用parent2的参数
                    child_param = dict(child.named_parameters())[name1]
                    child_param.data = param2.data.clone()
        
        return child
    
    def _mutate_model(self, model: nn.Module) -> nn.Module:
        """模型变异"""
        # 简化的变异：添加高斯噪声到参数
        for param in model.parameters():
            if random.random() < 0.1:  # 10%的参数进行变异
                noise = torch.randn_like(param) * 0.01
                param.data += noise
        
        return model
    
    def _calculate_generation_statistics(self, fitness: List[FitnessMetrics]) -> Dict[str, float]:
        """计算代际统计信息"""
        accuracies = [f.accuracy for f in fitness]
        efficiencies = [f.efficiency for f in fitness]
        complexities = [f.complexity for f in fitness]
        
        return {
            'best_accuracy': max(accuracies),
            'avg_accuracy': np.mean(accuracies),
            'best_efficiency': max(efficiencies),
            'avg_efficiency': np.mean(efficiencies),
            'min_complexity': min(complexities),
            'avg_complexity': np.mean(complexities)
        }


class DNMMultiObjectiveOptimization:
    """DNM多目标优化主控制器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.evolution = MultiObjectiveEvolution(**self.config['evolution'])
        self.optimization_history = []
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'evolution': {
                'population_size': 15,
                'max_generations': 20,
                'mutation_rate': 0.3,
                'crossover_rate': 0.7,
                'elitism_ratio': 0.15
            },
            'objectives': {
                'primary': ['accuracy', 'efficiency', 'complexity'],
                'secondary': ['memory_usage', 'training_speed'],
                'weights': {
                    'accuracy': 0.4,
                    'efficiency': 0.3,
                    'complexity': 0.2,
                    'memory_usage': 0.05,
                    'training_speed': 0.05
                }
            },
            'optimization': {
                'trigger_frequency': 15,  # 每15个epoch触发一次
                'performance_plateau_threshold': 0.01,
                'min_improvement_epochs': 5
            }
        }
    
    def optimize_architecture_population(self, 
                                       base_models: List[nn.Module],
                                       train_loader,
                                       val_loader,
                                       epoch: int) -> Dict[str, Any]:
        """优化架构种群"""
        
        # 检查是否需要触发优化
        if epoch % self.config['optimization']['trigger_frequency'] != 0:
            return {'optimized': False, 'message': 'Not optimization epoch'}
        
        logger.info(f"Starting multi-objective optimization at epoch {epoch}")
        
        # 执行多目标进化
        result = self.evolution.evolve_population(
            base_models, train_loader, val_loader,
            target_generation=self.config['evolution']['max_generations']
        )
        
        # 分析结果
        analysis = self._analyze_optimization_results(result)
        
        # 记录优化历史
        optimization_record = {
            'epoch': epoch,
            'initial_population_size': len(base_models),
            'final_pareto_front_size': len(result['best_models']),
            'analysis': analysis,
            'evolution_history': result['evolution_history'][-5:]  # 保留最后5代
        }
        
        self.optimization_history.append(optimization_record)
        
        logger.info(f"Multi-objective optimization completed: "
                   f"Pareto front size: {len(result['best_models'])}")
        
        return {
            'optimized': True,
            'best_models': result['best_models'],
            'best_fitness': result['best_fitness'],
            'analysis': analysis,
            'optimization_record': optimization_record
        }
    
    def _analyze_optimization_results(self, evolution_result: Dict) -> Dict[str, Any]:
        """分析优化结果"""
        best_fitness = evolution_result['best_fitness']
        
        if not best_fitness:
            return {'error': 'No valid fitness results'}
        
        # 计算帕累托前沿统计
        accuracies = [f.accuracy for f in best_fitness]
        efficiencies = [f.efficiency for f in best_fitness]
        complexities = [f.complexity for f in best_fitness]
        
        analysis = {
            'pareto_front_size': len(best_fitness),
            'accuracy_range': (min(accuracies), max(accuracies)),
            'efficiency_range': (min(efficiencies), max(efficiencies)),
            'complexity_range': (min(complexities), max(complexities)),
            'best_accuracy_model': accuracies.index(max(accuracies)),
            'most_efficient_model': efficiencies.index(max(efficiencies)),
            'simplest_model': complexities.index(min(complexities)),
            'diversity_score': self._calculate_diversity_score(best_fitness)
        }
        
        return analysis
    
    def _calculate_diversity_score(self, fitness_list: List[FitnessMetrics]) -> float:
        """计算种群多样性评分"""
        if len(fitness_list) < 2:
            return 0.0
        
        # 计算目标空间中的平均距离
        distances = []
        objectives = ['accuracy', 'efficiency', 'complexity']
        
        for i in range(len(fitness_list)):
            for j in range(i + 1, len(fitness_list)):
                f1, f2 = fitness_list[i], fitness_list[j]
                
                # 计算欧氏距离（标准化后）
                dist = 0.0
                for obj in objectives:
                    val1 = getattr(f1, obj)
                    val2 = getattr(f2, obj)
                    
                    # 简单标准化
                    if obj == 'accuracy':
                        norm_diff = abs(val1 - val2) / 100.0
                    elif obj == 'efficiency':
                        norm_diff = abs(val1 - val2) / max(val1 + val2, 1.0)
                    else:  # complexity
                        norm_diff = abs(val1 - val2) / max(val1 + val2, 1.0)
                    
                    dist += norm_diff ** 2
                
                distances.append(math.sqrt(dist))
        
        return np.mean(distances) if distances else 0.0
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化总结"""
        return {
            'total_optimizations': len(self.optimization_history),
            'optimization_history': self.optimization_history,
            'config': self.config
        }


# 测试函数
def test_multi_objective_optimization():
    """测试多目标优化"""
    print("🎯 Testing DNM Multi-Objective Optimization")
    
    # 创建测试模型
    def create_test_model(hidden_size=64):
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_size, 10)
        )
    
    # 创建初始种群
    initial_models = [
        create_test_model(32),
        create_test_model(64),
        create_test_model(96),
        create_test_model(128)
    ]
    
    # 创建虚拟数据
    from torch.utils.data import TensorDataset, DataLoader
    
    train_data = torch.randn(100, 3, 32, 32)
    train_labels = torch.randint(0, 10, (100,))
    val_data = torch.randn(50, 3, 32, 32)
    val_labels = torch.randint(0, 10, (50,))
    
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=16)
    val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=16)
    
    # 创建优化器
    optimizer = DNMMultiObjectiveOptimization()
    
    # 执行优化
    result = optimizer.optimize_architecture_population(
        initial_models, train_loader, val_loader, epoch=15
    )
    
    print(f"Optimization result: {result}")
    print(f"Summary: {optimizer.get_optimization_summary()}")
    
    return optimizer, result


if __name__ == "__main__":
    test_multi_objective_optimization()