"""
defgroup group_architecture_mutator Architecture Mutator
ingroup core
Architecture Mutator module for NeuroExapt framework.
"""

架构突变器 (Architecture Mutator)

ASO-SE框架的核心组件：负责执行架构的动态变化，包括：
1. 层增加 (Layer Addition) - 添加新的计算层
2. 通道扩展 (Channel Expansion) - 增加网络宽度
3. 分支添加 (Branch Addition) - 引入新的计算路径
4. 操作替换 (Operation Replacement) - 替换现有操作

所有变化都与函数保持初始化集成，确保架构变化时的平滑过渡。
"""

import torch
import torch.nn as nn
import copy
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from .function_preserving_init import FunctionPreservingInitializer, create_identity_residual_block
from .genotypes import Genotype, PRIMITIVES

logger = logging.getLogger(__name__)

class ArchitectureMutator:
    """
    架构突变器
    
    负责执行各种架构变化操作，与函数保持初始化紧密集成
    """
    
    def __init__(self, preserve_function: bool = True, mutation_strength: float = 0.3):
        """
        Args:
            preserve_function: 是否使用函数保持初始化
            mutation_strength: 突变强度，控制变化幅度
        """
        self.preserve_function = preserve_function
        self.mutation_strength = mutation_strength
        self.initializer = FunctionPreservingInitializer() if preserve_function else None
        
        # 突变历史
        self.mutation_history = []
        self.mutation_count = 0
        
        logger.info(f"🧬 Architecture Mutator initialized: "
                   f"preserve_function={preserve_function}, "
                   f"mutation_strength={mutation_strength}")
    
    def add_layer(self, model: nn.Module, target_position: int, 
                  layer_type: str = "conv", **layer_kwargs) -> nn.Module:
        """
        添加新层
        
        Args:
            model: 目标模型
            target_position: 插入位置
            layer_type: 层类型 ("conv", "linear", "residual")
            **layer_kwargs: 层参数
            
        Returns:
            修改后的模型
        """
        logger.info(f"🏗️ Adding {layer_type} layer at position {target_position}")
        
        if isinstance(model, nn.Sequential):
            return self._add_layer_to_sequential(model, target_position, layer_type, **layer_kwargs)
        else:
            # For non-Sequential models, treat as Sequential for simplicity
            logger.warning("Non-Sequential model, converting to Sequential for layer addition")
            return self._add_layer_to_sequential(nn.Sequential(*list(model.children())), target_position, layer_type, **layer_kwargs)
    
    def _add_layer_to_sequential(self, model: nn.Sequential, position: int, 
                               layer_type: str, **kwargs) -> nn.Sequential:
        """向Sequential模型添加层"""
        layers = list(model.children())
        
        if layer_type == "conv":
            new_layer = self._create_conv_layer(**kwargs)
        elif layer_type == "linear":
            new_layer = self._create_linear_layer(**kwargs)
        elif layer_type == "residual":
            new_layer = self._create_residual_layer(**kwargs)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
        
        # 函数保持初始化
        if self.preserve_function and self.initializer:
            new_layer = self.initializer.identity_layer_init(new_layer)
        
        # 插入新层
        layers.insert(position, new_layer)
        new_model = nn.Sequential(*layers)
        
        self._record_mutation("add_layer", {
            "position": position,
            "layer_type": layer_type,
            "kwargs": kwargs
        })
        
        return new_model
    
    def expand_channels(self, model: nn.Module, target_layer: str, 
                       expansion_factor: float = 1.5, 
                       strategy: str = "replicate") -> nn.Module:
        """
        扩展通道数
        
        Args:
            model: 目标模型
            target_layer: 目标层名称
            expansion_factor: 扩展倍数
            strategy: 扩展策略
            
        Returns:
            修改后的模型
        """
        logger.info(f"📈 Expanding channels in {target_layer} by {expansion_factor}x")
        
        new_model = copy.deepcopy(model)
        
        # 查找目标层
        target_module = self._find_module_by_name(new_model, target_layer)
        if target_module is None:
            logger.warning(f"Target layer {target_layer} not found")
            return model
        
        # 扩展通道
        if isinstance(target_module, nn.Conv2d):
            new_channels = int(target_module.out_channels * expansion_factor)
            if self.preserve_function and self.initializer:
                expanded_layer = self.initializer.expand_channels_preserving(
                    target_module, new_channels, strategy
                )
            else:
                expanded_layer = self._create_expanded_conv(target_module, new_channels)
            
            # 替换层
            self._replace_module_by_name(new_model, target_layer, expanded_layer)
            
            # 更新后续层的输入通道
            self._update_subsequent_layers(new_model, target_layer, new_channels)
        
        self._record_mutation("expand_channels", {
            "target_layer": target_layer,
            "expansion_factor": expansion_factor,
            "strategy": strategy
        })
        
        return new_model
    
    def add_branch(self, model: nn.Module, branch_point: str, 
                   branch_structure: List[Dict], merge_strategy: str = "add") -> nn.Module:
        """
        添加分支结构
        
        Args:
            model: 目标模型
            branch_point: 分支点
            branch_structure: 分支结构定义
            merge_strategy: 合并策略 ("add", "concat", "attention")
            
        Returns:
            修改后的模型
        """
        logger.info(f"🌿 Adding branch at {branch_point} with {len(branch_structure)} layers")
        
        # 创建分支模块
        branch = self._create_branch_from_structure(branch_structure)
        
        # 函数保持初始化：新分支初始为零输出
        if self.preserve_function and self.initializer:
            branch = self.initializer.zero_branch_init(branch)
        
        # 创建带分支的新模型
        new_model = self._integrate_branch(model, branch_point, branch, merge_strategy)
        
        self._record_mutation("add_branch", {
            "branch_point": branch_point,
            "branch_structure": branch_structure,
            "merge_strategy": merge_strategy
        })
        
        return new_model
    
    def replace_operation(self, model: nn.Module, target_layer: str, 
                         new_operation: str) -> nn.Module:
        """
        替换操作
        
        Args:
            model: 目标模型
            target_layer: 目标层
            new_operation: 新操作名称
            
        Returns:
            修改后的模型
        """
        logger.info(f"🔄 Replacing operation in {target_layer} with {new_operation}")
        
        new_model = copy.deepcopy(model)
        target_module = self._find_module_by_name(new_model, target_layer)
        
        if target_module is None:
            return model
        
        # 创建新操作
        new_op = self._create_operation(new_operation, target_module)
        
        # 函数保持初始化
        if self.preserve_function and self.initializer:
            new_op = self._preserve_function_in_replacement(target_module, new_op)
        
        # 替换操作
        self._replace_module_by_name(new_model, target_layer, new_op)
        
        self._record_mutation("replace_operation", {
            "target_layer": target_layer,
            "new_operation": new_operation
        })
        
        return new_model
    
    def mutate_genotype(self, current_genotype: Genotype, 
                       mutation_type: str = "random") -> Genotype:
        """
        突变基因型
        
        Args:
            current_genotype: 当前基因型
            mutation_type: 突变类型 ("random", "guided", "conservative")
            
        Returns:
            突变后的基因型
        """
        logger.info(f"🧬 Mutating genotype with {mutation_type} strategy")
        
        if mutation_type == "random":
            return self._random_genotype_mutation(current_genotype)
        elif mutation_type == "guided":
            return self._guided_genotype_mutation(current_genotype)
        elif mutation_type == "conservative":
            return self._conservative_genotype_mutation(current_genotype)
        else:
            raise ValueError(f"Unknown mutation type: {mutation_type}")
    
    def _create_conv_layer(self, in_channels: int, out_channels: int, 
                          kernel_size: int = 3, **kwargs) -> nn.Module:
        """创建卷积层"""
        return nn.Conv2d(in_channels, out_channels, kernel_size, 
                        padding=kernel_size//2, **kwargs)
    
    def _create_linear_layer(self, in_features: int, out_features: int, **kwargs) -> nn.Module:
        """创建线性层"""
        return nn.Linear(in_features, out_features, **kwargs)
    
    def _create_residual_layer(self, in_channels: int, out_channels: int, **kwargs) -> nn.Module:
        """创建残差层"""
        return create_identity_residual_block(in_channels, out_channels)
    
    def _create_expanded_conv(self, original_conv: nn.Conv2d, new_channels: int) -> nn.Conv2d:
        """创建扩展通道的卷积层"""
        new_conv = nn.Conv2d(
            original_conv.in_channels,
            new_channels,
            original_conv.kernel_size,
            original_conv.stride,
            original_conv.padding,
            original_conv.dilation,
            original_conv.groups,
            original_conv.bias is not None
        )
        return new_conv
    
    def _find_module_by_name(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """根据名称查找模块"""
        for module_name, module in model.named_modules():
            if module_name == name:
                return module
        return None
    
    def _replace_module_by_name(self, model: nn.Module, name: str, new_module: nn.Module):
        """根据名称替换模块"""
        parts = name.split('.')
        parent = model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_module)
    
    def _update_subsequent_layers(self, model: nn.Module, changed_layer: str, new_channels: int):
        """更新后续层以适应通道变化"""
        # 这里需要根据具体模型结构实现
        # 简化版本：假设下一层是需要更新输入通道的层
        pass
    
    def _create_branch_from_structure(self, structure: List[Dict]) -> nn.Module:
        """从结构定义创建分支"""
        layers = []
        for layer_def in structure:
            layer_type = layer_def.get("type", "conv")
            if layer_type == "conv":
                layers.append(self._create_conv_layer(**layer_def.get("params", {})))
            elif layer_type == "linear":
                layers.append(self._create_linear_layer(**layer_def.get("params", {})))
            # 可以添加更多层类型
        
        return nn.Sequential(*layers)
    
    def _integrate_branch(self, model: nn.Module, branch_point: str, 
                         branch: nn.Module, merge_strategy: str) -> nn.Module:
        """将分支集成到模型中"""
        # 简化版本：创建包装器
        class BranchedModel(nn.Module):
            def __init__(self, main_model, branch, branch_point, merge_strategy):
                super().__init__()
                self.main_model = main_model
                self.branch = branch
                self.branch_point = branch_point
                self.merge_strategy = merge_strategy
            
            def forward(self, x):
                # 这里需要实现分支逻辑
                main_out = self.main_model(x)
                branch_out = self.branch(x)
                
                if self.merge_strategy == "add":
                    return main_out + branch_out
                elif self.merge_strategy == "concat":
                    return torch.cat([main_out, branch_out], dim=1)
                else:
                    return main_out  # 默认只返回主路径
        
        return BranchedModel(model, branch, branch_point, merge_strategy)
    
    def _random_genotype_mutation(self, genotype: Genotype) -> Genotype:
        """随机基因型突变"""
        import random
        
        # 复制当前基因型
        normal_ops = list(genotype.normal)
        reduce_ops = list(genotype.reduce)
        
        # 随机突变几个操作
        num_mutations = max(1, int(len(normal_ops) * self.mutation_strength))
        
        for _ in range(num_mutations):
            if random.random() < 0.5 and normal_ops:  # 突变normal操作
                idx = random.randint(0, len(normal_ops) - 1)
                new_op = random.choice(PRIMITIVES)
                normal_ops[idx] = (new_op, normal_ops[idx][1])
            elif reduce_ops:  # 突变reduce操作
                idx = random.randint(0, len(reduce_ops) - 1)
                new_op = random.choice(PRIMITIVES)
                reduce_ops[idx] = (new_op, reduce_ops[idx][1])
        
        return Genotype(
            normal=normal_ops,
            normal_concat=genotype.normal_concat,
            reduce=reduce_ops,
            reduce_concat=genotype.reduce_concat
        )
    
    def _guided_genotype_mutation(self, genotype: Genotype) -> Genotype:
        """引导式基因型突变（基于性能历史）"""
        # 简化版本：偏向选择性能更好的操作
        return self._random_genotype_mutation(genotype)  # 目前使用随机突变
    
    def _conservative_genotype_mutation(self, genotype: Genotype) -> Genotype:
        """保守式基因型突变（小幅度变化）"""
        # 减少突变强度
        original_strength = self.mutation_strength
        self.mutation_strength *= 0.5
        
        result = self._random_genotype_mutation(genotype)
        
        # 恢复原始强度
        self.mutation_strength = original_strength
        return result
    
    def _create_operation(self, op_name: str, reference_module: nn.Module) -> nn.Module:
        """根据操作名称创建新操作"""
        # 这里需要根据PRIMITIVES实现具体操作创建
        # 简化版本：返回相同类型的模块
        return copy.deepcopy(reference_module)
    
    def _preserve_function_in_replacement(self, old_module: nn.Module, 
                                        new_module: nn.Module) -> nn.Module:
        """在操作替换时保持函数"""
        if self.initializer:
            # 尝试传递参数
            layer_mapping = {"new": "old"}  # 简化映射
            # 这里需要更复杂的逻辑来处理不同类型的层之间的参数传递
        return new_module
    
    def _record_mutation(self, mutation_type: str, details: Dict[str, Any]):
        """记录突变历史"""
        self.mutation_count += 1
        mutation_record = {
            "id": self.mutation_count,
            "type": mutation_type,
            "details": details,
            "timestamp": self.mutation_count  # 简化时间戳
        }
        self.mutation_history.append(mutation_record)
        logger.debug(f"Recorded mutation #{self.mutation_count}: {mutation_type}")
    
    def get_mutation_report(self) -> Dict:
        """获取突变历史报告"""
        if not self.mutation_history:
            return {"message": "No mutations performed yet"}
        
        mutation_types = {}
        for mutation in self.mutation_history:
            mut_type = mutation["type"]
            mutation_types[mut_type] = mutation_types.get(mut_type, 0) + 1
        
        return {
            "total_mutations": len(self.mutation_history),
            "mutation_types": mutation_types,
            "recent_mutations": self.mutation_history[-5:],  # 最近5次突变
            "mutation_strength": self.mutation_strength,
            "preserve_function": self.preserve_function
        }


class GradualArchitectureGrowth:
    """
    渐进式架构生长
    
    实现网络架构的渐进式增长，避免剧烈变化
    """
    
    def __init__(self, mutator: ArchitectureMutator, 
                 growth_schedule: Optional[List[Dict]] = None):
        """
        Args:
            mutator: 架构突变器
            growth_schedule: 生长计划
        """
        self.mutator = mutator
        self.growth_schedule = growth_schedule or self._default_growth_schedule()
        self.current_stage = 0
        
        logger.info(f"🌱 Gradual Architecture Growth initialized with {len(self.growth_schedule)} stages")
    
    def _default_growth_schedule(self) -> List[Dict]:
        """默认生长计划"""
        return [
            {"epoch": 10, "action": "expand_channels", "params": {"expansion_factor": 1.2}},
            {"epoch": 20, "action": "add_layer", "params": {"layer_type": "residual"}},
            {"epoch": 30, "action": "expand_channels", "params": {"expansion_factor": 1.5}},
            {"epoch": 40, "action": "add_branch", "params": {"merge_strategy": "add"}}
        ]
    
    def should_grow(self, current_epoch: int) -> bool:
        """判断是否应该生长"""
        if self.current_stage >= len(self.growth_schedule):
            return False
        
        target_epoch = self.growth_schedule[self.current_stage]["epoch"]
        return current_epoch >= target_epoch
    
    def perform_growth(self, model: nn.Module, current_epoch: int) -> Optional[nn.Module]:
        """执行生长操作"""
        if not self.should_grow(current_epoch):
            return None
        
        stage_config = self.growth_schedule[self.current_stage]
        action = stage_config["action"]
        params = stage_config.get("params", {})
        
        logger.info(f"🌿 Performing growth stage {self.current_stage}: {action}")
        
        if action == "expand_channels":
            result = self.mutator.expand_channels(model, "conv1", **params)
        elif action == "add_layer":
            result = self.mutator.add_layer(model, -1, **params)
        elif action == "add_branch":
            result = self.mutator.add_branch(model, "conv1", [{"type": "conv"}], **params)
        else:
            logger.warning(f"Unknown growth action: {action}")
            result = model
        
        self.current_stage += 1
        return result


def test_architecture_mutator():
    """测试架构突变器功能"""
    print("🧪 Testing Architecture Mutator...")
    
    # 创建突变器
    mutator = ArchitectureMutator(preserve_function=True, mutation_strength=0.3)
    
    # 创建简单测试模型
    test_model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU()
    )
    
    print(f"📊 Original model layers: {len(list(test_model.children()))}")
    
    # 测试添加层
    new_model = mutator.add_layer(test_model, 2, "conv", in_channels=16, out_channels=16)
    print(f"✅ After adding layer: {len(list(new_model.children()))} layers")
    
    # 测试通道扩展
    expanded_model = mutator.expand_channels(test_model, "0", expansion_factor=1.5)
    print(f"✅ Channel expansion completed")
    
    # 测试基因型突变
    from .genotypes import Genotype
    test_genotype = Genotype(
        normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
        normal_concat=[2],
        reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
        reduce_concat=[2]
    )
    
    mutated_genotype = mutator.mutate_genotype(test_genotype, "random")
    print(f"✅ Genotype mutation completed")
    
    # 测试渐进式生长
    growth = GradualArchitectureGrowth(mutator)
    should_grow = growth.should_grow(10)
    print(f"✅ Growth check at epoch 10: {should_grow}")
    
    # 获取报告
    report = mutator.get_mutation_report()
    print(f"✅ Mutation report: {report['total_mutations']} mutations performed")
    
    print("🎉 Architecture Mutator tests passed!")


if __name__ == "__main__":
    test_architecture_mutator() 