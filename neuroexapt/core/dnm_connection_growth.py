#!/usr/bin/env python3
"""
Dynamic Neural Morphogenesis - 连接生长模块

基于梯度引导的连接动态生长机制：
1. 分析跨层梯度相关性，发现潜在的有益连接
2. 动态添加跳跃连接、注意力机制
3. 实现层间信息流优化
4. 支持ResNet式跳跃连接和Transformer式注意力连接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class GradientCorrelationAnalyzer:
    """梯度相关性分析器"""
    
    def __init__(self, correlation_threshold=0.1, history_length=10):
        self.correlation_threshold = correlation_threshold
        self.history_length = history_length
        self.gradient_history = defaultdict(deque)
        self.correlation_cache = {}
        
    def collect_gradients(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """收集模型的梯度信息"""
        gradients = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                # 只收集卷积和线性层的梯度
                if any(layer_type in name for layer_type in ['conv', 'linear', 'fc']):
                    gradients[name] = param.grad.clone().detach()
        
        return gradients
    
    def update_gradient_history(self, gradients: Dict[str, torch.Tensor]) -> None:
        """更新梯度历史记录"""
        for name, grad in gradients.items():
            # 将梯度展平并添加到历史记录
            flat_grad = grad.view(-1)
            
            if len(self.gradient_history[name]) >= self.history_length:
                self.gradient_history[name].popleft()
            
            self.gradient_history[name].append(flat_grad)
    
    def calculate_layer_correlation(self, layer1_name: str, layer2_name: str) -> float:
        """计算两层之间的梯度相关性"""
        
        # 检查缓存
        cache_key = tuple(sorted([layer1_name, layer2_name]))
        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]
        
        # 获取梯度历史
        grad_history1 = self.gradient_history.get(layer1_name, [])
        grad_history2 = self.gradient_history.get(layer2_name, [])
        
        if len(grad_history1) < 3 or len(grad_history2) < 3:
            return 0.0
        
        # 计算时间序列相关性
        correlations = []
        min_length = min(len(grad_history1), len(grad_history2))
        
        for i in range(min_length):
            grad1 = grad_history1[i]
            grad2 = grad_history2[i]
            
            # 调整到相同维度
            min_size = min(grad1.size(0), grad2.size(0))
            if min_size < 10:  # 梯度太小，跳过
                continue
                
            grad1_sample = grad1[:min_size]
            grad2_sample = grad2[:min_size]
            
            # 计算皮尔逊相关系数
            correlation = self._pearson_correlation(grad1_sample, grad2_sample)
            if not math.isnan(correlation):
                correlations.append(abs(correlation))  # 使用绝对值
        
        # 计算平均相关性
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        # 缓存结果
        self.correlation_cache[cache_key] = avg_correlation
        
        return avg_correlation
    
    def _pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """计算皮尔逊相关系数"""
        try:
            # 标准化
            x_centered = x - torch.mean(x)
            y_centered = y - torch.mean(y)
            
            # 计算相关系数
            numerator = torch.sum(x_centered * y_centered)
            denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))
            
            if denominator > 1e-8:
                correlation = numerator / denominator
                return correlation.item()
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Correlation calculation failed: {e}")
            return 0.0
    
    def find_beneficial_connections(self, layer_names: List[str]) -> List[Dict]:
        """找到有益的连接候选"""
        beneficial_connections = []
        
        for i, layer1 in enumerate(layer_names):
            for j, layer2 in enumerate(layer_names):
                if i >= j:  # 避免重复和自连接
                    continue
                
                # 计算相关性
                correlation = self.calculate_layer_correlation(layer1, layer2)
                
                # 判断是否为有益连接
                if correlation > self.correlation_threshold:
                    # 分析连接类型和层间距离
                    layer_distance = abs(i - j)
                    connection_type = self._determine_connection_type(layer1, layer2, layer_distance)
                    
                    beneficial_connections.append({
                        'source_layer': layer1,
                        'target_layer': layer2,
                        'correlation': correlation,
                        'layer_distance': layer_distance,
                        'connection_type': connection_type,
                        'priority': self._calculate_connection_priority(correlation, layer_distance)
                    })
        
        # 按优先级排序
        beneficial_connections.sort(key=lambda x: x['priority'], reverse=True)
        
        return beneficial_connections
    
    def _determine_connection_type(self, layer1: str, layer2: str, distance: int) -> str:
        """确定连接类型"""
        
        # 基于层名称和距离确定连接类型
        if distance == 1:
            return 'adjacent'  # 相邻层，通常不需要额外连接
        elif distance == 2:
            return 'skip_connection'  # 跳跃连接
        elif distance >= 3:
            if 'conv' in layer1 and 'conv' in layer2:
                return 'feature_pyramid'  # 特征金字塔连接
            elif 'fc' in layer1 and 'fc' in layer2:
                return 'dense_connection'  # 密集连接
            else:
                return 'cross_stage'  # 跨阶段连接
        else:
            return 'unknown'
    
    def _calculate_connection_priority(self, correlation: float, distance: int) -> float:
        """计算连接优先级"""
        
        # 基础优先级 = 相关性强度
        base_priority = correlation
        
        # 距离惩罚：距离越远，优先级略微降低
        distance_penalty = 1.0 / (1.0 + 0.1 * distance)
        
        # 最终优先级
        priority = base_priority * distance_penalty
        
        return priority


class ConnectionBuilder:
    """连接构建器 - 实际构建神经网络连接"""
    
    def __init__(self):
        self.built_connections = set()
        self.connection_modules = {}
        
    def build_skip_connection(self, 
                            model: nn.Module, 
                            source_layer: str, 
                            target_layer: str,
                            connection_id: str) -> bool:
        """构建跳跃连接"""
        
        try:
            # 获取源层和目标层
            source_module = self._get_module_by_name(model, source_layer)
            target_module = self._get_module_by_name(model, target_layer)
            
            if source_module is None or target_module is None:
                logger.warning(f"Cannot find modules: {source_layer} or {target_layer}")
                return False
            
            # 分析层的输出维度
            source_shape = self._get_layer_output_shape(source_module)
            target_shape = self._get_layer_output_shape(target_module)
            
            if source_shape is None or target_shape is None:
                logger.warning(f"Cannot determine shapes for {source_layer} -> {target_layer}")
                return False
            
            # 创建适配器模块
            adapter = self._create_shape_adapter(source_shape, target_shape)
            
            if adapter is not None:
                # 将适配器添加到模型
                adapter_name = f"skip_adapter_{connection_id}"
                self._add_module_to_model(model, adapter_name, adapter)
                
                # 记录连接信息
                self.built_connections.add(connection_id)
                self.connection_modules[connection_id] = {
                    'type': 'skip_connection',
                    'source': source_layer,
                    'target': target_layer,
                    'adapter': adapter_name
                }
                
                logger.info(f"Built skip connection: {source_layer} -> {target_layer}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to build skip connection {source_layer} -> {target_layer}: {e}")
            return False
        
        return False
    
    def build_attention_connection(self, 
                                 model: nn.Module, 
                                 source_layer: str, 
                                 target_layer: str,
                                 connection_id: str) -> bool:
        """构建注意力连接"""
        
        try:
            # 获取层信息
            source_module = self._get_module_by_name(model, source_layer)
            target_module = self._get_module_by_name(model, target_layer)
            
            if source_module is None or target_module is None:
                return False
            
            # 分析特征维度
            source_shape = self._get_layer_output_shape(source_module)
            target_shape = self._get_layer_output_shape(target_module)
            
            # 创建注意力模块
            attention_module = self._create_attention_module(source_shape, target_shape)
            
            if attention_module is not None:
                # 添加到模型
                attention_name = f"attention_{connection_id}"
                self._add_module_to_model(model, attention_name, attention_module)
                
                # 记录连接
                self.built_connections.add(connection_id)
                self.connection_modules[connection_id] = {
                    'type': 'attention_connection',
                    'source': source_layer,
                    'target': target_layer,
                    'attention': attention_name
                }
                
                logger.info(f"Built attention connection: {source_layer} -> {target_layer}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to build attention connection {source_layer} -> {target_layer}: {e}")
            return False
        
        return False
    
    def _get_module_by_name(self, model: nn.Module, module_name: str) -> Optional[nn.Module]:
        """根据名称获取模块"""
        try:
            # 尝试直接获取
            if hasattr(model, 'get_submodule'):
                return model.get_submodule(module_name)
            else:
                # 兼容老版本
                parts = module_name.split('.')
                module = model
                for part in parts:
                    module = getattr(module, part)
                return module
        except Exception:
            return None
    
    def _get_layer_output_shape(self, module: nn.Module) -> Optional[Tuple]:
        """获取层的输出形状"""
        if isinstance(module, nn.Conv2d):
            return ('conv', module.out_channels, module.kernel_size, module.stride, module.padding)
        elif isinstance(module, nn.Linear):
            return ('linear', module.out_features)
        elif isinstance(module, nn.BatchNorm2d):
            return ('bn2d', module.num_features)
        elif isinstance(module, nn.BatchNorm1d):
            return ('bn1d', module.num_features)
        else:
            return None
    
    def _create_shape_adapter(self, source_shape: Tuple, target_shape: Tuple) -> Optional[nn.Module]:
        """创建形状适配器"""
        
        # Conv2d 到 Conv2d 的适配
        if source_shape[0] == 'conv' and target_shape[0] == 'conv':
            source_channels = source_shape[1]
            target_channels = target_shape[1]
            
            if source_channels != target_channels:
                # 通道数适配
                return nn.Conv2d(source_channels, target_channels, kernel_size=1, bias=False)
            else:
                # 形状相同，使用恒等映射
                return nn.Identity()
        
        # Linear 到 Linear 的适配
        elif source_shape[0] == 'linear' and target_shape[0] == 'linear':
            source_features = source_shape[1]
            target_features = target_shape[1]
            
            if source_features != target_features:
                return nn.Linear(source_features, target_features, bias=False)
            else:
                return nn.Identity()
        
        # Conv2d 到 Linear 的适配 (需要全局池化)
        elif source_shape[0] == 'conv' and target_shape[0] == 'linear':
            source_channels = source_shape[1]
            target_features = target_shape[1]
            
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(source_channels, target_features, bias=False)
            )
        
        else:
            logger.warning(f"Unsupported shape adaptation: {source_shape} -> {target_shape}")
            return None
    
    def _create_attention_module(self, source_shape: Tuple, target_shape: Tuple) -> Optional[nn.Module]:
        """创建注意力模块"""
        
        # 简化的注意力机制
        if source_shape[0] == 'conv' and target_shape[0] == 'conv':
            source_channels = source_shape[1]
            target_channels = target_shape[1]
            
            # 通道注意力
            return ChannelAttention(source_channels, target_channels)
        
        elif source_shape[0] == 'linear' and target_shape[0] == 'linear':
            source_features = source_shape[1]
            target_features = target_shape[1]
            
            # 特征注意力
            return FeatureAttention(source_features, target_features)
        
        else:
            return None
    
    def _add_module_to_model(self, model: nn.Module, module_name: str, module: nn.Module) -> None:
        """向模型添加新模块"""
        # 将模块添加到模型的一个特殊容器中
        if not hasattr(model, '_dnm_connections'):
            model._dnm_connections = nn.ModuleDict()
        
        model._dnm_connections[module_name] = module


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, source_channels: int, target_channels: int, reduction_ratio: int = 16):
        super().__init__()
        
        self.source_channels = source_channels
        self.target_channels = target_channels
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 注意力网络
        hidden_channels = max(source_channels // reduction_ratio, 1)
        self.attention_net = nn.Sequential(
            nn.Linear(source_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, target_channels),
            nn.Sigmoid()
        )
        
        # 通道适配器
        if source_channels != target_channels:
            self.channel_adapter = nn.Conv2d(source_channels, target_channels, 1, bias=False)
        else:
            self.channel_adapter = nn.Identity()
    
    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        # 计算注意力权重
        b, c, h, w = source_features.size()
        
        # 全局池化得到通道描述符
        channel_descriptor = self.global_avg_pool(source_features).view(b, c)
        
        # 计算注意力权重
        attention_weights = self.attention_net(channel_descriptor).view(b, self.target_channels, 1, 1)
        
        # 适配源特征的通道数
        adapted_source = self.channel_adapter(source_features)
        
        # 应用注意力权重
        attended_features = adapted_source * attention_weights
        
        # 与目标特征融合
        if attended_features.shape == target_features.shape:
            return target_features + attended_features
        else:
            # 尺寸不匹配时使用自适应池化
            attended_features = F.adaptive_avg_pool2d(attended_features, target_features.shape[2:])
            return target_features + attended_features


class FeatureAttention(nn.Module):
    """特征注意力模块"""
    
    def __init__(self, source_features: int, target_features: int):
        super().__init__()
        
        self.source_features = source_features
        self.target_features = target_features
        
        # 注意力权重计算
        hidden_size = max(min(source_features, target_features) // 4, 1)
        self.attention_net = nn.Sequential(
            nn.Linear(source_features, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, target_features),
            nn.Sigmoid()
        )
        
        # 特征适配器
        if source_features != target_features:
            self.feature_adapter = nn.Linear(source_features, target_features, bias=False)
        else:
            self.feature_adapter = nn.Identity()
    
    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        # 计算注意力权重
        attention_weights = self.attention_net(source_features)
        
        # 适配源特征
        adapted_source = self.feature_adapter(source_features)
        
        # 应用注意力
        attended_features = adapted_source * attention_weights
        
        # 与目标特征融合
        return target_features + attended_features


class DNMConnectionGrowth:
    """DNM连接生长主控制器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.analyzer = GradientCorrelationAnalyzer(**self.config['analyzer'])
        self.builder = ConnectionBuilder()
        self.connection_statistics = defaultdict(int)
        self.growth_history = []
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'analyzer': {
                'correlation_threshold': 0.15,
                'history_length': 8
            },
            'growth': {
                'max_new_connections': 3,
                'min_correlation_threshold': 0.1,
                'growth_frequency': 8,  # 每8个epoch分析一次
                'connection_types': ['skip_connection', 'attention_connection']
            },
            'filtering': {
                'min_layer_distance': 2,  # 最小层间距离
                'max_layer_distance': 6,  # 最大层间距离
                'avoid_redundant_connections': True
            }
        }
    
    def collect_and_analyze_gradients(self, model: nn.Module) -> None:
        """收集并分析梯度"""
        # 收集当前梯度
        gradients = self.analyzer.collect_gradients(model)
        
        # 更新梯度历史
        self.analyzer.update_gradient_history(gradients)
        
        logger.debug(f"Collected gradients from {len(gradients)} layers")
    
    def analyze_and_grow_connections(self, model: nn.Module, epoch: int) -> Dict[str, Any]:
        """分析并生长新连接"""
        
        # 检查是否到了生长时机
        if epoch % self.config['growth']['growth_frequency'] != 0:
            return {'connections_grown': 0, 'message': 'Not growth epoch'}
        
        # 获取所有层名称
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and any(t in name for t in ['conv', 'fc', 'linear']):
                layer_names.append(name)
        
        if len(layer_names) < 2:
            return {'connections_grown': 0, 'message': 'Not enough layers for connections'}
        
        # 找到有益连接
        beneficial_connections = self.analyzer.find_beneficial_connections(layer_names)
        
        # 过滤连接
        filtered_connections = self._filter_connections(beneficial_connections)
        
        # 执行连接生长
        connections_grown = self._grow_connections(model, filtered_connections)
        
        # 记录生长历史
        growth_record = {
            'epoch': epoch,
            'candidates_found': len(beneficial_connections),
            'connections_grown': connections_grown,
            'grown_connections': [conn for conn in filtered_connections[:connections_grown]]
        }
        self.growth_history.append(growth_record)
        
        # 更新统计
        for conn in filtered_connections[:connections_grown]:
            self.connection_statistics[conn['connection_type']] += 1
        
        result = {
            'connections_grown': connections_grown,
            'beneficial_connections_found': len(beneficial_connections),
            'connection_candidates': filtered_connections[:5],  # 前5个候选
            'growth_history': growth_record,
            'message': f'Successfully grown {connections_grown} connections'
        }
        
        logger.info(f"DNM Connection Growth completed: {connections_grown} connections grown")
        return result
    
    def _filter_connections(self, connections: List[Dict]) -> List[Dict]:
        """过滤连接候选"""
        filtered = []
        
        min_distance = self.config['filtering']['min_layer_distance']
        max_distance = self.config['filtering']['max_layer_distance']
        max_connections = self.config['growth']['max_new_connections']
        
        for conn in connections:
            # 距离过滤
            if not (min_distance <= conn['layer_distance'] <= max_distance):
                continue
            
            # 相关性过滤
            if conn['correlation'] < self.config['growth']['min_correlation_threshold']:
                continue
            
            # 避免重复连接
            connection_id = f"{conn['source_layer']}_to_{conn['target_layer']}"
            if self.config['filtering']['avoid_redundant_connections']:
                if connection_id in self.builder.built_connections:
                    continue
            
            # 添加连接ID
            conn['connection_id'] = connection_id
            filtered.append(conn)
            
            # 限制数量
            if len(filtered) >= max_connections:
                break
        
        return filtered
    
    def _grow_connections(self, model: nn.Module, connections: List[Dict]) -> int:
        """执行连接生长"""
        grown_count = 0
        
        for conn in connections:
            try:
                connection_type = conn['connection_type']
                connection_id = conn['connection_id']
                source_layer = conn['source_layer']
                target_layer = conn['target_layer']
                
                success = False
                
                # 根据连接类型生长
                if connection_type in ['skip_connection', 'dense_connection']:
                    success = self.builder.build_skip_connection(
                        model, source_layer, target_layer, connection_id
                    )
                elif connection_type in ['feature_pyramid', 'cross_stage']:
                    success = self.builder.build_attention_connection(
                        model, source_layer, target_layer, connection_id
                    )
                
                if success:
                    grown_count += 1
                    logger.info(f"Successfully grown {connection_type}: {source_layer} -> {target_layer}")
                else:
                    logger.warning(f"Failed to grow connection: {source_layer} -> {target_layer}")
                    
            except Exception as e:
                logger.error(f"Error growing connection {conn.get('connection_id', 'unknown')}: {e}")
                continue
        
        return grown_count
    
    def get_growth_summary(self) -> Dict[str, Any]:
        """获取连接生长总结"""
        return {
            'total_connections_grown': sum(self.connection_statistics.values()),
            'connections_by_type': dict(self.connection_statistics),
            'growth_history': self.growth_history,
            'built_connections': list(self.builder.built_connections),
            'connection_modules': self.builder.connection_modules,
            'config': self.config
        }


# 测试函数
def test_connection_growth():
    """测试连接生长功能"""
    print("🔗 Testing DNM Connection Growth")
    
    # 创建测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc1 = nn.Linear(128 * 4 * 4, 256)
            self.fc2 = nn.Linear(256, 10)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = TestModel()
    connection_growth = DNMConnectionGrowth()
    
    # 模拟训练过程，收集梯度
    dummy_input = torch.randn(8, 3, 32, 32)
    dummy_target = torch.randint(0, 10, (8,))
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(16):
        # 前向传播
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        
        # 反向传播
        model.zero_grad()
        loss.backward()
        
        # 收集梯度
        connection_growth.collect_and_analyze_gradients(model)
        
        # 尝试生长连接
        if epoch % 8 == 0 and epoch > 0:
            result = connection_growth.analyze_and_grow_connections(model, epoch)
            print(f"Epoch {epoch}: {result}")
    
    # 获取总结
    summary = connection_growth.get_growth_summary()
    print(f"Growth summary: {summary}")
    
    return model, connection_growth


if __name__ == "__main__":
    test_connection_growth()