# DNM Framework Examples

## 🧬 Dynamic Neural Morphogenesis (DNM) 框架示例

这个目录包含了DNM框架的使用示例，从简单测试到CIFAR-10基准测试。

### 📁 文件说明

1. **`dnm_simple_test.py`** - 简化的DNM测试
   - 修复了原始代码中的问题
   - 使用简单CNN模型验证DNM功能
   - 适合快速测试和调试

2. **`cifar10_dnm_benchmark.py`** - CIFAR-10基准测试
   - 使用ResNet-18变体作为初始架构
   - 目标：95%准确率
   - 完整的训练流程和性能分析

### 🚀 快速开始

#### 1. 简单测试

```bash
cd examples
python dnm_simple_test.py
```

这将运行一个30个epoch的简化测试，验证：
- DNM框架是否正常工作
- 神经元分裂是否发生
- 连接生长是否有效
- 模型参数是否实际增长

#### 2. CIFAR-10基准测试

```bash
cd examples
python cifar10_dnm_benchmark.py
```

这将运行完整的CIFAR-10基准测试：
- 200个epoch的训练
- 使用强化的数据增强
- 目标95%验证准确率
- 完整的DNM演化分析

### 🔧 解决的问题

相比原始的`dynamic_neural_morphogenesis.py`，新的示例解决了：

1. **`view size is not compatible`错误**
   - 增强的形状兼容性检查
   - 安全的tensor重塑操作
   - 更好的错误处理

2. **架构实际未演化**
   - 降低了分裂和连接阈值
   - 增加了演化频率
   - 更激进的配置参数

3. **性能不理想**
   - 优化的网络架构
   - 更好的数据增强
   - 改进的训练策略

### 📊 预期结果

#### 简单测试 (`dnm_simple_test.py`)
- **准确率**：60-80%
- **参数增长**：5-15%
- **形态发生事件**：3-8个
- **训练时间**：2-5分钟

#### CIFAR-10基准 (`cifar10_dnm_benchmark.py`)
- **准确率**：90-95%
- **参数增长**：10-30%
- **形态发生事件**：8-20个
- **训练时间**：30-60分钟

### 🎯 配置调优

#### 更激进的演化
```python
config = {
    'neuron_division': {
        'splitter': {
            'entropy_threshold': 0.3,    # 更低阈值
            'split_probability': 0.9     # 更高概率
        }
    },
    'framework': {
        'morphogenesis_frequency': 2  # 更频繁
    }
}
```

#### 更保守的演化
```python
config = {
    'neuron_division': {
        'splitter': {
            'entropy_threshold': 0.8,    # 更高阈值
            'split_probability': 0.3     # 更低概率
        }
    },
    'framework': {
        'morphogenesis_frequency': 8  # 更少频率
    }
}
```

### 🐛 调试技巧

1. **如果没有形态发生事件**：
   - 降低`entropy_threshold`和`overload_threshold`
   - 增加`split_probability`
   - 减少`morphogenesis_frequency`

2. **如果训练不稳定**：
   - 增加`inheritance_noise`的值
   - 减少`max_splits_per_layer`
   - 使用更小的学习率

3. **如果内存不足**：
   - 减少`population_size`
   - 减少batch size
   - 禁用多目标优化

### 📈 性能监控

训练过程中可以观察：

```
📈 Epoch 0: Train=15.20%, Val=18.40%, Params=425,356 (+0.0%)
📈 Epoch 6: Train=45.30%, Val=48.20%, Params=425,356 (+0.0%)
🔄 Triggering morphogenesis analysis...
   🧬 2 neuron splits executed
   🔗 1 connection grown
📈 Epoch 12: Train=62.10%, Val=65.30%, Params=438,892 (+3.2%)
```

### 🏆 成功标志

- **参数增长** > 0%：模型结构实际演化
- **形态发生事件** > 0：DNM机制正常工作
- **准确率提升**：演化带来性能改善

### 💡 最佳实践

1. **渐进式训练**：先用保守配置训练，再用激进配置演化
2. **监控资源**：注意GPU内存和训练时间
3. **保存检查点**：定期保存演化后的模型
4. **分析日志**：关注形态发生事件的触发模式

### 🔗 更多资源

- `../DNM_Framework_Integration_Guide.md` - 完整集成指南
- `../neuroexapt/core/` - DNM核心模块代码
- `../neuroexapt/math/` - 多目标优化模块

---

**让神经网络真正活起来！** 🧬✨ 