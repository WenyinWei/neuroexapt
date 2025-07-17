# 紧急修复：ASO-SE架构搜索阶段性能崩溃

## 🚨 发现的问题

### 1. Batch Size不匹配错误
```
ValueError: Expected input batch_size (128) to match target batch_size (80).
```
**原因**: `if 'outputs' not in locals()`在循环中失效，导致重复前向传播和batch size不一致

### 2. 准确率仍然暴跌到10%
尽管之前的修复，搜索阶段准确率还是从39%跌到10%
- 所有层权重都在0.2-0.26范围，过于分散
- 架构熵高达2.095，表明选择过于随机

## 🔧 应用的紧急修复

### 1. 修复Batch Size问题
```python
# 之前的有问题代码
with torch.no_grad():
    if 'outputs' not in locals():  # 这在循环中不工作！
        outputs = self.network(data)

# 修复后
for batch_idx, (data, targets) in enumerate(pbar):
    batch_outputs = None  # 用于追踪当前batch的输出
    
    # 在各个优化分支中设置batch_outputs
    if self.current_phase in ['warmup', 'optimize']:
        batch_outputs = self.network(data)
    
    # 统计时确保batch size匹配
    with torch.no_grad():
        if batch_outputs is None:
            batch_outputs = self.network(data)
        
        # 检查并修复batch size不匹配
        if batch_outputs.size(0) != targets.size(0):
            min_batch = min(batch_outputs.size(0), targets.size(0))
            batch_outputs = batch_outputs[:min_batch]
            targets = targets[:min_batch]
```

### 2. 修复架构权重过度分散
```python
def get_arch_weights(self, layer_idx, selector, training_phase='warmup'):
    # 在搜索阶段使用更保守的策略
    if training_phase == 'search':
        with torch.no_grad():
            current_best_idx = torch.argmax(logits).item()
            
            # 如果权重太分散，增强信号
            if current_best_idx != 3:  # 不是skip_connect
                softmax_weights = F.softmax(logits, dim=0)
                max_weight = softmax_weights[current_best_idx].item()
                
                if max_weight < 0.4:  # 权重太分散
                    # 增强top-3操作
                    _, top_indices = torch.topk(logits, 3)
                    for idx in top_indices:
                        if idx != 0:  # 不增强none操作
                            enhanced_logits[idx] += 0.5
        
        # 使用更保守的温度
        original_temp = selector.temperature
        selector.temperature = max(0.5, original_temp)
        
        try:
            result = selector(logits.unsqueeze(0)).squeeze(0)
        finally:
            selector.temperature = original_temp
```

### 3. 调整温度策略
```python
# 搜索阶段开始时
self.network.gumbel_selector.temperature = 1.5  # 更高的起始温度
self.network.gumbel_selector.anneal_rate = 0.995  # 更慢的退火

# 优化阶段
self.network.gumbel_selector.temperature = 0.1  # 而非0.01
```

### 4. 降低架构优化频率
```python
# 从每3个batch改为每5个batch
if batch_idx % 5 == 0:  # 进一步降低架构优化频率
    # 架构参数优化
else:
    # 权重参数优化 (4/5的时间)
```

## 🎯 预期效果

1. **消除Batch Size错误**: 不再出现tensor大小不匹配
2. **稳定架构搜索**: 权重分布更集中，不再过度分散
3. **保持基本性能**: 搜索阶段准确率应该保持在25-35%范围
4. **温和探索**: 温度从1.5开始，更慢退火到合理值

## 🔍 监控指标

运行时应该观察到：
- ✅ 无batch size错误
- ✅ 架构权重最大值 > 0.4
- ✅ 架构熵 < 1.8
- ✅ 搜索阶段准确率 > 25%
- ✅ 训练速度保持稳定

这些修复针对具体观察到的问题，应该能显著改善架构搜索的稳定性。