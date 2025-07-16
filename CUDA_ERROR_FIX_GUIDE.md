# 🔧 CUDA Runtime Error 修复指南

## 问题描述

在运行分离训练时出现 CUDA runtime error：
```
RuntimeError: CUDA error: unknown error
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
```

## 根本原因

该错误主要由以下原因引起：
1. **Triton内核兼容性问题**：Triton sepconv内核与当前CUDA环境不兼容
2. **GPU内存碎片化**：长时间训练导致的内存管理问题
3. **WSL2环境限制**：WSL2下的CUDA驱动可能存在稳定性问题

## 🚀 快速修复方案

### 方案1：使用安全模式（推荐）
```bash
python examples/basic_classification.py \
    --mode separated \
    --safe_mode \
    --epochs 50 \
    --batch_size 32 \
    --init_channels 16 \
    --layers 8
```

### 方案2：禁用Triton内核
```bash
python examples/basic_classification.py \
    --mode separated \
    --disable_triton \
    --force_pytorch_sepconv \
    --epochs 50 \
    --batch_size 32
```

### 方案3：保守配置
```bash
python examples/basic_classification.py \
    --mode separated \
    --safe_mode \
    --batch_size 16 \
    --init_channels 12 \
    --layers 6 \
    --epochs 30 \
    --use_checkpoint
```

## 📋 新增安全选项说明

### `--safe_mode`
- **功能**：启用全面安全模式
- **效果**：
  - 禁用Triton内核
  - 强制使用PyTorch sepconv
  - 禁用模型编译
  - 禁用高风险优化

### `--disable_triton`
- **功能**：仅禁用Triton内核
- **适用**：只想禁用Triton但保留其他优化

### `--force_pytorch_sepconv`
- **功能**：强制使用PyTorch sepconv实现
- **适用**：专门解决sepconv相关的CUDA错误

## 🎯 推荐的训练配置

### 小规模测试（快速验证）
```bash
python examples/basic_classification.py \
    --mode separated \
    --safe_mode \
    --epochs 10 \
    --batch_size 16 \
    --init_channels 8 \
    --layers 4 \
    --weight_epochs 2 \
    --arch_epochs 1
```

### 中规模训练（平衡性能与稳定性）
```bash
python examples/basic_classification.py \
    --mode separated \
    --safe_mode \
    --epochs 80 \
    --batch_size 32 \
    --init_channels 20 \
    --layers 12 \
    --weight_epochs 4 \
    --arch_epochs 1 \
    --warmup_epochs 10
```

### 大规模训练（稳定性优先）
```bash
python examples/basic_classification.py \
    --mode separated \
    --safe_mode \
    --epochs 150 \
    --batch_size 48 \
    --init_channels 28 \
    --layers 16 \
    --weight_epochs 5 \
    --arch_epochs 1 \
    --warmup_epochs 15 \
    --train_portion 0.8
```

## 🔧 修复实现原理

### 1. Triton内核禁用
```python
# 在setup_environment中自动禁用
if args.safe_mode or args.disable_triton:
    sepconv_module._TRITON_DISABLED = True
```

### 2. 强制PyTorch回退
```python
# 在operations.py中安全回退
if args.force_pytorch_sepconv or args.safe_mode:
    ops._SEPCONV_TRITON_SAFE = False
```

### 3. 错误处理增强
- 多层try-catch保护
- 自动CUDA缓存清理
- Graceful fallback机制

## 📊 性能影响预期

| 模式 | 训练速度 | 稳定性 | 内存使用 | 推荐场景 |
|------|----------|--------|----------|----------|
| **normal** | 100% | ⚠️ 中等 | 标准 | 测试环境 |
| **safe_mode** | 85-90% | ✅ 高 | 更高 | 生产训练 |
| **disable_triton** | 90-95% | ✅ 高 | 标准 | 轻度修复 |

## 🐛 故障排除

### 如果安全模式仍有问题
1. **减小batch size**：从32降到16或8
2. **减少层数**：从16层降到8层
3. **使用checkpoint**：添加`--use_checkpoint`
4. **增加内存清理**：每50步清理一次GPU缓存

### 验证修复是否生效
```bash
# 运行5分钟测试
timeout 300 python examples/basic_classification.py \
    --mode separated \
    --safe_mode \
    --epochs 5 \
    --batch_size 16 \
    --quiet
```

### 监控GPU状态
```bash
# 在另一个终端监控
watch -n 1 nvidia-smi
```

## 🚀 从安全模式过渡到优化模式

### 阶段1：验证稳定性（安全模式）
```bash
python examples/basic_classification.py --mode separated --safe_mode --epochs 10
```

### 阶段2：逐步启用优化
```bash
python examples/basic_classification.py --mode separated --disable_triton --epochs 20
```

### 阶段3：尝试标准模式（如果环境改善）
```bash
python examples/basic_classification.py --mode separated --epochs 30
```

## ✅ 修复验证清单

- [ ] 能够成功启动训练（前10个epoch）
- [ ] 没有CUDA runtime error
- [ ] GPU内存使用稳定
- [ ] 训练loss正常下降
- [ ] 验证准确率正常提升
- [ ] 能够完成完整训练周期

## 💡 长期解决方案

1. **升级PyTorch**：等待更稳定的Triton支持
2. **改善WSL2环境**：升级CUDA驱动和WSL2
3. **使用Docker**：考虑在容器中运行以获得更好的隔离
4. **迁移到Linux**：原生Linux环境具有更好的CUDA支持

## 📞 技术支持

如果问题仍然存在，请提供以下信息：
- 使用的命令行参数
- 完整的错误堆栈
- GPU型号和CUDA版本
- WSL2版本信息

---

**总结**：使用 `--safe_mode` 是解决CUDA错误的最可靠方案，虽然可能略微降低性能，但能确保训练的稳定性和可靠性。 