# DNM分支合并策略

## 🔍 当前状况

- **cursor/dnm-bbe5分支**: 基于多理论支撑的重构版本，专注于解决性能瓶颈
- **main分支**: 已包含不同的DNM实现，更综合的框架

## 🧬 两个版本的主要差异

### cursor/dnm-bbe5版本 (多理论支撑版本)
- **5种理论触发器**: 信息论、生物学原理、动力学系统、认知科学、网络科学
- **智能分裂策略**: 对称分裂、非对称分裂、功能分裂
- **设备兼容性**: 针对GPU/CPU设备不匹配问题的专门修复
- **专注目标**: 突破90%准确率瓶颈

### main分支版本 (综合框架版本)
- **三大核心模块**: 神经元分裂、连接生长、多目标优化
- **统一训练接口**: 更完整的训练流程集成
- **数学优化**: 包含Pareto优化等高级算法

## 🚀 推荐合并策略

### 选项1: 双版本共存（推荐）
将我们的版本重命名为`DNMMultiTheoryFramework`，与现有的`DNMFramework`共存：

```python
# 现有版本 (综合框架)
from neuroexapt.core.dnm_framework import DNMFramework

# 我们的版本 (多理论支撑)
from neuroexapt.core.dnm_multi_theory_framework import DNMMultiTheoryFramework
```

### 选项2: 功能集成
将我们的多理论触发器集成到现有的DNM框架中作为一个新的组件。

### 选项3: 替换升级
完全用我们的版本替换现有的实现（风险较高）。

## 📝 实施步骤（推荐选项1）

1. **重命名文件**:
   ```bash
   mv neuroexapt/core/dnm_framework.py neuroexapt/core/dnm_multi_theory_framework.py
   mv neuroexapt/core/dnm_neuron_division.py neuroexapt/core/dnm_multi_theory_division.py
   ```

2. **更新类名**:
   - `DNMFramework` → `DNMMultiTheoryFramework`
   - `AdaptiveNeuronDivision` → `MultiTheoryNeuronDivision`

3. **更新导入和文档**:
   - 修改测试文件的导入
   - 添加版本说明文档

4. **合并到main分支**:
   ```bash
   git checkout main
   git merge cursor/dnm-bbe5 --no-ff
   ```

## 🎯 优势

- **保持兼容性**: 不破坏现有的代码
- **功能互补**: 两个版本可以针对不同的使用场景
- **持续开发**: 可以分别优化两个版本

## 📋 待解决问题

1. **设备一致性**: 确保所有版本都能正确处理GPU/CPU切换
2. **性能对比**: 需要测试两个版本的性能差异
3. **文档统一**: 更新文档以说明两个版本的差异和使用场景

## 🔧 当前优先级

1. **先修复设备问题**: 确保当前版本可以稳定运行
2. **性能验证**: 确认能突破90%准确率
3. **再考虑合并**: 在功能验证后进行合并