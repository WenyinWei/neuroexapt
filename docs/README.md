# NeuroExapt Documentation

## 🧬 Welcome to DNM Framework Documentation

NeuroExapt现在基于**Dynamic Neural Morphogenesis (DNM) 框架**，提供革命性的神经网络自适应架构进化能力。

## 📚 Documentation Structure

### 🎓 循序渐进的学习路径

#### 🌱 初学者入门
1. **[快速入门](getting_started.md)** - 5分钟上手体验
2. **[DNM核心原理](dnm_principles.md)** - 理解生物学启发的设计思想
3. **[基础示例](../examples/)** - 实际代码示例

#### 🌿 进阶开发者
1. **[智能增长机制](intelligent_growth.md)** - 深度理解一步到位的架构优化
2. **[理论基础](theory.md)** - 数学原理与证明
3. **[符号说明](symbols.md)** - 完整的数学符号参考

#### 🌳 专家级应用
1. **[API文档](html/index.html)** - 完整的API参考
2. **[性能调优](../benchmarks/)** - 大规模部署优化
3. **[贡献指南](https://github.com/neuroexapt/neuroexapt)** - 开发和贡献

## 🚀 Quick Start

### 最简单的开始方式

```python
from neuroexapt.core.dnm_framework import train_with_dnm
import torch.nn as nn

# 创建基础模型
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.Linear(32*32*32, 10)
)

# 🧬 一行代码启动智能训练
result = train_with_dnm(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    target_accuracy=95.0
)
```

## 🔧 Documentation Generation

### 生成完整文档

```bash
# 安装依赖
pip install doxygen sphinx

# 生成HTML文档
python generate_docs.py

# 或直接使用doxygen
doxygen Doxyfile
```

### 文档内容说明

| 文档 | 内容 | 适合人群 |
|------|------|----------|
| `mainpage.md` | 项目概览和特性介绍 | 所有用户 |
| `getting_started.md` | 快速入门指南 | 新手 |
| `dnm_principles.md` | DNM核心原理详解 | 深度理解者 |
| `intelligent_growth.md` | 智能增长机制 | 高级用户 |
| `theory.md` | 数学理论基础 | 研究者 |
| `symbols.md` | 符号参考手册 | 开发者 |

## 🧹 文档清理说明

本次清理删除了以下过时内容：

### 已删除的过时文档
- **ASO-SE框架相关文档** (30+ 个文件) - 已被DNM框架完全替代
- **临时修复文档** (20+ 个文件) - 问题已解决，不再需要
- **重复的性能优化文档** - 内容已整合到主文档
- **碎片化的特性介绍** - 已整合为阶梯式学习路径

### 保留并重构的内容
- **核心算法原理** → 整合到`dnm_principles.md`
- **使用指南** → 重构为`getting_started.md`
- **高级特性** → 整合到`intelligent_growth.md`
- **理论基础** → 保持并增强`theory.md`

## 🌟 新文档特色

### 🎯 循循善诱的阶梯式设计
- **渐进式学习路径** - 从基础到专家级
- **实用代码示例** - 每个概念都有可执行代码
- **中英文混合** - 适合中文用户的专业表达

### 🧬 突出DNM框架优势
- **生物学启发** - 强调神经网络"生长"概念
- **一步到位** - 突出智能增长的革命性
- **性能对比** - 展示相比传统方法的优势

### 📊 实用性优先
- **快速上手** - 5分钟体验核心功能
- **实际案例** - 真实的性能提升数据
- **故障排除** - 常见问题和解决方案

## 🔗 相关资源

- **GitHub仓库**: [neuroexapt/neuroexapt](https://github.com/neuroexapt/neuroexapt)
- **在线文档**: [docs.neuroexapt.org](https://docs.neuroexapt.org)
- **示例代码**: [examples目录](../examples/)
- **性能测试**: [benchmarks目录](../benchmarks/)

---

*🧬 让神经网络像生物大脑一样智能进化！*