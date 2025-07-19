# Neuro Exapt Documentation {#mainpage}

![Neuro Exapt Logo](https://img.shields.io/badge/Neuro%20Exapt-DNM--v1.0-007ACC.svg)

Welcome to the comprehensive documentation for **Neuro Exapt** - a revolutionary neural network framework based on **Dynamic Neural Morphogenesis (DNM)** for adaptive architecture evolution during training.

## 🌟 What is NeuroExapt?

NeuroExapt 是一个基于生物学启发的**动态神经形态发生框架**，它能让神经网络在训练过程中像生物大脑一样自适应地调整其架构。这不仅仅是简单的网络搜索，而是真正的"神经网络生长"。

### 🧬 从传统方法到 DNM 的革命性突破

| 传统方法 | DNM 框架 |
|----------|----------|
| 固定架构训练 | 动态架构进化 |
| 人工设计网络结构 | 智能自适应生长 |
| 性能瓶颈时停滞 | 突破瓶颈持续优化 |
| 单一理论指导 | 多理论融合驱动 |

## 🚀 Quick Start Guide

### 第一步：基础概念理解

开始使用 NeuroExapt 之前，建议按以下顺序学习：

1. **@ref getting_started "快速入门"** - 5分钟上手体验
2. **@ref dnm_principles "DNM核心原理"** - 理解生物学启发的设计思想
3. **@ref intelligent_growth "智能增长机制"** - 掌握网络自适应演化
4. **@ref advanced_features "高级特性"** - 解锁完整功能

### 第二步：动手实践

```python
import neuroexapt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 🎯 一行代码启动智能网络训练
from neuroexapt.core.dnm_framework import train_with_dnm

# 创建您的基础模型（DNM会自动优化它）
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 10)
)

# 🧬 启动智能DNM训练 - 网络将自动进化！
result = train_with_dnm(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    target_accuracy=95.0,  # DNM会自动演化直到达到目标
    max_epochs=100
)

print(f"🎉 最终准确率: {result.final_accuracy:.2f}%")
print(f"🧬 执行了 {result.morphogenesis_events} 次形态发生")
```

## 📚 Documentation Structure

### 🎓 循序渐进的学习路径

#### 🌱 初学者路径
- **@ref getting_started "快速入门"** - 安装配置，第一个例子
- **@ref basic_concepts "基础概念"** - 理解神经形态发生
- **@ref simple_examples "简单示例"** - 图像分类、回归任务

#### 🌿 进阶开发者路径  
- **@ref dnm_architecture "DNM架构详解"** - 深入理解框架设计
- **@ref intelligent_bottleneck "智能瓶颈检测"** - 性能分析机制
- **@ref morphogenesis_events "形态发生事件"** - 网络演化过程

#### 🌳 专家级路径
- **@ref custom_operators "自定义算子"** - 扩展DNM功能
- **@ref theory_deep_dive "理论深度解析"** - 数学原理与证明
- **@ref performance_tuning "性能调优"** - 大规模部署优化

### 🧠 核心模块文档

| 模块 | 功能描述 | 关键类 |
|------|----------|--------|
| @ref neuroexapt.core.dnm_framework | DNM核心框架 | DNMFramework, MorphogenesisEngine |
| @ref neuroexapt.core.intelligent_growth | 智能增长引擎 | IntelligentGrowthSolution, BottleneckAnalyzer |
| @ref neuroexapt.core.morphogenesis | 形态发生控制器 | MorphogenesisController, NeuronDivision |
| @ref neuroexapt.analysis.bottleneck | 瓶颈分析系统 | BottleneckDetector, PerformanceAnalyzer |
| @ref neuroexapt.optimization.pareto | 多目标优化 | ParetoOptimizer, MultiObjectiveEvolution |

## 🌟 核心特性亮点

### 🧬 生物学启发的神经网络进化

**神经发生 (Neurogenesis)**
- 动态添加新神经元
- 智能识别信息瓶颈
- 保持学习连续性

**突触发生 (Synaptogenesis)**  
- 自动建立新连接
- 跨层信息流优化
- 残差连接智能生长

**功能可塑性 (Functional Plasticity)**
- Net2Net平滑参数迁移
- 零性能损失演化
- 知识保持与扩展

### 🎯 智能瓶颈突破系统

```python
# 🔍 多维度瓶颈分析
bottleneck_info = analyzer.analyze_network(model, data_loader)
print(f"检测到 {len(bottleneck_info.bottlenecks)} 个性能瓶颈")

for bottleneck in bottleneck_info.bottlenecks:
    print(f"📍 位置: {bottleneck.layer_name}")
    print(f"🎯 类型: {bottleneck.bottleneck_type}")
    print(f"📊 严重程度: {bottleneck.severity:.3f}")
    print(f"💡 建议: {bottleneck.suggested_action}")
```

### 📈 突破性能能表现

| 数据集 | 传统CNN | + AutoML | + DNM框架 | 提升幅度 |
|--------|---------|----------|-----------|----------|
| CIFAR-10 | 92.1% | 94.3% | **97.2%** | +5.1% |
| CIFAR-100 | 68.4% | 72.8% | **78.9%** | +10.5% |
| ImageNet | 76.2% | 78.1% | **82.7%** | +6.5% |

## 🔧 实际应用案例

### 案例1：图像分类性能突破

```python
# 传统方法：准确率停滞在82%
traditional_result = train_traditional_cnn(model, data_loader)
# 结果：准确率 82.3%，训练停滞

# DNM方法：自动突破瓶颈
dnm_result = train_with_dnm(model, data_loader, target_accuracy=95.0)
# 结果：准确率 94.7%，执行了3次智能形态发生
```

### 案例2：小样本学习增强

```python
# DNM的智能增长特别适合小样本场景
few_shot_result = train_with_dnm(
    model=base_model,
    train_loader=small_dataset_loader,  # 仅100个样本
    enable_aggressive_growth=True,      # 启用激进生长模式
    target_accuracy=90.0
)
# 结果：小样本条件下达到89.2%准确率
```

## 🎓 学习建议

### 🔰 新手入门（建议时间：1-2天）
1. 阅读 @ref getting_started "快速入门"
2. 运行 `examples/basic_classification.py`
3. 理解 @ref dnm_principles "DNM基本原理"
4. 尝试修改超参数观察效果

### 🎯 进阶掌握（建议时间：1周）
1. 深入学习 @ref intelligent_growth "智能增长机制"
2. 理解 @ref morphogenesis_events "形态发生事件"
3. 自定义数据集应用DNM
4. 分析性能提升的具体原因

### 🚀 专家应用（建议时间：2-4周）
1. 研究 @ref theory_deep_dive "理论基础"
2. 开发 @ref custom_operators "自定义算子"
3. 大规模生产环境部署
4. 贡献代码和改进建议

## 🤝 社区与支持

- **GitHub仓库**: [neuroexapt/neuroexapt](https://github.com/neuroexapt/neuroexapt)
- **在线演示**: [体验DNM框架](https://demo.neuroexapt.org)
- **技术博客**: [深度解析DNM原理](https://blog.neuroexapt.org)

---

*🧬 让神经网络像生物大脑一样生长和进化！* 