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

print(f"🎉 最终准确率: {result.final_accuracy:.2f}%")
print(f"🧬 执行了 {result.morphogenesis_events} 次形态发生")
```

## 📚 Documentation Structure

### 🎓 Learning Pathways

Choose your learning path based on your experience level:

#### 🌱 Beginners
- **@ref getting_started "Quick Start Guide"** - Get up and running in 5 minutes
- **@ref dnm_principles "DNM Core Principles"** - Understand the biological inspiration
- **@ref basic_examples "Basic Examples"** - Simple classification and regression tasks

#### 🌿 Advanced Developers
- **@ref dnm_architecture "DNM Architecture Deep Dive"** - Framework design details
- **@ref intelligent_bottleneck "Intelligent Bottleneck Detection"** - Performance analysis mechanisms
- **@ref morphogenesis_events "Morphogenesis Events"** - Network evolution process

#### 🌳 Expert Users
- **@ref custom_operators "Custom Operators"** - Extend DNM functionality
- **@ref theory_deep_dive "Theoretical Foundation"** - Mathematical principles and proofs
- **@ref performance_tuning "Performance Optimization"** - Large-scale deployment

### 🧠 Core Modules

| Module | Description | Key Classes |
|--------|-------------|-------------|
| @ref neuroexapt.core.dnm_framework | DNM core framework | DNMFramework, MorphogenesisEngine |
| @ref neuroexapt.core.intelligent_growth | Intelligent growth engine | IntelligentGrowthSolution, BottleneckAnalyzer |
| @ref neuroexapt.core.morphogenesis | Morphogenesis controller | MorphogenesisController, NeuronDivision |
| @ref neuroexapt.analysis.bottleneck | Bottleneck analysis system | BottleneckDetector, PerformanceAnalyzer |
| @ref neuroexapt.optimization.pareto | Multi-objective optimization | ParetoOptimizer, MultiObjectiveEvolution |

## 🌟 Key Features Overview

### 🧬 Biologically-Inspired Network Evolution

DNM framework draws inspiration from biological neural development:

- **Neurogenesis**: Dynamic addition of new neurons
- **Synaptogenesis**: Automatic establishment of new connections  
- **Functional Plasticity**: Net2Net smooth parameter migration
- **Specialization**: Task-based neuron differentiation

For detailed feature descriptions, see @ref core_features "Core Features".

### 📈 Performance Breakthrough

DNM consistently outperforms traditional methods. See @ref performance_benchmarks "Performance Benchmarks" for detailed comparisons and results.

### 🔧 Easy Integration

```python
# Drop-in replacement for traditional training
result = train_with_dnm(your_model, train_loader, val_loader)
```

For complete integration examples, see @ref integration_examples "Integration Examples".

## 🎓 Learning Recommendations

### 🔰 New to NeuroExapt? (Estimated time: 1-2 days)
1. Read @ref getting_started "Quick Start Guide"
2. Run `examples/basic_classification.py`
3. Understand @ref dnm_principles "DNM Core Principles"
4. Experiment with different hyperparameters

### 🎯 Ready to Go Deeper? (Estimated time: 1 week)
1. Study @ref intelligent_growth "Intelligent Growth Mechanisms"
2. Learn about @ref morphogenesis_events "Morphogenesis Events"
3. Apply DNM to your custom datasets
4. Analyze performance improvements

### 🚀 Expert-Level Application? (Estimated time: 2-4 weeks)
1. Explore @ref theory_deep_dive "Theoretical Foundation"
2. Develop @ref custom_operators "Custom Operators"
3. Deploy at scale with @ref performance_tuning "Performance Optimization"
4. Contribute improvements and feedback

## 🤝 Community & Support

- **GitHub Repository**: [neuroexapt/neuroexapt](https://github.com/neuroexapt/neuroexapt)
- **Online Demo**: [Experience DNM Framework](https://demo.neuroexapt.org)
- **Technical Blog**: [Deep Dive into DNM Principles](https://blog.neuroexapt.org)
- **Issues & Discussions**: [Get Help](https://github.com/neuroexapt/neuroexapt/issues)

---

*🧬 让神经网络像生物大脑一样生长和进化！*

*Make neural networks grow and evolve like biological brains!* 