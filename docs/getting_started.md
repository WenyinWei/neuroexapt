# NeuroExapt 快速入门指南 {#getting_started}

## 🚀 5分钟上手体验

### 安装配置

```bash
# 1. 安装NeuroExapt
pip install neuroexapt

# 或从源码安装（推荐获取最新特性）
git clone https://github.com/neuroexapt/neuroexapt.git
cd neuroexapt
pip install -e .

# 2. 验证安装
python -c "import neuroexapt; print('✅ NeuroExapt安装成功!')"
```

### 第一个DNM示例

让我们用一个简单的图像分类任务体验DNM的威力：

```python
# basic_dnm_example.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 🧬 导入DNM框架
from neuroexapt.core.dnm_framework import train_with_dnm

# 1. 准备数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

val_dataset = datasets.CIFAR10(root='./data', train=False, 
                              download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# 2. 创建基础模型（DNM将自动优化它！）
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 10)
)

# 3. 🎯 一行代码启动DNM训练！
print("🧬 启动DNM智能训练...")
result = train_with_dnm(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    target_accuracy=92.0,  # 目标准确率
    max_epochs=50,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 4. 查看结果
print(f"🎉 训练完成!")
print(f"📊 最终准确率: {result.final_accuracy:.2f}%")
print(f"🧬 执行的形态发生次数: {result.morphogenesis_count}")
print(f"⏱️  训练轮数: {result.total_epochs}")

# 查看演化历史
for event in result.morphogenesis_events:
    print(f"🌱 Epoch {event.epoch}: {event.type} - {event.description}")
```

**运行结果示例：**
```
🧬 启动DNM智能训练...
Epoch 1-15: 基础训练阶段...
📊 当前准确率: 78.3%

🧬 检测到性能瓶颈，执行形态发生...
🌱 Epoch 16: 神经元分裂 - Conv2d层从64→96通道
📊 形态发生后准确率: 84.1% (+5.8%)

Epoch 17-28: 稳定提升阶段...
📊 当前准确率: 89.7%

🧬 检测到梯度流问题，执行形态发生...
🌱 Epoch 29: 残差连接添加 - 跨层连接改善梯度流
📊 形态发生后准确率: 92.4% (+2.7%)

🎉 训练完成!
📊 最终准确率: 92.4%
🧬 执行的形态发生次数: 2
⏱️  训练轮数: 32
```

## 📦 核心特性体验

### 1. 🔍 自动批量大小优化

DNM会自动找到适合您GPU的最优批量大小：

```python
# 第一次运行 - 自动优化批量大小
python examples/basic_classification.py
# 输出: "🔍 寻找最优批量大小..."
# 输出: "✅ 最优批量大小: 928"
# 输出: "💾 已缓存最优批量大小供将来使用"

# 第二次运行 - 使用缓存值
python examples/basic_classification.py  
# 输出: "📦 使用缓存的最优批量大小: 928"
# 输出: "   GPU: NVIDIA GeForce RTX 3060"
# 输出: "   ⚠️  如果更换了GPU，请删除缓存: ~/.neuroexapt/cache"
```

**跳过批量大小优化**（如果您想手动控制）：
```bash
# Windows
set SKIP_BATCH_OPTIMIZATION=true
python examples/basic_classification.py

# Linux/Mac
SKIP_BATCH_OPTIMIZATION=true python examples/basic_classification.py
```

### 2. 🧠 智能训练配置

```python
from neuroexapt.core.intelligent_trainer import IntelligentTrainer

# 智能训练器会自动配置最佳参数
trainer = IntelligentTrainer(
    model=model,
    auto_lr_schedule=True,       # 自动学习率调度
    auto_data_augmentation=True, # 智能数据增强
    auto_regularization=True,    # 自适应正则化
    target_accuracy=95.0
)

# 一键训练，自动优化所有超参数
result = trainer.smart_train(train_loader, val_loader)
```

### 3. 🔬 实时形态发生监控

```python
from neuroexapt.visualization.morphogenesis_monitor import MorphogenesisMonitor

# 启动实时监控
monitor = MorphogenesisMonitor()

# 训练时实时可视化网络演化
result = train_with_dnm(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    monitor=monitor,  # 添加监控器
    target_accuracy=95.0
)

# 生成演化报告
monitor.generate_report("evolution_report.html")
```

## 🎯 常见使用场景

### 场景1：图像分类任务

```python
# 适用于CIFAR-10, ImageNet等
from neuroexapt.tasks.classification import DNMClassifier

classifier = DNMClassifier(
    num_classes=10,
    target_accuracy=95.0,
    aggressive_growth=False  # 保守增长模式
)

result = classifier.fit(train_loader, val_loader)
```

### 场景2：小样本学习

```python
# 针对数据稀少的场景优化
from neuroexapt.tasks.few_shot import DNMFewShotLearner

few_shot_learner = DNMFewShotLearner(
    shots_per_class=5,
    enable_aggressive_growth=True,  # 激进增长模式
    meta_learning=True
)

result = few_shot_learner.fit(support_set, query_set)
```

### 场景3：性能突破专用

```python
# 专门用于突破性能瓶颈
from neuroexapt.tasks.breakthrough import BreakthroughTrainer

breakthrough = BreakthroughTrainer(
    performance_threshold=90.0,    # 当前性能水平
    target_improvement=5.0,        # 期望提升5%
    max_morphogenesis_events=5     # 最多5次形态发生
)

result = breakthrough.breakthrough_training(model, train_loader, val_loader)
```

## 🔧 高级配置选项

### 自定义形态发生策略

```python
from neuroexapt.core.morphogenesis_config import MorphogenesisConfig

# 自定义DNM行为
config = MorphogenesisConfig(
    # 瓶颈检测敏感度
    bottleneck_threshold=0.02,
    
    # 形态发生类型偏好
    prefer_neuron_division=True,
    prefer_connection_growth=False,
    
    # 风险控制
    max_parameter_increase=2.0,  # 最多增加2倍参数
    performance_safety_margin=0.01,  # 1%性能安全边际
    
    # 高级选项
    enable_pruning=True,         # 启用智能剪枝
    enable_attention_growth=True, # 启用注意力机制生长
    enable_multi_objective=True   # 多目标优化
)

# 应用自定义配置
result = train_with_dnm(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)
```

### 监控和调试

```python
# 启用详细日志
import logging
neuroexapt.set_log_level(logging.DEBUG)

# 性能分析模式
result = train_with_dnm(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    profile_mode=True,          # 启用性能分析
    save_checkpoints=True,      # 保存检查点
    checkpoint_dir="./checkpoints"
)

# 分析训练过程
from neuroexapt.analysis.training_analyzer import TrainingAnalyzer

analyzer = TrainingAnalyzer()
report = analyzer.analyze_training_run(result)

print(f"📊 训练效率: {report.training_efficiency}")
print(f"🧬 形态发生效果: {report.morphogenesis_impact}")
print(f"💡 优化建议: {report.recommendations}")
```

## 🚀 下一步学习

### 🎓 学习路径建议

1. **理解原理** (30分钟)
   - 阅读 @ref dnm_principles "DNM核心原理"
   - 运行上面的基础示例

2. **深入实践** (2小时)
   - 尝试 `examples/` 目录下的所有示例
   - 在自己的数据集上应用DNM

3. **高级特性** (1天)
   - 学习 @ref intelligent_growth "智能增长机制"
   - 自定义形态发生策略

4. **生产应用** (1周)
   - 大规模数据集训练
   - 性能调优和部署

### 📚 推荐阅读顺序

1. @ref dnm_principles "DNM核心原理详解"
2. @ref basic_concepts "基础概念说明"  
3. @ref intelligent_growth "智能增长机制"
4. @ref morphogenesis_events "形态发生事件详解"
5. @ref advanced_features "高级特性与自定义"

### 🔗 实用资源

- **示例代码**: `examples/` 目录
- **API文档**: @ref neuroexapt.core "核心模块文档"
- **性能基准**: @ref benchmarks "性能测试结果"
- **常见问题**: @ref faq "FAQ与故障排除"

---

## 🤝 需要帮助？

- **GitHub Issues**: [报告问题](https://github.com/neuroexapt/neuroexapt/issues)
- **讨论区**: [技术讨论](https://github.com/neuroexapt/neuroexapt/discussions)
- **邮件支持**: team@neuroexapt.ai

**恭喜！您已经掌握了NeuroExapt的基础使用。让我们一起见证神经网络的智能进化！** 🧬✨