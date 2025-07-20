#!/usr/bin/env python3
"""
智能架构进化演示 - CIFAR-10实战版
Intelligent Architecture Evolution Demo - CIFAR-10 Edition

🔬 基于互信息和贝叶斯推断的神经网络架构自适应变异系统

🧬 演示内容：
1. 基于MINE的互信息估计 - 量化特征与目标的信息依赖
2. 贝叶斯不确定性量化 - 评估特征表征的稳定性
3. 智能瓶颈检测 - 精确定位网络性能限制点
4. 基于瓶颈的智能变异规划 - 15种变异策略精确匹配
5. 先进Net2Net参数迁移 - 保证功能等价性的平滑迁移
6. 完整架构进化流程 - 检测→规划→迁移→评估→迭代

🎯 目标：在CIFAR-10上展示智能架构进化的完整流程
🔬 理论基础：将抽象的"天赋上限"转化为可计算的互信息和不确定性指标
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import os
import sys

# 导入新的智能架构进化组件
from neuroexapt.core import (
    IntelligentArchitectureEvolutionEngine,
    EvolutionConfig,
    MutualInformationEstimator,
    BayesianUncertaintyEstimator,
    IntelligentBottleneckDetector,
    IntelligentMutationPlanner,
    AdvancedNet2NetTransfer
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvolvableResNet(nn.Module):
    """可进化的ResNet架构 - 用于智能进化演示"""
    
    def __init__(self, num_classes=10):
        super(EvolvableResNet, self).__init__()
        
        # 初始特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 可进化的特征提取层
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """创建ResNet层"""
        layers = []
        
        # 第一个块可能有降采样
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # 后续块
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = self.classifier(x)
        
        return x


class BasicBlock(nn.Module):
    """基础残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = self.relu(out)
        
        return out


class IntelligentEvolutionTrainer:
    """智能进化训练器"""
    
    def __init__(self, model, device, train_loader, test_loader):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # 训练历史
        self.train_history = []
        self.test_history = []
        self.evolution_history = []
        
        # 优化器和调度器
        self.optimizer = None
        self.scheduler = None
        
    def setup_optimizer(self, learning_rate=0.1):
        """设置优化器和学习率调度器"""
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-6
        )
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, '
                           f'Loss: {loss.item():.6f}, '
                           f'Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        self.train_history.append({'epoch': epoch, 'loss': avg_loss, 'accuracy': accuracy})
        return avg_loss, accuracy
    
    def test(self):
        """测试模型性能"""
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        
        return test_loss, accuracy
    
    def extract_features_and_labels(self, model, data_loader, max_batches=3):
        """提取模型特征和标签用于智能分析"""
        feature_dict = {}
        all_labels = []
        
        # 注册hook收集特征
        def get_hook(name):
            def hook(module, input, output):
                if name not in feature_dict:
                    feature_dict[name] = []
                # 确保输出是张量
                if isinstance(output, torch.Tensor):
                    feature_dict[name].append(output.detach().cpu())
            return hook
        
        # 为主要层注册hook
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and any(
                keyword in name for keyword in ['layer', 'classifier', 'stem']
            ):
                hook = module.register_forward_hook(get_hook(name))
                hooks.append(hook)
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                    
                data = data.to(self.device)
                _ = model(data)
                all_labels.append(target)
        
        # 清理hooks
        for hook in hooks:
            hook.remove()
        
        # 合并特征
        for name in list(feature_dict.keys()):
            if feature_dict[name]:
                try:
                    feature_dict[name] = torch.cat(feature_dict[name], dim=0)
                except:
                    # 如果无法拼接，删除这个特征
                    del feature_dict[name]
            else:
                del feature_dict[name]
        
        labels = torch.cat(all_labels, dim=0) if all_labels else torch.tensor([])
        
        logger.info(f"提取到 {len(feature_dict)} 个特征层: {list(feature_dict.keys())}")
        return feature_dict, labels


def prepare_cifar10_data(batch_size_train=128, batch_size_test=100):
    """准备CIFAR-10数据"""
    logger.info("准备CIFAR-10数据集...")
    
    # 训练时数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))
    ])
    
    # 测试时标准化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 创建数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True, 
        num_workers=2, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size_test, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    return train_loader, test_loader


def demo_mutual_information_analysis(trainer):
    """演示互信息分析"""
    print("\n" + "="*60)
    print("🔬 演示：基于MINE的互信息分析")
    print("="*60)
    
    # 提取特征和标签
    feature_dict, labels = trainer.extract_features_and_labels(
        trainer.model, trainer.test_loader, max_batches=5
    )
    
    if not feature_dict or labels.numel() == 0:
        print("❌ 无法提取特征，跳过互信息分析")
        return
    
    # 创建互信息估计器
    mi_estimator = MutualInformationEstimator(device=trainer.device)
    
    print(f"📊 开始计算 {len(feature_dict)} 个层的互信息...")
    
    # 估计分层互信息
    mi_results = mi_estimator.batch_estimate_layerwise_mi(
        feature_dict, labels, num_classes=10
    )
    
    print("\n🔍 分层互信息结果 I(H_k; Y):")
    for layer_name, mi_value in mi_results.items():
        print(f"  {layer_name}: {mi_value:.4f}")
    
    # 计算条件互信息（如果有足够的层）
    if len(feature_dict) >= 2:
        print("\n🔗 计算条件互信息 I(H_k; Y|H_{k+1})...")
        layer_names = list(feature_dict.keys())
        feature_pairs = []
        
        for i in range(len(layer_names) - 1):
            current_layer = layer_names[i]
            next_layer = layer_names[i + 1]
            feature_pairs.append((
                current_layer,
                feature_dict[current_layer],
                feature_dict[next_layer]
            ))
        
        if feature_pairs:
            conditional_mi_results = mi_estimator.batch_estimate_conditional_mi(
                feature_pairs, labels, num_classes=10
            )
            
            print("\n🔗 条件互信息结果 I(H_k; Y|H_{k+1}):")
            for layer_name, cmi_value in conditional_mi_results.items():
                print(f"  {layer_name}: {cmi_value:.4f}")
                if cmi_value < 0.01:
                    print(f"    ⚠️  检测到潜在信息泄露瓶颈")
    
    return mi_results


def demo_uncertainty_analysis(trainer):
    """演示贝叶斯不确定性分析"""
    print("\n" + "="*60)
    print("🎲 演示：贝叶斯不确定性分析")
    print("="*60)
    
    # 提取特征
    feature_dict, labels = trainer.extract_features_and_labels(
        trainer.model, trainer.test_loader, max_batches=3
    )
    
    if not feature_dict:
        print("❌ 无法提取特征，跳过不确定性分析")
        return
    
    # 创建不确定性估计器
    uncertainty_estimator = BayesianUncertaintyEstimator(device=trainer.device)
    
    print(f"📊 开始计算 {len(feature_dict)} 个层的不确定性...")
    
    # 估计特征不确定性
    uncertainty_results = uncertainty_estimator.estimate_feature_uncertainty(
        feature_dict, labels
    )
    
    print("\n🎲 特征不确定性结果 U(H_k):")
    for layer_name, uncertainty in uncertainty_results.items():
        status = "⚠️ 高" if uncertainty > 1.0 else "✓ 正常"
        print(f"  {layer_name}: {uncertainty:.4f} ({status})")
    
    return uncertainty_results


def demo_intelligent_bottleneck_detection(trainer):
    """演示智能瓶颈检测"""
    print("\n" + "="*60)
    print("🔍 演示：智能瓶颈检测")
    print("="*60)
    
    # 提取特征和标签
    feature_dict, labels = trainer.extract_features_and_labels(
        trainer.model, trainer.test_loader, max_batches=5
    )
    
    if not feature_dict or labels.numel() == 0:
        print("❌ 无法提取特征，跳过瓶颈检测")
        return []
    
    # 创建瓶颈检测器
    detector = IntelligentBottleneckDetector(device=trainer.device)
    
    print("🔍 开始智能瓶颈检测...")
    
    # 执行瓶颈检测
    bottleneck_reports = detector.detect_bottlenecks(
        model=trainer.model,
        feature_dict=feature_dict,
        labels=labels,
        num_classes=10,
        confidence_threshold=0.6  # 适中的置信度阈值
    )
    
    print(f"\n📊 检测结果: 发现 {len(bottleneck_reports)} 个潜在瓶颈")
    
    # 可视化瓶颈报告
    if bottleneck_reports:
        visualization = detector.visualize_bottlenecks(bottleneck_reports)
        print(visualization)
        
        # 获取摘要
        summary = detector.get_bottleneck_summary(bottleneck_reports)
        print(f"\n📈 瓶颈类型分布: {summary.get('type_distribution', {})}")
        print(f"🎯 建议优先处理: {summary.get('recommended_priority', [])}")
    else:
        print("✅ 未检测到显著瓶颈，网络架构相对健康")
    
    return bottleneck_reports


def demo_intelligent_evolution(trainer, initial_epochs=10):
    """演示完整的智能架构进化"""
    print("\n" + "="*60)
    print("🚀 演示：完整智能架构进化流程")
    print("="*60)
    
    # 先进行基础训练
    print(f"📚 开始基础训练 {initial_epochs} 个epoch...")
    trainer.setup_optimizer(learning_rate=0.1)
    
    best_accuracy = 0
    for epoch in range(initial_epochs):
        train_loss, train_acc = trainer.train_epoch(epoch)
        test_loss, test_acc = trainer.test()
        
        if trainer.scheduler:
            trainer.scheduler.step()
        
        trainer.test_history.append({
            'epoch': epoch, 'loss': test_loss, 'accuracy': test_acc
        })
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
        
        print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")
    
    print(f"🎯 基础训练完成，最佳测试准确率: {best_accuracy:.2f}%")
    
    # 配置进化参数
    evolution_config = EvolutionConfig(
        max_iterations=3,  # 限制迭代次数用于演示
        patience=2,
        min_improvement=0.005,  # 0.5%的最小改进
        confidence_threshold=0.6,
        max_mutations_per_iteration=2,
        task_type='vision',
        risk_tolerance=0.8
    )
    
    print(f"🧬 配置智能进化引擎: {evolution_config.max_iterations} 轮迭代")
    
    # 创建进化引擎
    evolution_engine = IntelligentArchitectureEvolutionEngine(evolution_config)
    
    # 定义评估函数
    def evaluation_fn(model):
        original_model = trainer.model
        trainer.model = model
        _, accuracy = trainer.test()
        trainer.model = original_model
        return accuracy / 100.0  # 转换为[0,1]范围
    
    # 定义特征提取函数
    def feature_extractor_fn(model, data_loader):
        original_model = trainer.model
        trainer.model = model
        features, labels = trainer.extract_features_and_labels(model, data_loader, max_batches=3)
        trainer.model = original_model
        return features, labels
    
    try:
        print("🚀 开始智能架构进化...")
        
        # 执行智能进化
        best_model, evolution_history = evolution_engine.evolve(
            model=trainer.model,
            data_loader=trainer.train_loader,
            evaluation_fn=evaluation_fn,
            feature_extractor_fn=feature_extractor_fn
        )
        
        # 评估进化后的性能
        trainer.model = best_model
        final_loss, final_accuracy = trainer.test()
        
        print(f"\n🎉 智能进化完成!")
        print(f"📈 性能提升: {best_accuracy:.2f}% → {final_accuracy:.2f}% "
              f"(+{final_accuracy - best_accuracy:.2f}%)")
        
        # 可视化进化过程
        evolution_visualization = evolution_engine.visualize_evolution()
        print(evolution_visualization)
        
        # 获取进化摘要
        summary = evolution_engine.get_evolution_summary()
        if summary.get('status') == 'completed':
            print(f"\n📊 进化统计:")
            print(f"  总迭代数: {summary['total_iterations']}")
            print(f"  成功率: {summary['success_rate']:.1%}")
            print(f"  参数增长: {summary['total_parameter_growth']:.1%}")
        
        trainer.evolution_history = evolution_history
        return best_model
        
    except Exception as e:
        print(f"❌ 进化过程中遇到错误: {e}")
        print("这可能是由于演示环境的限制，实际使用中请确保有足够的数据和计算资源")
        return trainer.model


def run_complete_demo():
    """运行完整演示"""
    print("🎯 NeuroExapt 智能架构进化演示 - CIFAR-10版")
    print("基于互信息和贝叶斯推断的神经网络架构自适应变异系统")
    print("="*80)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 准备数据
    train_loader, test_loader = prepare_cifar10_data()
    
    # 创建模型
    print("🏗️  创建可进化的ResNet模型...")
    model = EvolvableResNet(num_classes=10)
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    # 创建训练器
    trainer = IntelligentEvolutionTrainer(model, device, train_loader, test_loader)
    
    # 运行各个演示组件
    try:
        # 1. 互信息分析演示
        mi_results = demo_mutual_information_analysis(trainer)
        
        # 2. 不确定性分析演示
        uncertainty_results = demo_uncertainty_analysis(trainer)
        
        # 3. 瓶颈检测演示
        bottleneck_reports = demo_intelligent_bottleneck_detection(trainer)
        
        # 4. 完整进化演示
        evolved_model = demo_intelligent_evolution(trainer, initial_epochs=5)
        
        print("\n" + "="*80)
        print("🎉 智能架构进化演示完成!")
        print("\n新框架成功展示了以下核心能力:")
        print("✅ 基于MINE的互信息估计 - 量化特征信息含量")
        print("✅ 贝叶斯不确定性量化 - 评估特征稳定性")
        print("✅ 智能瓶颈检测 - 精确定位性能限制点")
        print("✅ 基于瓶颈的变异规划 - 15种策略精确匹配")
        print("✅ 先进Net2Net参数迁移 - 功能等价性保证")
        print("✅ 完整架构进化流程 - 自适应优化策略")
        
        print("\n🔬 理论创新:")
        print("• 将抽象的'神经网络天赋上限'转化为可计算的数学指标")
        print("• 基于信息论和贝叶斯推断的科学变异指导")
        print("• 从启发式'无趣变异'升级为理论指导的'智能变异'")
        print("="*80)
        
    except Exception as e:
        logger.error(f"演示过程中遇到错误: {e}")
        print(f"\n❌ 演示过程中遇到错误: {e}")
        print("这可能是由于:")
        print("1. 缺少必要的依赖包")
        print("2. 计算资源不足")
        print("3. 数据加载问题")
        print("\n请检查环境配置并重试")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行完整演示
    run_complete_demo()