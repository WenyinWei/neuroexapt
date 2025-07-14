"""
贝叶斯架构演化演示

这个演示展示了如何使用贝叶斯推断和信息论原理来指导神经网络架构的演化。
每次训练/架构调整都会更新最优架构估计的后验分布。

主要特点：
1. 基于贝叶斯推断的架构决策
2. 信息论指标（互信息、熵）指导演化
3. 后验分布的动态更新
4. 综合的架构建议系统
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import time
from typing import Dict, List

# Add the current directory to the path to import neuroexapt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt.core.bayesian_architecture_advisor import (
    BayesianArchitectureAdvisor,
    ArchitectureAction,
    ArchitectureState,
    ArchitectureEvidence
)
from neuroexapt.core.bayesian_depth_operators import (
    BayesianDepthOperator,
    get_bayesian_depth_operators
)
from neuroexapt.core.radical_evolution import RadicalEvolutionEngine


class BayesianEvolvableCNN(nn.Module):
    """
    为贝叶斯架构演化专门设计的CNN。
    
    这个网络会跟踪自己的架构演化历史，并与贝叶斯顾问集成。
    """
    
    def __init__(self, num_classes=10, initial_depth=8, initial_channels=32):
        super(BayesianEvolvableCNN, self).__init__()
        
        # 记录架构演化历史
        self.architecture_history = []
        self.performance_history = []
        self.bayesian_insights = []
        
        # 构建初始架构
        self.features = nn.Sequential(
            # 第一层：输入处理
            nn.Conv2d(3, initial_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True),
            
            # 第二层：特征提取
            nn.Conv2d(initial_channels, initial_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层：深度特征
            nn.Conv2d(initial_channels * 2, initial_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels * 4),
            nn.ReLU(inplace=True),
            
            # 第四层：高级特征
            nn.Conv2d(initial_channels * 4, initial_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(initial_channels * 8, initial_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(initial_channels * 4, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
        # 记录初始状态
        self._record_architecture_state("initial")
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _record_architecture_state(self, event_type: str):
        """记录架构状态"""
        
        # 计算架构统计信息
        total_params = sum(p.numel() for p in self.parameters())
        conv_layers = sum(1 for m in self.modules() if isinstance(m, nn.Conv2d))
        linear_layers = sum(1 for m in self.modules() if isinstance(m, nn.Linear))
        
        # 计算平均通道数
        conv_channels = [m.out_channels for m in self.modules() if isinstance(m, nn.Conv2d)]
        avg_channels = np.mean(conv_channels) if conv_channels else 0
        
        state = {
            'event_type': event_type,
            'timestamp': time.time(),
            'total_params': total_params,
            'conv_layers': conv_layers,
            'linear_layers': linear_layers,
            'avg_channels': avg_channels,
            'max_channels': max(conv_channels) if conv_channels else 0,
            'architecture_depth': conv_layers + linear_layers
        }
        
        self.architecture_history.append(state)
        
        print(f"📊 Architecture state recorded ({event_type}):")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Architecture depth: {state['architecture_depth']}")
        print(f"   Average channels: {avg_channels:.1f}")
    
    def update_performance_record(self, performance: Dict[str, float]):
        """更新性能记录"""
        self.performance_history.append({
            'timestamp': time.time(),
            'performance': performance
        })
    
    def get_architecture_summary(self) -> Dict[str, any]:
        """获取架构摘要信息"""
        
        if not self.architecture_history:
            return {}
        
        current_state = self.architecture_history[-1]
        initial_state = self.architecture_history[0]
        
        return {
            'evolution_steps': len(self.architecture_history) - 1,
            'parameter_growth': current_state['total_params'] - initial_state['total_params'],
            'depth_growth': current_state['architecture_depth'] - initial_state['architecture_depth'],
            'channel_growth': current_state['avg_channels'] - initial_state['avg_channels'],
            'current_state': current_state,
            'bayesian_insights': self.bayesian_insights[-5:] if self.bayesian_insights else []
        }


def load_demo_data(batch_size=32, num_samples=2000):
    """加载演示数据"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # 创建小数据集用于快速演示
    train_indices = torch.randperm(len(trainset))[:num_samples].tolist()
    test_indices = torch.randperm(len(testset))[:num_samples//4].tolist()
    
    train_subset = torch.utils.data.Subset(trainset, train_indices)
    test_subset = torch.utils.data.Subset(testset, test_indices)
    
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader


def evaluate_model(model, testloader, device):
    """评估模型性能"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, targets in testloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(testloader) if len(testloader) > 0 else 0.0
    
    model.train()
    return accuracy, avg_loss


def demonstrate_bayesian_evolution():
    """
    演示贝叶斯架构演化过程
    """
    
    print("🧠 贝叶斯架构演化演示")
    print("=" * 60)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 加载数据
    print("\n📊 加载演示数据...")
    trainloader, testloader = load_demo_data(batch_size=32, num_samples=2000)
    print(f"训练批次: {len(trainloader)}")
    print(f"测试批次: {len(testloader)}")
    
    # 创建模型
    print("\n🏗️ 创建贝叶斯进化CNN...")
    model = BayesianEvolvableCNN(num_classes=10, initial_depth=8, initial_channels=32).to(device)
    
    # 初始评估
    initial_accuracy, initial_loss = evaluate_model(model, testloader, device)
    print(f"初始性能: {initial_accuracy:.2f}% accuracy, {initial_loss:.3f} loss")
    
    # 创建贝叶斯深度操作器
    print("\n🧠 初始化贝叶斯深度操作器...")
    bayesian_operators = get_bayesian_depth_operators(
        initial_depth_prior=(10.0, 5.0),  # 先验：最优深度约10层，不确定性5层
        initial_channel_prior=(64.0, 32.0),  # 先验：最优通道数约64，不确定性32
        information_threshold=0.8,
        entropy_threshold=0.6
    )
    
    # 创建演化引擎
    evolution_engine = RadicalEvolutionEngine(
        model=model,
        operators=bayesian_operators,
        input_shape=(3, 32, 32),
        evolution_probability=1.0,  # 每次都尝试演化
        max_mutations_per_epoch=1,
        enable_validation=True
    )
    
    # 训练和演化循环
    print("\n🔄 开始贝叶斯架构演化...")
    print("=" * 60)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    num_evolution_cycles = 5
    epochs_per_cycle = 3
    
    for cycle in range(num_evolution_cycles):
        print(f"\n🔄 演化周期 {cycle + 1}/{num_evolution_cycles}")
        print("-" * 40)
        
        # 训练几个epoch
        for epoch in range(epochs_per_cycle):
            print(f"\n📈 训练 Epoch {epoch + 1}/{epochs_per_cycle}")
            
            # 训练循环
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, targets) in enumerate(trainloader):
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                if batch_idx % 20 == 0:
                    print(f"   Batch {batch_idx:3d}: Loss {loss.item():.3f}")
            
            train_accuracy = 100 * correct / total
            avg_loss = epoch_loss / len(trainloader)
            
            # 评估
            test_accuracy, test_loss = evaluate_model(model, testloader, device)
            
            print(f"   训练: {train_accuracy:.2f}% | 测试: {test_accuracy:.2f}% | 损失: {avg_loss:.3f}")
            
            # 更新模型的性能记录
            performance = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_loss': avg_loss,
                'test_loss': test_loss,
                'epoch': epoch + 1,
                'cycle': cycle + 1
            }
            model.update_performance_record(performance)
        
        # 贝叶斯架构演化
        print(f"\n🧠 贝叶斯架构分析和演化...")
        
        # 获取当前性能指标
        current_performance = {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'convergence_speed': 0.5,  # 简化
            'overfitting': max(0, train_accuracy - test_accuracy) / 100.0,
            'gradient_flow': 0.7,  # 简化
            'efficiency': 0.6,  # 简化
            'epoch': cycle * epochs_per_cycle + epochs_per_cycle
        }
        
        # 尝试演化
        evolved_model, evolution_action = evolution_engine.evolve(
            epoch=cycle + 1,
            dataloader=trainloader,
            criterion=criterion,
            performance_metrics=current_performance
        )
        
        if evolution_action and evolved_model:
            print(f"   ✅ 演化成功: {evolution_action}")
            
            # 更新模型
            model = evolved_model
            
            # 重新创建优化器
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 评估演化后的性能
            post_evolution_accuracy, post_evolution_loss = evaluate_model(model, testloader, device)
            
            print(f"   📊 演化后性能: {post_evolution_accuracy:.2f}% accuracy")
            print(f"   📈 性能变化: {post_evolution_accuracy - test_accuracy:+.2f}%")
            
            # 记录架构状态
            model._record_architecture_state(f"evolution_cycle_{cycle + 1}")
            
            # 获取贝叶斯顾问的洞察
            if hasattr(evolution_engine.operators[0], 'get_advisor_insights'):
                insights = evolution_engine.operators[0].get_advisor_insights()
                model.bayesian_insights.append(insights)
                
                print(f"   🧠 贝叶斯洞察:")
                print(f"      最优深度估计: {insights['optimal_depth_estimate']:.1f} ± {insights['depth_uncertainty']:.1f}")
                print(f"      最优通道估计: {insights['optimal_channel_estimate']:.1f} ± {insights['channel_uncertainty']:.1f}")
                print(f"      决策次数: {insights['decisions_made']}")
                print(f"      近期行动: {insights['recent_actions']}")
            
            # 更新操作器的结果
            if hasattr(evolution_engine.operators[0], 'update_with_outcome'):
                outcome_performance = {
                    'accuracy': post_evolution_accuracy,
                    'loss': post_evolution_loss,
                    'convergence_speed': 0.5,
                    'overfitting': 0.0,
                    'gradient_flow': 0.7,
                    'efficiency': 0.6,
                    'epoch': cycle + 1
                }
                evolution_engine.operators[0].update_with_outcome(model, outcome_performance)
        
        else:
            print(f"   ℹ️ 此周期未执行架构演化")
    
    # 最终结果
    print("\n📋 最终结果")
    print("=" * 60)
    
    final_accuracy, final_loss = evaluate_model(model, testloader, device)
    architecture_summary = model.get_architecture_summary()
    
    print(f"最终性能:")
    print(f"  准确率: {final_accuracy:.2f}% (初始: {initial_accuracy:.2f}%)")
    print(f"  改进: {final_accuracy - initial_accuracy:+.2f}%")
    print(f"  损失: {final_loss:.3f}")
    
    print(f"\n架构演化摘要:")
    print(f"  演化步数: {architecture_summary['evolution_steps']}")
    print(f"  参数增长: {architecture_summary['parameter_growth']:+,}")
    print(f"  深度增长: {architecture_summary['depth_growth']:+.0f}")
    print(f"  通道增长: {architecture_summary['channel_growth']:+.1f}")
    
    # 演化引擎统计
    engine_stats = evolution_engine.get_evolution_stats()
    print(f"\n演化引擎统计:")
    print(f"  总突变数: {engine_stats['total_mutations']}")
    print(f"  成功突变数: {engine_stats['successful_mutations']}")
    print(f"  成功率: {engine_stats['overall_success_rate']:.2%}")
    
    # 贝叶斯洞察历史
    if model.bayesian_insights:
        print(f"\n贝叶斯洞察演化:")
        for i, insight in enumerate(model.bayesian_insights):
            print(f"  周期 {i+1}: 深度={insight['optimal_depth_estimate']:.1f}±{insight['depth_uncertainty']:.1f}, "
                  f"通道={insight['optimal_channel_estimate']:.1f}±{insight['channel_uncertainty']:.1f}")
    
    print(f"\n✅ 贝叶斯架构演化演示完成!")
    
    return model, final_accuracy, architecture_summary


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        model, accuracy, summary = demonstrate_bayesian_evolution()
        print(f"\n🎉 演示成功完成!")
        print(f"最终准确率: {accuracy:.2f}%")
        print(f"架构演化步数: {summary['evolution_steps']}")
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc() 