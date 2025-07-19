"""
测试保守的信息论进化策略
这个脚本用于快速验证调整后的框架是否能够保持更高的准确率
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

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt.core.structural_evolution import StructuralEvolution
from neuroexapt.core.operators import PruneByEntropy, AddBlock, ExpandWithMI, create_conv_block
from neuroexapt.neuroexapt import NeuroExapt


class TestCNN(nn.Module):
    """简单的测试CNN"""
    def __init__(self, num_classes=10):
        super(TestCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),    # features.0
            nn.ReLU(inplace=True),             # features.1
            nn.MaxPool2d(2, 2),                # features.2
            nn.Conv2d(32, 64, 3, padding=1),  # features.3
            nn.ReLU(inplace=True),             # features.4
            nn.MaxPool2d(2, 2),                # features.5
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_test_data(batch_size=64):
    """加载测试数据"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    # 使用小子集进行快速测试
    trainset.data = trainset.data[:1000]  # 只使用1000个样本
    trainset.targets = trainset.targets[:1000]
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform
    )
    testset.data = testset.data[:200]  # 只使用200个测试样本
    testset.targets = testset.targets[:200]
    
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader


def evaluate_model(model, testloader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in testloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    model.train()
    return accuracy


def main():
    print("🧪 测试保守的信息论进化策略")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("📊 加载测试数据...")
    trainloader, testloader = load_test_data()
    
    # 创建模型
    print("🏗️ 创建测试模型...")
    model = TestCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"初始参数数量: {initial_params:,}")
    
    # 获取初始准确率
    initial_acc = evaluate_model(model, testloader, device)
    print(f"初始准确率: {initial_acc:.2f}%")
    
    # 设置保守的操作符
    operators = [
        # 只针对非关键层进行剪枝
        PruneByEntropy('features.1'),  # ReLU层
        PruneByEntropy('features.4'),  # ReLU层
        
        # 优先扩展操作
        AddBlock('features.2', lambda channels: create_conv_block(channels)),
        AddBlock('features.5', lambda channels: create_conv_block(channels)),
        
        # 保守的扩展
        ExpandWithMI(expansion_factor=1.1),
    ]
    
    # 创建NeuroExapt实例
    neuroexapt = NeuroExapt(
        model=model,
        criterion=criterion,
        dataloader=trainloader,
        operators=operators,
        device=device,
        lambda_entropy=0.005,    # 降低正则化
        lambda_bayesian=0.002,
        enable_validation=True
    )
    
    print("\n🧠 开始进化测试...")
    
    # 进行几轮训练和进化
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(5):  # 只测试5个epoch
        print(f"\nEpoch {epoch + 1}/5")
        
        # 简单训练
        model.train()
        for i, (data, targets) in enumerate(trainloader):
            if i >= 5:  # 每个epoch只训练5个batch
                break
                
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # 评估
        current_acc = evaluate_model(model, testloader, device)
        print(f"当前准确率: {current_acc:.2f}%")
        
        # 尝试进化
        performance_metrics = {
            'val_accuracy': current_acc,
            'train_accuracy': current_acc,  # 简化测试
            'loss': loss.item()
        }
        
        prev_params = sum(p.numel() for p in model.parameters())
        evolved_model, action_taken = neuroexapt.evolution_engine.step(
            epoch=epoch + 1,
            performance_metrics=performance_metrics,
            dataloader=trainloader,
            criterion=criterion
        )
        
        if action_taken:
            model = evolved_model
            neuroexapt.model = model
            new_params = sum(p.numel() for p in model.parameters())
            
            print(f"✅ 进化动作: {action_taken}")
            print(f"📊 参数变化: {prev_params:,} -> {new_params:,} (Δ{new_params - prev_params:+,})")
            
            # 重新评估
            post_evolution_acc = evaluate_model(model, testloader, device)
            print(f"🎯 进化后准确率: {post_evolution_acc:.2f}%")
            
            # 重置优化器
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        else:
            print("ℹ️ 未进行进化")
    
    # 最终结果
    final_acc = evaluate_model(model, testloader, device)
    final_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n📊 最终结果:")
    print(f"   初始准确率: {initial_acc:.2f}%")
    print(f"   最终准确率: {final_acc:.2f}%")
    print(f"   准确率变化: {final_acc - initial_acc:+.2f}%")
    print(f"   初始参数: {initial_params:,}")
    print(f"   最终参数: {final_params:,}")
    print(f"   参数变化: {final_params - initial_params:+,}")
    print(f"   进化次数: {neuroexapt.stats['evolutions']}")
    
    # 清理
    neuroexapt.cleanup()
    
    if final_acc >= initial_acc - 2.0:  # 允许小幅下降
        print("✅ 测试通过：保守策略成功保持了准确率！")
        return True
    else:
        print("❌ 测试失败：准确率下降过多")
        return False


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        success = main()
        if success:
            print("\n🎉 保守进化策略测试成功！")
        else:
            print("\n⚠️ 需要进一步调整策略")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 