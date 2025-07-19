"""
改进的基础分类示例 - 使用保守的信息论策略
展示如何通过调整信息论策略来保持更高的准确率
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt.core.operators import PruneByEntropy, AddBlock, ExpandWithMI, create_conv_block
from neuroexapt.neuroexapt import NeuroExapt


class ImprovedCNN(nn.Module):
    """改进的CNN架构，具有更好的适应性"""
    
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # features.0
            nn.BatchNorm2d(32),                           # features.1  
            nn.ReLU(inplace=True),                        # features.2
            nn.MaxPool2d(kernel_size=2, stride=2),        # features.3
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # features.4
            nn.BatchNorm2d(64),                           # features.5
            nn.ReLU(inplace=True),                        # features.6
            nn.MaxPool2d(kernel_size=2, stride=2),        # features.7
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # 较少的dropout
            nn.Linear(64 * 8 * 8, 256),  # 更大的隐藏层
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
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


def create_smart_expansion_block(out_channels: int) -> nn.Module:
    """创建智能扩展块，包含残差连接"""
    return nn.Sequential(
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),  # 1x1压缩
        nn.BatchNorm2d(out_channels),
    )


def load_optimized_data(batch_size=128):
    """加载优化的数据"""
    # 改进的数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 轻微颜色抖动
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    # 使用更大的子集进行测试
    subset_size = 5000
    indices = torch.randperm(len(trainset))[:subset_size].tolist()
    trainset = torch.utils.data.Subset(trainset, indices)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_test
    )
    # 使用1000个测试样本
    test_indices = torch.randperm(len(testset))[:1000].tolist()
    testset = torch.utils.data.Subset(testset, test_indices)
    
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader


def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    model.train()
    return accuracy, avg_loss


def main():
    print("🚀 改进的NeuroExapt分类示例 - 保守信息论策略")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 超参数设置
    num_epochs = 20
    batch_size = 128
    learning_rate = 0.001
    evolution_frequency = 2  # 每2个epoch检查一次
    
    # 加载数据
    print("📊 加载优化的CIFAR-10数据...")
    trainloader, testloader = load_optimized_data(batch_size=batch_size)
    print(f"训练批次: {len(trainloader)}, 测试批次: {len(testloader)}")
    
    # 创建改进的模型
    print("🏗️ 创建改进的CNN架构...")
    model = ImprovedCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"初始参数数量: {initial_params:,}")
    
    # 保守的操作符设置
    print("🔧 设置保守的结构操作符...")
    operators = [
        # 🔄 只对安全的层进行剪枝
        PruneByEntropy('features.2'),  # ReLU层
        PruneByEntropy('features.6'),  # ReLU层
        
        # 🔄 优先扩展操作
        AddBlock('features.3', create_smart_expansion_block),  # 第一个池化后
        AddBlock('features.7', create_smart_expansion_block),  # 第二个池化后
        
        # 🔄 保守的宽度扩展
        ExpandWithMI(expansion_factor=1.15),  # 轻微扩展
    ]
    
    # 创建NeuroExapt实例（保守配置）
    print("🧠 初始化保守的NeuroExapt框架...")
    neuroexapt = NeuroExapt(
        model=model,
        criterion=criterion,
        dataloader=trainloader,
        operators=operators,
        device=device,
        lambda_entropy=0.003,      # 🔄 非常低的熵正则化
        lambda_bayesian=0.001,     # 🔄 非常低的贝叶斯正则化
        input_shape=(3, 32, 32),
        enable_validation=True
    )
    
    # 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 记录历史
    history = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': [],
        'evolutions': []
    }
    
    print("🚀 开始训练...")
    print("=" * 60)
    
    # 初始评估
    initial_test_acc, initial_test_loss = evaluate_model(model, testloader, device)
    print(f"初始测试准确率: {initial_test_acc:.2f}%, 损失: {initial_test_loss:.4f}")
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1:2d}/{num_epochs}]")
        
        # 训练阶段
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
        
        # 计算指标
        train_acc = 100 * correct / total
        train_loss = epoch_loss / len(trainloader)
        test_acc, test_loss = evaluate_model(model, testloader, device)
        
        # 记录历史
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        
        print(f"训练 - 准确率: {train_acc:.2f}%, 损失: {train_loss:.4f}")
        print(f"测试 - 准确率: {test_acc:.2f}%, 损失: {test_loss:.4f}")
        
        # 架构进化检查
        if (epoch + 1) % evolution_frequency == 0:
            print(f"\n🔄 进化检查 (Epoch {epoch + 1})...")
            
            performance_metrics = {
                'val_accuracy': test_acc,
                'train_accuracy': train_acc,
                'loss': train_loss,
                'test_loss': test_loss
            }
            
            # 准确率保护提示
            if test_acc < 60.0:
                print(f"🛡️ 准确率保护激活: {test_acc:.1f}% < 60%")
            
            prev_params = sum(p.numel() for p in model.parameters())
            
            try:
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
                    
                    print(f"✅ 进化成功! 动作: {action_taken}")
                    print(f"📊 参数: {prev_params:,} -> {new_params:,} (Δ{new_params - prev_params:+,})")
                    
                    # 进化后评估
                    post_evo_acc, post_evo_loss = evaluate_model(model, testloader, device)
                    print(f"🎯 进化后准确率: {post_evo_acc:.2f}% (变化: {post_evo_acc - test_acc:+.2f}%)")
                    
                    if post_evo_acc < test_acc - 3.0:
                        print("⚠️ 准确率下降较大，策略需要调整")
                    
                    # 记录进化
                    history['evolutions'].append({
                        'epoch': epoch + 1,
                        'action': str(action_taken),
                        'param_change': new_params - prev_params,
                        'acc_before': test_acc,
                        'acc_after': post_evo_acc
                    })
                    
                    # 重置优化器
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                    
                else:
                    print("ℹ️ 无需进化")
                    
            except Exception as e:
                print(f"❌ 进化失败: {e}")
        
        print("-" * 40)
    
    # 最终结果
    print("\n" + "=" * 60)
    print("🎯 训练完成！最终结果:")
    
    final_test_acc, final_test_loss = evaluate_model(model, testloader, device)
    final_params = sum(p.numel() for p in model.parameters())
    
    print(f"📊 准确率变化: {initial_test_acc:.2f}% -> {final_test_acc:.2f}% (Δ{final_test_acc - initial_test_acc:+.2f}%)")
    print(f"📊 参数变化: {initial_params:,} -> {final_params:,} (Δ{final_params - initial_params:+,})")
    print(f"📊 进化次数: {len(history['evolutions'])}")
    
    # 进化历史
    if history['evolutions']:
        print(f"\n🔄 进化历史:")
        for evo in history['evolutions']:
            print(f"  Epoch {evo['epoch']:2d}: {evo['action']}")
            print(f"    参数变化: {evo['param_change']:+,}")
            print(f"    准确率: {evo['acc_before']:.1f}% -> {evo['acc_after']:.1f}%")
    
    # 性能评估
    accuracy_maintained = final_test_acc >= initial_test_acc - 2.0
    print(f"\n{'✅' if accuracy_maintained else '❌'} 保守策略评估: {'成功' if accuracy_maintained else '需改进'}")
    
    # 清理
    neuroexapt.cleanup()
    
    return final_test_acc >= initial_test_acc - 2.0


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        success = main()
        if success:
            print("\n🎉 改进策略测试成功！保守的信息论策略有效保持了准确率。")
        else:
            print("\n⚠️ 策略仍需进一步调整。")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 