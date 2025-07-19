"""
测试平滑通道扩展 - 验证参数迁移的平滑性
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

from neuroexapt.core.smart_channel_expander import SmartChannelExpander

class TestCNN(nn.Module):
    """简单的测试CNN"""
    def __init__(self):
        super(TestCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_cifar10_sample(batch_size=64):
    """加载CIFAR-10数据集的小样本用于测试"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 使用小样本进行快速测试
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 只使用前1000个样本
    indices = list(range(1000))
    subset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return dataloader

def evaluate_model(model, dataloader, device):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            try:
                outputs = model(data)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            except Exception as e:
                print(f"Warning: Evaluation failed: {e}")
                continue
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    model.train()
    return accuracy, avg_loss

def train_brief(model, dataloader, device, num_epochs=3):
    """简短训练用于测试"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx >= 10:  # 只训练几个batch
                break
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/(batch_idx+1):.4f}")

def test_smooth_expansion():
    """测试平滑通道扩展"""
    print("=" * 60)
    print("🧪 平滑通道扩展测试")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 创建模型
    model = TestCNN().to(device)
    
    # 加载数据
    dataloader = load_cifar10_sample(batch_size=64)
    
    # 初始训练
    print("\n📚 Initial training...")
    train_brief(model, dataloader, device, num_epochs=5)
    
    # 评估初始性能
    initial_accuracy, initial_loss = evaluate_model(model, dataloader, device)
    print(f"\n📊 Initial Performance:")
    print(f"   Accuracy: {initial_accuracy:.2f}%")
    print(f"   Loss: {initial_loss:.4f}")
    
    # 检查初始参数
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {initial_params:,}")
    
    # 创建平滑扩展器
    expander = SmartChannelExpander(accuracy_threshold=0.7)
    
    # 模拟低准确率状态来触发扩展
    test_state = {
        'current_performance': initial_accuracy / 100.0,
        'val_accuracy': initial_accuracy,
        'epoch': 5
    }
    
    print(f"\n🔄 Applying smooth channel expansion...")
    print(f"   Simulated accuracy: {initial_accuracy:.2f}%")
    
    # 应用平滑扩展
    evolved_model = expander(model, test_state)
    
    if evolved_model is not None:
        print("✅ Channel expansion successful!")
        
        # 检查扩展后的参数
        expanded_params = sum(p.numel() for p in evolved_model.parameters())
        print(f"   Parameters: {initial_params:,} → {expanded_params:,}")
        print(f"   Increase: {expanded_params - initial_params:,} (+{((expanded_params - initial_params) / initial_params * 100):.1f}%)")
        
        # 立即评估扩展后的性能（无需重新训练）
        immediate_accuracy, immediate_loss = evaluate_model(evolved_model, dataloader, device)
        print(f"\n📊 Immediate Post-Expansion Performance:")
        print(f"   Accuracy: {immediate_accuracy:.2f}%")
        print(f"   Loss: {immediate_loss:.4f}")
        print(f"   Accuracy Change: {immediate_accuracy - initial_accuracy:+.2f}%")
        
        # 验证平滑性
        accuracy_drop = initial_accuracy - immediate_accuracy
        if accuracy_drop > 5.0:  # 如果准确率下降超过5%
            print(f"⚠️  WARNING: Significant accuracy drop detected: {accuracy_drop:.2f}%")
            print("   This indicates the parameter migration is not smooth enough!")
        elif accuracy_drop > 0:
            print(f"✅ Acceptable accuracy drop: {accuracy_drop:.2f}%")
            print("   Parameter migration is working smoothly!")
        else:
            print(f"🎉 Accuracy improved immediately: {-accuracy_drop:+.2f}%")
            print("   Excellent parameter migration!")
        
        # 短暂重新训练以验证恢复能力
        print(f"\n📚 Brief retraining after expansion...")
        train_brief(evolved_model, dataloader, device, num_epochs=3)
        
        # 最终评估
        final_accuracy, final_loss = evaluate_model(evolved_model, dataloader, device)
        print(f"\n📊 Final Performance After Retraining:")
        print(f"   Accuracy: {final_accuracy:.2f}%")
        print(f"   Loss: {final_loss:.4f}")
        print(f"   Total Change: {final_accuracy - initial_accuracy:+.2f}%")
        
        # 分析结果
        print(f"\n🔍 Smooth Migration Analysis:")
        print(f"   Initial → Immediate: {initial_accuracy:.2f}% → {immediate_accuracy:.2f}% ({immediate_accuracy - initial_accuracy:+.2f}%)")
        print(f"   Immediate → Final: {immediate_accuracy:.2f}% → {final_accuracy:.2f}% ({final_accuracy - immediate_accuracy:+.2f}%)")
        print(f"   Overall Improvement: {final_accuracy - initial_accuracy:+.2f}%")
        
        # 成功标准
        if accuracy_drop < 5.0 and final_accuracy >= initial_accuracy:
            print(f"\n🎉 SMOOTH MIGRATION SUCCESS!")
            print(f"   ✅ Immediate drop: {accuracy_drop:.2f}% < 5.0%")
            print(f"   ✅ Final improvement: {final_accuracy - initial_accuracy:+.2f}%")
        else:
            print(f"\n❌ SMOOTH MIGRATION NEEDS IMPROVEMENT!")
            print(f"   ⚠️ Immediate drop: {accuracy_drop:.2f}%")
            print(f"   ⚠️ Final change: {final_accuracy - initial_accuracy:+.2f}%")
        
    else:
        print("❌ Channel expansion failed!")
        print("   No expansion was applied.")
    
    print("\n" + "=" * 60)
    print("🧪 平滑通道扩展测试完成")
    print("=" * 60)

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        test_smooth_expansion()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 