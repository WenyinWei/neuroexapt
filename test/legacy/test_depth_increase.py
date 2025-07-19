"""
测试深度增加操作 - 验证是否有张量形状不匹配和内存泄露问题
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
import gc
import psutil
import time

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt.core.depth_expansion_operators import DepthExpansionOperator, get_depth_expansion_operators

class TestCNN(nn.Module):
    """用于测试深度增加的简单CNN"""
    def __init__(self):
        super(TestCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_test_data(batch_size=32):
    """加载测试数据"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 只使用前500个样本进行快速测试
    indices = list(range(500))
    subset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def evaluate_model(model, dataloader, device):
    """评估模型"""
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
                print(f"❌ Forward pass failed: {e}")
                return None, None
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    model.train()
    return accuracy, avg_loss

def test_tensor_shape_compatibility(model, dataloader, device):
    """测试张量形状兼容性"""
    print("\n🔍 Testing tensor shape compatibility...")
    
    model.eval()
    with torch.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            
            try:
                outputs = model(data)
                print(f"   Batch {i+1}: Input {tuple(data.shape)} → Output {tuple(outputs.shape)} ✅")
                
                if i >= 5:  # 测试前5个batch
                    break
                    
            except Exception as e:
                print(f"   Batch {i+1}: FAILED - {e} ❌")
                return False
    
    print("   ✅ All tensor shapes compatible!")
    return True

def test_memory_leaks(model, dataloader, device, num_iterations=10):
    """测试内存泄露"""
    print(f"\n🧠 Testing memory leaks ({num_iterations} iterations)...")
    
    initial_memory = get_memory_usage()
    print(f"   Initial memory: {initial_memory:.1f} MB")
    
    memory_readings = [initial_memory]
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for iteration in range(num_iterations):
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if batch_idx >= 3:  # 只做几个batch
                break
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        current_memory = get_memory_usage()
        memory_readings.append(current_memory)
        print(f"   Iteration {iteration+1}: {current_memory:.1f} MB")
    
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    print(f"   Final memory: {final_memory:.1f} MB")
    print(f"   Memory increase: {memory_increase:.1f} MB")
    
    # 判断是否有严重的内存泄露
    if memory_increase > 100:  # 超过100MB认为有问题
        print(f"   ⚠️ Potential memory leak detected!")
        return False
    else:
        print(f"   ✅ Memory usage acceptable!")
        return True

def test_depth_increase():
    """测试深度增加操作"""
    print("=" * 60)
    print("🧪 深度增加测试")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 记录初始内存
    initial_memory = get_memory_usage()
    print(f"Initial system memory: {initial_memory:.1f} MB")
    
    # 创建模型
    model = TestCNN().to(device)
    dataloader = load_test_data(batch_size=32)
    
    # 初始架构分析
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Initial Architecture:")
    print(f"   Parameters: {initial_params:,}")
    
    # 检查层数的更安全方式
    features_layers = len(list(model.features.children())) if isinstance(model.features, nn.Sequential) else "N/A"
    classifier_layers = len(list(model.classifier.children())) if isinstance(model.classifier, nn.Sequential) else "N/A"
    
    print(f"   Layers in features: {features_layers}")
    print(f"   Layers in classifier: {classifier_layers}")
    
    # 测试初始张量形状兼容性
    initial_compatibility = test_tensor_shape_compatibility(model, dataloader, device)
    if not initial_compatibility:
        print("❌ Initial model has tensor shape issues!")
        return
    
    # 评估初始性能
    initial_accuracy, initial_loss = evaluate_model(model, dataloader, device)
    if initial_accuracy is None:
        print("❌ Initial model evaluation failed!")
        return
    
    print(f"\n📊 Initial Performance:")
    print(f"   Accuracy: {initial_accuracy:.2f}%")
    print(f"   Loss: {initial_loss:.4f}")
    
    # 创建深度扩展操作器
    depth_expander = DepthExpansionOperator(min_accuracy_for_pruning=0.9)  # 设置高阈值确保触发
    
    # 模拟低准确率状态来触发深度扩展
    test_state = {
        'current_performance': initial_accuracy / 100.0,
        'val_accuracy': initial_accuracy,
        'epoch': 5
    }
    
    print(f"\n🏗️ Applying depth expansion...")
    print(f"   Simulated accuracy: {initial_accuracy:.2f}%")
    
    # 应用深度扩展
    evolved_model = depth_expander(model, test_state)
    
    if evolved_model is not None:
        print("✅ Depth expansion successful!")
        
        # 检查扩展后的架构
        expanded_params = sum(p.numel() for p in evolved_model.parameters())
        print(f"\n📊 Expanded Architecture:")
        print(f"   Parameters: {initial_params:,} → {expanded_params:,}")
        print(f"   Increase: {expanded_params - initial_params:,} (+{((expanded_params - initial_params) / initial_params * 100):.1f}%)")
        
        # 检查扩展后的层数
        evolved_features_layers = len(list(evolved_model.features.children())) if isinstance(evolved_model.features, nn.Sequential) else "N/A"
        evolved_classifier_layers = len(list(evolved_model.classifier.children())) if isinstance(evolved_model.classifier, nn.Sequential) else "N/A"
        
        print(f"   Features layers: {features_layers} → {evolved_features_layers}")
        print(f"   Classifier layers: {classifier_layers} → {evolved_classifier_layers}")
        
        # 测试张量形状兼容性
        post_expansion_compatibility = test_tensor_shape_compatibility(evolved_model, dataloader, device)
        
        if post_expansion_compatibility:
            print("✅ Post-expansion tensor shapes compatible!")
            
            # 立即评估扩展后的性能
            immediate_accuracy, immediate_loss = evaluate_model(evolved_model, dataloader, device)
            
            if immediate_accuracy is not None:
                print(f"\n📊 Immediate Post-Expansion Performance:")
                print(f"   Accuracy: {immediate_accuracy:.2f}%")
                print(f"   Loss: {immediate_loss:.4f}")
                print(f"   Accuracy Change: {immediate_accuracy - initial_accuracy:+.2f}%")
                
                # 测试内存泄露
                memory_ok = test_memory_leaks(evolved_model, dataloader, device)
                
                if memory_ok:
                    print("✅ No significant memory leaks detected!")
                else:
                    print("⚠️ Potential memory issues detected!")
                
                # 最终评估
                print(f"\n🎯 Depth Increase Assessment:")
                accuracy_change = immediate_accuracy - initial_accuracy
                
                if post_expansion_compatibility and memory_ok:
                    if accuracy_change >= -2.0:  # 允许小幅度下降
                        print(f"🎉 DEPTH INCREASE SUCCESS!")
                        print(f"   ✅ Shape compatibility: OK")
                        print(f"   ✅ Memory usage: OK")
                        print(f"   ✅ Accuracy change: {accuracy_change:+.2f}% (acceptable)")
                        print(f"   ✅ Architecture expansion: {((expanded_params - initial_params) / initial_params * 100):.1f}%")
                    else:
                        print(f"⚠️ DEPTH INCREASE PARTIAL SUCCESS")
                        print(f"   ✅ Shape compatibility: OK")
                        print(f"   ✅ Memory usage: OK")
                        print(f"   ⚠️ Accuracy drop: {accuracy_change:+.2f}% (significant)")
                else:
                    print(f"❌ DEPTH INCREASE FAILED")
                    print(f"   {'✅' if post_expansion_compatibility else '❌'} Shape compatibility")
                    print(f"   {'✅' if memory_ok else '❌'} Memory usage")
            else:
                print("❌ Post-expansion evaluation failed!")
        else:
            print("❌ Post-expansion tensor shape compatibility failed!")
    else:
        print("❌ Depth expansion failed!")
        print("   No depth expansion was applied.")
    
    # 清理内存
    del model
    if 'evolved_model' in locals():
        del evolved_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    final_memory = get_memory_usage()
    print(f"\nFinal system memory: {final_memory:.1f} MB")
    print(f"Total memory change: {final_memory - initial_memory:+.1f} MB")
    
    print("\n" + "=" * 60)
    print("🧪 深度增加测试完成")
    print("=" * 60)

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        test_depth_increase()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 