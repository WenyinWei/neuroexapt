"""
通道扩展 vs 深度增加对比测试
比较两种扩展方式的效果，确定哪种更适合提升准确率
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
import copy

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt.core.smart_channel_expander import SmartChannelExpander
from neuroexapt.core.depth_expansion_operators import DepthExpansionOperator

class ComparisonCNN(nn.Module):
    """用于对比测试的CNN"""
    def __init__(self):
        super(ComparisonCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 10)
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
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 使用前2000个样本进行测试
    indices = list(range(2000))
    subset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return dataloader

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
                print(f"❌ Evaluation failed: {e}")
                return None, None
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    model.train()
    return accuracy, avg_loss

def train_brief(model, dataloader, device, num_epochs=5):
    """简短训练"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        batches_processed = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches_processed += 1
            
            if batch_idx >= 15:  # 限制每个epoch的batch数
                break
        
        avg_loss = epoch_loss / batches_processed if batches_processed > 0 else 0
        print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

def test_expansion_method(model, dataloader, device, method_name, expander):
    """测试单一扩展方法"""
    print(f"\n🧪 Testing {method_name}...")
    
    # 克隆模型以避免互相影响
    test_model = copy.deepcopy(model).to(device)
    
    # 初始训练
    print(f"   📚 Initial training...")
    test_model = train_brief(test_model, dataloader, device, num_epochs=3)
    
    # 评估初始性能
    initial_accuracy, initial_loss = evaluate_model(test_model, dataloader, device)
    initial_params = sum(p.numel() for p in test_model.parameters())
    
    print(f"   📊 Initial Performance:")
    print(f"      Accuracy: {initial_accuracy:.2f}%")
    print(f"      Loss: {initial_loss:.4f}")
    print(f"      Parameters: {initial_params:,}")
    
    # 应用扩展
    test_state = {
        'current_performance': initial_accuracy / 100.0,
        'val_accuracy': initial_accuracy,
        'epoch': 5
    }
    
    print(f"   🔄 Applying {method_name}...")
    evolved_model = expander(test_model, test_state)
    
    if evolved_model is not None:
        # 检查扩展后的架构
        expanded_params = sum(p.numel() for p in evolved_model.parameters())
        param_increase = expanded_params - initial_params
        param_increase_pct = (param_increase / initial_params) * 100
        
        print(f"   ✅ {method_name} successful!")
        print(f"      Parameters: {initial_params:,} → {expanded_params:,}")
        print(f"      Increase: +{param_increase:,} (+{param_increase_pct:.1f}%)")
        
        # 立即评估
        immediate_accuracy, immediate_loss = evaluate_model(evolved_model, dataloader, device)
        immediate_change = immediate_accuracy - initial_accuracy
        
        print(f"   📊 Immediate Post-Expansion:")
        print(f"      Accuracy: {immediate_accuracy:.2f}% ({immediate_change:+.2f}%)")
        print(f"      Loss: {immediate_loss:.4f}")
        
        # 短暂重新训练
        print(f"   📚 Brief retraining...")
        evolved_model = train_brief(evolved_model, dataloader, device, num_epochs=3)
        
        # 最终评估
        final_accuracy, final_loss = evaluate_model(evolved_model, dataloader, device)
        final_change = final_accuracy - initial_accuracy
        recovery = final_accuracy - immediate_accuracy
        
        print(f"   📊 Final Performance:")
        print(f"      Accuracy: {final_accuracy:.2f}% ({final_change:+.2f}%)")
        print(f"      Loss: {final_loss:.4f}")
        print(f"      Recovery: {recovery:+.2f}%")
        
        return {
            'method': method_name,
            'success': True,
            'initial_accuracy': initial_accuracy,
            'immediate_accuracy': immediate_accuracy,
            'final_accuracy': final_accuracy,
            'immediate_change': immediate_change,
            'final_change': final_change,
            'recovery': recovery,
            'initial_params': initial_params,
            'expanded_params': expanded_params,
            'param_increase': param_increase,
            'param_increase_pct': param_increase_pct
        }
    else:
        print(f"   ❌ {method_name} failed!")
        return {
            'method': method_name,
            'success': False,
            'initial_accuracy': initial_accuracy,
            'initial_params': initial_params
        }

def test_combined_expansion():
    """对比测试通道扩展和深度增加"""
    print("=" * 80)
    print("🧪 通道扩展 vs 深度增加对比测试")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 加载数据
    dataloader = load_test_data(batch_size=64)
    
    # 创建基础模型
    base_model = ComparisonCNN().to(device)
    
    # 创建扩展器
    channel_expander = SmartChannelExpander(accuracy_threshold=0.9)  # 确保触发
    depth_expander = DepthExpansionOperator(min_accuracy_for_pruning=0.9)  # 确保触发
    
    # 测试两种方法
    results = []
    
    # 测试通道扩展
    channel_result = test_expansion_method(
        base_model, dataloader, device, 
        "Channel Expansion", channel_expander
    )
    results.append(channel_result)
    
    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 测试深度增加
    depth_result = test_expansion_method(
        base_model, dataloader, device, 
        "Depth Increase", depth_expander
    )
    results.append(depth_result)
    
    # 比较结果
    print(f"\n" + "=" * 80)
    print("📊 COMPARISON RESULTS")
    print("=" * 80)
    
    for result in results:
        if result['success']:
            print(f"\n🔸 {result['method']}:")
            print(f"   Immediate Impact: {result['immediate_change']:+.2f}%")
            print(f"   Final Improvement: {result['final_change']:+.2f}%")
            print(f"   Recovery Ability: {result['recovery']:+.2f}%")
            print(f"   Parameter Increase: +{result['param_increase_pct']:.1f}%")
            print(f"   Efficiency: {result['final_change']/result['param_increase_pct']:.3f} accuracy/param%")
        else:
            print(f"\n🔸 {result['method']}: FAILED")
    
    # 确定最佳方法
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) >= 2:
        # 比较指标
        channel_res = next(r for r in successful_results if 'Channel' in r['method'])
        depth_res = next(r for r in successful_results if 'Depth' in r['method'])
        
        print(f"\n🏆 HEAD-TO-HEAD COMPARISON:")
        print(f"   Final Accuracy:")
        print(f"      Channel Expansion: {channel_res['final_change']:+.2f}%")
        print(f"      Depth Increase:    {depth_res['final_change']:+.2f}%")
        
        print(f"   Immediate Stability:")
        print(f"      Channel Expansion: {channel_res['immediate_change']:+.2f}%")
        print(f"      Depth Increase:    {depth_res['immediate_change']:+.2f}%")
        
        print(f"   Parameter Efficiency:")
        channel_efficiency = channel_res['final_change'] / channel_res['param_increase_pct']
        depth_efficiency = depth_res['final_change'] / depth_res['param_increase_pct']
        print(f"      Channel Expansion: {channel_efficiency:.4f}")
        print(f"      Depth Increase:    {depth_efficiency:.4f}")
        
        # 综合评分
        channel_score = (
            channel_res['final_change'] * 0.4 +  # 最终改进权重40%
            max(0, channel_res['immediate_change']) * 0.3 +  # 立即稳定性权重30%
            channel_efficiency * 10 * 0.3  # 参数效率权重30%
        )
        
        depth_score = (
            depth_res['final_change'] * 0.4 +
            max(0, depth_res['immediate_change']) * 0.3 +
            depth_efficiency * 10 * 0.3
        )
        
        print(f"\n🎯 RECOMMENDATION:")
        if channel_score > depth_score:
            print(f"   🥇 CHANNEL EXPANSION is recommended!")
            print(f"      Score: {channel_score:.2f} vs {depth_score:.2f}")
            print(f"      Reasons: Better accuracy improvement and/or stability")
        elif depth_score > channel_score:
            print(f"   🥇 DEPTH INCREASE is recommended!")
            print(f"      Score: {depth_score:.2f} vs {channel_score:.2f}")
            print(f"      Reasons: Better accuracy improvement and/or stability")
        else:
            print(f"   🤝 Both methods are equally effective!")
            print(f"      Scores: Channel {channel_score:.2f}, Depth {depth_score:.2f}")
        
        print(f"\n💡 PRACTICAL INSIGHTS:")
        if abs(channel_res['immediate_change']) < abs(depth_res['immediate_change']):
            print(f"   • Channel expansion is more stable (less immediate accuracy drop)")
        else:
            print(f"   • Depth increase is more stable (less immediate accuracy drop)")
        
        if channel_res['final_change'] > depth_res['final_change']:
            print(f"   • Channel expansion provides better final accuracy improvement")
        else:
            print(f"   • Depth increase provides better final accuracy improvement")
        
        if channel_efficiency > depth_efficiency:
            print(f"   • Channel expansion is more parameter-efficient")
        else:
            print(f"   • Depth increase is more parameter-efficient")
    
    elif len(successful_results) == 1:
        winner = successful_results[0]
        print(f"\n🏆 WINNER: {winner['method']}")
        print(f"   Only method that worked successfully!")
    
    else:
        print(f"\n❌ Both methods failed!")
        print(f"   需要进一步调试和优化")
    
    print(f"\n" + "=" * 80)
    print("🧪 对比测试完成")
    print("=" * 80)

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        test_combined_expansion()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 