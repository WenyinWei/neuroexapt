"""
Test channel expansion to debug the tensor shape mismatch issue.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt.core.smart_channel_expander import SmartChannelExpander


class TestCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),      # features.0
            nn.BatchNorm2d(32),                              # features.1  
            nn.ReLU(inplace=True),                           # features.2
            nn.Conv2d(32, 32, kernel_size=3, padding=1),     # features.3
            nn.BatchNorm2d(32),                              # features.4
            nn.ReLU(inplace=True),                           # features.5
            nn.Conv2d(32, 64, kernel_size=3, padding=1),     # features.6
            nn.BatchNorm2d(64),                              # features.7
            nn.ReLU(inplace=True),                           # features.8
        )
    
    def forward(self, x):
        return self.features(x)


def test_single_expansion():
    """测试单个层的扩展"""
    print("=" * 60)
    print("测试单个层的通道扩展")
    print("=" * 60)
    
    # 创建测试模型
    model = TestCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print("原始模型结构:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            if isinstance(module, nn.Conv2d):
                print(f"  {name}: Conv2d({module.in_channels}, {module.out_channels})")
            else:
                print(f"  {name}: BatchNorm2d({module.num_features})")
    
    # 测试前向传播
    test_input = torch.randn(2, 3, 32, 32, device=device)
    print(f"\n测试输入形状: {test_input.shape}")
    
    with torch.no_grad():
        output = model(test_input)
        print(f"测试输出形状: {output.shape}")
    
    # 创建扩展器
    expander = SmartChannelExpander()
    
    # 执行扩展 - 只扩展第一层
    print(f"\n执行通道扩展 features.0...")
    
    success = expander._expand_layer_smart(
        model, 
        'features.0', 
        model.features[0],  # features.0
        factor=2.0
    )
    
    print(f"扩展结果: {'成功' if success else '失败'}")
    
    print("\n扩展后模型结构:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            if isinstance(module, nn.Conv2d):
                print(f"  {name}: Conv2d({module.in_channels}, {module.out_channels})")
            else:
                print(f"  {name}: BatchNorm2d({module.num_features})")
    
    # 测试扩展后的前向传播
    print(f"\n测试扩展后的前向传播...")
    try:
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ 成功! 输出形状: {output.shape}")
    except Exception as e:
        print(f"❌ 失败: {e}")
        
        # 详细诊断
        print("\n详细诊断:")
        x = test_input
        for i, layer in enumerate(model.features):
            try:
                x = layer(x)
                print(f"  Layer {i} ({type(layer).__name__}): {x.shape}")
            except Exception as layer_error:
                print(f"  ❌ Layer {i} ({type(layer).__name__}) 失败: {layer_error}")
                if isinstance(layer, nn.Conv2d):
                    print(f"    期望输入: {layer.in_channels} 通道")
                    print(f"    实际输入: {x.shape[1]} 通道")
                    print(f"    权重形状: {layer.weight.shape}")
                break


if __name__ == "__main__":
    test_single_expansion() 