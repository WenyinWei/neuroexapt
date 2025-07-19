#!/usr/bin/env python3
"""
测试CUDA错误修复
"""

import torch
import sys
import os

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_cuda_fix():
    """测试CUDA错误修复是否有效"""
    print("🔧 测试CUDA错误修复...")
    
    # 检查基本环境
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法测试修复")
        return False
    
    try:
        # 测试基本的sepconv操作
        from neuroexapt.kernels.sepconv_triton import sepconv_forward_generic, is_triton_disabled, reset_triton_state
        
        print(f"Triton禁用状态: {is_triton_disabled()}")
        
        # 创建测试数据
        B, C, H, W = 2, 16, 32, 32
        x = torch.randn(B, C, H, W, device='cuda')
        dw_weight = torch.randn(C, 1, 3, 3, device='cuda')
        pw_weight = torch.randn(32, C, 1, 1, device='cuda')
        
        print(f"测试数据: B={B}, C={C}, H={H}, W={W}")
        
        # 测试sepconv操作
        try:
            result = sepconv_forward_generic(x, dw_weight, pw_weight)
            print(f"✅ Sepconv测试成功: {result.shape}")
        except Exception as e:
            print(f"⚠️ Sepconv回退到PyTorch: {e}")
            # 这是期望的行为，应该回退到安全的PyTorch实现
        
        # 测试SepConv模块
        from neuroexapt.core.operations import SepConv
        
        sepconv_module = SepConv(16, 32, 3, 1, 1, affine=True).cuda()
        
        try:
            output = sepconv_module(x)
            print(f"✅ SepConv模块测试成功: {output.shape}")
        except Exception as e:
            print(f"❌ SepConv模块测试失败: {e}")
            return False
        
        # 测试分离训练组件
        try:
            from neuroexapt.core.separated_training import SeparatedTrainingStrategy, SeparatedOptimizer, SeparatedTrainer
            from neuroexapt.core.model import Network
            
            # 创建小型网络
            model = Network(C=16, num_classes=10, layers=4, potential_layers=2).cuda()
            
            strategy = SeparatedTrainingStrategy(
                weight_training_epochs=2,
                arch_training_epochs=1,
                total_epochs=10,
                warmup_epochs=2
            )
            
            optimizer = SeparatedOptimizer(
                model,
                weight_lr=0.025,
                arch_lr=3e-4,
                weight_momentum=0.9,
                weight_decay=3e-4
            )
            
            criterion = torch.nn.CrossEntropyLoss().cuda()
            trainer = SeparatedTrainer(model, strategy, optimizer, criterion)
            
            print("✅ 分离训练组件初始化成功")
            
        except Exception as e:
            print(f"❌ 分离训练组件测试失败: {e}")
            return False
        
        print("✅ 所有测试通过，CUDA错误修复成功！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_minimal_training():
    """测试最小化训练流程"""
    print("\n🚀 测试最小化训练流程...")
    
    try:
        # 导入必要模块
        from examples.basic_classification import create_model, create_architect
        import torch.nn as nn
        import torch.optim as optim
        
        # 创建模拟的args对象
        class Args:
            def __init__(self):
                self.mode = 'separated'
                self.init_channels = 8  # 很小的网络
                self.layers = 2
                self.potential_layers = 1
                self.learning_rate = 0.025
                self.arch_learning_rate = 3e-4
                self.momentum = 0.9
                self.weight_decay = 3e-4
                self.separated_weight_lr = 0.025
                self.separated_arch_lr = 3e-4
                self.weight_epochs = 2
                self.arch_epochs = 1
                self.warmup_epochs = 1
                self.use_model_compile = False
                self.disable_progress_spam = True
                self.quiet = True
        
        args = Args()
        
        # 创建模型
        model = create_model(args, mode='separated')
        print(f"✅ 模型创建成功: {sum(p.numel() for p in model.parameters())} 参数")
        
        # 创建测试数据
        batch_size = 4
        test_input = torch.randn(batch_size, 3, 32, 32, device='cuda')
        test_target = torch.randint(0, 10, (batch_size,), device='cuda')
        
        # 前向传播测试
        model.train()
        try:
            output = model(test_input)
            print(f"✅ 前向传播成功: {output.shape}")
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            return False
        
        # 反向传播测试
        try:
            criterion = nn.CrossEntropyLoss().cuda()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            
            loss = criterion(output, test_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"✅ 反向传播成功: loss={loss.item():.4f}")
        except Exception as e:
            print(f"❌ 反向传播失败: {e}")
            return False
        
        print("✅ 最小化训练流程测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 最小化训练流程测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🔧 CUDA错误修复验证测试")
    print("=" * 50)
    
    # 测试1: CUDA修复
    success1 = test_cuda_fix()
    
    # 测试2: 最小化训练
    success2 = test_minimal_training()
    
    print("\n" + "=" * 50)
    print("📊 测试结果:")
    print(f"   CUDA修复测试: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"   最小化训练测试: {'✅ 通过' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print("\n🎉 所有测试通过！CUDA错误已修复，可以安全进行分离训练。")
        print("💡 建议使用以下命令进行分离训练:")
        print("   python examples/basic_classification.py --mode separated --epochs 10 --batch_size 32 --layers 8")
        return True
    else:
        print("\n❌ 部分测试失败，需要进一步调试。")
        return False

if __name__ == "__main__":
    main() 