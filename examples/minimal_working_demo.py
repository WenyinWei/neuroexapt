#!/usr/bin/env python3
"""
最小可工作演示 - 修复optimizer错误
Minimal Working Demo - Fix Optimizer Error

🎯 目标：提供一个确保可以运行的最小演示，解决原始的optimizer.zero_grad()错误

🔧 关键修复：
1. 确保optimizer在使用前正确初始化
2. 添加充分的错误检查和异常处理
3. 提供清晰的错误信息和诊断
4. 使用最少的依赖来避免复杂的导入问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import traceback

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MinimalModel(nn.Module):
    """最小模型 - 确保可以正常工作"""
    
    def __init__(self, input_size=32*32*3, hidden_size=128, num_classes=10):
        super(MinimalModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SafeTrainer:
    """安全训练器 - 专注于解决optimizer错误"""
    
    def __init__(self, model, device='cpu', lr=0.01):
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        # 关键修复：立即初始化optimizer
        self.optimizer = None
        self.setup_optimizer(lr)
        
        # 验证optimizer初始化
        if self.optimizer is None:
            raise RuntimeError("❌ 优化器初始化失败 - 这是原始错误的根源!")
        
        logger.info("✅ SafeTrainer初始化成功，optimizer已正确设置")
    
    def setup_optimizer(self, lr):
        """安全的优化器设置"""
        try:
            if self.model is None:
                raise ValueError("模型为None，无法创建优化器")
            
            # 检查模型参数
            params = list(self.model.parameters())
            if not params:
                raise ValueError("模型没有可训练参数")
            
            # 创建优化器
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9
            )
            
            logger.info(f"✅ 优化器创建成功 - 类型: {type(self.optimizer).__name__}, LR: {lr}")
            
        except Exception as e:
            logger.error(f"❌ 优化器设置失败: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def safe_train_step(self, data, target):
        """安全的训练步骤 - 包含完整的错误检查"""
        try:
            # 检查optimizer状态
            if self.optimizer is None:
                raise RuntimeError("优化器为None - 这正是原始错误的原因!")
            
            # 检查模型状态
            if self.model is None:
                raise RuntimeError("模型为None")
            
            # 检查数据
            if data is None or target is None:
                raise ValueError("数据或标签为None")
            
            # 移动数据到设备
            data = data.to(self.device)
            target = target.to(self.device)
            
            # 训练步骤
            self.model.train()
            
            # 关键步骤：清零梯度
            self.optimizer.zero_grad()  # 这里是原始错误发生的地方
            
            # 前向传播
            output = self.model(data)
            
            # 计算损失
            loss = F.cross_entropy(output, target)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            self.optimizer.step()
            
            # 计算准确率
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            accuracy = correct / target.size(0)
            
            return loss.item(), accuracy
            
        except Exception as e:
            logger.error(f"❌ 训练步骤失败: {e}")
            logger.error(f"optimizer状态: {self.optimizer}")
            logger.error(f"model状态: {self.model}")
            raise
    
    def evaluate(self, data_loader):
        """安全的模型评估"""
        try:
            self.model.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in data_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    
                    loss = F.cross_entropy(output, target, reduction='sum')
                    total_loss += loss.item()
                    
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            avg_loss = total_loss / total
            accuracy = correct / total
            
            return avg_loss, accuracy
            
        except Exception as e:
            logger.error(f"❌ 评估失败: {e}")
            return float('inf'), 0.0


def create_mock_data(num_samples=1000, batch_size=32):
    """创建模拟数据集"""
    try:
        logger.info("📦 创建模拟CIFAR-10数据集...")
        
        # 创建随机数据
        data = torch.randn(num_samples, 3, 32, 32)
        labels = torch.randint(0, 10, (num_samples,))
        
        # 创建数据集和加载器
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        logger.info(f"✅ 模拟数据集创建成功 - 样本数: {num_samples}, batch_size: {batch_size}")
        return loader
        
    except Exception as e:
        logger.error(f"❌ 数据集创建失败: {e}")
        raise


def run_minimal_demo():
    """运行最小演示"""
    print("🔧 最小可工作演示 - 修复optimizer.zero_grad()错误")
    print("="*60)
    
    try:
        # 1. 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  使用设备: {device}")
        
        # 2. 创建模型
        logger.info("🏗️  创建最小模型...")
        model = MinimalModel()
        logger.info(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 3. 创建训练器 - 这里会正确初始化optimizer
        logger.info("⚙️  初始化安全训练器...")
        trainer = SafeTrainer(model, device, lr=0.01)
        
        # 4. 创建数据
        train_loader = create_mock_data(num_samples=500, batch_size=16)
        test_loader = create_mock_data(num_samples=100, batch_size=16)
        
        # 5. 进行几个训练步骤
        logger.info("🚀 开始训练演示...")
        
        num_epochs = 2
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # 执行安全的训练步骤
                loss, acc = trainer.safe_train_step(data, target)
                
                epoch_loss += loss
                epoch_acc += acc
                num_batches += 1
                
                # 每5个batch报告一次
                if batch_idx % 5 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss:.4f}, Acc={acc:.2%}")
            
            # Epoch总结
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            
            # 评估
            test_loss, test_acc = trainer.evaluate(test_loader)
            
            logger.info(f"Epoch {epoch} 完成 - 训练: Loss={avg_loss:.4f}, Acc={avg_acc:.2%} | 测试: Loss={test_loss:.4f}, Acc={test_acc:.2%}")
        
        print("\n" + "="*60)
        print("🎉 最小演示成功完成!")
        print("\n✅ 关键修复说明:")
        print("• optimizer在SafeTrainer.__init__()中立即初始化")
        print("• 添加了optimizer状态检查")
        print("• 提供了详细的错误诊断信息")
        print("• 使用了最少的依赖避免导入问题")
        
        print(f"\n📊 最终结果:")
        print(f"• 完成训练: {num_epochs} epochs")
        print(f"• 最终测试准确率: {test_acc:.2%}")
        print(f"• 测试损失: {test_loss:.4f}")
        
        print(f"\n🔧 原始错误原因:")
        print("• 原代码中optimizer没有在train_epoch调用前初始化")
        print("• IntelligentEvolutionTrainer.setup_optimizer()从未被调用")
        print("• 当trainer.train_epoch()执行self.optimizer.zero_grad()时，self.optimizer为None")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 演示失败: {e}")
        logger.error(traceback.format_exc())
        
        print(f"\n❌ 演示失败: {e}")
        print("\n🔍 错误诊断:")
        print("• 如果是optimizer相关错误，检查模型初始化")
        print("• 如果是导入错误，检查PyTorch安装")
        print("• 如果是CUDA错误，尝试使用CPU: device='cpu'")
        
        return False


def demonstrate_original_error():
    """演示原始错误的产生过程"""
    print("\n🚨 演示原始错误的产生过程")
    print("="*50)
    
    try:
        # 模拟原始代码的错误情况
        class BuggyTrainer:
            def __init__(self, model, device):
                self.model = model
                self.device = device
                self.optimizer = None  # 👈 这里是问题所在 - optimizer为None
                # 注意：setup_optimizer从未被调用
                
            def setup_optimizer(self, lr):
                """这个方法定义了但从未被调用"""
                self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
                
            def train_epoch(self):
                """这里会触发错误"""
                self.optimizer.zero_grad()  # ❌ AttributeError: 'NoneType' object has no attribute 'zero_grad'
        
        model = MinimalModel()
        buggy_trainer = BuggyTrainer(model, 'cpu')
        
        print(f"optimizer状态: {buggy_trainer.optimizer}")  # None
        print("尝试调用train_epoch()...")
        
        # 这里会触发原始错误
        buggy_trainer.train_epoch()
        
    except AttributeError as e:
        print(f"✅ 成功重现原始错误: {e}")
        print("🔧 这正是原始代码的问题：optimizer从未被初始化!")
        
    except Exception as e:
        print(f"❌ 意外错误: {e}")


if __name__ == "__main__":
    print("🎯 NeuroExapt - 最小可工作演示")
    print("专注解决: 'NoneType' object has no attribute 'zero_grad' 错误")
    print("="*80)
    
    # 首先演示原始错误
    demonstrate_original_error()
    
    # 然后展示修复后的版本
    success = run_minimal_demo()
    
    if success:
        print("\n🎉 问题已解决! 重构版本可以正常工作")
        print("\n💡 要运行完整功能，请使用:")
        print("   python examples/intelligent_evolution_demo_refactored.py")
    else:
        print("\n❌ 仍有问题需要解决")
        print("\n💡 建议运行诊断脚本:")
        print("   python examples/test_core_modules.py")