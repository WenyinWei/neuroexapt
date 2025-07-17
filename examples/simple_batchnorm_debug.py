#!/usr/bin/env python3
"""
🔧 简化的BatchNorm同步调试脚本
直接测试DNM框架中的BatchNorm同步问题
"""

import torch
import torch.nn as nn
import logging
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 直接导入DNM模块，避免整个neuroexapt包的依赖问题
try:
    # 直接从文件导入，绕过__init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dnm_neuron_division", 
        "/workspace/neuroexapt/core/dnm_neuron_division.py"
    )
    dnm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dnm_module)
    DNMNeuronDivision = dnm_module.DNMNeuronDivision
    logger.info("✅ 成功导入DNM模块")
except Exception as e:
    logger.error(f"❌ 无法导入DNM模块: {e}")
    sys.exit(1)

class SimpleTestNet(nn.Module):
    """简单的测试网络，模拟ResNet结构"""
    
    def __init__(self):
        super().__init__()
        
        # 简单的stem结构 (类似ResNet)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),  # stem.0
            nn.BatchNorm2d(64),  # stem.1
            nn.ReLU(inplace=True),  # stem.2
        )
        
        # 简单的layer1结构
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),  # layer1.0
            nn.BatchNorm2d(64),  # layer1.1
            nn.ReLU(inplace=True),  # layer1.2
        )
        
        # 分类头
        self.classifier = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def test_batchnorm_sync():
    """测试BatchNorm同步功能"""
    
    logger.info("🧪 开始BatchNorm同步测试")
    
    # 创建测试网络
    model = SimpleTestNet()
    logger.info(f"原始模型参数: {sum(p.numel() for p in model.parameters())}")
    
    # 打印模型结构
    logger.info("📋 模型结构:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            if isinstance(module, nn.Conv2d):
                logger.info(f"  {name}: Conv2d(in={module.in_channels}, out={module.out_channels})")
            else:
                logger.info(f"  {name}: BatchNorm2d(features={module.num_features})")
    
    # 创建DNM管理器配置
    config = {
        'splitter': {
            'entropy_threshold': 0.3,  # 降低阈值，确保会触发分裂
            'overload_threshold': 0.3,
            'split_probability': 0.8,  # 提高分裂概率
            'max_splits_per_layer': 5,
            'inheritance_noise': 0.1
        },
        'monitoring': {
            'target_layers': ['conv', 'linear'],
            'analysis_frequency': 1,  # 每次都分析
            'min_epoch_before_split': 0  # 立即开始分裂
        }
    }
    
    dnm_manager = DNMNeuronDivision(config)
    
    # 注册hooks
    dnm_manager.register_model_hooks(model)
    
    # 创建测试数据
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    
    logger.info("🔍 执行前向传播以收集统计信息")
    with torch.no_grad():
        y = model(x)
        logger.info(f"输出形状: {y.shape}")
    
    # 执行神经元分裂
    logger.info("⚡ 执行DNM神经元分裂")
    try:
        result = dnm_manager.analyze_and_split(model, epoch=0)
        splits_made = result.get('splits_executed', 0)
        logger.info(f"✅ 分裂完成，执行了 {splits_made} 次分裂")
    except Exception as e:
        logger.error(f"❌ 分裂过程中发生错误: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    # 检查模型参数变化
    new_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"分裂后模型参数: {new_param_count}")
    
    # 打印分裂后的模型结构
    logger.info("📋 分裂后模型结构:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            if isinstance(module, nn.Conv2d):
                logger.info(f"  {name}: Conv2d(in={module.in_channels}, out={module.out_channels})")
            else:
                logger.info(f"  {name}: BatchNorm2d(features={module.num_features})")
    
    # 测试前向传播是否正常
    logger.info("🔍 测试分裂后的前向传播")
    try:
        with torch.no_grad():
            y_new = model(x)
            logger.info(f"✅ 前向传播成功，输出形状: {y_new.shape}")
            return True
    except Exception as e:
        logger.error(f"❌ 前向传播失败: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("🚀 启动BatchNorm同步调试")
    
    success = test_batchnorm_sync()
    
    if success:
        logger.info("🎉 BatchNorm同步测试成功！")
    else:
        logger.error("💥 BatchNorm同步测试失败！")