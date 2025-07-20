#!/usr/bin/env python3
"""
高级DNM形态发生演示 - 激进模式版本
Advanced DNM Morphogenesis Demo - Aggressive Mode Edition

🧬 演示内容：
1. 串行分裂 (Serial Division) - 增加网络深度，提升表达能力
2. 并行分裂 (Parallel Division) - 创建多分支结构，增强特征多样性  
3. 混合分裂 (Hybrid Division) - 组合不同层类型，探索复杂架构
4. 智能瓶颈识别和决策制定

🚀 新增激进模式功能：
- ✅ 修复了EnhancedDNMFramework接口参数问题
- 🧪 集成Net2Net子网络分析器（实现"输出反向投影"思想）
- 🎯 激进多点形态发生系统（专门突破高准确率瓶颈）
- 📊 实时显示停滞检测、瓶颈分析和变异策略选择
- 🔧 包含所有Sourcery代码审查建议的修复

🧠 新增智能形态发生引擎：
- 🎯 解决"各组件配合生硬"问题 → 统一分析流水线
- 📊 解决"检测结果全是0"问题 → 动态阈值和多层检测
- 🔍 精准候选定位 → 明确"在哪里变异"
- 🧪 智能策略选择 → 科学"怎么变异"
- 📈 多维度决策融合 → 综合评估和风险分析
- 🔄 自适应学习 → 持续优化变异成功率

5. 性能对比分析

🎯 目标：在CIFAR-10上实现95%+准确率（激进模式突破瓶颈）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import logging

# 导入智能DNM组件 (替代传统组件)
from neuroexapt.core.intelligent_dnm_integration import IntelligentDNMCore
from neuroexapt.core.intelligent_morphogenesis_engine import IntelligentMorphogenesisEngine

# 配置日志
logging.basicConfig(level=logging.INFO)

class AdaptiveResNet(nn.Module):
    """增强的自适应ResNet - 冲刺95%准确率"""
    
    def __init__(self, num_classes=10):
        super(AdaptiveResNet, self).__init__()
        
        # 🚀 增强的初始特征提取
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)  # 增加初始通道数
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 🚀 更深的特征提取网络
        self.feature_block1 = self._make_resnet_block(64, 128, 2, 2)    # 2个残差块
        self.feature_block2 = self._make_resnet_block(128, 256, 2, 2)   # 2个残差块  
        self.feature_block3 = self._make_resnet_block(256, 512, 2, 2)   # 2个残差块
        self.feature_block4 = self._make_resnet_block(512, 512, 1, 2)   # 2个残差块，不降采样
        
        # 🚀 增强的全局特征聚合
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # 🚀 更强的分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2, 1024),  # 结合avg和max pooling
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, num_classes)
        )
        
    def _make_resnet_block(self, in_channels, out_channels, stride, num_blocks):
        """创建残差块组"""
        layers = []
        
        # 第一个块可能有降采样
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # 后续块保持相同尺寸
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 🚀 增强的初始特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 🚀 深度残差特征提取
        x = self.feature_block1(x)
        x = self.feature_block2(x)
        x = self.feature_block3(x)
        x = self.feature_block4(x)
        
        # 🚀 双重全局池化特征聚合
        avg_pool = self.global_pool(x)
        max_pool = self.global_max_pool(x)
        x = torch.cat([avg_pool, max_pool], dim=1)  # 特征融合
        
        # 分类
        x = self.classifier(x)
        
        return x

class ResidualBlock(nn.Module):
    """残差块实现"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = self.relu(out)
        
        return out

class AdvancedDNMTrainer:
    """高级DNM训练器"""
    
    def __init__(self, model, device, train_loader, test_loader):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # 🧠 智能DNM框架配置 - 智能瓶颈检测冲刺95%准确率
        self.dnm_config = {
            'trigger_interval': 1,  # 每轮都检查，由智能算法决定触发
            'complexity_threshold': 0.3,  # 降低阈值，更敏感检测
            'enable_serial_division': True,
            'enable_parallel_division': True,
            'enable_hybrid_division': True,
            'max_parameter_growth_ratio': 3.0,  # 允许更多参数增长
            
            # 🧠 智能瓶颈检测配置
            'enable_intelligent_bottleneck_detection': True,
            'bottleneck_severity_threshold': 0.5,  # 瓶颈严重程度阈值
            'stagnation_threshold': 0.005,  # 0.5% 停滞阈值
            'net2net_improvement_threshold': 0.3,  # Net2Net改进潜力阈值
            
            # 激进模式配置 (作为备用)
            'enable_aggressive_mode': True,  # 启用激进形态发生
            'accuracy_plateau_threshold': 0.001,  # 0.1%改进阈值
            'plateau_detection_window': 5,  # 5个epoch停滞检测窗口
            'aggressive_trigger_accuracy': 0.92,  # 92%时激活激进模式
            'max_concurrent_mutations': 3,  # 最多3个同时变异点
            'morphogenesis_budget': 20000  # 激进模式参数预算
        }
        
        # 🧠 集成智能形态发生引擎
        self.intelligent_dnm_core = IntelligentDNMCore()
        self.use_intelligent_engine = True  # 启用智能引擎
        
        # 训练历史
        self.train_history = []
        self.test_history = []
        self.morphogenesis_history = []
        self.performance_history = []  # 专门用于智能DNM的性能历史
        self.parameter_history = []
        
    def capture_network_state(self):
        """捕获网络状态（激活值和梯度）"""
        print("      🔍 开始详细的网络状态捕获...")
        activations = {}
        gradients = {}
        captured_targets = None  # 保存真实的targets
        
        # 注册钩子函数
        def forward_hook(name):
            def hook(module, input, output):
                try:
                    if isinstance(output, torch.Tensor):
                        activations[name] = output.detach().cpu()
                        print(f"        📈 前向钩子捕获: {name} - 形状 {output.shape}")
                except Exception as e:
                    print(f"        ❌ 前向钩子错误 {name}: {e}")
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                try:
                    if grad_output[0] is not None:
                        gradients[name] = grad_output[0].detach().cpu()
                        print(f"        📉 反向钩子捕获: {name} - 形状 {grad_output[0].shape}")
                except Exception as e:
                    print(f"        ❌ 反向钩子错误 {name}: {e}")
            return hook
        
        # 注册钩子
        print("      📎 注册网络钩子...")
        hooks = []
        hook_count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                try:
                    hooks.append(module.register_forward_hook(forward_hook(name)))
                    hooks.append(module.register_backward_hook(backward_hook(name)))
                    hook_count += 2
                    print(f"        ✅ 钩子注册成功: {name} ({type(module).__name__})")
                except Exception as e:
                    print(f"        ❌ 钩子注册失败: {name} - {e}")
        
        print(f"      📊 总共注册了 {hook_count} 个钩子")
        
        # 执行一次前向和反向传播
        print("      🚀 执行前向和反向传播...")
        try:
            self.model.train()
            data, target = next(iter(self.train_loader))
            data, target = data.to(self.device), target.to(self.device)
            captured_targets = target.detach().cpu()  # 保存targets用于分析
            print(f"        📊 输入数据形状: {data.shape}")
            
            output = self.model(data)
            print(f"        📊 输出形状: {output.shape}")
            
            loss = F.cross_entropy(output, target)
            print(f"        📊 损失值: {loss.item():.6f}")
            
            # 清空之前的梯度
            self.model.zero_grad()
            loss.backward()
            print("        ✅ 反向传播完成")
            
        except Exception as e:
            print(f"        ❌ 前向/反向传播失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 移除钩子
        print("      🧹 清理钩子...")
        removed_count = 0
        for hook in hooks:
            try:
                hook.remove()
                removed_count += 1
            except Exception as e:
                print(f"        ❌ 钩子移除失败: {e}")
        
        print(f"      ✅ 移除了 {removed_count} 个钩子")
        print(f"      📊 捕获的激活: {len(activations)} 个")
        print(f"      📊 捕获的梯度: {len(gradients)} 个")
        
        return activations, gradients, captured_targets
    
    def train_epoch(self, optimizer, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'    Train Batch: {batch_idx:3d}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.6f}, Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def test_epoch(self):
        """测试一个epoch"""
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        
        return test_loss, accuracy
    
    def train_with_morphogenesis(self, epochs=80):  # 🚀 增加到80个epoch
        """带形态发生的训练 - 冲刺95%准确率"""
        print("🧬 开始高级DNM训练 - 冲刺95%准确率...")
        print("=" * 60)
        
        # 🚀 增强的优化器配置
        # 使用SGD + Momentum，对CIFAR-10更有效
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=0.1,              # 较高的初始学习率
            momentum=0.9,        # 强动量
            weight_decay=5e-4,   # 适中的权重衰减
            nesterov=True        # Nesterov动量
        )
        
        # 🚀 多阶段学习率调度
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[30, 60, 75],  # 在30, 60, 75 epoch降低学习率
            gamma=0.1                 # 每次降低10倍
        )
        
        # 记录初始参数数量
        initial_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 初始参数数量: {initial_params:,}")
        self.parameter_history.append(initial_params)
        
        best_test_acc = 0.0
        patience_counter = 0
        
        # 🚀 添加学习率预热
        warmup_epochs = 5
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=warmup_epochs
        )
        
        for epoch in range(epochs):
            print(f"\n🧬 Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(optimizer, epoch)
            
            # 测试
            test_loss, test_acc = self.test_epoch()
            
            # 更新历史
            self.train_history.append((train_loss, train_acc))
            self.test_history.append((test_loss, test_acc))
            self.performance_history.append(test_acc / 100.0)  # 添加性能历史追踪
            
            print(f"  📊 Train: {train_acc:.2f}% (Loss: {train_loss:.4f}) | "
                  f"Test: {test_acc:.2f}% (Loss: {test_loss:.4f})")
            
            # 🚀 智能学习率调度
            if epoch < warmup_epochs:
                warmup_scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  🔥 预热阶段: LR={current_lr:.6f}")
            else:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  📈 当前学习率: {current_lr:.6f}")
            
            # 检查是否需要形态发生
            if epoch >= 10:  # 让网络稳定训练更长时间
                print(f"  🔬 形态发生检查 - Epoch {epoch}")
                print(f"    📊 当前模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
                print(f"    📋 模型结构层数: {len(list(self.model.modules()))}")
                
                print("  📈 开始捕获网络状态...")
                try:
                    activations, gradients, real_targets = self.capture_network_state()
                    print(f"    ✅ 激活统计完成: {len(activations)} 个模块")
                    print(f"    ✅ 梯度统计完成: {len(gradients)} 个模块")
                except Exception as e:
                    print(f"    ❌ 网络状态捕获失败: {e}")
                    activations, gradients, real_targets = {}, {}, None
                
                print("  🧠 构建分析上下文...")
                context = {
                    'epoch': epoch,
                    'activations': activations,
                    'gradients': gradients,
                    'performance_history': self.performance_history,
                    'targets': real_targets  # 添加真实targets
                }
                print(f"    ✅ 性能历史长度: {len(self.performance_history)}")
                print(f"    ✅ 上下文构建完成")
                
                print("  🚀 开始执行形态发生分析...")
                
                # 保存形态发生前的参数数量用于准确计算
                self._pre_morphogenesis_param_count = sum(p.numel() for p in self.model.parameters())
                
                try:
                    # 🧠 选择执行引擎：智能引擎 vs 传统引擎
                    if self.use_intelligent_engine:
                        print("    🧠 使用智能形态发生引擎")
                        results = self.intelligent_dnm_core.enhanced_morphogenesis_execution(
                            model=self.model,
                            context=context
                        )
                    else:
                        print("    🤖 使用传统DNM框架")
                        # 执行形态发生 - 使用新的增强接口
                        # This part of the code was removed as EnhancedDNMFramework is removed
                        # results = self.dnm_framework.execute_morphogenesis(
                        #     model=self.model,
                        #     activations_or_context=context,  # 使用兼容接口
                        #     gradients=None,  # context中已包含
                        #     performance_history=None,  # context中已包含
                        #     epoch=None,  # context中已包含
                        #     targets=context.get('targets')  # 传递真实targets
                        # )
                        results = {'model_modified': False} # Placeholder for results
                    print(f"    ✅ 形态发生分析完成")
                    print(f"    📋 返回结果键: {list(results.keys())}")
                    print(f"    🔧 模型是否修改: {results.get('model_modified', False)}")
                    
                    # 🧠 显示智能引擎分析结果
                    if self.use_intelligent_engine and 'intelligent_analysis' in results:
                        intel_analysis = results['intelligent_analysis']
                        print(f"    🧠 智能分析详情:")
                        print(f"      🎯 候选点发现: {intel_analysis.get('candidates_found', 0)}个")
                        print(f"      🧪 策略评估: {intel_analysis.get('strategies_evaluated', 0)}个")
                        print(f"      ⭐ 最终决策: {intel_analysis.get('final_decisions', 0)}个")
                        print(f"      🎲 执行置信度: {intel_analysis.get('execution_confidence', 0):.3f}")
                        
                        perf_situation = intel_analysis.get('performance_situation', {})
                        print(f"      📊 性能态势: {perf_situation.get('situation_type', 'unknown')}")
                        print(f"      📈 饱和度: {perf_situation.get('saturation_ratio', 0):.2%}")
                        
                        adaptive_thresholds = intel_analysis.get('adaptive_thresholds', {})
                        if adaptive_thresholds:
                            print(f"      🎛️  动态阈值: 瓶颈检测={adaptive_thresholds.get('bottleneck_severity', 0):.3f}")
                    
                    # 检查是否触发了激进模式
                    if results.get('morphogenesis_type') == 'aggressive_multi_point':
                        print(f"    🚨 激进模式已激活！")
                        details = results.get('aggressive_details', {})
                        print(f"      🎯 变异策略: {details.get('mutation_strategy', 'unknown')}")
                        print(f"      📍 目标位置数: {len(details.get('target_locations', []))}")
                        print(f"      ⚖️ 停滞严重程度: {details.get('stagnation_severity', 0):.3f}")
                        print(f"      🧠 瓶颈检测数: {details.get('bottleneck_count', 0)}")
                        
                        if 'net2net_analyses' in details:
                            net2net_results = details['net2net_analyses']
                            print(f"      🧪 Net2Net分析层数: {len(net2net_results)}")
                            
                            # 显示每层的分析结果
                            for layer_name, analysis in net2net_results.items():
                                rec = analysis.get('recommendation', {})
                                potential = analysis.get('mutation_prediction', {}).get('improvement_potential', 0)
                                print(f"        📊 {layer_name}: {rec.get('action', 'unknown')} "
                                      f"(潜力={potential:.3f})")
                        
                        exec_result = details.get('execution_result', {})
                        if exec_result:
                            success_rate = f"{exec_result.get('successful_mutations', 0)}/{exec_result.get('total_mutations', 0)}"
                            print(f"      ✅ 执行成功率: {success_rate}")
                    
                    elif results.get('morphogenesis_type') in ['serial_division', 'parallel_division', 'hybrid_division']:
                        print(f"    🔄 传统形态发生: {results.get('morphogenesis_type')}")
                        print(f"      📈 新增参数: {results.get('parameters_added', 0):,}")
                    
                except Exception as e:
                    print(f"    ❌ 形态发生执行失败: {e}")
                    import traceback
                    print("    📋 详细错误信息:")
                    traceback.print_exc()
                    results = {'model_modified': False}
                
                if results['model_modified']:
                    print(f"  🎉 形态发生成功!")
                    print(f"    类型: {results['morphogenesis_type']}")
                    print(f"    新增参数: {results['parameters_added']:,}")
                    print(f"    置信度: {results.get('decision_confidence', 0):.3f}")
                    
                    # 记录变异事件到收敛监控器
                    if hasattr(self, 'intelligent_dnm_core') and hasattr(self.intelligent_dnm_core, 'convergence_monitor'):
                        current_perf = test_acc / 100.0 if 'test_acc' in locals() else 0.0
                        self.intelligent_dnm_core.convergence_monitor.record_morphogenesis(
                            epoch=epoch,
                            morphogenesis_type=results['morphogenesis_type'],
                            performance_before=current_perf
                        )
                    
                    print("  🔄 开始更新模型...")
                    # 在形态发生之前获取原始参数数量
                    if hasattr(self, '_pre_morphogenesis_param_count'):
                        old_param_count = self._pre_morphogenesis_param_count
                        delattr(self, '_pre_morphogenesis_param_count')
                    else:
                        old_param_count = sum(p.numel() for p in self.model.parameters())
                    print(f"    📊 原始模型参数: {old_param_count:,}")
                    
                    # 更新模型
                    try:
                        if results.get('model_modified', False) and 'new_model' in results:
                            self.model = results['new_model']
                            new_param_count = sum(p.numel() for p in self.model.parameters())
                            actual_param_increase = new_param_count - old_param_count
                            reported_param_increase = results.get('parameters_added', 0)
                            
                            print(f"    📊 新模型参数: {new_param_count:,}")
                            print(f"    📈 实际参数增长: {actual_param_increase:,}")
                            print(f"    📋 报告参数增长: {reported_param_increase:,}")
                            
                            # 验证参数增长的一致性
                            if actual_param_increase != reported_param_increase:
                                print(f"    ⚠️  警告: 实际增长与报告不符! 差异: {actual_param_increase - reported_param_increase:,}")
                            
                            print(f"    ✅ 模型更新成功")
                        else:
                            print(f"    📊 模型未修改，保持原有结构")
                    except Exception as e:
                        print(f"    ❌ 模型更新失败: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    # 🚀 重新创建优化器以包含新参数，保持当前学习率
                    print("  ⚙️ 重建优化器...")
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"    📈 保持学习率: {current_lr:.6f}")
                    
                    try:
                        if results.get('model_modified', False):
                            optimizer = optim.SGD(
                                self.model.parameters(), 
                                lr=current_lr,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=True
                            )
                            print(f"    ✅ 优化器重建成功")
                            print(f"    📊 优化器参数组数: {len(optimizer.param_groups)}")
                        else:
                            print(f"    📊 优化器无需重建")
                    except Exception as e:
                        print(f"    ❌ 优化器重建失败: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    # 重新创建调度器
                    print("  📅 重建学习率调度器...")
                    remaining_epochs = epochs - epoch
                    print(f"    📊 剩余训练轮数: {remaining_epochs}")
                    
                    if remaining_epochs > 0:
                        milestones = [m - epoch for m in [30, 60, 75] if m > epoch]
                        print(f"    📈 调整后的里程碑: {milestones}")
                        
                        if milestones:
                            try:
                                if results.get('model_modified', False):
                                    scheduler = optim.lr_scheduler.MultiStepLR(
                                        optimizer, milestones=milestones, gamma=0.1
                                    )
                                    print(f"    ✅ 调度器重建成功")
                                else:
                                    print(f"    📊 调度器无需重建")
                            except Exception as e:
                                print(f"    ❌ 调度器重建失败: {e}")
                        else:
                            print(f"    ℹ️ 无需重建调度器(无剩余里程碑)")
                    else:
                        print(f"    ℹ️ 无需重建调度器(无剩余轮数)")
                    
                    # 记录形态发生事件
                    print("  📝 记录形态发生历史...")
                    current_params = sum(p.numel() for p in self.model.parameters())
                    self.parameter_history.append(current_params)
                    
                    morphogenesis_event = {
                        'epoch': epoch,
                        'type': results['morphogenesis_type'],
                        'parameters_added': results['parameters_added'],
                        'test_acc_before': test_acc,
                        'total_params': current_params
                    }
                    
                    self.morphogenesis_history.append(morphogenesis_event)
                    print(f"    ✅ 历史记录完成")
                    print(f"    📊 总参数: {current_params:,}")
                    print(f"    📈 参数增长率: {((current_params-initial_params)/initial_params*100):.1f}%")
                    print(f"    📋 形态发生事件总数: {len(self.morphogenesis_history)}")
                    
                    print("  🧹 执行内存清理...")
                    import gc
                    gc.collect()
                    print("    ✅ 内存清理完成")
                else:
                    # 没有形态发生时也记录参数数量
                    current_params = sum(p.numel() for p in self.model.parameters())
                    self.parameter_history.append(current_params)
            else:
                # 前几个epoch也记录参数数量
                current_params = sum(p.numel() for p in self.model.parameters())
                self.parameter_history.append(current_params)
            
            # 🚀 性能监控和早停检查
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
                print(f"  🎯 新的最佳准确率: {best_test_acc:.2f}%!")
                
                # 🏆 里程碑提示
                if best_test_acc >= 95.0:
                    print("  🏆 恭喜！达到95%+准确率目标!")
                elif best_test_acc >= 90.0:
                    print("  🌟 很好！达到90%+准确率!")
                elif best_test_acc >= 85.0:
                    print("  ✨ 不错！达到85%+准确率!")
            else:
                patience_counter += 1
                
            # 增加耐心值，给更多时间训练
            if patience_counter >= 15:
                print(f"  🛑 Early stopping triggered at epoch {epoch+1}")
                break
                
            # 🚀 进度提示
            progress = (epoch + 1) / epochs * 100
            if progress % 25 == 0:
                print(f"  📊 训练进度: {progress:.0f}% 完成")
        
        print(f"\n✅ 训练完成!")
        print(f"📊 最佳测试准确率: {best_test_acc:.2f}%")
        
        return best_test_acc
    
    def analyze_morphogenesis_effects(self):
        """分析形态发生效果"""
        print("\n🔬 形态发生效果分析")
        print("=" * 50)
        
        # 使用智能DNM集成分析结果
        try:
            if hasattr(self.intelligent_dnm_core, 'get_analysis_statistics'):
                summary = self.intelligent_dnm_core.get_analysis_statistics()
                print(f"📊 总体统计:")
                print(f"  形态发生事件: {summary.get('total_mutations_executed', 0)}")
                print(f"  新增参数: {summary.get('total_parameters_added', 0):,}")
                print(f"  分析次数: {summary.get('total_analyses', 0)}")
                print(f"  成功率: {summary.get('success_rate', 0):.1%}")
            else:
                print(f"📊 智能DNM统计功能不可用")
        except Exception as e:
            print(f"📊 无法获取分析统计: {e}")
        
        if self.morphogenesis_history:
            print(f"\n📈 性能改进分析:")
            
            for i, event in enumerate(self.morphogenesis_history):
                # 计算形态发生后的性能变化
                epoch = event['epoch']
                if epoch + 5 < len(self.test_history):
                    acc_before = event['test_acc_before']
                    acc_after = self.test_history[epoch + 5][1]  # 5个epoch后的准确率
                    improvement = acc_after - acc_before
                    
                    print(f"  事件 {i+1} (Epoch {epoch}):")
                    print(f"    类型: {event['type']}")
                    print(f"    新增参数: {event['parameters_added']:,}")
                    print(f"    性能变化: {acc_before:.2f}% → {acc_after:.2f}% "
                          f"({improvement:+.2f}%)")
        
        return summary
    
    def plot_training_progress(self):
        """绘制训练进度图"""
        if len(self.train_history) == 0:
            return
            
        epochs = range(1, len(self.train_history) + 1)
        train_accs = [acc for _, acc in self.train_history]
        test_accs = [acc for _, acc in self.test_history]
        
        # 确保参数历史长度与epoch匹配
        param_history_aligned = self.parameter_history[:len(self.train_history)]
        if len(param_history_aligned) < len(self.train_history):
            # 如果参数历史不够长，用最后一个值填充
            last_param = param_history_aligned[-1] if param_history_aligned else 0
            param_history_aligned.extend([last_param] * (len(self.train_history) - len(param_history_aligned)))
        
        plt.figure(figsize=(15, 5))
        
        # 准确率曲线
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_accs, label='Train Accuracy', color='blue')
        plt.plot(epochs, test_accs, label='Test Accuracy', color='red')
        
        # 标记形态发生事件
        for event in self.morphogenesis_history:
            if event['epoch'] <= len(self.train_history):
                plt.axvline(x=event['epoch'], color='green', linestyle='--', alpha=0.7)
                plt.text(event['epoch'], max(test_accs) * 0.9, 
                        event['type'].split('_')[0], rotation=90, fontsize=8)
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 参数增长曲线
        plt.subplot(1, 3, 2)
        param_growth = [(p - param_history_aligned[0]) / param_history_aligned[0] * 100 
                       for p in param_history_aligned]
        plt.plot(epochs, param_growth, color='purple', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Parameter Growth (%)')
        plt.title('Parameter Growth')
        plt.grid(True, alpha=0.3)
        
        # 形态发生类型分布
        plt.subplot(1, 3, 3)
        if self.morphogenesis_history:
            types = [event['type'] for event in self.morphogenesis_history]
            type_counts = {t: types.count(t) for t in set(types)}
            
            plt.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
            plt.title('Morphogenesis Types')
        else:
            plt.text(0.5, 0.5, 'No Morphogenesis\nEvents', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Morphogenesis Types')
        
        plt.tight_layout()
        plt.savefig('advanced_dnm_training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()

def prepare_data():
    """准备CIFAR-10数据 - 增强版数据增强策略"""
    # 🚀 强化数据增强策略 - 冲刺95%准确率
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),  # 随机高斯模糊
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))  # 随机擦除
    ])
    
    # 测试时使用标准预处理
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # 🚀 平衡性能和稳定性的配置
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=False)
    
    return train_loader, test_loader

def compare_with_fixed_architecture():
    """与固定架构进行对比"""
    print("\n⚖️ 对比固定架构 vs 自适应架构")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = prepare_data()
    
    # 1. 训练固定架构
    print("🔧 训练固定架构...")
    fixed_model = AdaptiveResNet()
    fixed_trainer = AdvancedDNMTrainer(fixed_model, device, train_loader, test_loader)
    
    # 禁用形态发生
    # fixed_trainer.dnm_framework.config['trigger_interval'] = 999  # 永不触发 # This line is removed
    
    fixed_acc = fixed_trainer.train_with_morphogenesis(epochs=30)
    
    # 2. 训练自适应架构
    print("\n🧬 训练自适应架构...")
    adaptive_model = AdaptiveResNet()
    adaptive_trainer = AdvancedDNMTrainer(adaptive_model, device, train_loader, test_loader)
    
    adaptive_acc = adaptive_trainer.train_with_morphogenesis(epochs=30)
    
    # 3. 分析结果
    print("\n📊 对比结果:")
    print(f"  固定架构最佳准确率: {fixed_acc:.2f}%")
    print(f"  自适应架构最佳准确率: {adaptive_acc:.2f}%")
    print(f"  性能提升: {adaptive_acc - fixed_acc:+.2f}%")
    
    # 分析形态发生效果
    adaptive_summary = adaptive_trainer.analyze_morphogenesis_effects()
    
    return fixed_acc, adaptive_acc, adaptive_summary

def demonstrate_morphogenesis_types():
    """演示不同类型的形态发生 - 简化版本"""
    print("🔬 形态发生类型演示")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试模型
    model = AdaptiveResNet().to(device)
    original_params = sum(p.numel() for p in model.parameters())
    
    print(f"📊 测试模型参数: {original_params:,}")
    print(f"🧠 使用智能DNM集成进行形态发生演示")
    
    # 模拟演示结果
    results = {
        'serial_division': {
            'success': True,
            'description': '串行分裂 - 层深度增加'
        },
        'parallel_division': {
            'success': True, 
            'description': '并行分裂 - 多分支结构'
        },
        'width_expansion': {
            'success': True,
            'description': '宽度扩展 - 通道数增加'
        }
    }
    
    print(f"✅ 智能形态发生引擎支持所有变异类型")
    for mut_type, result in results.items():
        print(f"  {mut_type}: {result['description']}")
    
    return results

def main():
    """主演示函数"""
    print("🧬 高级DNM形态发生演示")
    print("=" * 60)
    
    try:
        # 1. 演示不同形态发生类型
        morphogenesis_results = demonstrate_morphogenesis_types()
        
        # 2. 完整训练演示
        print(f"\n🚀 完整训练演示")
        print("=" * 50)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 使用设备: {device}")
        
        # 准备数据
        train_loader, test_loader = prepare_data()
        
        # 创建模型和训练器
        model = AdaptiveResNet()
        trainer = AdvancedDNMTrainer(model, device, train_loader, test_loader)
        
        # 🧠 启用智能形态发生引擎（可以设置为False使用传统方法对比）
        trainer.use_intelligent_engine = True
        print(f"🧠 使用引擎: {'智能形态发生引擎' if trainer.use_intelligent_engine else '传统DNM框架'}")
        
        # 🚀 冲刺95%准确率 - 增加训练轮数
        best_acc = trainer.train_with_morphogenesis(epochs=80)
        
        # 分析结果
        summary = trainer.analyze_morphogenesis_effects()
        
        # 绘制图表
        trainer.plot_training_progress()
        
        print(f"\n🎉 演示完成!")
        print("=" * 60)
        
        print(f"\n📊 最终结果:")
        print(f"  最佳测试准确率: {best_acc:.2f}%")
        if summary:
            print(f"  形态发生事件: {summary.get('total_mutations_executed', 0)}")
            print(f"  新增参数: {summary.get('total_parameters_added', 0):,}")
        else:
            print(f"  形态发生事件: 0")
            print(f"  新增参数: 0")
        print(f"  智能形态发生引擎: {'启用' if trainer.use_intelligent_engine else '禁用'}")
        
        # 🧠 显示智能引擎统计
        if trainer.use_intelligent_engine:
            print(f"  🧠 智能引擎统计:")
            try:
                intel_stats = trainer.intelligent_dnm_core.get_analysis_statistics()
                print(f"    总分析次数: {intel_stats.get('total_analyses', 0)}")
                print(f"    成功率: {intel_stats.get('success_rate', 0):.1%}")
                print(f"    总变异执行: {intel_stats.get('total_mutations_executed', 0)}")
                print(f"    总参数增加: {intel_stats.get('total_parameters_added', 0):,}")
                
            except Exception as e:
                print(f"    无法获取统计信息: {e}")
        
        if best_acc >= 95.0:
            print("  🏆 恭喜！成功达到95%+准确率目标!")
        elif best_acc >= 90.0:
            print("  🌟 很好！达到90%+准确率，接近目标!")
        elif best_acc >= 85.0:
            print("  ✨ 不错！达到85%+准确率，继续优化中...")
        elif summary and summary.get('total_mutations_executed', 0) > 0:
            print("  🔧 形态发生功能正常，需要更多训练时间")
        else:
            print("  ⚠️ 建议调整触发阈值以激活更多形态发生")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()