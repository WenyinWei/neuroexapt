"""
ASO-SE 稳定训练器
重新设计的四阶段训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import math
from .aso_se_architecture import ProgressiveArchitectureNetwork


class StableASO_SETrainer:
    """稳定的ASO-SE训练器"""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 训练状态
        self.current_epoch = 0
        self.current_phase = 'warmup'
        self.phase_epochs = 0
        self.best_accuracy = 0.0
        self.training_history = []
        
        # 数据加载器
        self.train_loader = None
        self.test_loader = None
        
        # 模型和优化器
        self.network = None
        self.weight_optimizer = None
        self.arch_optimizer = None
        self.scheduler = None
        
        print(f"🚀 稳定ASO-SE训练器初始化完成")
        print(f"   设备: {self.device}")
        print(f"   配置: {self.config}")
    
    def _default_config(self):
        """默认配置"""
        return {
            'dataset': 'CIFAR-10',
            'batch_size': 128,
            'num_epochs': 100,
            'init_channels': 32,
            'init_depth': 4,
            'max_depth': 8,
            
            # 学习率设置
            'weight_lr': 0.025,
            'arch_lr': 3e-4,
            'momentum': 0.9,
            'weight_decay': 3e-4,
            
            # 阶段设置
            'warmup_epochs': 15,
            'search_epochs': 30,
            'growth_epochs': 35,
            'optimize_epochs': 20,
            
            # 搜索控制
            'arch_update_freq': 5,  # 每5个batch更新一次架构
            'growth_patience': 8,   # 性能停滞8个epoch后生长
            'growth_threshold': 0.01,  # 性能提升阈值
        }
    
    def setup_data(self):
        """设置数据加载器"""
        if self.config['dataset'] == 'CIFAR-10':
            # 数据增强
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            train_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=train_transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=test_transform
            )
            
            self.train_loader = DataLoader(
                train_dataset, batch_size=self.config['batch_size'], 
                shuffle=True, num_workers=2, pin_memory=True
            )
            self.test_loader = DataLoader(
                test_dataset, batch_size=self.config['batch_size'], 
                shuffle=False, num_workers=2, pin_memory=True
            )
            
            print(f"📊 CIFAR-10数据加载完成: 训练集 {len(train_dataset)}, 测试集 {len(test_dataset)}")
        else:
            raise ValueError(f"不支持的数据集: {self.config['dataset']}")
    
    def setup_model(self):
        """设置模型"""
        self.network = ProgressiveArchitectureNetwork(
            input_channels=3,
            init_channels=self.config['init_channels'],
            num_classes=10,
            init_depth=self.config['init_depth'],
            max_depth=self.config['max_depth']
        ).to(self.device)
        
        # 使用skip_biased初始化
        self.network.arch_manager.init_strategy = 'skip_biased'
        
        print(f"🏗️ 网络初始化完成:")
        info = self.network.get_architecture_info()
        print(f"   深度: {info['depth']}")
        print(f"   参数量: {info['parameters']:,}")
        print(f"   初始架构: {info['architecture']}")
    
    def setup_optimizers(self):
        """设置优化器"""
        # 分离权重参数和架构参数
        weight_params = []
        arch_params = []
        
        for name, param in self.network.named_parameters():
            if 'arch_manager.alpha' in name:
                arch_params.append(param)
            else:
                weight_params.append(param)
        
        # 权重优化器
        self.weight_optimizer = optim.SGD(
            weight_params,
            lr=self.config['weight_lr'],
            momentum=self.config['momentum'],
            weight_decay=self.config['weight_decay']
        )
        
        # 架构优化器
        self.arch_optimizer = optim.Adam(
            arch_params,
            lr=self.config['arch_lr'],
            betas=(0.5, 0.999),
            weight_decay=1e-3
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.weight_optimizer,
            T_max=self.config['num_epochs'],
            eta_min=1e-4
        )
        
        print(f"⚙️ 优化器设置完成:")
        print(f"   权重参数: {len(weight_params)}")
        print(f"   架构参数: {len(arch_params)}")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.network.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # 根据训练阶段选择优化策略
            if self.current_phase == 'warmup':
                # warmup阶段只优化权重
                self._optimize_weights(data, targets)
            
            elif self.current_phase in ['search', 'growth']:
                # 搜索和生长阶段交替优化
                if batch_idx % self.config['arch_update_freq'] == 0:
                    # 优化架构参数
                    self._optimize_architecture(data, targets)
                else:
                    # 优化权重参数
                    self._optimize_weights(data, targets)
            
            elif self.current_phase == 'optimize':
                # 优化阶段只优化权重
                self._optimize_weights(data, targets)
            
            # 统计
            with torch.no_grad():
                outputs = self.network(data)
                loss = F.cross_entropy(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            accuracy = 100. * correct / total
            arch_info = self.network.get_architecture_info()
            
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.3f}',
                'Acc': f'{accuracy:.2f}%',
                'Phase': self.current_phase,
                'Temp': f'{arch_info["temperature"]:.3f}',
                'Entropy': f'{arch_info["entropy"]:.2f}'
            })
        
        return total_loss / len(self.train_loader), accuracy
    
    def _optimize_weights(self, data, targets):
        """优化权重参数"""
        self.weight_optimizer.zero_grad()
        outputs = self.network(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
        
        self.weight_optimizer.step()
        return loss.item()
    
    def _optimize_architecture(self, data, targets):
        """优化架构参数"""
        self.arch_optimizer.zero_grad()
        outputs = self.network(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        
        # 梯度裁剪
        arch_params = [p for name, p in self.network.named_parameters() if 'arch_manager.alpha' in name]
        torch.nn.utils.clip_grad_norm_(arch_params, 5.0)
        
        self.arch_optimizer.step()
        
        # 温度退火
        self.network.arch_manager.anneal_temperature()
        
        return loss.item()
    
    def evaluate(self):
        """评估模型"""
        self.network.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.network(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def update_phase(self):
        """更新训练阶段"""
        self.phase_epochs += 1
        old_phase = self.current_phase
        
        # 阶段转换逻辑
        if (self.current_phase == 'warmup' and 
            self.phase_epochs >= self.config['warmup_epochs']):
            self.current_phase = 'search'
            self.phase_epochs = 0
            self.network.set_training_phase('search')
            self.network.arch_manager.smooth_transition_to_search()
            print(f"🔄 进入搜索阶段，温度: {self.network.arch_manager.sampler.tau:.3f}")
        
        elif (self.current_phase == 'search' and 
              self.phase_epochs >= self.config['search_epochs']):
            self.current_phase = 'growth'
            self.phase_epochs = 0
            self.network.set_training_phase('growth')
            print(f"🔄 进入生长阶段")
        
        elif (self.current_phase == 'growth' and 
              self.phase_epochs >= self.config['growth_epochs']):
            self.current_phase = 'optimize'
            self.phase_epochs = 0
            self.network.set_training_phase('optimize')
            print(f"🔄 进入优化阶段")
        
        # 如果阶段发生变化，打印架构信息
        if old_phase != self.current_phase:
            self._print_architecture_analysis()
    
    def _print_architecture_analysis(self):
        """打印架构分析"""
        info = self.network.get_architecture_info()
        print(f"\n🔍 架构分析:")
        print(f"   深度: {info['depth']}")
        print(f"   参数量: {info['parameters']:,}")
        print(f"   架构熵: {info['entropy']:.3f}")
        print(f"   置信度: {info['confidence']:.3f}")
        print(f"   当前架构: {info['architecture']}")
    
    def _should_grow(self, current_accuracy):
        """判断是否应该生长"""
        if len(self.training_history) < self.config['growth_patience']:
            return False
        
        # 检查最近几个epoch的性能
        recent_accuracies = [h['test_acc'] for h in self.training_history[-self.config['growth_patience']:]]
        improvement = max(recent_accuracies) - min(recent_accuracies)
        
        return improvement < self.config['growth_threshold']
    
    def train(self):
        """完整训练流程"""
        print(f"\n🔧 开始ASO-SE训练")
        print(f"{'='*60}")
        
        # 设置
        self.setup_data()
        self.setup_model()
        self.setup_optimizers()
        
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 评估
            test_acc = self.evaluate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            epoch_info = {
                'epoch': epoch,
                'phase': self.current_phase,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'architecture': self.network.get_architecture_info()
            }
            self.training_history.append(epoch_info)
            
            # 生长控制
            if (self.current_phase == 'growth' and 
                self._should_grow(test_acc) and 
                self.network.current_depth < self.network.max_depth):
                
                print(f"🌱 触发网络生长")
                self.network.grow_depth(1)
                self._update_optimizers_after_growth()
            
            # 更新最佳精度
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                self._save_checkpoint('best')
            
            # 更新阶段
            self.update_phase()
            
            # 定期汇报
            if (epoch + 1) % 5 == 0:
                elapsed = time.time() - start_time
                print(f"\n📊 Epoch {epoch+1}/{self.config['num_epochs']} | 阶段: {self.current_phase}")
                print(f"   训练损失: {train_loss:.4f} | 训练精度: {train_acc:.2f}%")
                print(f"   测试精度: {test_acc:.2f}% | 最佳: {self.best_accuracy:.2f}%")
                print(f"   耗时: {elapsed/60:.1f}分钟")
                
                if self.current_phase == 'search':
                    self._print_architecture_analysis()
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n🎉 训练完成!")
        print(f"   最佳精度: {self.best_accuracy:.2f}%")
        print(f"   总耗时: {total_time/60:.1f}分钟")
        print(f"   最终架构: {self.network.get_architecture_info()['architecture']}")
        
        return self.training_history, self.best_accuracy
    
    def _update_optimizers_after_growth(self):
        """生长后更新优化器"""
        try:
            # 重新设置优化器以包含新参数
            self.setup_optimizers()
            print(f"✅ 优化器已更新")
        except Exception as e:
            print(f"⚠️ 优化器更新失败: {e}")
    
    def _save_checkpoint(self, name):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'phase': self.current_phase,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.weight_optimizer.state_dict(),
            'arch_optimizer_state_dict': self.arch_optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': self.config,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, f'aso_se_{name}.pth')
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.weight_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.arch_optimizer.load_state_dict(checkpoint['arch_optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.current_phase = checkpoint['phase']
        self.best_accuracy = checkpoint['best_accuracy']
        self.training_history = checkpoint['training_history']
        
        print(f"✅ 检查点加载完成: {path}") 