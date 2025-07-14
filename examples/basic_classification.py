"""
Efficient Classification Example with NeuroExapt Simple Expansion

This example demonstrates a new efficient expansion system that:
1. Makes conservative changes (1.2x-1.5x expansion, not 5x)
2. Applies changes directly without time-wasting evaluation
3. Focuses on training efficiency over complex mutation testing
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
import time
from typing import List
import argparse

# Add the parent directory to the path to import neuroexapt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import simple and efficient operators
from neuroexapt.core.simple_expansion_operators import get_simple_operators
from neuroexapt.math.optimization import AdaptiveLearningRateScheduler


class AdaptiveClassifier(nn.Module):
    """
    Highly adaptive classifier that can handle architectural changes.
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, dropout_rate: float = 0.5):
        super(AdaptiveClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Build classifier layers
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Dropout(dropout_rate),
                nn.Linear(current_size, hidden_size),
                nn.ReLU(inplace=True)
            ])
            current_size = hidden_size
        
        layers.extend([
            nn.Dropout(dropout_rate),
            nn.Linear(current_size, num_classes)
        ])
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Auto-adapt to changing input dimensions
        if x.size(1) != self.input_size:
            self._adapt_to_input_size(x.size(1))
        
        return self.classifier(x)
    
    def _adapt_to_input_size(self, new_input_size: int):
        """Dynamically adapt classifier to new input dimensions."""
        try:
            print(f"🔧 Adapting classifier: {self.input_size} → {new_input_size}")
            
            device = next(self.classifier.parameters()).device
            
            # Find the first linear layer
            first_linear_idx = None
            for i, layer in enumerate(self.classifier):
                if isinstance(layer, nn.Linear):
                    first_linear_idx = i
                    break
            
            if first_linear_idx is None:
                return
            
            old_layer = self.classifier[first_linear_idx]
            if not isinstance(old_layer, nn.Linear):
                return
            
            new_layer = nn.Linear(new_input_size, old_layer.out_features).to(device)
            
            # Transfer weights intelligently
            if new_input_size < self.input_size:
                # Downsample by taking most important features
                new_layer.weight.data = old_layer.weight.data[:, :new_input_size]
            elif new_input_size > self.input_size:
                # Upsample by padding with averaged weights
                new_layer.weight.data[:, :self.input_size] = old_layer.weight.data
                avg_weights = old_layer.weight.data.mean(dim=1, keepdim=True)
                new_layer.weight.data[:, self.input_size:] = avg_weights.repeat(1, new_input_size - self.input_size)
            else:
                new_layer.weight.data = old_layer.weight.data
            
            if old_layer.bias is not None and new_layer.bias is not None:
                new_layer.bias.data = old_layer.bias.data
            
            # Replace the layer
            self.classifier[first_linear_idx] = new_layer
            self.input_size = new_input_size
            
        except Exception as e:
            print(f"Warning: Classifier adaptation failed: {e}")


class EfficientCNN(nn.Module):
    """
    An efficient CNN designed for accuracy-guided expansion without wasted time.
    """
    
    def __init__(self, num_classes=10, initial_channels=32):
        super(EfficientCNN, self).__init__()
        
        # Feature extraction with expandable architecture
        self.features = nn.Sequential(
            # Block 1: Initial convolution
            nn.Conv2d(3, initial_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True),
            
            # Block 2: First layer
            nn.Conv2d(initial_channels, initial_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: Second layer
            nn.Conv2d(initial_channels, initial_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels * 2),
            nn.ReLU(inplace=True),
            
            # Block 4: Deep features
            nn.Conv2d(initial_channels * 2, initial_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5: Advanced features
            nn.Conv2d(initial_channels * 2, initial_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels * 4),
            nn.ReLU(inplace=True),
            
            # Block 6: High-level features
            nn.Conv2d(initial_channels * 4, initial_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 7: Final feature extraction
            nn.Conv2d(initial_channels * 4, initial_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels * 8),
            nn.ReLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Adaptive classifier
        self.classifier = AdaptiveClassifier(
            input_size=initial_channels * 8,
            hidden_sizes=[256, 128],
            num_classes=num_classes,
            dropout_rate=0.4
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        try:
            x = self.features(x)
            x = self.classifier(x)
            return x
        except RuntimeError as e:
            print(f"❌ Forward pass error: {e}")
            # Auto-recovery attempt
            self._auto_recovery()
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    def _initialize_weights(self):
        """Initialize weights with better variance."""
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
    
    def _auto_recovery(self):
        """Attempt to recover from architectural inconsistencies."""
        print("🔄 Attempting auto-recovery...")
        try:
            # Test forward pass through features
            with torch.no_grad():
                test_input = torch.randn(1, 3, 32, 32)
                if next(self.parameters()).is_cuda:
                    test_input = test_input.cuda()
                
                feature_output = self.features(test_input)
                feature_size = feature_output.view(1, -1).size(1)
                
                # Update classifier input size
                self.classifier.input_size = feature_size
                self.classifier._adapt_to_input_size(feature_size)
                
        except Exception as e:
            print(f"Auto-recovery failed: {e}")


class SimpleExpansionEngine:
    """
    简单高效的扩展引擎 - 不浪费时间评估
    """
    
    def __init__(self, model: nn.Module, operators: List):
        self.model = model
        self.operators = operators
        self.expansion_count = 0
        self.last_expansion_epoch = 0
        
        print(f"🚀 SimpleExpansionEngine initialized with {len(operators)} operators")
        print(f"   Strategy: Direct application, no time-wasting evaluation")
    
    def should_expand(self, epoch: int, accuracy: float) -> bool:
        """判断是否应该扩展"""
        # 基于准确度和间隔决定
        if accuracy >= 0.95:
            return False  # 已经很好了
        
        # 防止过于频繁的扩展
        if epoch - self.last_expansion_epoch < 3:
            return False
        
        # 根据准确度决定扩展频率
        if accuracy < 0.6:
            return epoch % 2 == 0  # 每2轮扩展一次
        elif accuracy < 0.8:
            return epoch % 4 == 0  # 每4轮扩展一次
        else:
            return epoch % 8 == 0  # 每8轮扩展一次
    
    def expand(self, epoch: int, accuracy: float) -> bool:
        """执行扩展"""
        try:
            print(f"\n🔧 SIMPLE EXPANSION - Epoch {epoch}")
            print(f"   Current accuracy: {accuracy:.2f}%")
            
            # 选择合适的操作器
            if len(self.operators) == 0:
                print("   ❌ No operators available")
                return False
            
            # 根据准确度选择操作器
            if accuracy < 60:
                # 低准确度：使用更积极的扩展
                selected_op = self.operators[2] if len(self.operators) > 2 else self.operators[0]
            elif accuracy < 80:
                # 中准确度：使用中等扩展
                selected_op = self.operators[1] if len(self.operators) > 1 else self.operators[0]
            else:
                # 高准确度：使用保守扩展
                selected_op = self.operators[0]
            
            print(f"   🎯 Selected operator: {selected_op.__class__.__name__}")
            
            # 准备状态
            state = {
                'val_accuracy': accuracy,
                'current_performance': accuracy / 100.0,
                'epoch': epoch
            }
            
            # 直接应用操作器（不评估）
            result = selected_op(self.model, state)
            
            if result is not None:
                self.model = result
                self.expansion_count += 1
                self.last_expansion_epoch = epoch
                
                print(f"   ✅ Expansion successful! Total expansions: {self.expansion_count}")
                print(f"   📊 New model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
                return True
            else:
                print(f"   ❌ Expansion failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Expansion error: {e}")
            return False


def load_cifar10_data(batch_size=64, download=True):
    """
    Load CIFAR-10 with enhanced data augmentation.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=download, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                           num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=download, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                          num_workers=2, pin_memory=True)

    return trainloader, testloader


def evaluate_model(model, testloader, device):
    """
    Model evaluation.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(testloader):
            data, targets = data.to(device), targets.to(device)
            try:
                outputs = model(data)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            except RuntimeError as e:
                print(f"Warning: Evaluation batch {batch_idx} failed: {e}")
                continue
    
    accuracy = 100 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(testloader) if len(testloader) > 0 else 0.0
    
    model.train()
    return accuracy, avg_loss


def main(num_epochs: int = 100, quick_mode: bool = False):
    """
    Main training function with efficient expansion.
    """
    print("🚀 NeuroExapt Efficient Expansion Classification")
    print("=" * 60)
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    
    # Load data
    print("📊 Loading CIFAR-10 dataset...")
    trainloader, testloader = load_cifar10_data(batch_size=batch_size)
    
    # Create model
    print("🧬 Creating efficient CNN...")
    model = EfficientCNN(num_classes=10, initial_channels=32).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Print initial architecture
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initial parameters: {total_params:,}")
    
    # Create simple expansion engine
    print("🔧 Initializing Simple Expansion Engine...")
    operators = get_simple_operators()
    expansion_engine = SimpleExpansionEngine(model, operators)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    expansion_log = []
    
    print("🎯 Starting efficient training...")
    print("=" * 60)
    
    # Initial evaluation
    initial_test_acc, initial_test_loss = evaluate_model(model, testloader, device)
    print(f"Initial test accuracy: {initial_test_acc:.2f}%")
    test_accuracies.append(initial_test_acc)
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(trainloader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            try:
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
            except RuntimeError as e:
                print(f"Warning: Training batch {batch_idx} failed: {e}")
                continue
        
        # Calculate metrics
        avg_loss = epoch_loss / len(trainloader)
        train_acc = 100 * correct / total
        test_acc, test_loss = evaluate_model(model, testloader, device)
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"Loss: {avg_loss:.4f} | "
              f"Train: {train_acc:.2f}% | "
              f"Test: {test_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
        
        # Simple expansion phase
        if expansion_engine.should_expand(epoch + 1, test_acc):
            print(f"\n{'🔧 SIMPLE EXPANSION PHASE 🔧':^60}")
            
            # Apply expansion (no evaluation, just do it)
            expansion_success = expansion_engine.expand(epoch + 1, test_acc)
            
            if expansion_success:
                model = expansion_engine.model
                
                # Update optimizer for new architecture
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate * 0.9, weight_decay=0.01)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-epoch)
                
                # Log expansion
                expansion_log.append({
                    'epoch': epoch + 1,
                    'accuracy_before': test_acc,
                    'parameters': sum(p.numel() for p in model.parameters()),
                    'expansion_count': expansion_engine.expansion_count
                })
                
                print(f"✅ Expansion applied successfully!")
                print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
                
            print(f"{'='*60}")
        
        # Early stopping if target reached
        if test_acc >= 93.0:
            print(f"\n🎉 TARGET ACCURACY REACHED: {test_acc:.2f}%!")
            break
        
        # Memory management
        if (epoch + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final evaluation and summary
    print("\n" + "="*60)
    print("🎉 EFFICIENT TRAINING COMPLETED!")
    print("="*60)
    
    final_test_acc, final_test_loss = evaluate_model(model, testloader, device)
    final_params = sum(p.numel() for p in model.parameters())
    
    print(f"📊 Final Results:")
    print(f"   Initial Accuracy: {initial_test_acc:.2f}%")
    print(f"   Final Accuracy: {final_test_acc:.2f}%")
    print(f"   Improvement: {final_test_acc - initial_test_acc:+.2f}%")
    print(f"   Final Parameters: {final_params:,}")
    print(f"   Expansion Count: {expansion_engine.expansion_count}")
    
    # Expansion summary
    if expansion_log:
        print(f"\n🔧 Expansion Summary:")
        for i, exp in enumerate(expansion_log):
            print(f"   {i+1}. Epoch {exp['epoch']:2d}: {exp['parameters']:,} parameters")
    
    # Save model
    save_path = 'efficient_cifar10_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'expansion_log': expansion_log,
        'final_accuracy': final_test_acc,
        'expansion_count': expansion_engine.expansion_count
    }, save_path)
    print(f"💾 Model saved to {save_path}")
    
    return model, final_test_acc, expansion_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficient Expansion CIFAR-10 Demo")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        model, accuracy, expansion_log = main(num_epochs=args.epochs, quick_mode=args.quick)
        print(f"\n✅ Training completed successfully!")
        print(f"Final accuracy: {accuracy:.2f}%")
        print(f"Total expansions: {len(expansion_log)}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 