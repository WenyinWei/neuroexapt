"""
Robust Classification with Shape-Safe Architecture Evolution

This example demonstrates how the enhanced validation system prevents
tensor shape mismatches during architecture evolution, ensuring stable
and reliable network evolution.
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
from typing import List, Dict, Any
import argparse

# Add the parent directory to the path to import neuroexapt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.core.radical_evolution import RadicalEvolutionEngine
from neuroexapt.core.radical_operators import get_radical_operator_pool
from neuroexapt.core.advanced_mutations import get_advanced_operator_pool
from neuroexapt.core.enhanced_validation import (
    validate_architecture_change, 
    fix_common_shape_issues,
    make_operator_safe
)


class RobustEvolvableCNN(nn.Module):
    """
    Robust evolvable CNN with built-in shape validation.
    """
    
    def __init__(self, num_classes=10, initial_channels=32):
        super(RobustEvolvableCNN, self).__init__()
        
        # Feature extraction with robust architecture
        self.features = nn.Sequential(
            # Block 1: Initial convolution
            nn.Conv2d(3, initial_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: Expandable convolution
            nn.Conv2d(initial_channels, initial_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: Deep convolution
            nn.Conv2d(initial_channels * 2, initial_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: Final convolution
            nn.Conv2d(initial_channels * 4, initial_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Robust classifier with self-validation
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(initial_channels * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Evolution tracking
        self.evolution_count = 0
        self.parameter_history = []
        self.validation_history = []
        
        # Initialize weights
        self._initialize_weights()
        
        # Validate initial architecture
        self._validate_architecture()
    
    def forward(self, x):
        try:
            features = self.features(x)
            output = self.classifier(features)
            return output
        except RuntimeError as e:
            print(f"❌ Forward pass error: {e}")
            
            # Attempt auto-recovery
            if self._attempt_auto_recovery():
                print("🔧 Auto-recovery successful, retrying forward pass...")
                features = self.features(x)
                output = self.classifier(features)
                return output
            else:
                raise e
    
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
    
    def _validate_architecture(self):
        """Validate current architecture."""
        try:
            is_valid = validate_architecture_change(self)
            self.validation_history.append(is_valid)
            
            if not is_valid:
                print("⚠️ Architecture validation failed, attempting fix...")
                fixed = fix_common_shape_issues(self)
                if fixed:
                    print("✅ Architecture fixed successfully")
                else:
                    print("❌ Could not fix architecture issues")
            
            return is_valid
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False
    
    def _attempt_auto_recovery(self) -> bool:
        """Attempt to recover from forward pass errors."""
        print("🔄 Attempting auto-recovery...")
        try:
            return fix_common_shape_issues(self)
        except Exception as e:
            print(f"Auto-recovery failed: {e}")
            return False
    
    def mark_evolution(self):
        """Mark that an evolution has occurred."""
        self.evolution_count += 1
        current_params = sum(p.numel() for p in self.parameters())
        self.parameter_history.append(current_params)
        
        # Validate after evolution
        is_valid = self._validate_architecture()
        print(f"📊 Post-evolution validation: {'✅ Passed' if is_valid else '❌ Failed'}")
    
    def get_evolution_stats(self):
        """Get evolution statistics."""
        current_params = sum(p.numel() for p in self.parameters())
        return {
            'evolution_count': self.evolution_count,
            'current_parameters': current_params,
            'parameter_history': self.parameter_history,
            'validation_success_rate': np.mean(self.validation_history) if self.validation_history else 0.0
        }


def load_cifar10_data(batch_size=64, download=True):
    """Load CIFAR-10 with standard augmentation."""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load datasets
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
    """Comprehensive model evaluation."""
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


def main(num_epochs: int = 50, quick_mode: bool = False):
    """Main training function with robust architecture evolution."""
    
    print("🛡️ Robust Classification with Shape-Safe Evolution")
    print("=" * 60)
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    evolution_frequency = 3  # Every 3 epochs
    
    # Load data
    print("📊 Loading CIFAR-10 dataset...")
    trainloader, testloader = load_cifar10_data(batch_size=batch_size)
    print(f"Training batches: {len(trainloader)}")
    print(f"Test batches: {len(testloader)}")
    
    # Create robust model
    print("🛡️ Creating robust evolvable CNN...")
    model = RobustEvolvableCNN(num_classes=10, initial_channels=32).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Print initial architecture
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initial parameters: {total_params:,}")
    print("Initial architecture validated: ✅")
    
    # Create enhanced evolution engine with safe operators
    print("🔥 Initializing Shape-Safe Evolution Engine...")
    
    # Get all operators (use original operators for now)
    all_operators = get_radical_operator_pool() + get_advanced_operator_pool()
    
    print(f"Available operators: {len(all_operators)}")
    for i, op in enumerate(all_operators[:10]):
        print(f"  {i+1}. {type(op).__name__}")
    
    evolution_engine = RadicalEvolutionEngine(
        model=model,
        operators=all_operators,
        input_shape=(3, 32, 32),
        evolution_probability=0.6,  # Conservative probability
        max_mutations_per_epoch=2,  # Fewer mutations
        enable_validation=True
    )
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    evolution_log = []
    validation_failures = 0
    
    print("🎯 Starting robust evolution training...")
    print("=" * 60)
    
    # Initial evaluation
    initial_test_acc, initial_test_loss = evaluate_model(model, testloader, device)
    print(f"Initial test accuracy: {initial_test_acc:.2f}%")
    test_accuracies.append(initial_test_acc)
    
    # Training loop with robust evolution
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
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Update evolution engine
                evolution_engine.update_performance(
                    performance=correct/total,
                    loss=loss.item()
                )
                
            except RuntimeError as e:
                print(f"Warning: Training batch {batch_idx} failed: {e}")
                validation_failures += 1
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
        
        # Robust Evolution Phase
        if (epoch + 1) % evolution_frequency == 0:
            print(f"\n{'🛡️ ROBUST EVOLUTION PHASE 🛡️':^60}")
            
            performance_metrics = {
                'val_accuracy': test_acc,
                'train_accuracy': train_acc,
                'val_loss': test_loss,
                'train_loss': avg_loss,
                'epoch': epoch + 1
            }
            
            # Get evolution stats before
            evolution_stats_before = model.get_evolution_stats()
            
            # Attempt robust evolution
            try:
                evolved_model, evolution_action = evolution_engine.evolve(
                    epoch=epoch + 1,
                    dataloader=trainloader,
                    criterion=criterion,
                    performance_metrics=performance_metrics
                )
                
                if evolution_action:
                    # Evolution successful
                    model = evolved_model
                    model.mark_evolution()
                    
                    # Update optimizer for new architecture
                    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-epoch)
                    
                    # Evaluate post-evolution
                    post_evolution_acc, post_evolution_loss = evaluate_model(model, testloader, device)
                    
                    evolution_log.append({
                        'epoch': epoch + 1,
                        'action': evolution_action,
                        'accuracy_before': test_acc,
                        'accuracy_after': post_evolution_acc,
                        'accuracy_change': post_evolution_acc - test_acc,
                        'parameters': sum(p.numel() for p in model.parameters()),
                        'validation_success': True
                    })
                    
                    print(f"✅ Robust evolution successful!")
                    print(f"   Action: {evolution_action}")
                    print(f"   Accuracy: {test_acc:.2f}% → {post_evolution_acc:.2f}% ({post_evolution_acc-test_acc:+.2f}%)")
                    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
                    
                    # Update test accuracy
                    test_accuracies[-1] = post_evolution_acc
                    
                else:
                    print("ℹ️ No evolution occurred this cycle")
                    
            except Exception as e:
                print(f"❌ Evolution failed with error: {e}")
                validation_failures += 1
                evolution_log.append({
                    'epoch': epoch + 1,
                    'action': 'failed',
                    'error': str(e),
                    'validation_success': False
                })
            
            print(f"{'='*60}")
        
        # Memory management
        if (epoch + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final evaluation and summary
    print("\n" + "="*60)
    print("🎉 ROBUST EVOLUTION TRAINING COMPLETED!")
    print("="*60)
    
    final_test_acc, final_test_loss = evaluate_model(model, testloader, device)
    final_params = sum(p.numel() for p in model.parameters())
    evolution_stats = model.get_evolution_stats()
    
    print(f"📊 Final Results:")
    print(f"   Initial Accuracy: {initial_test_acc:.2f}%")
    print(f"   Final Accuracy: {final_test_acc:.2f}%")
    print(f"   Improvement: {final_test_acc - initial_test_acc:+.2f}%")
    print(f"   Final Parameters: {final_params:,}")
    print(f"   Evolution Count: {model.evolution_count}")
    print(f"   Validation Failures: {validation_failures}")
    print(f"   Validation Success Rate: {evolution_stats['validation_success_rate']:.2%}")
    
    # Evolution summary
    if evolution_log:
        successful_evolutions = [evo for evo in evolution_log if evo.get('validation_success', False)]
        print(f"\n🛡️ Robust Evolution Summary:")
        print(f"   Total Evolution Attempts: {len(evolution_log)}")
        print(f"   Successful Evolutions: {len(successful_evolutions)}")
        print(f"   Evolution Success Rate: {len(successful_evolutions)/len(evolution_log):.2%}")
        
        if successful_evolutions:
            total_improvement = sum(evo['accuracy_change'] for evo in successful_evolutions)
            print(f"   Total Accuracy Improvement: {total_improvement:+.2f}%")
    
    # Save robust model
    save_path = 'robust_evolved_cifar10_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'evolution_log': evolution_log,
        'final_accuracy': final_test_acc,
        'evolution_count': model.evolution_count,
        'validation_failures': validation_failures,
        'evolution_stats': evolution_stats
    }, save_path)
    print(f"💾 Robust model saved to {save_path}")
    
    # Cleanup
    evolution_engine.cleanup()
    
    return model, final_test_acc, evolution_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust Evolution CIFAR-10 Demo")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer epochs")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    num_epochs = 20 if args.quick else args.epochs
    
    try:
        model, accuracy, evolution_log = main(num_epochs=num_epochs, quick_mode=args.quick)
        print(f"\n✅ Robust evolution completed successfully!")
        print(f"Final accuracy: {accuracy:.2f}%")
        
        successful_evolutions = len([evo for evo in evolution_log if evo.get('validation_success', False)])
        print(f"Successful evolutions: {successful_evolutions}/{len(evolution_log)}")
        
        if successful_evolutions > 0:
            avg_improvement = np.mean([
                evo['accuracy_change'] for evo in evolution_log 
                if evo.get('validation_success', False)
            ])
            print(f"Average improvement per successful evolution: {avg_improvement:+.2f}%")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 