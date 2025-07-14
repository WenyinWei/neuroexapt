"""
Test script for the new depth expansion operators.

This script demonstrates:
1. Depth expansion operators that prioritize adding layers when accuracy is low
2. Channel expansion operators that increase network width
3. Residual block expansion for skip connections
4. Accuracy-based operator selection strategy
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

# Add the parent directory to the path to import neuroexapt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt.core.radical_evolution import RadicalEvolutionEngine
from neuroexapt.core.depth_expansion_operators import get_depth_expansion_operators
from neuroexapt.core.radical_operators import get_radical_operator_pool
from neuroexapt.core.advanced_mutations import get_advanced_operator_pool


class TestCNN(nn.Module):
    """Simple CNN for testing depth expansion"""
    
    def __init__(self, num_classes=10):
        super(TestCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_small_cifar10(batch_size=32, num_samples=1000):
    """Load a small subset of CIFAR-10 for quick testing"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Create small subsets
    train_indices = torch.randperm(len(trainset))[:num_samples].tolist()
    test_indices = torch.randperm(len(testset))[:num_samples//5].tolist()
    
    train_subset = torch.utils.data.Subset(trainset, train_indices)
    test_subset = torch.utils.data.Subset(testset, test_indices)
    
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader


def evaluate_model(model, testloader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in testloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    model.train()
    return 100 * correct / total if total > 0 else 0.0


def test_depth_expansion_operators():
    """Test the depth expansion operators"""
    
    print("🚀 Testing Depth Expansion Operators")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    model = TestCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Load small dataset
    trainloader, testloader = load_small_cifar10(batch_size=32, num_samples=1000)
    
    # Get initial metrics
    initial_params = sum(p.numel() for p in model.parameters())
    initial_accuracy = evaluate_model(model, testloader, device)
    
    print(f"Initial model:")
    print(f"  Parameters: {initial_params:,}")
    print(f"  Accuracy: {initial_accuracy:.2f}%")
    print(f"  Architecture: {len(list(model.modules()))} modules")
    
    # Create evolution engine with depth expansion priority
    depth_operators = get_depth_expansion_operators(min_accuracy_for_pruning=0.75)
    radical_operators = get_radical_operator_pool()
    advanced_operators = get_advanced_operator_pool()
    
    # Prioritize depth expansion operators
    all_operators = depth_operators + radical_operators + advanced_operators
    
    print(f"\nOperator pool:")
    print(f"  🚀 Depth expansion: {len(depth_operators)}")
    print(f"  🔥 Radical: {len(radical_operators)}")
    print(f"  🧬 Advanced: {len(advanced_operators)}")
    print(f"  Total: {len(all_operators)}")
    
    evolution_engine = RadicalEvolutionEngine(
        model=model,
        operators=all_operators,
        input_shape=(3, 32, 32),
        evolution_probability=1.0,  # Always evolve for testing
        max_mutations_per_epoch=1,
        enable_validation=True
    )
    
    # Test evolution at different accuracy levels
    print("\n🧪 Testing Evolution at Different Accuracy Levels")
    print("=" * 50)
    
    # Simulate different accuracy levels
    accuracy_levels = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95]
    
    for accuracy in accuracy_levels:
        print(f"\n📊 Testing with simulated accuracy: {accuracy:.1%}")
        
        # Create performance metrics
        performance_metrics = {
            'val_accuracy': accuracy * 100,
            'train_accuracy': accuracy * 100,
            'val_loss': 2.0 * (1 - accuracy),
            'train_loss': 2.0 * (1 - accuracy),
            'epoch': 1
        }
        
        # Update evolution engine performance
        evolution_engine.update_performance(accuracy, 2.0 * (1 - accuracy))
        
        # Test operator selection
        state = {
            'current_performance': accuracy,
            'gradient_urgency': 0.5,
            'entropy_metrics': {}
        }
        
        # Test multiple selections to see pattern
        selected_operators = []
        for i in range(5):
            selected_op = evolution_engine._select_operator(all_operators, state)
            if selected_op:
                selected_operators.append(type(selected_op).__name__)
        
        print(f"   Selected operators: {selected_operators}")
        
        # Test actual evolution
        try:
            model_copy = evolution_engine._create_safe_copy(model)
            evolved_model, action = evolution_engine.evolve(
                epoch=1,
                dataloader=trainloader,
                criterion=criterion,
                performance_metrics=performance_metrics
            )
            
            if action:
                evolved_params = sum(p.numel() for p in evolved_model.parameters())
                param_change = evolved_params - initial_params
                print(f"   ✅ Evolution successful: {action}")
                print(f"   📈 Parameter change: {param_change:+,} ({param_change/initial_params:+.1%})")
            else:
                print(f"   ❌ Evolution failed or not needed")
                
        except Exception as e:
            print(f"   ❌ Evolution error: {e}")
    
    # Final model comparison
    print("\n📋 Final Model Comparison")
    print("=" * 50)
    
    try:
        # Try one more evolution with low accuracy
        performance_metrics = {
            'val_accuracy': 30.0,
            'train_accuracy': 30.0,
            'val_loss': 1.5,
            'train_loss': 1.5,
            'epoch': 1
        }
        
        evolved_model, action = evolution_engine.evolve(
            epoch=1,
            dataloader=trainloader,
            criterion=criterion,
            performance_metrics=performance_metrics
        )
        
        if evolved_model and action:
            final_params = sum(p.numel() for p in evolved_model.parameters())
            final_accuracy = evaluate_model(evolved_model, testloader, device)
            
            print(f"Final evolved model:")
            print(f"  Parameters: {final_params:,} ({final_params-initial_params:+,})")
            print(f"  Accuracy: {final_accuracy:.2f}% ({final_accuracy-initial_accuracy:+.2f}%)")
            print(f"  Architecture: {len(list(evolved_model.modules()))} modules")
            print(f"  Last action: {action}")
            
        else:
            print("No successful evolution in final test")
    
    except Exception as e:
        print(f"Final evolution test failed: {e}")
    
    # Evolution engine statistics
    print(f"\n📊 Evolution Engine Statistics:")
    stats = evolution_engine.get_evolution_stats()
    print(f"  Total mutations: {stats['total_mutations']}")
    print(f"  Successful mutations: {stats['successful_mutations']}")
    print(f"  Success rate: {stats['overall_success_rate']:.2%}")
    print(f"  Recent success rate: {stats['recent_success_rate']:.2%}")
    
    print("\n✅ Depth expansion test completed!")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        test_depth_expansion_operators()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 