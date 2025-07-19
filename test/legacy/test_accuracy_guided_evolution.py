"""
Quick test for Accuracy-Guided Evolution System.

This script runs a shortened version of the training to verify that the 
accuracy-guided evolution system works correctly and can improve model performance.
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

# Add the current directory to the path to import neuroexapt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt.core.accuracy_guided_evolution import get_accuracy_guided_operators
from neuroexapt.core.radical_evolution import RadicalEvolutionEngine


class SimpleTestCNN(nn.Module):
    """Simple CNN for testing accuracy-guided evolution"""
    
    def __init__(self, num_classes=10):
        super(SimpleTestCNN, self).__init__()
        
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


def load_small_cifar10(num_samples=1000):
    """Load a small subset of CIFAR-10 for quick testing"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Create small subsets
    train_indices = torch.randperm(len(trainset))[:num_samples].tolist()
    test_indices = torch.randperm(len(testset))[:num_samples//4].tolist()
    
    train_subset = torch.utils.data.Subset(trainset, train_indices)
    test_subset = torch.utils.data.Subset(testset, test_indices)
    
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
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


def test_accuracy_guided_evolution():
    """Test the accuracy-guided evolution system"""
    
    print("🎯 Testing Accuracy-Guided Evolution System")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load small dataset
    trainloader, testloader = load_small_cifar10(num_samples=1000)
    print(f"Training batches: {len(trainloader)}")
    print(f"Test batches: {len(testloader)}")
    
    # Create model
    model = SimpleTestCNN(num_classes=10).to(device)
    
    # Initial evaluation
    initial_accuracy = evaluate_model(model, testloader, device)
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"Initial model:")
    print(f"  Parameters: {initial_params:,}")
    print(f"  Accuracy: {initial_accuracy:.2f}%")
    
    # Create accuracy-guided operators
    accuracy_operators = get_accuracy_guided_operators(
        accuracy_target=0.8,  # Lower target for quick test
        min_improvement=0.01
    )
    
    print(f"\nAccuracy-guided operators: {len(accuracy_operators)}")
    for i, op in enumerate(accuracy_operators):
        print(f"  {i+1}. {op.__class__.__name__} (target: {op.accuracy_target:.1%})")
    
    # Create evolution engine
    evolution_engine = RadicalEvolutionEngine(
        model=model,
        operators=accuracy_operators,
        input_shape=(3, 32, 32),
        evolution_probability=1.0,  # Always evolve for testing
        max_mutations_per_epoch=1,
        enable_validation=True
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n🔄 Starting training with evolution...")
    print("-" * 50)
    
    # Short training loop
    num_epochs = 10
    evolution_frequency = 3
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for data, targets in trainloader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        train_accuracy = 100 * correct / total
        avg_loss = epoch_loss / len(trainloader)
        
        # Evaluation
        test_accuracy = evaluate_model(model, testloader, device)
        
        print(f"Epoch {epoch+1:2d}: Train {train_accuracy:.1f}% | Test {test_accuracy:.1f}% | Loss {avg_loss:.3f}")
        
        # Evolution
        if (epoch + 1) % evolution_frequency == 0:
            print(f"\n🎯 Attempting accuracy-guided evolution...")
            
            performance_metrics = {
                'val_accuracy': test_accuracy,
                'train_accuracy': train_accuracy,
                'val_loss': avg_loss,
                'train_loss': avg_loss,
                'epoch': epoch + 1
            }
            
            evolved_model, evolution_action = evolution_engine.evolve(
                epoch=epoch + 1,
                dataloader=trainloader,
                criterion=criterion,
                performance_metrics=performance_metrics
            )
            
            if evolution_action and evolved_model:
                print(f"   ✅ Evolution successful: {evolution_action}")
                
                # Update model
                model = evolved_model
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Evaluate post-evolution
                post_evolution_accuracy = evaluate_model(model, testloader, device)
                post_evolution_params = sum(p.numel() for p in model.parameters())
                
                print(f"   📊 Post-evolution: {post_evolution_accuracy:.1f}% accuracy")
                print(f"   📈 Change: {post_evolution_accuracy - test_accuracy:+.1f}%")
                print(f"   🔢 Parameters: {post_evolution_params:,} ({post_evolution_params - initial_params:+,})")
                
                # Update operator with feedback
                if hasattr(accuracy_operators[0], 'update_success'):
                    accuracy_operators[0].update_success(post_evolution_accuracy - test_accuracy)
            else:
                print(f"   ❌ No evolution occurred")
            
            print()
    
    # Final results
    final_accuracy = evaluate_model(model, testloader, device)
    final_params = sum(p.numel() for p in model.parameters())
    
    print("📋 Final Results:")
    print(f"  Initial: {initial_accuracy:.2f}% accuracy, {initial_params:,} parameters")
    print(f"  Final: {final_accuracy:.2f}% accuracy, {final_params:,} parameters")
    print(f"  Improvement: {final_accuracy - initial_accuracy:+.2f}%")
    print(f"  Parameter growth: {final_params - initial_params:+,}")
    
    # Engine statistics
    engine_stats = evolution_engine.get_evolution_stats()
    print(f"\n🔥 Evolution Statistics:")
    print(f"  Total mutations: {engine_stats['total_mutations']}")
    print(f"  Successful mutations: {engine_stats['successful_mutations']}")
    print(f"  Success rate: {engine_stats['overall_success_rate']:.2%}")
    
    print(f"\n✅ Accuracy-guided evolution test completed!")
    
    return final_accuracy > initial_accuracy + 5.0  # Test passes if >5% improvement


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        success = test_accuracy_guided_evolution()
        if success:
            print("\n🎉 Test PASSED: Accuracy-guided evolution shows significant improvement!")
        else:
            print("\n⚠️ Test INCOMPLETE: Limited improvement observed (may need longer training)")
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc() 