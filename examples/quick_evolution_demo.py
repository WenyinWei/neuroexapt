"""
Quick Radical Evolution Demo - Shows Architecture Changes in Action
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt.core.radical_evolution import RadicalEvolutionEngine
from neuroexapt.core.radical_operators import get_radical_operator_pool
from neuroexapt.core.advanced_mutations import get_advanced_operator_pool


class SimpleEvolvableCNN(nn.Module):
    """Simple CNN for quick evolution demo."""
    
    def __init__(self, num_classes=10):
        super(SimpleEvolvableCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        
        self.evolution_count = 0
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def mark_evolution(self):
        self.evolution_count += 1


def quick_evaluate(model, dataloader, device):
    """Quick evaluation on a few batches."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            if i >= 3:  # Only use 3 batches for speed
                break
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    model.train()
    return 100 * correct / total if total > 0 else 0.0


def main():
    print("🚀 Quick Radical Evolution Demo")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load small CIFAR-10 subset for speed
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                          download=True, transform=transform)
    
    # Use only 1000 samples for speed
    train_subset = Subset(trainset, range(1000))
    trainloader = DataLoader(train_subset, batch_size=64, shuffle=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
    test_subset = Subset(testset, range(200))
    testloader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Test samples: {len(test_subset)}")
    
    # Create model
    model = SimpleEvolvableCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"Initial parameters: {initial_params:,}")
    print("Initial architecture:")
    print(model)
    
    # Create evolution engine
    operators = get_radical_operator_pool() + get_advanced_operator_pool()
    evolution_engine = RadicalEvolutionEngine(
        model=model,
        operators=operators,
        input_shape=(3, 32, 32),
        evolution_probability=1.0,  # 100% evolution probability
        max_mutations_per_epoch=2,
        enable_validation=True
    )
    
    print(f"\nAvailable operators: {len(operators)}")
    for i, op in enumerate(operators[:5]):  # Show first 5
        print(f"  {i+1}. {op.__class__.__name__}")
    
    # Quick training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n🎯 Starting evolution demo...")
    
    for epoch in range(3):  # Only 3 epochs for demo
        print(f"\nEpoch {epoch + 1}/3")
        
        # Quick training
        model.train()
        for i, (data, targets) in enumerate(trainloader):
            if i >= 5:  # Only 5 batches per epoch
                break
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Update evolution engine
            evolution_engine.update_performance(
                performance=0.5,  # Dummy performance
                loss=loss.item()
            )
        
        # Evaluate
        accuracy = quick_evaluate(model, testloader, device)
        print(f"Quick accuracy: {accuracy:.1f}%")
        
        # Try evolution
        print("\n🧬 Attempting radical evolution...")
        
        performance_metrics = {
            'val_accuracy': accuracy,
            'train_accuracy': accuracy,
            'val_loss': 1.0,
            'train_loss': 1.0,
            'epoch': epoch + 1
        }
        
        evolved_model, evolution_action = evolution_engine.evolve(
            epoch=epoch + 1,
            dataloader=trainloader,
            criterion=criterion,
            performance_metrics=performance_metrics
        )
        
        if evolution_action:
            print(f"✅ Evolution successful!")
            print(f"   Action: {evolution_action}")
            
            model = evolved_model
            model.mark_evolution()
            
            new_params = sum(p.numel() for p in model.parameters())
            param_change = new_params - initial_params
            
            print(f"   Parameters: {initial_params:,} → {new_params:,} ({param_change:+,})")
            
            # Update optimizer for new model
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Quick post-evolution evaluation
            post_acc = quick_evaluate(model, testloader, device)
            print(f"   Post-evolution accuracy: {post_acc:.1f}%")
            
            print("\n🏗️ New Architecture:")
            print(model)
            
        else:
            print("ℹ️ No evolution occurred")
    
    # Final results
    final_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Demo Results:")
    print(f"   Initial parameters: {initial_params:,}")
    print(f"   Final parameters: {final_params:,}")
    print(f"   Parameter change: {final_params - initial_params:+,}")
    print(f"   Evolution count: {model.evolution_count}")
    
    evolution_stats = evolution_engine.get_evolution_stats()
    print(f"   Total mutations attempted: {evolution_stats['total_mutations']}")
    print(f"   Successful mutations: {evolution_stats['successful_mutations']}")
    if evolution_stats['total_mutations'] > 0:
        print(f"   Success rate: {evolution_stats['overall_success_rate']:.1%}")
    
    print("\n✅ Quick demo completed!")
    
    # Cleanup
    evolution_engine.cleanup()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc() 