#!/usr/bin/env python3
"""
Demonstration of Intelligent Architecture Evolution System.

This demo showcases the new intelligent architecture evolution system that:
1. Analyzes network bottlenecks using information theory
2. Predicts architecture changes using Bayesian inference
3. Detects performance plateaus intelligently
4. Applies information-guided operators dynamically
5. Provides real-time evolution insights

Run this to see how the system intelligently evolves network architecture
during training for optimal performance.
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
import argparse
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Import our intelligent evolution system
from neuroexapt.core.intelligent_evolution_engine import IntelligentEvolutionEngine, EvolutionConfig
from neuroexapt.utils.gpu_manager import GPUManager


class DemoEvolutionVisualization:
    """Visualizes the evolution process in real-time."""
    
    def __init__(self, max_epochs: int = 100):
        self.max_epochs = max_epochs
        self.epochs = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.evolution_points = []
        self.bottleneck_scores = []
        self.information_flow_scores = []
        self.network_health_scores = []
        self.parameter_counts = []
        
        # Set up the plot
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('🧠 Intelligent Architecture Evolution - Real-Time Analysis', fontsize=16)
        
        # Configure subplots
        self.axes[0, 0].set_title('📈 Performance Evolution')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Accuracy')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        self.axes[0, 1].set_title('🔬 Information Theory Metrics')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Score')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        self.axes[1, 0].set_title('🏥 Network Health')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Health Score')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        self.axes[1, 1].set_title('🔢 Parameter Count Evolution')
        self.axes[1, 1].set_xlabel('Epoch')
        self.axes[1, 1].set_ylabel('Parameters')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def update(self, epoch: int, train_acc: float, val_acc: float, 
               evolution_result: Dict[str, Any], model: nn.Module):
        """Update the visualization with new data."""
        self.epochs.append(epoch)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.parameter_counts.append(sum(p.numel() for p in model.parameters()))
        
        # Check if evolution occurred
        if evolution_result.get('evolution_result', {}).get('applied', False):
            self.evolution_points.append(epoch)
        
        # Extract metrics
        network_analysis = evolution_result.get('network_analysis', {})
        self.bottleneck_scores.append(len(network_analysis.get('bottlenecks', [])))
        self.information_flow_scores.append(network_analysis.get('information_flow_efficiency', 0.5))
        self.network_health_scores.append(network_analysis.get('network_health', {}).get('health_score', 0.5))
        
        # Update plots
        self._update_plots()
        
    def _update_plots(self):
        """Update all plot displays."""
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Performance Evolution
        self.axes[0, 0].plot(self.epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        self.axes[0, 0].plot(self.epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        
        # Mark evolution points
        for evo_epoch in self.evolution_points:
            if evo_epoch in self.epochs:
                idx = self.epochs.index(evo_epoch)
                self.axes[0, 0].axvline(x=evo_epoch, color='green', linestyle='--', alpha=0.7)
                self.axes[0, 0].scatter([evo_epoch], [self.val_accuracies[idx]], 
                                     color='green', s=100, marker='*', zorder=5)
        
        self.axes[0, 0].set_title('📈 Performance Evolution')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Accuracy')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Information Theory Metrics
        self.axes[0, 1].plot(self.epochs, self.bottleneck_scores, 'orange', label='Bottlenecks', linewidth=2)
        self.axes[0, 1].plot(self.epochs, self.information_flow_scores, 'purple', label='Info Flow', linewidth=2)
        self.axes[0, 1].set_title('🔬 Information Theory Metrics')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Score')
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Network Health
        self.axes[1, 0].plot(self.epochs, self.network_health_scores, 'green', linewidth=2)
        self.axes[1, 0].fill_between(self.epochs, self.network_health_scores, alpha=0.3, color='green')
        self.axes[1, 0].set_title('🏥 Network Health')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Health Score')
        self.axes[1, 0].set_ylim(0, 1)
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Parameter Count Evolution
        self.axes[1, 1].plot(self.epochs, self.parameter_counts, 'brown', linewidth=2)
        
        # Mark evolution points
        for evo_epoch in self.evolution_points:
            if evo_epoch in self.epochs:
                self.axes[1, 1].axvline(x=evo_epoch, color='green', linestyle='--', alpha=0.7)
        
        self.axes[1, 1].set_title('🔢 Parameter Count Evolution')
        self.axes[1, 1].set_xlabel('Epoch')
        self.axes[1, 1].set_ylabel('Parameters')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.1)


class DemoNetwork(nn.Module):
    """A simple CNN for demonstration purposes."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_cifar10_data(batch_size: int = 128, num_workers: int = 2) -> tuple:
    """Load CIFAR-10 dataset."""
    
    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return trainloader, testloader


def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple:
    """Evaluate model on given dataloader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return accuracy, avg_loss


def calculate_gradient_norm(model: nn.Module) -> float:
    """Calculate the gradient norm for the model."""
    total_norm = 0.0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    return (total_norm ** 0.5) / max(param_count, 1)


def print_evolution_summary(evolution_result: Dict[str, Any], epoch: int):
    """Print a summary of the evolution analysis."""
    print(f"\n{'='*60}")
    print(f"🔬 INTELLIGENT EVOLUTION ANALYSIS - EPOCH {epoch}")
    print(f"{'='*60}")
    
    # Network analysis
    network_analysis = evolution_result.get('network_analysis', {})
    bottlenecks = network_analysis.get('bottlenecks', [])
    redundancies = network_analysis.get('redundancies', [])
    network_health = network_analysis.get('network_health', {})
    
    print(f"🏥 Network Health: {network_health.get('status', 'unknown').upper()}")
    print(f"   Health Score: {network_health.get('health_score', 0):.3f}")
    print(f"   Bottlenecks: {len(bottlenecks)}")
    print(f"   Redundancies: {len(redundancies)}")
    
    # Information flow
    info_flow = network_analysis.get('information_flow_efficiency', 0)
    print(f"📊 Information Flow Efficiency: {info_flow:.3f}")
    
    # Most severe bottleneck
    if bottlenecks:
        severe_bottleneck = max(bottlenecks, key=lambda b: b.severity)
        print(f"🚨 Most Severe Bottleneck:")
        print(f"   Layers: {severe_bottleneck.layer1} → {severe_bottleneck.layer2}")
        print(f"   Mutual Information: {severe_bottleneck.mutual_information:.3f}")
        print(f"   Severity: {severe_bottleneck.severity:.3f}")
    
    # Evolution decision
    evolution_decision = evolution_result.get('evolution_decision')
    if evolution_decision:
        print(f"🎯 Evolution Decision:")
        print(f"   Should Evolve: {'YES' if evolution_decision.should_evolve else 'NO'}")
        print(f"   Confidence: {evolution_decision.confidence:.3f}")
        print(f"   Reasoning: {evolution_decision.reasoning}")
        if evolution_decision.should_evolve:
            print(f"   Expected Benefit: {evolution_decision.expected_benefit:.4f}")
            print(f"   Information Basis: {evolution_decision.information_basis}")
    
    # Evolution result
    evolution_result_data = evolution_result.get('evolution_result', {})
    if evolution_result_data.get('applied', False):
        if evolution_result_data.get('success', False):
            print(f"✅ Evolution Applied Successfully!")
            print(f"   Operation: {evolution_result_data.get('operation', 'unknown')}")
        else:
            print(f"❌ Evolution Failed!")
            print(f"   Reason: {evolution_result_data.get('reason', 'unknown')}")
    
    # Plateau detection
    plateau_info = evolution_result.get('plateau_info')
    if plateau_info:
        print(f"🏔️ Plateau Detected:")
        print(f"   Type: {plateau_info.plateau_type.value}")
        print(f"   Duration: {plateau_info.duration} epochs")
        print(f"   Severity: {plateau_info.severity:.3f}")
        print(f"   Confidence: {plateau_info.confidence:.3f}")
    
    print(f"{'='*60}")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description='Intelligent Architecture Evolution Demo')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--visualize', action='store_true', help='Enable real-time visualization')
    parser.add_argument('--evolution-freq', type=int, default=1, help='Evolution analysis frequency')
    parser.add_argument('--cooldown', type=int, default=3, help='Evolution cooldown period')
    
    args = parser.parse_args()
    
    # Setup device
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Starting Intelligent Architecture Evolution Demo")
    print(f"   Device: {device}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Evolution Frequency: {args.evolution_freq}")
    print(f"   Evolution Cooldown: {args.cooldown}")
    
    # Load data
    print("\n📊 Loading CIFAR-10 dataset...")
    trainloader, testloader = load_cifar10_data(batch_size=args.batch_size)
    
    # Create model
    print("\n🏗️ Creating initial network...")
    model = DemoNetwork(num_classes=10).to(device)
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"   Initial parameters: {initial_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Setup intelligent evolution engine
    print("\n🧠 Initializing Intelligent Evolution Engine...")
    evolution_config = EvolutionConfig(
        analysis_frequency=args.evolution_freq,
        evolution_cooldown=args.cooldown,
        confidence_threshold=0.6,
        information_threshold=0.05,
        enable_proactive_evolution=True,
        enable_reactive_evolution=True,
        max_consecutive_failures=3
    )
    
    evolution_engine = IntelligentEvolutionEngine(
        model=model,
        config=evolution_config,
        input_shape=(3, 32, 32)
    )
    
    # Setup visualization
    visualization = None
    if args.visualize:
        print("\n📈 Setting up real-time visualization...")
        visualization = DemoEvolutionVisualization(max_epochs=args.epochs)
    
    # Initial evaluation
    print("\n🔍 Initial evaluation...")
    val_accuracy, val_loss = evaluate_model(model, testloader, criterion, device)
    print(f"   Initial validation accuracy: {val_accuracy:.2f}%")
    print(f"   Initial validation loss: {val_loss:.4f}")
    
    # Training loop with intelligent evolution
    print(f"\n🎯 Starting training with intelligent evolution...")
    print(f"{'='*80}")
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Calculate training metrics
        train_accuracy = 100.0 * train_correct / train_total
        train_loss = train_loss / len(trainloader)
        
        # Evaluation phase
        val_accuracy, val_loss = evaluate_model(model, testloader, criterion, device)
        
        # Calculate gradient norm
        gradient_norm = calculate_gradient_norm(model)
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        # Intelligent evolution step
        print(f"\n🧠 Epoch {epoch+1}/{args.epochs} - Intelligent Evolution Analysis")
        print(f"   📊 Performance: Train {train_accuracy:.2f}%, Val {val_accuracy:.2f}%")
        print(f"   📉 Loss: Train {train_loss:.4f}, Val {val_loss:.4f}")
        print(f"   🔄 Gradient Norm: {gradient_norm:.6f}")
        
        # Apply intelligent evolution
        evolved_model, evolution_result = evolution_engine.step(
            epoch=epoch,
            train_performance=train_accuracy / 100.0,
            val_performance=val_accuracy / 100.0,
            train_loss=train_loss,
            val_loss=val_loss,
            gradient_norm=gradient_norm,
            learning_rate=current_lr,
            dataloader=trainloader,
            criterion=criterion,
            optimizer=optimizer
        )
        
        # Update model reference if evolved
        if evolved_model is not model:
            model = evolved_model
            # Update optimizer for new model parameters
            optimizer = optim.Adam(model.parameters(), lr=current_lr)
        
        # Print evolution summary
        if not evolution_result.get('skipped', False):
            print_evolution_summary(evolution_result, epoch + 1)
        
        # Update visualization
        if visualization:
            visualization.update(epoch, train_accuracy, val_accuracy, evolution_result, model)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        current_params = sum(p.numel() for p in model.parameters())
        param_change = current_params - initial_params
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   🎯 Accuracy: {val_accuracy:.2f}% ({val_accuracy - val_accuracy:.2f}%)")
        print(f"   📈 Parameters: {current_params:,} ({param_change:+,})")
        print(f"   ⏱️ Time: {epoch_time:.1f}s")
        print(f"   🔄 Learning Rate: {current_lr:.6f}")
        
        print(f"{'='*80}")
    
    # Final evaluation and report
    print(f"\n🎉 Training Complete!")
    print(f"{'='*80}")
    
    final_accuracy, final_loss = evaluate_model(model, testloader, criterion, device)
    final_params = sum(p.numel() for p in model.parameters())
    
    print(f"📊 Final Results:")
    print(f"   🎯 Final Accuracy: {final_accuracy:.2f}%")
    print(f"   📉 Final Loss: {final_loss:.4f}")
    print(f"   📈 Final Parameters: {final_params:,}")
    print(f"   🔄 Parameter Change: {final_params - initial_params:+,}")
    print(f"   📊 Parameter Efficiency: {final_accuracy / (final_params / 1000):.3f} acc/K params")
    
    # Get evolution report
    evolution_report = evolution_engine.get_evolution_report()
    stats = evolution_report['statistics']
    
    print(f"\n🧠 Evolution Engine Report:")
    print(f"   🔧 Total Evolutions: {stats['total_evolutions']}")
    print(f"   ✅ Successful Evolutions: {stats['successful_evolutions']}")
    print(f"   🎯 Success Rate: {stats['successful_evolutions'] / max(stats['total_evolutions'], 1):.2%}")
    print(f"   🏆 Best Performance: {stats['best_performance']:.3f}")
    print(f"   🔄 Consecutive Failures: {stats['consecutive_failures']}")
    
    # Evolution trigger analysis
    if stats['evolution_trigger_reasons']:
        print(f"\n🎯 Evolution Triggers:")
        for reason, count in stats['evolution_trigger_reasons'].items():
            print(f"   {reason.capitalize()}: {count}")
    
    # Keep visualization open
    if visualization:
        print(f"\n📈 Visualization will remain open. Close the window to exit.")
        input("Press Enter to close...")
    
    # Cleanup
    evolution_engine.cleanup()
    
    print(f"\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main() 