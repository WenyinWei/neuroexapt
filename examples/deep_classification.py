"""
NeuroExapt V3 - Deep Classification Example

This example demonstrates the new NeuroExapt V3 framework on a deeper CNN
using the CIFAR-10 dataset. It shows how to apply the framework to a more
complex, ResNet-like architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.trainer_v3 import TrainerV3
from neuroexapt.core.operators import PruneByEntropy, ExpandWithMI

# --- 1. Define a standard, non-evolving Deep CNN ---
# This ResNet-like model is a good candidate for evolution, but contains no
# evolution logic itself. All architectural changes are handled by NeuroExapt.

class BasicBlock(nn.Module):
    """A standard basic block for ResNet-like architectures."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DeepCNN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(DeepCNN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return DeepCNN(BasicBlock, [2, 2, 2, 2])


# --- 2. Setup DataLoaders ---

def get_cifar10_loaders(batch_size=128):
    """Downloads CIFAR-10 and provides DataLoaders."""
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

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

# --- 3. Main Training Execution ---

def main():
    print("🚀 Starting NeuroExapt V3 Deep Classification Example 🚀")
    
    # Setup environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get DataLoaders
    train_loader, val_loader = get_cifar10_loaders()
    
    # Initialize a ResNet-18 model, criterion, and optimizer
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    print(f"Initial ResNet-18 model has {sum(p.numel() for p in model.parameters()):,} parameters.")

    # Define Structural Operators for NeuroExapt
    operators = [
        PruneByEntropy(threshold=0.2, layers_to_prune=1),
        ExpandWithMI(mi_threshold=0.85),
    ]
    
    # Initialize and run the Trainer
    trainer = TrainerV3(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        operators=operators
    )
    
    # Start the training and evolution process
    final_model = trainer.fit(epochs=20, evolution_frequency=2)
    
    print("\n🎉 Deep network training and evolution complete! 🎉")
    print(f"Final model has {sum(p.numel() for p in final_model.parameters()):,} parameters.")
    
if __name__ == "__main__":
    main()