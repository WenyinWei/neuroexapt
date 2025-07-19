#!/usr/bin/env python3
"""
Simple test script for dataset download using torchvision with improved handling.
This bypasses the problematic Chinese mirrors and uses more reliable sources.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import sys
import time

def download_with_progress(dataset_name="CIFAR10"):
    """Download dataset with progress tracking using torchvision."""
    print(f"🚀 Downloading {dataset_name} dataset...")
    
    # Create data directory
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    start_time = time.time()
    
    try:
        if dataset_name == "CIFAR10":
            # Download CIFAR-10
            trainset = torchvision.datasets.CIFAR10(
                root=data_dir, 
                train=True, 
                download=True, 
                transform=transform
            )
            testset = torchvision.datasets.CIFAR10(
                root=data_dir, 
                train=False, 
                download=True, 
                transform=transform
            )
        elif dataset_name == "CIFAR100":
            # Download CIFAR-100
            trainset = torchvision.datasets.CIFAR100(
                root=data_dir, 
                train=True, 
                download=True, 
                transform=transform
            )
            testset = torchvision.datasets.CIFAR100(
                root=data_dir, 
                train=False, 
                download=True, 
                transform=transform
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        download_time = time.time() - start_time
        print(f"✅ {dataset_name} download completed in {download_time:.1f} seconds!")
        
        # Create data loaders
        train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
        test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
        
        print(f"📊 Dataset info:")
        print(f"   Train samples: {len(trainset)}")
        print(f"   Test samples: {len(testset)}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Test data loading
        print(f"\n🧪 Testing data loading:")
        for i, (data, target) in enumerate(train_loader):
            print(f"   Batch {i+1}: {data.shape}, targets: {target.shape}")
            if i >= 2:  # Just test first 3 batches
                break
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to test dataset downloads."""
    print("=" * 60)
    print("🧪 Simple Dataset Download Test")
    print("=" * 60)
    
    # Test CIFAR-10
    print("\n📥 Testing CIFAR-10 download:")
    success_cifar10 = download_with_progress("CIFAR10")
    
    if success_cifar10:
        print("\n📥 Testing CIFAR-100 download:")
        success_cifar100 = download_with_progress("CIFAR100")
    else:
        print("❌ Skipping CIFAR-100 due to CIFAR-10 failure")
        success_cifar100 = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Download Summary:")
    print(f"   CIFAR-10: {'✅ Success' if success_cifar10 else '❌ Failed'}")
    print(f"   CIFAR-100: {'✅ Success' if success_cifar100 else '❌ Failed'}")
    print("=" * 60)

if __name__ == "__main__":
    main() 