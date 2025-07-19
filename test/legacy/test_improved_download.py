#!/usr/bin/env python3
"""
Test script for improved dataset download functionality.
Demonstrates robust download with chunked transfers and better error handling.
"""

import sys
import os
sys.path.append('.')

from neuroexapt.utils.dataset_loader import AdvancedDatasetLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_improved_download():
    """Test the improved download functionality."""
    print("=" * 60)
    print("🧪 Testing Improved Dataset Download")
    print("=" * 60)
    
    # Initialize loader with improved settings
    loader = AdvancedDatasetLoader(
        cache_dir="./test_cache",
        download_dir="./test_data", 
        use_p2p=False,  # Disable P2P for testing
        max_retries=2
    )
    
    print("\n📊 Cache Information:")
    cache_info = loader.get_cache_info()
    print(f"Cache directory: {cache_info['cache_dir']}")
    print(f"Total cached size: {cache_info['total_size'] / (1024*1024):.1f}MB")
    
    # Test CIFAR-10 download
    print("\n🚀 Testing CIFAR-10 Download:")
    try:
        success = loader.download_dataset('cifar10', force_download=False)
        if success:
            print("✅ CIFAR-10 download successful!")
            
            # Get data loaders
            train_loader, test_loader = loader.get_cifar10_dataloaders(
                batch_size=32, 
                num_workers=0,  # Use 0 for testing
                download=False  # Already downloaded
            )
            
            print(f"✅ Data loaders created successfully!")
            print(f"   Train batches: {len(train_loader)}")
            print(f"   Test batches: {len(test_loader)}")
            
            # Test a few batches
            print("\n🧪 Testing data loading:")
            for i, (data, target) in enumerate(train_loader):
                print(f"   Batch {i+1}: {data.shape}, targets: {target.shape}")
                if i >= 2:  # Just test first 3 batches
                    break
                    
        else:
            print("❌ CIFAR-10 download failed!")
            
    except Exception as e:
        print(f"❌ Error during download: {e}")
        import traceback
        traceback.print_exc()
    
    # Show final cache info
    print("\n📊 Final Cache Information:")
    cache_info = loader.get_cache_info()
    print(f"Cache directory: {cache_info['cache_dir']}")
    print(f"Total cached size: {cache_info['total_size'] / (1024*1024):.1f}MB")
    
    if cache_info['datasets']:
        for dataset, info in cache_info['datasets'].items():
            print(f"   {dataset}: {info['size'] / (1024*1024):.1f}MB ({info['files']} files)")
    
    print("\n" + "=" * 60)
    print("🏁 Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_improved_download() 