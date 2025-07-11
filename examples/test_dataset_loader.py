"""
Test script for the Advanced Dataset Loader.

This script demonstrates the P2P acceleration and caching capabilities
of the advanced dataset loader, optimized for Chinese users.
"""

import sys
import os
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.utils.dataset_loader import AdvancedDatasetLoader
from neuroexapt.utils.xunlei_downloader import XunleiDownloader


def test_dataset_loader():
    """Test the advanced dataset loader functionality."""
    
    print("=" * 60)
    print("Testing Advanced Dataset Loader")
    print("=" * 60)
    
    # Initialize the advanced loader
    loader = AdvancedDatasetLoader(
        cache_dir="./data_cache",
        download_dir="./data",
        use_p2p=True,
        max_retries=3
    )
    
    print(f"Cache directory: {loader.cache.cache_dir}")
    print(f"Download directory: {loader.download_dir}")
    print(f"P2P enabled: {loader.p2p_downloader is not None}")
    
    # Test mirror connectivity
    print("\n" + "-" * 40)
    print("Testing Mirror Connectivity")
    print("-" * 40)
    
    for mirror in loader.mirrors:
        print(f"{mirror.name} ({mirror.region}): {'✓' if mirror.is_available else '✗'} "
              f"[{mirror.response_time:.3f}s]")
    
    # Test cache functionality
    print("\n" + "-" * 40)
    print("Cache Information")
    print("-" * 40)
    
    cache_info = loader.get_cache_info()
    print(f"Cache directory: {cache_info['cache_dir']}")
    print(f"Total cache size: {cache_info['total_size'] / (1024**3):.2f} GB")
    
    for dataset, info in cache_info['datasets'].items():
        print(f"  {dataset}: {info['files']} files, {info['size'] / (1024**2):.2f} MB")
    
    # Test CIFAR-10 download
    print("\n" + "-" * 40)
    print("Testing CIFAR-10 Download")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        train_loader, test_loader = loader.get_cifar10_dataloaders(
            batch_size=32,
            num_workers=2,
            download=True,
            force_download=False
        )
        
        download_time = time.time() - start_time
        
        print(f"✓ CIFAR-10 download completed in {download_time:.2f} seconds")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test a few batches
        print("\nTesting data loading...")
        for i, (data, target) in enumerate(train_loader):
            print(f"  Batch {i+1}: {data.shape}, {target.shape}")
            if i >= 2:  # Only test first 3 batches
                break
        
    except Exception as e:
        print(f"✗ CIFAR-10 download failed: {e}")
        return False
    
    # Test cache after download
    print("\n" + "-" * 40)
    print("Cache After Download")
    print("-" * 40)
    
    cache_info = loader.get_cache_info()
    for dataset, info in cache_info['datasets'].items():
        print(f"  {dataset}: {info['files']} files, {info['size'] / (1024**2):.2f} MB")
    
    # Test CIFAR-100 download
    print("\n" + "-" * 40)
    print("Testing CIFAR-100 Download")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        train_loader, test_loader = loader.get_cifar100_dataloaders(
            batch_size=32,
            num_workers=2,
            download=True,
            force_download=False
        )
        
        download_time = time.time() - start_time
        
        print(f"✓ CIFAR-100 download completed in {download_time:.2f} seconds")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
        
    except Exception as e:
        print(f"✗ CIFAR-100 download failed: {e}")
    
    # Test cache clearing
    print("\n" + "-" * 40)
    print("Testing Cache Management")
    print("-" * 40)
    
    print("Cache before clearing:")
    cache_info = loader.get_cache_info()
    for dataset, info in cache_info['datasets'].items():
        print(f"  {dataset}: {info['files']} files, {info['size'] / (1024**2):.2f} MB")
    
    # Clear cache for a specific dataset
    loader.clear_cache("cifar10")
    
    print("\nCache after clearing CIFAR-10:")
    cache_info = loader.get_cache_info()
    for dataset, info in cache_info['datasets'].items():
        print(f"  {dataset}: {info['files']} files, {info['size'] / (1024**2):.2f} MB")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
    return True


def test_mirror_selection():
    """Test mirror selection and fallback mechanisms."""
    
    print("\n" + "=" * 60)
    print("Testing Mirror Selection")
    print("=" * 60)
    
    loader = AdvancedDatasetLoader()
    
    # Test mirror selection
    best_mirror = loader._get_best_mirror()
    if best_mirror:
        print(f"Best available mirror: {best_mirror.name} ({best_mirror.region})")
        print(f"  Response time: {best_mirror.response_time:.3f}s")
        print(f"  Priority: {best_mirror.priority}")
    else:
        print("No available mirrors found")
    
    # Test mirror URLs
    if best_mirror:
        test_url = best_mirror.get_download_url("cifar10", "cifar-10-python.tar.gz")
        print(f"Sample download URL: {test_url}")


if __name__ == "__main__":
    # Test basic functionality
    success = test_dataset_loader()
    
    # Test mirror selection
    test_mirror_selection()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 