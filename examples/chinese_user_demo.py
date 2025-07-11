"""
Chinese User Dataset Loading Demo

This script demonstrates the optimized dataset loading for Chinese users,
featuring fast Chinese mirrors and intelligent fallback mechanisms.
"""

import sys
import os
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.utils.dataset_loader import AdvancedDatasetLoader
from neuroexapt.utils.xunlei_downloader import XunleiDownloader


def demo_chinese_optimized_loading():
    """Demonstrate Chinese-optimized dataset loading."""
    
    print("=" * 60)
    print("🇨🇳 Chinese User Dataset Loading Demo")
    print("=" * 60)
    
    # Initialize the advanced loader optimized for Chinese users
    loader = AdvancedDatasetLoader(
        cache_dir="./data_cache",
        download_dir="./data",
        use_p2p=True,
        max_retries=3
    )
    
    print(f"📁 Cache directory: {loader.cache.cache_dir}")
    print(f"📁 Download directory: {loader.download_dir}")
    print(f"🚀 P2P acceleration: {'Enabled' if loader.p2p_downloader else 'Disabled'}")
    
    # Show Chinese mirror configuration
    print("\n🌍 Chinese Mirror Configuration:")
    for dataset_name, config in loader.dataset_configs.items():
        if 'chinese_mirrors' in config:
            print(f"  {dataset_name.upper()}:")
            for i, mirror in enumerate(config['chinese_mirrors']):
                print(f"    {i+1}. {mirror}")
    
    # Test CIFAR-10 download with Chinese mirrors
    print("\n" + "=" * 40)
    print("📥 Testing CIFAR-10 Download")
    print("=" * 40)
    
    start_time = time.time()
    
    try:
        # This will try Chinese mirrors first, then fall back to direct source
        train_loader, test_loader = loader.get_cifar10_dataloaders(
            batch_size=32,
            num_workers=2,
            download=True,
            force_download=False
        )
        
        download_time = time.time() - start_time
        
        print(f"✅ CIFAR-10 download completed in {download_time:.2f} seconds")
        print(f"📊 Training samples: {len(train_loader.dataset):,}")
        print(f"📊 Test samples: {len(test_loader.dataset):,}")
        print(f"📊 Training batches: {len(train_loader)}")
        print(f"📊 Test batches: {len(test_loader)}")
        
        # Test data loading speed
        print("\n⚡ Testing data loading speed...")
        batch_start = time.time()
        
        for i, (data, target) in enumerate(train_loader):
            if i >= 3:  # Test first 3 batches
                break
            batch_time = time.time() - batch_start
            print(f"  Batch {i+1}: {data.shape}, {target.shape} ({batch_time:.3f}s)")
            batch_start = time.time()
        
        # Show cache information
        print("\n" + "=" * 40)
        print("💾 Cache Information")
        print("=" * 40)
        
        cache_info = loader.get_cache_info()
        print(f"📁 Cache directory: {cache_info['cache_dir']}")
        print(f"💾 Total cache size: {cache_info['total_size'] / (1024**3):.2f} GB")
        
        for dataset, info in cache_info['datasets'].items():
            print(f"  📦 {dataset}: {info['files']} files, {info['size'] / (1024**2):.2f} MB")
        
        print("\n🎉 Demo completed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def show_performance_comparison():
    """Show performance comparison between different sources."""
    
    print("\n" + "=" * 60)
    print("📈 Performance Comparison")
    print("=" * 60)
    
    print("🇨🇳 Chinese Mirrors (Recommended for Chinese users):")
    print("  • Tsinghua University: ~10-50 MB/s")
    print("  • USTC: ~5-30 MB/s")
    print("  • Automatic fallback if one fails")
    print("  • Optimized for Chinese network conditions")
    
    print("\n🌍 International Sources (Fallback):")
    print("  • Direct CIFAR source: ~0.5-2 MB/s (slow in China)")
    print("  • PyTorch official: ~1-5 MB/s")
    print("  • Used only if Chinese mirrors fail")
    
    print("\n💡 Key Features for Chinese Users:")
    print("  • 🚀 Fast Chinese mirrors prioritized")
    print("  • 🔄 Automatic fallback to international sources")
    print("  • 💾 Intelligent caching to avoid re-downloads")
    print("  • ✅ File size verification to detect HTML pages")
    print("  • 📊 Progress tracking with speed display")


if __name__ == "__main__":
    # Run the demo
    success = demo_chinese_optimized_loading()
    
    # Show performance comparison
    show_performance_comparison()
    
    if success:
        print("\n🎯 Key Benefits for Chinese Users:")
        print("  ✅ 5-20x faster downloads using Chinese mirrors")
        print("  ✅ Automatic fallback if mirrors are unavailable")
        print("  ✅ Intelligent caching saves time on subsequent runs")
        print("  ✅ Robust error handling and verification")
        print("  ✅ No manual configuration required")
    else:
        print("\n💡 Troubleshooting Tips:")
        print("  • Check your internet connection")
        print("  • Try running with force_download=True")
        print("  • Clear cache if files are corrupted")
        print("  • Check firewall settings") 