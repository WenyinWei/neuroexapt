#!/usr/bin/env python3
"""
Test script for 迅雷 (Xunlei/Thunder) integration.
Demonstrates how Chinese users can use 迅雷 to accelerate dataset downloads.
"""

import sys
import os
sys.path.append('.')

from neuroexapt.utils.xunlei_downloader import XunleiDatasetDownloader, XunleiDownloader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_xunlei_detection():
    """Test 迅雷 detection."""
    print("🔍 Testing 迅雷 detection...")
    
    downloader = XunleiDownloader()
    
    if downloader.is_available:
        print(f"✅ 迅雷 detected at: {downloader.xunlei_path}")
        return True
    else:
        print("❌ 迅雷 not detected")
        print("\n📥 To install 迅雷:")
        print("   • Windows: Download from https://www.xunlei.com/")
        print("   • macOS: Download from Mac App Store or https://www.xunlei.com/")
        print("   • Linux: Download from https://www.xunlei.com/")
        return False

def test_dataset_downloader():
    """Test dataset downloader with 迅雷."""
    print("\n🚀 Testing 迅雷 dataset downloader...")
    
    downloader = XunleiDatasetDownloader(data_dir="./xunlei_data")
    
    # Show status
    print("\n📊 Current dataset status:")
    status = downloader.get_status()
    for name, info in status['datasets'].items():
        if info['downloaded']:
            if info['complete']:
                print(f"   ✅ {name}: Complete ({info['size'] / (1024*1024):.1f}MB)")
            else:
                print(f"   ⏳ {name}: Partial ({info['progress']:.1f}%)")
        else:
            print(f"   ❌ {name}: Not downloaded")
    
    return downloader

def demo_cifar10_download(downloader):
    """Demo CIFAR-10 download using 迅雷."""
    print("\n🎯 Demo: Downloading CIFAR-10 with 迅雷")
    print("=" * 50)
    
    # Show instructions
    instructions = downloader.create_download_instructions('cifar10')
    print(instructions)
    
    # Ask user if they want to proceed
    response = input("\n🤔 Do you want to start the download? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\n🚀 Starting 迅雷 download...")
        success = downloader.download_dataset('cifar10', wait_for_completion=False)
        
        if success:
            print("✅ Download started successfully!")
            print("📋 Please check 迅雷 for download progress")
            print("💡 Tip: 迅雷 can significantly speed up downloads in China")
        else:
            print("❌ Failed to start download")
            print("💡 Try manual download method:")
            print(f"   1. Open 迅雷")
            print(f"   2. Copy URL: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
            print(f"   3. Paste into 迅雷 and start download")
    else:
        print("⏸️ Download skipped")

def show_manual_instructions():
    """Show manual download instructions."""
    print("\n" + "=" * 60)
    print("📋 Manual 迅雷 Download Instructions")
    print("=" * 60)
    
    datasets = {
        'CIFAR-10': {
            'url': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'size': '162MB',
            'description': 'CIFAR-10 image classification dataset'
        },
        'CIFAR-100': {
            'url': 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
            'size': '161MB',
            'description': 'CIFAR-100 image classification dataset'
        }
    }
    
    for name, info in datasets.items():
        print(f"\n📁 {name}:")
        print(f"   Description: {info['description']}")
        print(f"   Size: {info['size']}")
        print(f"   URL: {info['url']}")
        print(f"   Steps:")
        print(f"     1. Open 迅雷")
        print(f"     2. Copy the URL above")
        print(f"     3. Paste into 迅雷 download dialog")
        print(f"     4. Set save path to: ./data")
        print(f"     5. Start download")
    
    print("\n💡 迅雷 Tips for Chinese Users:")
    print("   • 迅雷 uses P2P technology for faster downloads")
    print("   • Enable '迅雷加速' for maximum speed")
    print("   • Consider 迅雷 VIP for even faster speeds")
    print("   • 迅雷 can resume interrupted downloads")
    print("   • Use 迅雷's batch download feature for multiple files")

def main():
    """Main demo function."""
    print("=" * 60)
    print("🇨🇳 迅雷 (Xunlei/Thunder) Integration Demo")
    print("=" * 60)
    
    # Test 迅雷 detection
    xunlei_available = test_xunlei_detection()
    
    if xunlei_available:
        # Test dataset downloader
        downloader = test_dataset_downloader()
        
        # Demo download
        demo_cifar10_download(downloader)
    else:
        print("\n⚠️ 迅雷 not available, showing manual instructions...")
        show_manual_instructions()
    
    # Show final tips
    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. Install 迅雷 if not already installed")
    print("   2. Use 迅雷 to download datasets")
    print("   3. Place downloaded files in ./data directory")
    print("   4. Use with Neuro Exapt framework")
    print("\n🌐 Download 迅雷: https://www.xunlei.com/")

if __name__ == "__main__":
    main() 