#!/usr/bin/env python3
"""
Complete demo for Chinese users showing all download options including 迅雷 integration.
This script demonstrates the full range of download capabilities optimized for Chinese users.
"""

import sys
import os
sys.path.append('.')

from neuroexapt.utils.dataset_loader import AdvancedDatasetLoader
from neuroexapt.utils.xunlei_downloader import XunleiDatasetDownloader, XunleiDownloader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_welcome():
    """Show welcome message for Chinese users."""
    print("=" * 70)
    print("🇨🇳 欢迎使用 Neuro Exapt - 中国用户优化版")
    print("🇨🇳 Welcome to Neuro Exapt - Optimized for Chinese Users")
    print("=" * 70)
    print()
    print("🚀 本演示将展示专为中国用户优化的数据集下载功能")
    print("🚀 This demo showcases dataset download features optimized for Chinese users")
    print()

def test_download_options():
    """Test all available download options."""
    print("🔍 检测下载选项...")
    print("🔍 Detecting download options...")
    print()
    
    # Test 迅雷 availability
    xunlei_downloader = XunleiDownloader()
    xunlei_available = xunlei_downloader.is_available
    
    if xunlei_available:
        print(f"✅ 迅雷已检测到: {xunlei_downloader.xunlei_path}")
        print(f"✅ 迅雷 detected at: {xunlei_downloader.xunlei_path}")
    else:
        print("❌ 未检测到迅雷，建议安装以获得最佳下载速度")
        print("❌ 迅雷 not detected, recommended to install for best download speed")
    
    print()
    
    # Test advanced dataset loader
    print("🔧 初始化高级数据集加载器...")
    print("🔧 Initializing advanced dataset loader...")
    
    loader = AdvancedDatasetLoader(
        cache_dir="./chinese_cache",
        download_dir="./chinese_data",
        use_p2p=True,
        use_xunlei=True,
        max_retries=3
    )
    
    print("✅ 高级数据集加载器初始化完成")
    print("✅ Advanced dataset loader initialized")
    print()
    
    return loader, xunlei_available

def show_download_methods():
    """Show available download methods."""
    print("📋 可用的下载方法:")
    print("📋 Available download methods:")
    print()
    
    methods = [
        {
            "name": "迅雷下载 (Xunlei Download)",
            "description": "使用迅雷P2P技术加速下载",
            "speed": "1-10MB/s",
            "recommended": "⭐⭐⭐⭐⭐"
        },
        {
            "name": "P2P加速 (P2P Acceleration)",
            "description": "使用内置P2P技术",
            "speed": "500KB-2MB/s",
            "recommended": "⭐⭐⭐⭐"
        },
        {
            "name": "分块下载 (Chunked Download)",
            "description": "支持断点续传的稳健下载",
            "speed": "100KB-1MB/s",
            "recommended": "⭐⭐⭐"
        },
        {
            "name": "镜像站点 (Mirror Sites)",
            "description": "使用中国镜像站点",
            "speed": "500KB-5MB/s",
            "recommended": "⭐⭐⭐⭐"
        }
    ]
    
    for i, method in enumerate(methods, 1):
        print(f"{i}. {method['name']}")
        print(f"   {method['description']}")
        print(f"   速度/Speed: {method['speed']}")
        print(f"   推荐度/Recommendation: {method['recommended']}")
        print()

def demo_xunlei_download():
    """Demo 迅雷 download functionality."""
    print("🎯 演示: 迅雷下载功能")
    print("🎯 Demo: 迅雷 Download Functionality")
    print("=" * 50)
    
    downloader = XunleiDatasetDownloader(data_dir="./xunlei_demo")
    
    # Show current status
    print("\n📊 当前数据集状态:")
    print("📊 Current dataset status:")
    status = downloader.get_status()
    for name, info in status['datasets'].items():
        if info['downloaded']:
            if info['complete']:
                print(f"   ✅ {name}: 已完成 ({info['size'] / (1024*1024):.1f}MB)")
            else:
                print(f"   ⏳ {name}: 部分下载 ({info['progress']:.1f}%)")
        else:
            print(f"   ❌ {name}: 未下载")
    
    print()
    
    # Show download instructions
    print("📋 CIFAR-10 迅雷下载说明:")
    print("📋 CIFAR-10 迅雷 Download Instructions:")
    instructions = downloader.create_download_instructions('cifar10')
    print(instructions)
    
    return downloader

def demo_advanced_loader(loader):
    """Demo advanced dataset loader."""
    print("\n🎯 演示: 高级数据集加载器")
    print("🎯 Demo: Advanced Dataset Loader")
    print("=" * 50)
    
    print("\n🔧 配置信息:")
    print("🔧 Configuration:")
    print(f"   缓存目录/Cache dir: {loader.cache.cache_dir}")
    print(f"   下载目录/Download dir: {loader.download_dir}")
    print(f"   P2P加速/P2P enabled: {loader.p2p_downloader is not None}")
    print(f"   迅雷集成/Xunlei enabled: {loader.xunlei_downloader is not None}")
    print(f"   最大重试/Max retries: {loader.max_retries}")
    
    print("\n📊 缓存信息:")
    print("📊 Cache information:")
    cache_info = loader.get_cache_info()
    print(f"   总大小/Total size: {cache_info['total_size'] / (1024*1024):.1f}MB")
    for dataset, info in cache_info['datasets'].items():
        print(f"   {dataset}: {info['size'] / (1024*1024):.1f}MB ({info['files']} files)")

def show_performance_comparison():
    """Show performance comparison for different download methods."""
    print("\n📈 下载性能对比 (中国用户)")
    print("📈 Download Performance Comparison (Chinese Users)")
    print("=" * 60)
    
    comparison = [
        {
            "method": "直接下载 (Direct Download)",
            "speed": "10-50KB/s",
            "stability": "不稳定/Unstable",
            "recommendation": "不推荐/Not recommended"
        },
        {
            "method": "镜像站点 (Mirror Sites)",
            "speed": "500KB-5MB/s",
            "stability": "较稳定/Moderate",
            "recommendation": "推荐/Recommended"
        },
        {
            "method": "P2P加速 (P2P Acceleration)",
            "speed": "500KB-2MB/s",
            "stability": "稳定/Stable",
            "recommendation": "推荐/Recommended"
        },
        {
            "method": "迅雷下载 (Xunlei Download)",
            "speed": "1-10MB/s",
            "stability": "非常稳定/Very stable",
            "recommendation": "强烈推荐/Highly recommended"
        },
        {
            "method": "迅雷VIP (Xunlei VIP)",
            "speed": "10-50MB/s",
            "stability": "极其稳定/Extremely stable",
            "recommendation": "最佳选择/Best choice"
        }
    ]
    
    print(f"{'方法/Method':<25} {'速度/Speed':<15} {'稳定性/Stability':<20} {'推荐度/Recommendation'}")
    print("-" * 80)
    
    for method in comparison:
        print(f"{method['method']:<25} {method['speed']:<15} {method['stability']:<20} {method['recommendation']}")

def show_usage_examples():
    """Show usage examples for Chinese users."""
    print("\n💡 使用示例 (Usage Examples)")
    print("=" * 50)
    
    examples = [
        {
            "title": "基本使用 (Basic Usage)",
            "code": """
from neuroexapt.utils.dataset_loader import AdvancedDatasetLoader

# 自动使用最佳下载方式 (包括迅雷)
loader = AdvancedDatasetLoader(use_xunlei=True)
train_loader, test_loader = loader.get_cifar10_dataloaders()
""",
            "description": "自动选择最佳下载方式，包括迅雷集成"
        },
        {
            "title": "仅使用迅雷 (Xunlei Only)",
            "code": """
from neuroexapt.utils.xunlei_downloader import XunleiDatasetDownloader

downloader = XunleiDatasetDownloader()
downloader.download_dataset('cifar10')
""",
            "description": "专门使用迅雷下载数据集"
        },
        {
            "title": "自定义配置 (Custom Configuration)",
            "code": """
loader = AdvancedDatasetLoader(
    cache_dir="./my_cache",
    download_dir="./my_data",
    use_p2p=True,
    use_xunlei=True,
    max_retries=5
)
""",
            "description": "自定义缓存和下载目录"
        }
    ]
    
    for example in examples:
        print(f"\n📝 {example['title']}")
        print(f"   描述: {example['description']}")
        print(f"   代码:")
        print(example['code'])

def show_troubleshooting():
    """Show troubleshooting tips for Chinese users."""
    print("\n🔧 故障排除 (Troubleshooting)")
    print("=" * 50)
    
    issues = [
        {
            "problem": "下载速度很慢",
            "solution": "启用迅雷下载，检查网络连接，使用有线网络"
        },
        {
            "problem": "迅雷未检测到",
            "solution": "安装迅雷，或手动指定迅雷路径"
        },
        {
            "problem": "下载中断",
            "solution": "使用支持断点续传的下载方式，如迅雷或分块下载"
        },
        {
            "problem": "文件损坏",
            "solution": "验证文件完整性，重新下载损坏的文件"
        },
        {
            "problem": "网络连接问题",
            "solution": "检查防火墙设置，尝试使用VPN，联系网络管理员"
        }
    ]
    
    for issue in issues:
        print(f"❓ {issue['problem']}")
        print(f"   💡 {issue['solution']}")
        print()

def main():
    """Main demo function."""
    show_welcome()
    
    # Test download options
    loader, xunlei_available = test_download_options()
    
    # Show download methods
    show_download_methods()
    
    # Demo 迅雷 functionality
    if xunlei_available:
        xunlei_downloader = demo_xunlei_download()
    else:
        print("⚠️ 跳过迅雷演示 (迅雷未安装)")
        print("⚠️ Skipping 迅雷 demo (迅雷 not installed)")
    
    # Demo advanced loader
    demo_advanced_loader(loader)
    
    # Show performance comparison
    show_performance_comparison()
    
    # Show usage examples
    show_usage_examples()
    
    # Show troubleshooting
    show_troubleshooting()
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 演示完成! (Demo Completed!)")
    print("=" * 70)
    print()
    print("💡 下一步 (Next Steps):")
    print("   1. 安装迅雷以获得最佳下载体验")
    print("      Install 迅雷 for best download experience")
    print("   2. 使用高级数据集加载器下载数据集")
    print("      Use advanced dataset loader to download datasets")
    print("   3. 开始使用 Neuro Exapt 框架")
    print("      Start using the Neuro Exapt framework")
    print()
    print("🌐 相关链接 (Related Links):")
    print("   • 迅雷官网: https://www.xunlei.com/")
    print("   • Neuro Exapt 文档: docs/html/index.html")
    print("   • 中国镜像站点: https://mirrors.tuna.tsinghua.edu.cn/")
    print()
    print("🇨🇳 专为中国用户优化，享受更快的下载速度！")
    print("🇨🇳 Optimized for Chinese users, enjoy faster downloads!")

if __name__ == "__main__":
    main() 