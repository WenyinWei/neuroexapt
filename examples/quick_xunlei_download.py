#!/usr/bin/env python3
"""
快速迅雷下载示例 - 自动指定文件路径和文件名
Quick 迅雷 Download Example - Automatic Path and Filename Specification
"""

import sys
import os
sys.path.append('.')

from neuroexapt.utils.xunlei_downloader import XunleiDatasetDownloader, XunleiDownloader
from neuroexapt.utils.dataset_loader import AdvancedDatasetLoader
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_download_example():
    """快速下载示例 - 无需手动设置路径"""
    print("=" * 60)
    print("🚀 快速迅雷下载示例")
    print("🚀 Quick 迅雷 Download Example")
    print("=" * 60)
    
    # 检测迅雷
    xunlei = XunleiDownloader()
    if not xunlei.is_available:
        print("❌ 未检测到迅雷，请先安装")
        print("❌ 迅雷 not detected, please install first")
        return
    
    print(f"✅ 迅雷已检测到: {xunlei.xunlei_path}")
    print()
    
    # 创建下载器，指定数据目录
    downloader = XunleiDatasetDownloader(data_dir="./datasets")
    
    # 显示当前状态
    print("📊 当前数据集状态:")
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
    
    # 下载CIFAR-10 - 自动指定路径和文件名
    print("🎯 开始下载 CIFAR-10 数据集...")
    print("💡 文件将自动保存到: ./datasets/cifar-10-python.tar.gz")
    
    success = downloader.download_dataset('cifar10', wait_for_completion=False)
    
    if success:
        print("✅ 下载已启动！")
        print("📋 迅雷会自动处理以下设置:")
        print("   • 下载URL: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        print("   • 保存路径: ./datasets/")
        print("   • 文件名: cifar-10-python.tar.gz")
        print("   • 文件大小: 162MB")
        print("   • 如未自动下载，请手动双击 .thunder 文件")
        print()
        print("🎉 无需手动设置，直接开始下载！")
    else:
        print("❌ 下载启动失败")
    
    print()
    print("💡 提示:")
    print("   • 迅雷会自动创建目录")
    print("   • 文件名已预设好")
    print("   • 支持断点续传")
    print("   • 下载完成后可直接使用")

def batch_download_example():
    """批量下载示例"""
    print("\n" + "=" * 60)
    print("📦 批量下载示例")
    print("📦 Batch Download Example")
    print("=" * 60)
    
    downloader = XunleiDatasetDownloader(data_dir="./datasets")
    
    datasets = ['cifar10', 'cifar100']
    
    for dataset in datasets:
        print(f"\n🚀 下载 {dataset.upper()}...")
        print(f"📁 保存路径: ./datasets/")
        
        success = downloader.download_dataset(dataset, wait_for_completion=False)
        
        if success:
            print(f"✅ {dataset} 下载已启动")
        else:
            print(f"❌ {dataset} 下载失败")
    
    print("\n🎉 所有下载任务已启动！")
    print("💡 迅雷会并行处理多个下载任务")

def custom_path_example():
    """自定义路径示例"""
    print("\n" + "=" * 60)
    print("🔧 自定义路径示例")
    print("🔧 Custom Path Example")
    print("=" * 60)
    
    # 自定义数据目录
    custom_dir = "./datasets"
    downloader = XunleiDatasetDownloader(data_dir=custom_dir)
    
    print(f"📁 数据目录: {custom_dir}")
    print(f"📄 CIFAR-10 将保存为: {custom_dir}/cifar-10-python.tar.gz")
    
    success = downloader.download_dataset('cifar10', wait_for_completion=False)
    
    if success:
        print("✅ 下载已启动！")
        print(f"💡 文件将保存到: {custom_dir}/cifar-10-python.tar.gz")
        print("   • 如未自动下载，请手动双击 .thunder 文件")
    else:
        print("❌ 下载失败")

def main():
    """主函数"""
    print("🇨🇳 欢迎使用快速迅雷下载！")
    print("🇨🇳 Welcome to Quick 迅雷 Download!")
    print()
    print("✨ 特性:")
    print("   • 自动检测迅雷")
    print("   • 自动指定保存路径")
    print("   • 自动设置文件名")
    print("   • 无需手动配置")
    print()
    
    # 快速下载示例
    quick_download_example()
    
    # 批量下载示例
    batch_download_example()
    
    # 自定义路径示例
    custom_path_example()
    
    print("\n" + "=" * 60)
    print("🎉 示例完成！")
    print("🎉 Examples completed!")
    print("=" * 60)
    print()
    print("💡 使用建议:")
    print("   • 确保迅雷已安装并运行")
    print("   • 检查网络连接")
    print("   • 确保有足够的磁盘空间")
    print("   • 下载完成后验证文件完整性")
    print()
    print("🌐 迅雷官网: https://www.xunlei.com/")

if __name__ == "__main__":
    main() 