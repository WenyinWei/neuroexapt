#!/usr/bin/env python3
"""
测试迅雷默认路径设置功能
"""

import os
import sys
sys.path.append('.')

from neuroexapt.utils.xunlei_downloader import XunleiDownloader

def main():
    # 创建下载器
    downloader = XunleiDownloader()
    
    print("🚀 测试迅雷默认路径设置...")
    print(f"✅ 迅雷可用性: {downloader.is_available}")
    
    # 测试设置默认路径
    test_path = "./datasets"
    success = downloader._set_xunlei_default_path(test_path)
    
    if success:
        print(f"✅ 成功设置迅雷默认下载路径: {test_path}")
    else:
        print(f"❌ 设置迅雷默认下载路径失败")
    
    # 测试下载
    print("\n🚀 测试下载功能...")
    download_success = downloader.download_with_xunlei(
        url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
        save_path='./datasets',
        filename='cifar-10-python.tar.gz'
    )
    
    if download_success:
        print("✅ 迅雷下载启动成功！")
        print("💡 请检查迅雷下载窗口中的保存路径是否正确")
    else:
        print("❌ 迅雷下载启动失败")

if __name__ == "__main__":
    main() 