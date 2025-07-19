#!/usr/bin/env python3
"""
测试ThunderOpenSDK静默下载功能
"""

import os
import sys
import logging
sys.path.append('.')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_thunder_sdk():
    """测试ThunderOpenSDK静默下载"""
    print("🚀 测试ThunderOpenSDK静默下载...")
    
    try:
        from neuroexapt.utils.thunder_sdk_downloader import download_with_thunder_sdk
        
        # 测试下载
        success = download_with_thunder_sdk(
            url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            save_path="./datasets",
            filename="cifar-10-python.tar.gz"
        )
        
        if success:
            print("✅ ThunderOpenSDK静默下载成功！")
            print("💡 文件已直接保存到指定路径，无需用户操作")
        else:
            print("❌ ThunderOpenSDK静默下载失败")
            
    except Exception as e:
        print(f"❌ ThunderOpenSDK测试异常: {e}")

def test_integrated_download():
    """测试集成的下载功能"""
    print("\n🚀 测试集成的下载功能...")
    
    try:
        from neuroexapt.utils.xunlei_downloader import XunleiDownloader
        
        downloader = XunleiDownloader()
        success = downloader.download_with_xunlei(
            url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            save_path="./datasets",
            filename="cifar-10-python.tar.gz"
        )
        
        if success:
            print("✅ 集成下载功能成功！")
        else:
            print("❌ 集成下载功能失败")
            
    except Exception as e:
        print(f"❌ 集成下载测试异常: {e}")

def main():
    print("🧪 ThunderOpenSDK 静默下载测试")
    print("=" * 50)
    
    # 测试ThunderOpenSDK
    test_thunder_sdk()
    
    # 测试集成功能
    test_integrated_download()
    
    print("\n" + "=" * 50)
    print("📋 测试完成")

if __name__ == "__main__":
    main() 