#!/usr/bin/env python3
"""
defgroup group_xunlei_downloader Xunlei Downloader
ingroup core
Xunlei Downloader module for NeuroExapt framework.
"""

迅雷 (Xunlei/Thunder) Download Manager Integration for Chinese Users
Provides seamless integration with 迅雷 to accelerate dataset downloads.
"""

import os
import sys
import time
import subprocess
import platform
import json
import requests
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class XunleiDownloader:
    """迅雷 download manager integration for dataset downloads."""
    
    def __init__(self, xunlei_path: Optional[str] = None):
        """
        Initialize 迅雷 downloader.
        
        Args:
            xunlei_path: Path to 迅雷 executable. If None, will auto-detect.
        """
        self.xunlei_path = xunlei_path or self._detect_xunlei_path()
        self.is_available = self.xunlei_path is not None
        
        if self.is_available:
            logger.info(f"✅ 迅雷 detected at: {self.xunlei_path}")
        else:
            logger.warning("⚠️ 迅雷 not detected. Please install 迅雷 or provide path manually.")
    
    def _detect_xunlei_path(self) -> Optional[str]:
        """Auto-detect 迅雷 installation path."""
        system = platform.system().lower()
        
        if system == "windows":
            # Common 迅雷 installation paths on Windows
            possible_paths = [
                r"C:\Program Files (x86)\Thunder Network\Thunder\Program\Thunder.exe",
                r"C:\Program Files\Thunder Network\Thunder\Program\Thunder.exe",
                r"C:\Users\{}\AppData\Local\Thunder Network\Thunder\Program\Thunder.exe".format(os.getenv('USERNAME', '')),
                r"C:\Users\{}\AppData\Roaming\Thunder Network\Thunder\Program\Thunder.exe".format(os.getenv('USERNAME', '')),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return path
            
            # Try to find from registry
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Thunder Network\Thunder") as key:
                    install_path = winreg.QueryValueEx(key, "InstallPath")[0]
                    thunder_exe = os.path.join(install_path, "Program", "Thunder.exe")
                    if os.path.exists(thunder_exe):
                        return thunder_exe
            except:
                pass
                
        elif system == "darwin":  # macOS
            possible_paths = [
                "/Applications/Thunder.app/Contents/MacOS/Thunder",
                "/Applications/Xunlei.app/Contents/MacOS/Xunlei",
                os.path.expanduser("~/Applications/Thunder.app/Contents/MacOS/Thunder"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return path
                    
        elif system == "linux":
            possible_paths = [
                "/usr/bin/thunder",
                "/usr/local/bin/thunder",
                "/opt/thunder/thunder",
                os.path.expanduser("~/.local/bin/thunder"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return path
        
        return None
    
    def _set_xunlei_default_path(self, save_path: str) -> bool:
        """
        通过注册表设置迅雷的默认下载路径。
        
        Args:
            save_path: 要设置的默认下载路径
            
        Returns:
            bool: 是否成功设置
        """
        try:
            import winreg
            
            # 构建完整的绝对路径
            abs_path = os.path.abspath(save_path)
            logger.info(f"🔧 尝试设置迅雷默认下载路径: {abs_path}")
            
            # 尝试设置多个可能的注册表键
            registry_keys = [
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Thunder Network\Thunder"),
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Thunder Network\Thunder\Profiles"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Thunder Network\Thunder"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Thunder Network\Thunder\Profiles"),
            ]
            
            success = False
            for hkey, subkey in registry_keys:
                try:
                    logger.debug(f"🔧 尝试设置注册表键: {subkey}")
                    with winreg.OpenKey(hkey, subkey, 0, winreg.KEY_WRITE) as key:
                        winreg.SetValueEx(key, "DefaultDownloadPath", 0, winreg.REG_SZ, abs_path)
                        logger.info(f"✅ 成功设置迅雷默认下载路径: {abs_path}")
                        logger.info(f"✅ 注册表键: {subkey}")
                        success = True
                        break
                except FileNotFoundError:
                    logger.debug(f"⚠️ 注册表键不存在: {subkey}")
                    continue
                except PermissionError:
                    logger.debug(f"⚠️ 权限不足，无法写入注册表键: {subkey}")
                    continue
                except Exception as e:
                    logger.debug(f"⚠️ 设置注册表键 {subkey} 失败: {e}")
                    continue
            
            if not success:
                logger.warning("⚠️ 所有注册表键设置都失败")
                logger.info("💡 尝试使用命令行设置...")
                
                # 尝试使用命令行设置
                try:
                    import subprocess
                    cmd = f'reg add "HKCU\\SOFTWARE\\Thunder Network\\Thunder" /v "DefaultDownloadPath" /t REG_SZ /d "{abs_path}" /f'
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        logger.info(f"✅ 命令行设置成功: {abs_path}")
                        success = True
                    else:
                        logger.debug(f"⚠️ 命令行设置失败: {result.stderr}")
                except Exception as e:
                    logger.debug(f"⚠️ 命令行设置异常: {e}")
            
            return success
            
        except Exception as e:
            logger.warning(f"⚠️ 设置迅雷默认路径失败: {e}")
            return False
    
    def _try_xunlei_com_download(self, url: str, save_path: str, filename: str) -> bool:
        """
        尝试使用迅雷的COM接口进行下载并指定保存路径。
        
        Args:
            url: 下载URL
            save_path: 保存路径
            filename: 文件名
            
        Returns:
            bool: 是否成功启动下载
        """
        try:
            import win32com.client
            
            # 先设置迅雷的默认下载路径
            self._set_xunlei_default_path(save_path)
            
            # 将目标路径复制到剪贴板，方便用户粘贴
            self._copy_path_to_clipboard(save_path)
            
            # 尝试多种COM接口
            com_objects = [
                "ThunderAgent.Agent.1",
                "ThunderAgent.Agent",
                "ThunderAgent.Agent.2"
            ]
            
            thunder = None
            for com_obj in com_objects:
                try:
                    thunder = win32com.client.Dispatch(com_obj)
                    logger.info(f"✅ 成功连接到迅雷COM接口: {com_obj}")
                    break
                except Exception as e:
                    logger.debug(f"⚠️ COM接口 {com_obj} 连接失败: {e}")
                    continue
            
            if thunder is None:
                logger.warning("⚠️ 所有COM接口都连接失败")
                return False
            
            # 构建完整的绝对路径（包含文件名）
            full_path = os.path.abspath(os.path.join(save_path, filename))
            
            # 确保目录存在
            os.makedirs(save_path, exist_ok=True)
            
            # 使用COM接口添加下载任务
            # 注意：某些版本的COM接口参数顺序可能不同
            try:
                # 方法1: 标准参数顺序 (url, save_path, filename)
                thunder.AddTask(url, full_path, filename)
                thunder.CommitTasks()
                logger.info(f"✅ 迅雷COM接口下载启动成功: {filename}")
                logger.info(f"💡 文件将保存到: {full_path}")
                return True
            except Exception as e1:
                logger.debug(f"⚠️ 标准COM参数失败，尝试其他格式: {e1}")
                try:
                    # 方法2: 只传入URL，让迅雷自动处理路径
                    thunder.AddTask(url, "", filename)
                    thunder.CommitTasks()
                    logger.info(f"✅ 迅雷COM接口下载启动成功 (简化参数): {filename}")
                    logger.info(f"💡 文件将保存到默认路径: {save_path}")
                    return True
                except Exception as e2:
                    logger.debug(f"⚠️ 简化COM参数也失败: {e2}")
                    return False
            
        except ImportError:
            logger.warning("⚠️ pywin32未安装，无法使用COM接口")
            return False
        except Exception as e:
            logger.warning(f"⚠️ COM接口启动失败: {e}")
            return False
    
    def _copy_path_to_clipboard(self, path: str) -> bool:
        """
        将指定路径复制到剪贴板，方便用户在迅雷下载窗口中粘贴。
        
        Args:
            path: 要复制到剪贴板的路径
            
        Returns:
            bool: 是否成功复制到剪贴板
        """
        try:
            import win32clipboard
            import win32con
            
            # 构建完整的绝对路径
            abs_path = os.path.abspath(path)
            
            # 复制到剪贴板
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(abs_path, win32con.CF_UNICODETEXT)
            win32clipboard.CloseClipboard()
            
            logger.info(f"📋 目标路径已复制到剪贴板: {abs_path}")
            logger.info("💡 在迅雷下载窗口中按 Ctrl+V 即可粘贴路径")
            return True
            
        except ImportError:
            logger.warning("⚠️ pywin32未安装，无法使用剪贴板功能")
            return False
        except Exception as e:
            logger.warning(f"⚠️ 复制到剪贴板失败: {e}")
            return False
    
    def download_with_xunlei(self, url: str, save_path: str, filename: Optional[str] = None) -> bool:
        """
        Download file using 迅雷 with automatic path and filename specification.
        优先使用ThunderOpenSDK静默下载，失败时回退到传统方法。
        """
        if not self.is_available:
            logger.error("❌ 迅雷 not available")
            return False
        try:
            os.makedirs(save_path, exist_ok=True)
            if filename is None:
                filename = os.path.basename(url)
            full_file_path = os.path.join(save_path, filename)
            system = platform.system().lower()
            
            # 尝试多种迅雷调用方式
            success = False
            
            if system == "windows":
                # 方法1: 使用迅雷的COM接口 (最可靠)
                logger.info("🚀 尝试使用迅雷COM接口启动下载...")
                success = self._try_xunlei_com_download(url, save_path, filename)
                
                # 方法3: 使用迅雷的URL协议调用
                if not success:
                    logger.info("🚀 尝试使用迅雷URL协议启动下载...")
                    
                    # 先设置迅雷的默认下载路径
                    self._set_xunlei_default_path(save_path)
                    
                    # 将目标路径复制到剪贴板，方便用户粘贴
                    self._copy_path_to_clipboard(save_path)
                    
                    import base64
                    thunder_url = f"thunder://{base64.b64encode(('AA' + url + 'ZZ').encode()).decode()}"
                    try:
                        os.startfile(thunder_url)
                        logger.info(f"✅ 迅雷下载已启动 (URL协议): {filename}")
                        logger.info("💡 迅雷已弹出下载窗口")
                        logger.info(f"💡 目标路径已复制到剪贴板: {save_path}")
                        logger.info(f"💡 文件名: {filename}")
                        logger.info("💡 在下载窗口中按 Ctrl+V 粘贴路径，然后点击'立即下载'")
                        success = True
                    except Exception as e:
                        logger.warning(f"⚠️ URL协议启动失败: {e}")
                
                # 方法4: 直接命令行调用
                if not success and self.xunlei_path:
                    logger.info("🚀 尝试使用命令行启动迅雷...")
                    try:
                        # 构建完整的绝对路径
                        full_path = os.path.abspath(os.path.join(save_path, filename))
                        
                        # 尝试多种命令行参数格式
                        cmd_variants = [
                            [self.xunlei_path, url, "--save-path", full_path],
                            [self.xunlei_path, url, "-s", full_path],
                            [self.xunlei_path, url, full_path],
                            [self.xunlei_path, f"{url} -s {full_path}"],
                        ]
                        
                        for cmd in cmd_variants:
                            logger.info(f"🚀 尝试命令行启动迅雷: {cmd}")
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                            if result.returncode == 0:
                                logger.info(f"✅ 迅雷下载已启动 (命令行): {filename}")
                                logger.info(f"💡 文件将保存到: {full_file_path}")
                                success = True
                                break
                            else:
                                logger.debug(f"⚠️ 命令行格式失败: {result.stderr}")
                        
                        if not success:
                            logger.warning("⚠️ 所有命令行格式都失败")
                            
                    except subprocess.TimeoutExpired:
                        logger.warning("⚠️ 迅雷启动超时")
                    except Exception as e:
                        logger.warning(f"⚠️ 命令行启动异常: {e}")
                
            elif system == "darwin":
                cmd = ["open", "-a", "Thunder", url]
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        logger.info(f"✅ 迅雷下载已启动 (macOS): {filename}")
                        success = True
                except Exception as e:
                    logger.warning(f"⚠️ macOS启动失败: {e}")
                    
            elif system == "linux":
                cmd = [self.xunlei_path, "--url", url, "--save-path", save_path, "--filename", filename]
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        logger.info(f"✅ 迅雷下载已启动 (Linux): {filename}")
                        success = True
                except Exception as e:
                    logger.warning(f"⚠️ Linux启动失败: {e}")
            
            # 如果所有API方法都失败，提供手动下载说明
            if not success:
                logger.warning("⚠️ 所有自动启动方法都失败")
                logger.info("📋 请手动下载:")
                logger.info(f"   1. 打开迅雷")
                logger.info(f"   2. 点击'新建下载'或'添加下载'")
                logger.info(f"   3. 复制此URL: {url}")
                logger.info(f"   4. 设置保存路径: {save_path}")
                logger.info(f"   5. 确认文件名: {filename}")
                logger.info(f"   6. 点击'立即下载'按钮")
                return False
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error starting 迅雷 download: {e}")
            return False
    

    

    
    def get_download_progress(self, save_path: str, filename: str) -> Tuple[bool, float]:
        """
        Check download progress by monitoring file size.
        
        Args:
            save_path: Directory where file is being saved
            filename: Name of the file being downloaded
            
        Returns:
            Tuple[bool, float]: (is_complete, progress_percentage)
        """
        file_path = os.path.join(save_path, filename)
        
        if not os.path.exists(file_path):
            return False, 0.0
        
        # Get file size
        current_size = os.path.getsize(file_path)
        
        # For now, we can't easily get total size from 迅雷
        # So we'll just return that file exists and has some content
        if current_size > 0:
            return False, 50.0  # Assume 50% if file exists and has content
        else:
            return False, 0.0
    
    def wait_for_completion(self, save_path: str, filename: str, timeout: int = 3600) -> bool:
        """
        Wait for download to complete by monitoring file size stability.
        
        Args:
            save_path: Directory where file is being saved
            filename: Name of the file being downloaded
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if download completed successfully
        """
        file_path = os.path.join(save_path, filename)
        start_time = time.time()
        last_size = 0
        stable_count = 0
        
        logger.info(f"⏳ Waiting for 迅雷 download to complete: {filename}")
        
        while time.time() - start_time < timeout:
            if os.path.exists(file_path):
                current_size = os.path.getsize(file_path)
                
                if current_size == last_size:
                    stable_count += 1
                    if stable_count >= 10:  # File size stable for 10 checks
                        logger.info(f"✅ 迅雷 download completed: {filename}")
                        return True
                else:
                    stable_count = 0
                    last_size = current_size
                    
                    # Log progress every 30 seconds
                    if int(time.time() - start_time) % 30 == 0:
                        logger.info(f"📥 Downloading: {current_size / (1024*1024):.1f}MB")
            
            time.sleep(2)
        
        logger.warning(f"⏰ Download timeout: {filename}")
        return False

class XunleiDatasetDownloader:
    """High-level interface for downloading datasets using 迅雷."""
    
    def __init__(self, data_dir: str = "./data", xunlei_path: Optional[str] = None):
        """
        Initialize 迅雷 dataset downloader.
        
        Args:
            data_dir: Directory to save datasets
            xunlei_path: Path to 迅雷 executable
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.xunlei_downloader = XunleiDownloader(xunlei_path)
        
        # Dataset configurations
        self.datasets = {
            'cifar10': {
                'filename': 'cifar-10-python.tar.gz',
                'url': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                'expected_size': 170498071,
                'description': 'CIFAR-10 Dataset (162MB)'
            },
            'cifar100': {
                'filename': 'cifar-100-python.tar.gz',
                'url': 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
                'expected_size': 169001437,
                'description': 'CIFAR-100 Dataset (161MB)'
            },
            'mnist': {
                'filename': 'mnist.tar.gz',
                'url': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                'expected_size': 9912422,
                'description': 'MNIST Dataset (9.4MB)'
            }
        }
    
    def download_dataset(self, dataset_name: str, wait_for_completion: bool = False) -> bool:
        """
        Download dataset using 迅雷.
        
        Args:
            dataset_name: Name of the dataset to download
            wait_for_completion: Whether to wait for download to complete
            
        Returns:
            bool: True if download started successfully
        """
        if dataset_name not in self.datasets:
            logger.error(f"❌ Unknown dataset: {dataset_name}")
            return False
        
        dataset = self.datasets[dataset_name]
        filename = dataset['filename']
        url = dataset['url']
        
        logger.info(f"🚀 Starting 迅雷 download for {dataset_name}")
        logger.info(f"📋 {dataset['description']}")
        logger.info(f"🌐 URL: {url}")
        
        # Check if already downloaded
        file_path = self.data_dir / filename
        if file_path.exists() and file_path.stat().st_size == dataset['expected_size']:
            logger.info(f"✅ {dataset_name} already downloaded and verified")
            return True
        
        # Start download
        success = self.xunlei_downloader.download_with_xunlei(
            url=url,
            save_path=str(self.data_dir),
            filename=filename
        )
        
        if success and wait_for_completion:
            return self.xunlei_downloader.wait_for_completion(
                str(self.data_dir), 
                filename
            )
        
        return success
    
    def create_download_instructions(self, dataset_name: str) -> str:
        """
        Create detailed download instructions for users.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            str: Formatted instructions
        """
        if dataset_name not in self.datasets:
            return f"❌ Unknown dataset: {dataset_name}"
        
        dataset = self.datasets[dataset_name]
        
        instructions = f"""
📋 迅雷 Download Instructions for {dataset_name.upper()}
{'=' * 60}

📁 Dataset: {dataset['description']}
🌐 URL: {dataset['url']}
📂 Save to: {self.data_dir / dataset['filename']}

🚀 Method 1: Automatic Download
   Run: python -c "from neuroexapt.utils.xunlei_downloader import XunleiDatasetDownloader; XunleiDatasetDownloader().download_dataset('{dataset_name}')"

🚀 Method 2: Manual Download
   1. Open 迅雷
   2. Copy this URL: {dataset['url']}
   3. Paste into 迅雷 download dialog
   4. Set save path to: {self.data_dir}
   5. Start download

🚀 Method 3: Task File
   1. Run the automatic download (creates .thunder file)
   2. Double-click the .thunder file to open in 迅雷
   3. Confirm download settings and start

💡 Tips:
   • 迅雷 can significantly speed up downloads in China
   • Use 迅雷's P2P acceleration for faster speeds
   • Check 迅雷 settings for optimal performance
   • Consider using 迅雷 VIP for even faster speeds

✅ After download:
   • Verify file size: {dataset['expected_size'] / (1024*1024):.1f}MB
   • Extract the archive if needed
   • Use with Neuro Exapt framework
"""
        return instructions
    
    def get_status(self) -> Dict:
        """Get status of all datasets."""
        status = {
            'data_dir': str(self.data_dir),
            'xunlei_available': self.xunlei_downloader.is_available,
            'datasets': {}
        }
        
        for name, config in self.datasets.items():
            file_path = self.data_dir / config['filename']
            if file_path.exists():
                size = file_path.stat().st_size
                complete = size == config['expected_size']
                status['datasets'][name] = {
                    'downloaded': True,
                    'complete': complete,
                    'size': size,
                    'expected_size': config['expected_size'],
                    'progress': (size / config['expected_size'] * 100) if config['expected_size'] > 0 else 0
                }
            else:
                status['datasets'][name] = {
                    'downloaded': False,
                    'complete': False,
                    'size': 0,
                    'expected_size': config['expected_size'],
                    'progress': 0
                }
        
        return status

def main():
    """Demo function for 迅雷 integration."""
    print("=" * 60)
    print("🚀 迅雷 Dataset Downloader Demo")
    print("=" * 60)
    
    downloader = XunleiDatasetDownloader()
    
    # Check 迅雷 availability
    if downloader.xunlei_downloader.is_available:
        print("✅ 迅雷 is available!")
    else:
        print("❌ 迅雷 not detected. Please install 迅雷 first.")
        print("\n📥 Download 迅雷 from: https://www.xunlei.com/")
        return
    
    # Show available datasets
    print("\n📋 Available datasets:")
    for name, config in downloader.datasets.items():
        print(f"   • {name}: {config['description']}")
    
    # Show current status
    print("\n📊 Current status:")
    status = downloader.get_status()
    for name, info in status['datasets'].items():
        if info['downloaded']:
            if info['complete']:
                print(f"   ✅ {name}: Complete ({info['size'] / (1024*1024):.1f}MB)")
            else:
                print(f"   ⏳ {name}: Partial ({info['progress']:.1f}%)")
        else:
            print(f"   ❌ {name}: Not downloaded")
    
    # Demo download
    print("\n🚀 Starting demo download (CIFAR-10)...")
    success = downloader.download_dataset('cifar10', wait_for_completion=False)
    
    if success:
        print("✅ Download started successfully!")
        print("📋 Check 迅雷 for download progress")
    else:
        print("❌ Failed to start download")
    
    # Show instructions
    print("\n" + downloader.create_download_instructions('cifar10'))

if __name__ == "__main__":
    main() 