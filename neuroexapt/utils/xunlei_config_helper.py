#!/usr/bin/env python3
"""
"""
\defgroup group_xunlei_config_helper Xunlei Config Helper
\ingroup core
Xunlei Config Helper module for NeuroExapt framework.
"""


迅雷配置助手 - 帮助用户配置迅雷的默认下载路径
"""

import os
import json
import winreg
import platform
from pathlib import Path
from typing import Optional, Dict, List

class XunleiConfigHelper:
    """迅雷配置助手，帮助用户设置默认下载路径"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.config_paths = self._get_config_paths()
    
    def _get_config_paths(self) -> Dict[str, str]:
        """获取迅雷配置文件路径"""
        paths = {}
        
        if self.system == "windows":
            # 迅雷配置文件路径
            username = os.getenv('USERNAME', '')
            possible_paths = [
                f"C:\\Users\\{username}\\AppData\\Roaming\\Thunder Network\\Thunder\\Profiles\\config.ini",
                f"C:\\Users\\{username}\\AppData\\Local\\Thunder Network\\Thunder\\Profiles\\config.ini",
                "C:\\Program Files (x86)\\Thunder Network\\Thunder\\Profiles\\config.ini",
                "C:\\Program Files\\Thunder Network\\Thunder\\Profiles\\config.ini",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    paths['config_ini'] = path
                    break
            
            # 注册表路径
            paths['registry'] = r"SOFTWARE\Thunder Network\Thunder"
            
        return paths
    
    def get_current_download_path(self) -> Optional[str]:
        """获取迅雷当前的默认下载路径"""
        if self.system != "windows":
            return None
            
        try:
            # 尝试从注册表读取
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.config_paths['registry']) as key:
                try:
                    download_path = winreg.QueryValueEx(key, "DefaultDownloadPath")[0]
                    return download_path
                except FileNotFoundError:
                    pass
        except Exception:
            pass
        
        # 尝试从配置文件读取
        if 'config_ini' in self.config_paths:
            try:
                with open(self.config_paths['config_ini'], 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 查找下载路径配置
                    for line in content.split('\n'):
                        if 'DefaultDownloadPath' in line or 'DownloadPath' in line:
                            if '=' in line:
                                path = line.split('=')[1].strip()
                                if path and os.path.exists(path):
                                    return path
            except Exception:
                pass
        
        return None
    
    def set_download_path(self, new_path: str) -> bool:
        """设置迅雷的默认下载路径"""
        if self.system != "windows":
            return False
            
        try:
            # 确保路径存在
            os.makedirs(new_path, exist_ok=True)
            
            # 尝试通过注册表设置
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.config_paths['registry'], 0, winreg.KEY_WRITE) as key:
                    winreg.SetValueEx(key, "DefaultDownloadPath", 0, winreg.REG_SZ, new_path)
                    print(f"✅ 已通过注册表设置下载路径: {new_path}")
                    return True
            except Exception as e:
                print(f"⚠️ 注册表设置失败: {e}")
            
            # 尝试通过配置文件设置
            if 'config_ini' in self.config_paths:
                try:
                    with open(self.config_paths['config_ini'], 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 替换或添加下载路径配置
                    lines = content.split('\n')
                    updated = False
                    
                    for i, line in enumerate(lines):
                        if 'DefaultDownloadPath' in line or 'DownloadPath' in line:
                            lines[i] = f"DefaultDownloadPath={new_path}"
                            updated = True
                            break
                    
                    if not updated:
                        lines.append(f"DefaultDownloadPath={new_path}")
                    
                    with open(self.config_paths['config_ini'], 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))
                    
                    print(f"✅ 已通过配置文件设置下载路径: {new_path}")
                    return True
                    
                except Exception as e:
                    print(f"⚠️ 配置文件设置失败: {e}")
            
            return False
            
        except Exception as e:
            print(f"❌ 设置下载路径失败: {e}")
            return False
    
    def create_download_guide(self, target_path: str) -> str:
        """创建迅雷下载配置指南"""
        guide = f"""
🚀 迅雷下载路径配置指南
{'='*50}

📁 目标下载路径: {target_path}

🔧 配置方法 (选择其中一种):

方法1: 自动配置 (推荐)
- 运行配置助手: python -m neuroexapt.utils.xunlei_config_helper
- 选择自动设置下载路径

方法2: 手动配置
1. 打开迅雷
2. 点击右上角设置图标 ⚙️
3. 选择"下载设置"
4. 在"默认下载目录"中设置: {target_path}
5. 点击"确定"保存

方法3: 注册表配置
1. 按 Win+R，输入 regedit
2. 导航到: HKEY_CURRENT_USER\\SOFTWARE\\Thunder Network\\Thunder
3. 创建字符串值: DefaultDownloadPath
4. 设置值为: {target_path}

💡 配置完成后，迅雷下载时会自动使用指定路径
💡 如果仍有问题，请手动在迅雷下载窗口中设置保存路径

📋 当前配置状态:
"""
        
        current_path = self.get_current_download_path()
        if current_path:
            guide += f"✅ 当前下载路径: {current_path}\n"
            if current_path == target_path:
                guide += "✅ 路径已正确配置！\n"
            else:
                guide += f"⚠️ 路径不匹配，建议更新为: {target_path}\n"
        else:
            guide += "❌ 未检测到下载路径配置\n"
        
        return guide
    
    def interactive_setup(self):
        """交互式配置迅雷下载路径"""
        print("🚀 迅雷下载路径配置助手")
        print("=" * 50)
        
        # 获取项目数据集路径
        project_path = Path.cwd() / "datasets"
        project_path.mkdir(exist_ok=True)
        
        print(f"📁 建议的下载路径: {project_path}")
        print()
        
        # 显示当前配置
        current_path = self.get_current_download_path()
        if current_path:
            print(f"📋 当前下载路径: {current_path}")
        else:
            print("📋 当前下载路径: 未配置")
        print()
        
        # 询问是否设置
        choice = input("是否将迅雷默认下载路径设置为项目数据集目录? (y/n): ").lower().strip()
        
        if choice in ['y', 'yes', '是']:
            if self.set_download_path(str(project_path)):
                print("✅ 配置成功！")
                print("💡 现在使用迅雷下载时，文件会自动保存到项目目录")
            else:
                print("❌ 自动配置失败，请手动配置")
                print(self.create_download_guide(str(project_path)))
        else:
            print("📋 配置指南:")
            print(self.create_download_guide(str(project_path)))

def main():
    """主函数"""
    helper = XunleiConfigHelper()
    helper.interactive_setup()

if __name__ == "__main__":
    main() 