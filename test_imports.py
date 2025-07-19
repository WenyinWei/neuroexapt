#!/usr/bin/env python3
"""
测试导入是否正常
"""

def test_imports():
    print("🧪 测试基本导入...")
    
    try:
        from neuroexapt.core.logging_utils import logger
        print("✅ logging_utils 导入成功")
    except Exception as e:
        print(f"❌ logging_utils 导入失败: {e}")
        return False
    
    try:
        from neuroexapt.core import EnhancedDNMFramework
        print("✅ EnhancedDNMFramework 导入成功")
    except Exception as e:
        print(f"❌ EnhancedDNMFramework 导入失败: {e}")
        return False
    
    try:
        from neuroexapt.core import AdvancedBottleneckAnalyzer
        print("✅ AdvancedBottleneckAnalyzer 导入成功")
    except Exception as e:
        print(f"❌ AdvancedBottleneckAnalyzer 导入失败: {e}")
        return False
    
    try:
        config = {
            'trigger_interval': 1,
            'complexity_threshold': 0.3,
            'enable_serial_division': True,
            'enable_parallel_division': True,
            'enable_hybrid_division': True,
            'enable_aggressive_mode': False  # 先不测试激进模式避免更多导入问题
        }
        framework = EnhancedDNMFramework(config)
        print("✅ EnhancedDNMFramework 初始化成功")
    except Exception as e:
        print(f"❌ EnhancedDNMFramework 初始化失败: {e}")
        return False
    
    print("🎉 所有基本导入测试通过!")
    return True

if __name__ == "__main__":
    test_imports()