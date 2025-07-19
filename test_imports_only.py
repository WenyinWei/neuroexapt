#!/usr/bin/env python3
"""
纯导入测试 - 不依赖torch
Pure import test - no torch dependency
"""

def test_basic_imports():
    """测试基本导入"""
    print("🧪 测试基本导入...")
    
    try:
        from neuroexapt.core.logging_utils import logger
        print("✅ logging_utils 导入成功")
    except Exception as e:
        print(f"❌ logging_utils 导入失败: {e}")
        return False
    
    try:
        from neuroexapt.core.advanced_morphogenesis import AdvancedBottleneckAnalyzer
        print("✅ AdvancedBottleneckAnalyzer 导入成功")
    except Exception as e:
        print(f"❌ AdvancedBottleneckAnalyzer 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from neuroexapt.core import EnhancedDNMFramework
        print("✅ EnhancedDNMFramework 导入成功")
    except Exception as e:
        print(f"❌ EnhancedDNMFramework 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("🎉 基本导入测试通过!")
    return True

def test_docstring_fixes():
    """测试文档字符串修复是否成功"""
    print("\n📄 测试文档字符串修复...")
    
    # 检查几个关键文件
    files_to_check = [
        'neuroexapt/core/device_manager.py',
        'neuroexapt/core/fast_operations.py',
        'neuroexapt/core/enhanced_dnm_framework.py'
    ]
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if '\\defgroup' in content:
                print(f"❌ {file_path} 仍有未修复的转义序列")
                return False
            else:
                print(f"✅ {file_path} 文档字符串已修复")
        except Exception as e:
            print(f"❌ 检查 {file_path} 失败: {e}")
            return False
    
    print("🎉 文档字符串修复验证通过!")
    return True

if __name__ == "__main__":
    print("🚀 开始无依赖导入测试...\n")
    
    success1 = test_docstring_fixes()
    success2 = test_basic_imports()
    
    if success1 and success2:
        print("\n🎉 所有测试通过!")
        print("✅ 语法错误已修复，可以正常导入!")
        print("\n💡 现在可以运行:")
        print("   python examples/intelligent_dnm_demo.py")
        print("   python examples/advanced_dnm_demo.py")
    else:
        print("\n❌ 测试失败，需要进一步调试")