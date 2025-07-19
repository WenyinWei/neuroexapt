#!/usr/bin/env python3
"""
简单的语法检查脚本，验证代码没有基本错误
"""

import sys
import os
import ast
import traceback

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_syntax(file_path):
    """检查Python文件的语法"""
    print(f"🔧 Checking syntax: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # 解析AST
        ast.parse(source)
        print(f"✅ Syntax OK: {file_path}")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax Error in {file_path}:")
        print(f"   Line {e.lineno}: {e.text}")
        print(f"   {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Error checking {file_path}: {e}")
        return False

def main():
    """主函数"""
    print("🚀 ASO-SE Code Syntax Check")
    print("=" * 50)
    
    # 要检查的文件列表
    files_to_check = [
        "neuroexapt/core/fast_operations.py",
        "neuroexapt/math/fast_math.py", 
        "examples/aso_se_classification_optimized.py"
    ]
    
    all_good = True
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            if not check_syntax(file_path):
                all_good = False
        else:
            print(f"❌ File not found: {file_path}")
            all_good = False
    
    if all_good:
        print("\n🎉 All syntax checks passed!")
        print("\n💡 Next step: Install PyTorch and run:")
        print("   python examples/aso_se_classification_optimized.py")
    else:
        print("\n❌ Some syntax errors found. Please fix them first.")

if __name__ == "__main__":
    main()