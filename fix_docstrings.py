#!/usr/bin/env python3
"""
批量修复文档字符串中的无效转义序列
Fix invalid escape sequences in documentation strings
"""

import os
import re

def fix_file_docstring(file_path):
    """修复单个文件的文档字符串"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含问题模式
        if '\\defgroup' not in content and '\\ingroup' not in content:
            return False
        
        original_content = content
        
        # 修复模式1: """后面直接跟\defgroup
        pattern1 = r'"""(\s*)\\\s*defgroup\s+([^\n]+)\n\\\s*ingroup\s+([^\n]+)\n([^"]*?)"""'
        def replace1(match):
            indent, defgroup, ingroup, description = match.groups()
            return f'"""{indent}defgroup {defgroup}\ningroup {ingroup}\n{description}"""'
        content = re.sub(pattern1, replace1, content, flags=re.MULTILINE | re.DOTALL)
        
        # 修复模式2: 单独的\defgroup和\ingroup行
        content = re.sub(r'^\\defgroup\s+', 'defgroup ', content, flags=re.MULTILINE)
        content = re.sub(r'^\\ingroup\s+', 'ingroup ', content, flags=re.MULTILINE)
        
        # 修复模式3: 字符串内的\defgroup
        content = re.sub(r'"""([^"]*?)\\defgroup\s+([^\n]+)', r'"""\1defgroup \2', content)
        content = re.sub(r'\\ingroup\s+([^\n]+)', r'ingroup \1', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ 修复文件 {file_path} 失败: {e}")
        return False

def fix_all_docstrings():
    """批量修复所有文档字符串"""
    print("🔧 开始批量修复文档字符串...")
    
    # 需要修复的目录
    directories = [
        'neuroexapt/core',
        'neuroexapt/math', 
        'neuroexapt/utils',
        'neuroexapt/cuda_ops',
        'neuroexapt'
    ]
    
    fixed_count = 0
    total_count = 0
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"⚠️ 目录不存在: {directory}")
            continue
            
        print(f"\n📁 处理目录: {directory}")
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    total_count += 1
                    
                    if fix_file_docstring(file_path):
                        print(f"  ✅ 修复: {file_path}")
                        fixed_count += 1
                    else:
                        print(f"  ⚪ 跳过: {file_path}")
    
    print(f"\n🎉 修复完成!")
    print(f"📊 统计: 修复 {fixed_count}/{total_count} 个文件")

if __name__ == "__main__":
    fix_all_docstrings()