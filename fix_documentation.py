#!/usr/bin/env python3
"""
Script to fix incorrect documentation strings in NeuroExapt codebase.
Fixes patterns where defgroup statements are outside of triple quotes.
"""

import os
import re
import glob

def fix_file_documentation(filepath):
    """Fix documentation strings in a single file."""
    print(f"Processing: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: """<newline>"""<newline>defgroup... (broken pattern in fast_operations.py)
    # This should become: """<newline>defgroup...<newline>"""
    pattern1 = re.compile(r'^("""\s*\n)("""\s*\n)(defgroup[^"]+?)("""\s*\n)', re.MULTILINE | re.DOTALL)
    content = pattern1.sub(r'"""\n\3"""\n\n', content)
    
    # Pattern 2: Find standalone defgroup blocks not enclosed in quotes
    # Look for lines starting with 'defgroup' that are not inside triple quotes
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a standalone defgroup line (not inside quotes)
        if line.strip().startswith('defgroup') and i > 0:
            # Look backwards to see if we're after empty triple quotes
            prev_line = lines[i-1].strip() if i > 0 else ""
            prev_prev_line = lines[i-2].strip() if i > 1 else ""
            
            # Check if we have the pattern: """ followed by """ followed by defgroup
            if prev_line == '"""' and prev_prev_line == '"""':
                # Remove the previous empty triple quotes
                fixed_lines.pop()  # Remove the empty """
                fixed_lines.pop()  # Remove the opening """
                
                # Start collecting the defgroup block
                doc_block = []
                j = i
                while j < len(lines):
                    current_line = lines[j]
                    # Stop if we hit a line that's clearly code (import, class, def, etc.)
                    if (current_line.strip().startswith(('import ', 'from ', 'class ', 'def ', 'if __name__')) or
                        (current_line.strip() and not current_line.strip().startswith(('defgroup', 'ingroup', '@', '#')) and 
                         not any(word in current_line.lower() for word in ['module', 'framework', 'optimization', '优化', '模块']))):
                        break
                    if current_line.strip() == '"""':
                        j += 1
                        break
                    doc_block.append(current_line)
                    j += 1
                
                # Create properly formatted docstring
                if doc_block:
                    fixed_lines.append('"""')
                    fixed_lines.extend(doc_block)
                    fixed_lines.append('"""')
                    fixed_lines.append('')  # Add blank line after docstring
                
                i = j
                continue
        
        fixed_lines.append(line)
        i += 1
    
    content = '\n'.join(fixed_lines)
    
    # Clean up any remaining issues
    # Remove any remaining standalone """ that might be left
    content = re.sub(r'^\s*"""\s*\n^\s*"""\s*\n', '"""', content, flags=re.MULTILINE)
    
    # If content changed, write it back
    if content != original_content:
        print(f"  -> Fixed documentation in {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    else:
        print(f"  -> No changes needed in {filepath}")
        return False

def main():
    """Main function to fix all Python files."""
    print("Fixing documentation strings in NeuroExapt codebase...")
    
    # Find all Python files with defgroup
    python_files = []
    for root, dirs, files in os.walk('neuroexapt'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                # Check if file contains defgroup
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        if 'defgroup' in f.read():
                            python_files.append(filepath)
                except Exception as e:
                    print(f"Warning: Could not read {filepath}: {e}")
    
    print(f"Found {len(python_files)} files with defgroup statements")
    
    fixed_count = 0
    for filepath in python_files:
        try:
            if fix_file_documentation(filepath):
                fixed_count += 1
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    print(f"\nCompleted! Fixed {fixed_count} files out of {len(python_files)} files.")

if __name__ == "__main__":
    main()