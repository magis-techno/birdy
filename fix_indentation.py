#!/usr/bin/env python3
"""
自动修复Python文件的缩进问题
"""

def fix_indentation(input_file, output_file):
    """修复Python文件的缩进问题"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_class = False
    in_function = False
    base_indent = 0
    
    for i, line in enumerate(lines):
        original_line = line
        stripped = line.strip()
        
        if not stripped:
            fixed_lines.append('\n')
            continue
            
        # 检测类定义
        if stripped.startswith('class '):
            in_class = True
            base_indent = 0
            fixed_lines.append(stripped + '\n')
            continue
            
        # 检测函数/方法定义
        if stripped.startswith('def '):
            if in_class:
                # 类方法
                fixed_lines.append('    ' + stripped + '\n')
                in_function = True
                base_indent = 8  # 类方法体缩进
            else:
                # 独立函数
                fixed_lines.append(stripped + '\n')
                in_function = True
                base_indent = 4  # 函数体缩进
            continue
            
        # 处理docstring
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if in_function:
                fixed_lines.append(' ' * base_indent + stripped + '\n')
            else:
                fixed_lines.append(' ' * 4 + stripped + '\n')
            continue
            
        # 处理注释
        if stripped.startswith('#'):
            if in_function:
                fixed_lines.append(' ' * base_indent + stripped + '\n')
            elif in_class:
                fixed_lines.append(' ' * 4 + stripped + '\n')
            else:
                fixed_lines.append(stripped + '\n')
            continue
            
        # 处理import语句
        if stripped.startswith('import ') or stripped.startswith('from '):
            fixed_lines.append(stripped + '\n')
            continue
            
        # 检测控制流语句
        control_keywords = ['if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except ', 'finally:', 'with ']
        is_control = any(stripped.startswith(kw) for kw in control_keywords)
        
        # 检测return/break/continue等语句
        single_keywords = ['return ', 'break', 'continue', 'pass', 'raise ']
        is_single = any(stripped.startswith(kw) for kw in single_keywords)
        
        # 根据上下文调整缩进
        if in_function:
            if is_control:
                fixed_lines.append(' ' * base_indent + stripped + '\n')
                # 下一行可能需要额外缩进
            elif is_single:
                fixed_lines.append(' ' * base_indent + stripped + '\n')
            else:
                # 普通语句
                if (stripped.startswith('(') or 
                    (i > 0 and lines[i-1].strip().endswith(('(', '[', '{', ',', '=')))):
                    # 继续行，额外缩进
                    fixed_lines.append(' ' * (base_indent + 4) + stripped + '\n')
                else:
                    fixed_lines.append(' ' * base_indent + stripped + '\n')
        elif in_class:
            # 类级别的语句
            fixed_lines.append(' ' * 4 + stripped + '\n')
        else:
            # 模块级别
            fixed_lines.append(stripped + '\n')
    
    # 写入修复后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ 缩进修复完成: {input_file} -> {output_file}")

def main():
    fix_indentation('restored_original.py', 'restored_original_fixed.py')

if __name__ == "__main__":
    main()
