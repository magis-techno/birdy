#!/usr/bin/env python3
"""
清理从网页复制的part文件并恢复原始代码
"""

import os
import glob
import re
import subprocess

def clean_file(input_file, output_file):
    """清理文件中的多余字符和空行"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 移除非标准Unicode字符（如聽）
        content = re.sub(r'[^\x00-\x7F\u4e00-\u9fff\n\r\t ]', '', content)
        
        # 替换全角空格为半角空格
        content = content.replace('\u3000', ' ')
        
        # 清理多余的空行，但保留traceback结构需要的空行
        lines = content.splitlines()
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            line = line.rstrip()  # 移除行尾空白
            
            if not line:  # 空行
                if not prev_empty:  # 避免连续空行
                    cleaned_lines.append('')
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
        
        # 写入清理后的内容
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))
        
        print(f"✅ 清理完成: {input_file} -> {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ 清理失败 {input_file}: {e}")
        return False

def main():
    # 创建清理后的文件目录
    cleaned_dir = "encrypt_cleaned"
    os.makedirs(cleaned_dir, exist_ok=True)
    
    # 查找所有part文件
    part_files = sorted(glob.glob("encrypt/part*.txt"))
    
    if not part_files:
        print("❌ 未找到part文件")
        return
    
    print(f"📁 找到 {len(part_files)} 个part文件")
    
    # 清理所有文件
    cleaned_files = []
    for part_file in part_files:
        filename = os.path.basename(part_file)
        cleaned_file = os.path.join(cleaned_dir, filename)
        
        if clean_file(part_file, cleaned_file):
            cleaned_files.append(cleaned_file)
    
    if len(cleaned_files) != len(part_files):
        print("⚠️ 部分文件清理失败")
        return
    
    print(f"\n🔄 正在恢复文件...")
    
    # 构建恢复命令
    cmd = ["python", "disguise.py", "decrypt"] + cleaned_files + ["--output", "restored_original.py"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ 恢复成功! 文件保存为: restored_original.py")
        
        # 验证语法
        print("\n🔍 验证语法...")
        syntax_check = subprocess.run(["python", "-m", "py_compile", "restored_original.py"], 
                                     capture_output=True, text=True)
        
        if syntax_check.returncode == 0:
            print("✅ 语法检查通过")
        else:
            print("⚠️ 语法检查失败:")
            print(syntax_check.stderr)
            print("可能需要手动修复一些代码")
        
        # 显示文件信息
        if os.path.exists("restored_original.py"):
            size = os.path.getsize("restored_original.py")
            with open("restored_original.py", 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
            print(f"\n📄 恢复的文件信息:")
            print(f"   文件大小: {size} 字节")
            print(f"   代码行数: {lines} 行")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 恢复失败: {e}")
        print(f"错误输出: {e.stderr}")

if __name__ == "__main__":
    main()
