#!/usr/bin/env python3
"""
自动恢复加密文件的脚本
用法: python auto_recover.py <加密目录> <输出文件名>
"""

import os
import sys
import subprocess
import glob

def auto_recover(encrypted_dir, output_file):
    """自动恢复加密目录中的part文件"""
    
    # 查找所有part文件
    part_pattern = os.path.join(encrypted_dir, "part*.txt")
    part_files = sorted(glob.glob(part_pattern))
    
    if not part_files:
        print(f"❌ 在目录 {encrypted_dir} 中未找到part文件")
        return False
    
    print(f"📁 找到 {len(part_files)} 个part文件:")
    for f in part_files:
        print(f"   - {os.path.basename(f)}")
    
    # 构建恢复命令
    cmd = ["python", "disguise.py", "decrypt"] + part_files + ["--output", output_file]
    
    print(f"\n🔄 正在恢复文件...")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        # 执行恢复命令
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ 恢复成功! 文件保存为: {output_file}")
        
        # 验证语法
        print(f"\n🔍 验证语法...")
        syntax_check = subprocess.run(["python", "-m", "py_compile", output_file], 
                                     capture_output=True, text=True)
        
        if syntax_check.returncode == 0:
            print("✅ 语法检查通过")
        else:
            print("⚠️  语法检查失败:")
            print(syntax_check.stderr)
            print("建议手动检查并修复恢复的文件")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 恢复失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def main():
    if len(sys.argv) != 3:
        print("用法: python auto_recover.py <加密目录> <输出文件名>")
        print("示例: python auto_recover.py encrypted_parts recovered_code.py")
        sys.exit(1)
    
    encrypted_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.isdir(encrypted_dir):
        print(f"❌ 目录不存在: {encrypted_dir}")
        sys.exit(1)
    
    success = auto_recover(encrypted_dir, output_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
