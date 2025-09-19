import argparse
import os
import textwrap
import random
import re

def encrypt_code_as_tracebacks(code: str, parts: int = 6):
    """拆分代码 -> 多份伪装成 traceback"""
    lines = code.splitlines()
    chunk_size = len(lines) // parts + 1
    chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
    
    # 真实的Python文件名和路径
    fake_paths = [
        "/usr/lib/python3.11/site-packages/requests/sessions.py",
        "/usr/lib/python3.11/site-packages/urllib3/connectionpool.py", 
        "/home/user/project/main.py",
        "/opt/venv/lib/python3.11/site-packages/flask/app.py",
        "/usr/lib/python3.11/json/decoder.py",
        "/home/user/.local/lib/python3.11/site-packages/pandas/core/frame.py"
    ]
    
    # 各种错误类型和消息
    error_types = [
        ("SyntaxError", "invalid syntax"),
        ("IndentationError", "expected an indented block"),
        ("NameError", "name 'undefined_var' is not defined"),
        ("TypeError", "unsupported operand type(s) for +: 'str' and 'int'"),
        ("ValueError", "invalid literal for int() with base 10"),
        ("AttributeError", "module has no attribute"),
        ("ImportError", "cannot import name"),
        ("KeyError", "'missing_key'")
    ]
    
    outputs = []
    for idx, chunk in enumerate(chunks, start=1):
        # 随机选择错误类型和路径
        error_type, error_msg = random.choice(error_types)
        fake_path = random.choice(fake_paths)
        line_num = random.randint(50, 500)
        
        # 将代码块伪装成函数内容，添加更多上下文
        disguised_lines = []
        for line in chunk:
            if line.strip():
                # 添加随机的变量名替换使代码看起来更像库代码
                disguised_line = _disguise_code_line(line)
                disguised_lines.append(disguised_line)
        
        # 创建多层调用栈
        call_stack = _generate_call_stack(fake_path, line_num)
        
        # 格式化代码块
        code_block = "\n".join([f"    {line}" for line in disguised_lines])
        
        fake_tb = f"""{call_stack}
{code_block}
{error_type}: {error_msg}"""
        
        outputs.append(fake_tb)
    return outputs


def _disguise_code_line(line):
    """将代码行伪装得更像库代码"""
    # 替换常见变量名
    replacements = {
        r'\bdata\b': 'response_data',
        r'\bresult\b': 'parsed_result', 
        r'\bitem\b': 'element',
        r'\bvalue\b': 'attribute_value',
        r'\bkey\b': 'dict_key',
        r'\bfile\b': 'file_handler',
        r'\bpath\b': 'file_path'
    }
    
    disguised = line
    for pattern, replacement in replacements.items():
        disguised = re.sub(pattern, replacement, disguised)
    
    return disguised


def _generate_call_stack(file_path, line_num):
    """生成看起来真实的调用栈"""
    stacks = [
        f"""Traceback (most recent call last):
  File "/home/user/main.py", line 23, in <module>
    process_data()
  File "/home/user/utils.py", line 45, in process_data
    return parser.parse(content)
  File "{file_path}", line {line_num}, in parse""",
        
        f"""Traceback (most recent call last):
  File "/usr/bin/python3", line 8, in <module>
    sys.exit(main())
  File "/home/user/app.py", line 156, in main
    server.run()
  File "{file_path}", line {line_num}, in run""",
        
        f"""Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/user/project/core.py", line 89, in execute
    return self._process()
  File "{file_path}", line {line_num}, in _process"""
    ]
    return random.choice(stacks)


def decrypt_tracebacks(files):
    """从多份 traceback 文件还原源码"""
    restored = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            tb = fh.read()
        
        # 提取代码块：查找错误行上方的缩进代码
        lines = tb.splitlines()
        code_lines = []
        
        # 找到最后一个 "File" 行的位置
        last_file_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('File "') and ', line ' in line:
                last_file_idx = i
        
        if last_file_idx != -1:
            # 从该位置后开始提取代码，直到错误行
            for i in range(last_file_idx + 1, len(lines)):
                line = lines[i]
                # 如果遇到错误类型行就停止
                if any(err in line for err in ['Error:', 'Exception:']):
                    break
                # 提取缩进的代码行
                if line.startswith('    ') and line.strip():
                    code_line = line[4:]  # 去掉4个空格的缩进
                    # 还原变量名
                    restored_line = _restore_code_line(code_line)
                    code_lines.append(restored_line)
        
        if code_lines:
            restored.append("\n".join(code_lines))
    
    return "\n".join(restored)


def _restore_code_line(line):
    """还原被伪装的代码行"""
    # 反向替换变量名
    restorations = {
        'response_data': 'data',
        'parsed_result': 'result',
        'element': 'item', 
        'attribute_value': 'value',
        'dict_key': 'key',
        'file_handler': 'file',
        'file_path': 'path'
    }
    
    restored = line
    for disguised, original in restorations.items():
        restored = restored.replace(disguised, original)
    
    return restored


def main():
    parser = argparse.ArgumentParser(description="Disguise code as tracebacks")
    subparsers = parser.add_subparsers(dest="command")

    # encrypt
    enc = subparsers.add_parser("encrypt")
    enc.add_argument("input", help="输入源码文件")
    enc.add_argument("--parts", type=int, default=6, help="拆分份数")
    enc.add_argument("--outdir", default="encrypted", help="输出目录")

    # decrypt
    dec = subparsers.add_parser("decrypt")
    dec.add_argument("inputs", nargs="+", help="多个 traceback 文件")
    dec.add_argument("--output", default="restored.py", help="还原输出文件")

    args = parser.parse_args()

    if args.command == "encrypt":
        with open(args.input, "r", encoding="utf-8") as f:
            code = f.read()
        parts = encrypt_code_as_tracebacks(code, args.parts)
        os.makedirs(args.outdir, exist_ok=True)
        for i, p in enumerate(parts, start=1):
            path = os.path.join(args.outdir, f"part{i}.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(p)
        print(f"已生成 {len(parts)} 份，保存在 {args.outdir}/")

    elif args.command == "decrypt":
        restored = decrypt_tracebacks(args.inputs)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(restored)
        print(f"已还原代码到 {args.output}")


if __name__ == "__main__":
    main()
