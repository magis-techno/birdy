import argparse
import os
import textwrap
import random
import re

def encrypt_code_as_tracebacks(code: str, parts: int = 6):
    """拆分代码 -> 多份伪装成混合调试内容"""
    lines = code.splitlines()
    chunk_size = len(lines) // parts + 1
    chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
    
    outputs = []
    for idx, chunk in enumerate(chunks, start=1):
        # 为每个大块创建混合伪装内容
        mixed_content = _create_mixed_disguise(chunk, idx)
        outputs.append(mixed_content)
    
    return outputs


def _create_mixed_disguise(code_lines, chunk_id):
    """为代码块创建混合的伪装内容（traceback + REPL + 调试输出）"""
    result_lines = []
    
    # 添加初始的调试会话头部
    session_header = _generate_debug_session_header()
    result_lines.extend(session_header)
    
    # 将代码行分成小段，每段5-8行
    segments = _split_into_segments(code_lines, segment_size=random.randint(5, 8))
    
    for i, segment in enumerate(segments):
        # 在段之间插入不同类型的伪装内容
        if i > 0:
            disguise_type = random.choice(['traceback', 'repl', 'debug_output', 'exception'])
            separator = _generate_disguise_separator(disguise_type)
            result_lines.extend(separator)
        
        # 处理当前代码段
        disguised_segment = _process_code_segment(segment)
        result_lines.extend(disguised_segment)
    
    # 添加结尾的错误信息
    footer = _generate_error_footer()
    result_lines.extend(footer)
    
    return "\n".join(result_lines)


def _split_into_segments(lines, segment_size):
    """将代码行分割成小段"""
    segments = []
    current_segment = []
    
    for line in lines:
        if line.strip():  # 只处理非空行
            current_segment.append(line)
            if len(current_segment) >= segment_size:
                segments.append(current_segment)
                current_segment = []
    
    if current_segment:  # 添加剩余的行
        segments.append(current_segment)
    
    return segments


def _generate_debug_session_header():
    """生成调试会话的头部"""
    headers = [
        [
            "Python 3.11.2 (main, Feb  8 2023, 14:49:24) [GCC 9.4.0] on linux",
            "Type \"help\", \"copyright\", \"credits\" or \"license\" for more information.",
            ">>> import sys, traceback",
            ">>> # Debugging session started"
        ],
        [
            "IPython 8.12.0 -- An enhanced Interactive Python. Type '?' for help.",
            "",
            "In [1]: %debug",
            "> /usr/lib/python3.11/site-packages/IPython/core/debugger.py(45)__init__()"
        ],
        [
            "pdb trace started...",
            "(Pdb) l",
            "Current function: process_data"
        ]
    ]
    return random.choice(headers)


def _generate_disguise_separator(disguise_type):
    """生成不同类型的伪装分隔内容"""
    if disguise_type == 'traceback':
        return _generate_mini_traceback()
    elif disguise_type == 'repl':
        return _generate_repl_interaction()
    elif disguise_type == 'debug_output':
        return _generate_debug_output()
    elif disguise_type == 'exception':
        return _generate_exception_context()
    else:
        return ["# --- Debug checkpoint ---"]


def _generate_mini_traceback():
    """生成小型的traceback片段"""
    fake_paths = [
        "/usr/lib/python3.11/site-packages/requests/sessions.py",
        "/usr/lib/python3.11/site-packages/urllib3/connectionpool.py", 
        "/home/user/project/utils.py",
        "/opt/venv/lib/python3.11/site-packages/flask/app.py"
    ]
    
    path = random.choice(fake_paths)
    line_num = random.randint(50, 500)
    
    return [
        f"Traceback (most recent call last):",
        f'  File "{path}", line {line_num}, in process_request',
        f"    response = self._handle_request(data)",
        f"  File \"{path}\", line {line_num + 15}, in _handle_request"
    ]


def _generate_repl_interaction():
    """生成REPL交互式内容"""
    interactions = [
        [
            ">>> print(f'Processing {len(data)} items...')",
            f"Processing {random.randint(10, 999)} items...",
            ">>> # Continue execution"
        ],
        [
            ">>> vars().keys()",
            "dict_keys(['__name__', '__doc__', '__package__', 'data', 'result', 'temp_var'])",
            ">>> type(data)",
            "<class 'list'>"
        ],
        [
            "In [15]: %timeit process_function()",
            f"{random.randint(1, 99)} ms ± {random.randint(1, 9)} ms per loop (mean ± std. dev. of 7 runs, 10 loops each)",
            "In [16]: # Performance analysis complete"
        ]
    ]
    return random.choice(interactions)


def _generate_debug_output():
    """生成调试输出内容"""
    outputs = [
        [
            f"DEBUG: Variable state at checkpoint {random.randint(1, 20)}",
            f"  - data_length: {random.randint(10, 1000)}",
            f"  - current_index: {random.randint(0, 100)}",
            f"  - processing_time: {random.uniform(0.1, 5.0):.3f}s"
        ],
        [
            "(Pdb) p locals()",
            "{'data': [...], 'index': 42, 'temp_result': None}",
            "(Pdb) n"
        ],
        [
            f"[{random.randint(10, 23)}:{random.randint(10, 59)}:{random.randint(10, 59)}] INFO: Processing step {random.randint(1, 100)}",
            f"[{random.randint(10, 23)}:{random.randint(10, 59)}:{random.randint(10, 59)}] DEBUG: Memory usage: {random.randint(50, 500)}MB"
        ]
    ]
    return random.choice(outputs)


def _generate_exception_context():
    """生成异常上下文信息"""
    exceptions = [
        [
            "Exception occurred during processing:",
            "  Context: data validation phase",
            f"  Item count: {random.randint(1, 1000)}",
            "  Continuing execution..."
        ],
        [
            "Warning: Performance degradation detected",
            f"  Expected time: {random.uniform(0.1, 1.0):.2f}s",
            f"  Actual time: {random.uniform(1.0, 5.0):.2f}s",
            "  Reason: Large dataset processing"
        ]
    ]
    return random.choice(exceptions)


def _process_code_segment(lines):
    """处理代码段，添加伪装和格式化"""
    processed = []
    
    for line in lines:
        if line.strip():
            # 伪装变量名
            disguised_line = _disguise_code_line(line)
            # 添加适当的缩进，模拟在traceback中的代码
            processed.append(f"    {disguised_line}")
    
    return processed


def _generate_error_footer():
    """生成错误结尾信息"""
    error_types = [
        ("SyntaxError", "invalid syntax"),
        ("IndentationError", "expected an indented block"),
        ("NameError", "name 'undefined_var' is not defined"),
        ("TypeError", "unsupported operand type(s)"),
        ("ValueError", "invalid literal for int()"),
        ("AttributeError", "object has no attribute"),
        ("RuntimeError", "maximum recursion depth exceeded")
    ]
    
    error_type, error_msg = random.choice(error_types)
    return [f"{error_type}: {error_msg}"]


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
    """从多份混合伪装文件中还原源码"""
    restored = []
    
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            content = fh.read()
        
        # 提取真实代码行
        code_lines = _extract_real_code_lines(content)
        if code_lines:
            restored.extend(code_lines)
    
    return "\n".join(restored)


def _extract_real_code_lines(content):
    """从混合伪装内容中提取真实的代码行"""
    lines = content.splitlines()
    code_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # 跳过各种伪装内容
        if _is_disguise_line(line):
            i += 1
            continue
        
        # 检测并提取真实代码行（以4个空格开头的缩进行）
        if line.startswith('    ') and line.strip():
            # 确保这不是伪装的traceback输出
            if not _is_fake_traceback_code(line):
                code_line = line[4:]  # 去掉4个空格的缩进
                restored_line = _restore_code_line(code_line)
                code_lines.append(restored_line)
        
        i += 1
    
    return code_lines


def _is_disguise_line(line):
    """判断是否为伪装行"""
    line = line.strip()
    
    # Python解释器启动信息
    if line.startswith('Python ') and 'on linux' in line:
        return True
    
    # IPython相关
    if line.startswith('IPython ') or line.startswith('In [') or line.startswith('Out['):
        return True
    
    # REPL交互
    if line.startswith('>>> ') or line == '>>>':
        return True
    
    # 调试器相关
    if line.startswith('(Pdb)') or line.startswith('pdb trace'):
        return True
    
    # Traceback信息
    if line.startswith('Traceback (most recent call last):'):
        return True
    
    # 文件路径行
    if line.startswith('File "') and ', line ' in line:
        return True
    
    # 错误类型行
    error_patterns = [
        'Error:', 'Exception:', 'Warning:', 'DEBUG:', 'INFO:',
        'SyntaxError:', 'NameError:', 'TypeError:', 'ValueError:',
        'AttributeError:', 'ImportError:', 'KeyError:', 'RuntimeError:',
        'IndentationError:'
    ]
    if any(pattern in line for pattern in error_patterns):
        return True
    
    # 调试输出
    if line.startswith('DEBUG:') or line.startswith('  - '):
        return True
    
    # 时间戳日志
    if re.match(r'\[\d{2}:\d{2}:\d{2}\]', line):
        return True
    
    # 性能测试输出
    if ' ms ± ' in line and 'per loop' in line:
        return True
    
    # 字典输出
    if line.startswith("dict_keys(") or line.startswith("<class '"):
        return True
    
    # 帮助信息
    if 'Type "help"' in line or 'for more information' in line:
        return True
    
    # 处理上下文信息
    if line.startswith('Context:') or line.startswith('Reason:'):
        return True
    
    # 空行和注释
    if not line or line.startswith('#'):
        return True
    
    return False


def _is_fake_traceback_code(line):
    """判断是否为伪造的traceback中的代码行"""
    code_content = line.strip()
    
    # 检查是否包含明显的伪装代码特征
    fake_patterns = [
        'response = self._handle_request(data)',
        'return parser.parse(content)',
        'sys.exit(main())',
        'server.run()',
        'return self._process()'
    ]
    
    return any(pattern in code_content for pattern in fake_patterns)


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
