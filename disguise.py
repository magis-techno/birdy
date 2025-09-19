import argparse
import os
import textwrap
import random
import re
import ast
from datetime import datetime, timedelta

def encrypt_code_as_tracebacks(code: str, parts: int = 6):
    """拆分代码 -> 多份伪装成日志和异常输出"""
    lines = code.splitlines()
    chunk_size = len(lines) // parts + 1
    chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
    
    # 分析代码结构
    function_density_map = _analyze_code_structure(code, chunks)
    
    outputs = []
    base_time = datetime.now()
    
    for idx, chunk in enumerate(chunks, start=1):
        # 根据函数密度调整伪装策略
        density = function_density_map.get(idx, 'normal')
        chunk_content = _create_logging_disguise(chunk, idx, base_time, density)
        outputs.append(chunk_content)
        
        # 时间递增
        base_time += timedelta(seconds=random.randint(1, 5))
    
    return outputs


def _analyze_code_structure(code, chunks):
    """分析代码结构，识别函数密集区域"""
    try:
        tree = ast.parse(code)
        function_lines = set()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                function_lines.add(node.lineno)
        
        # 分析每个chunk的函数密度
        density_map = {}
        lines_processed = 0
        
        for idx, chunk in enumerate(chunks, start=1):
            chunk_start = lines_processed + 1
            chunk_end = lines_processed + len([l for l in chunk if l.strip()])
            
            # 计算这个chunk中的函数定义数量
            func_count = len([line for line in range(chunk_start, chunk_end + 1) 
                            if line in function_lines])
            
            if func_count >= 3:
                density_map[idx] = 'high'
            elif func_count >= 1:
                density_map[idx] = 'medium'
            else:
                density_map[idx] = 'normal'
            
            lines_processed = chunk_end
        
        return density_map
    
    except SyntaxError:
        # 如果AST解析失败，返回默认密度
        return {i: 'normal' for i in range(1, len(chunks) + 1)}


def _create_logging_disguise(code_lines, chunk_id, base_time, density):
    """创建基于日志和异常的伪装内容"""
    result_lines = []
    current_time = base_time
    
    # 根据密度选择伪装策略
    if density == 'high':
        result_lines.extend(_generate_module_loading_header(current_time))
    else:
        result_lines.extend(_generate_simple_log_header(current_time))
    
    # 将代码行分成小段
    segments = _split_into_segments(code_lines, segment_size=random.randint(4, 7))
    
    for i, segment in enumerate(segments):
        current_time += timedelta(milliseconds=random.randint(10, 50))
        
        # 在段之间插入伪装内容
        if i > 0:
            if density == 'high':
                separator = _generate_function_loading_context(current_time)
            else:
                separator = _generate_simple_log_separator(current_time)
            result_lines.extend(separator)
        
        # 处理当前代码段
        disguised_segment = _process_code_segment_simple(segment)
        result_lines.extend(disguised_segment)
    
    # 添加结尾
    current_time += timedelta(milliseconds=random.randint(20, 100))
    footer = _generate_log_footer(current_time, density)
    result_lines.extend(footer)
    
    return "\n".join(result_lines)


def _generate_module_loading_header(current_time):
    """生成模块加载的头部日志"""
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    headers = [
        [
            f"{timestamp} - INFO - Module initialization started",
            f"{timestamp} - DEBUG - Loading function definitions",
            f"{timestamp} - INFO - Processing class definitions"
        ],
        [
            f"{timestamp} - INFO - Starting function registration phase",
            f"{timestamp} - DEBUG - Analyzing code structure",
            f"{timestamp} - INFO - Preparing execution context"
        ]
    ]
    return random.choice(headers)


def _generate_simple_log_header(current_time):
    """生成简单的日志头部"""
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    headers = [
        [f"{timestamp} - INFO - Processing code block"],
        [f"{timestamp} - DEBUG - Execution phase started"],
        [f"{timestamp} - INFO - Loading module components"]
    ]
    return random.choice(headers)


def _generate_function_loading_context(current_time):
    """生成函数加载上下文（高密度区域使用）"""
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    
    contexts = [
        [
            f"{timestamp} - DEBUG - Function definition detected",
            "Exception occurred during function loading:",
            "  Context: module initialization phase",
            f"  Location: line {random.randint(45, 200)}, in <module>",
            "  Recovered: continuing execution"
        ],
        [
            f"{timestamp} - INFO - Registering function handler",
            "Traceback (most recent call last):",
            f'  File "/usr/lib/python3.11/importlib/__init__.py", line {random.randint(100, 300)}, in import_module',
            f"    return _bootstrap._gcd_import(name[level:], package, level)",
            f'  File "/usr/lib/python3.11/importlib/_bootstrap.py", line {random.randint(800, 1200)}, in _gcd_import'
        ],
        [
            f"{timestamp} - DEBUG - Processing method definitions",
            f"{timestamp} - INFO - Function validation complete"
        ]
    ]
    return random.choice(contexts)


def _generate_simple_log_separator(current_time):
    """生成简单的日志分隔（普通区域使用）"""
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    
    separators = [
        [f"{timestamp} - DEBUG - Processing next code segment"],
        [f"{timestamp} - INFO - Execution checkpoint reached"],
        [
            "Exception in execution flow:",
            f"  Timestamp: {timestamp}",
            "  Status: recovered, continuing"
        ]
    ]
    return random.choice(separators)


def _process_code_segment_simple(lines):
    """简化的代码段处理"""
    processed = []
    
    for line in lines:
        if line.strip():
            # 伪装变量名
            disguised_line = _disguise_code_line(line)
            # 添加适当的缩进
            processed.append(f"    {disguised_line}")
    
    return processed


def _generate_log_footer(current_time, density):
    """生成日志结尾"""
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    
    if density == 'high':
        footers = [
            [
                f"{timestamp} - INFO - Function loading completed",
                f"{timestamp} - DEBUG - {random.randint(5, 15)} functions registered successfully"
            ],
            [
                "Traceback (most recent call last):",
                f'  File "/home/user/app.py", line {random.randint(20, 100)}, in <module>',
                f"    import {random.choice(['config', 'utils', 'handlers', 'processors'])}",
                f"ModuleNotFoundError: No module named 'temp_module_{random.randint(1, 999)}'"
            ]
        ]
    else:
        footers = [
            [f"{timestamp} - INFO - Code block processing complete"],
            [f"{timestamp} - DEBUG - Memory cleanup initiated"]
        ]
    
    return random.choice(footers)


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
    """判断是否为伪装行（简化版）"""
    line = line.strip()
    
    # 时间戳日志格式
    if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - (INFO|DEBUG|WARNING|ERROR)', line):
        return True
    
    # Traceback信息
    if line.startswith('Traceback (most recent call last):'):
        return True
    
    # 文件路径行
    if line.startswith('File "') and ', line ' in line:
        return True
    
    # 异常相关
    if line.startswith('Exception ') or 'Exception:' in line:
        return True
    
    # 错误类型行
    error_patterns = [
        'Error:', 'SyntaxError:', 'NameError:', 'TypeError:', 'ValueError:',
        'AttributeError:', 'ImportError:', 'KeyError:', 'RuntimeError:',
        'IndentationError:', 'ModuleNotFoundError:'
    ]
    if any(pattern in line for pattern in error_patterns):
        return True
    
    # 特定的伪装import语句
    if line.startswith('import ') and line in ['import handlers', 'import config', 'import utils', 'import processors']:
        return True
    
    # 上下文信息
    if line.startswith('Context:') or line.startswith('Location:') or line.startswith('Timestamp:'):
        return True
    
    # 状态信息
    if line.startswith('Status:') or line.startswith('Recovered:'):
        return True
    
    # 缩进的上下文信息
    if line.startswith('  ') and ('Context:' in line or 'Location:' in line or 'Status:' in line):
        return True
    
    # 空行
    if not line:
        return True
    
    return False


def _is_fake_traceback_code(line):
    """判断是否为伪造的traceback中的代码行"""
    code_content = line.strip()
    
    # 检查是否包含明显的伪装代码特征
    fake_patterns = [
        'return _bootstrap._gcd_import(',
        '_bootstrap._gcd_import(name[level:], package, level)',
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
