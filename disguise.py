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
    """创建紧凑的traceback伪装内容"""
    result_lines = []
    
    # 将代码行分成小段，针对函数密集区域使用更小的段
    if density == 'high':
        segments = _split_into_function_segments(code_lines)
    else:
        segments = _split_into_segments(code_lines, segment_size=random.randint(6, 10))
    
    for i, segment in enumerate(segments):
        # 为每个段创建独立的traceback
        traceback_content = _create_compact_traceback(segment, density)
        result_lines.extend(traceback_content)
        
        # 在段之间只添加极简分隔（可选）
        if i < len(segments) - 1 and len(segments) > 1:
            if random.choice([True, False]):  # 50%概率添加分隔
                result_lines.append("")  # 仅添加空行
    
    return "\n".join(result_lines)


def _split_into_function_segments(code_lines):
    """智能分割函数密集区域，每个函数尽量独立"""
    segments = []
    current_segment = []
    
    for line in code_lines:
        if line.strip():
            current_segment.append(line)
            
            # 检测函数定义结束 - 当遇到下一个函数定义时分割
            if (line.strip().startswith('def ') or line.strip().startswith('class ')) and len(current_segment) > 1:
                # 保存前一个段
                segments.append(current_segment[:-1])
                # 开始新段
                current_segment = [line]
            # 或者当段落过长时分割
            elif len(current_segment) >= 8:
                segments.append(current_segment)
                current_segment = []
    
    if current_segment:
        segments.append(current_segment)
    
    return segments


def _create_compact_traceback(code_segment, density):
    """为代码段创建真实的多层traceback结构"""
    if not code_segment:
        return []
    
    traceback_lines = ["Traceback (most recent call last):"]
    
    # 为每行代码创建独立的调用栈层级
    for i, line in enumerate(code_segment):
        if not line.strip():
            continue
            
        # 分析代码行，选择相关的错误和上下文
        line_info = _analyze_code_line(line)
        file_path = _select_contextual_path(line_info['operation'])
        line_num = random.randint(20, 300)
        
        # 添加调用栈层级
        traceback_lines.append(f'  File "{file_path}", line {line_num}, in {line_info["function"]}')
        
        # 添加该层级的代码行（使用正确的traceback缩进）
        disguised_line = _disguise_code_line(line)
        traceback_lines.append(f"    {disguised_line}")
        
        # 如果是最后一行或者是关键操作，添加错误
        if i == len(code_segment) - 1 or line_info['should_error']:
            error_info = _select_contextual_error(line_info['operation'], line)
            traceback_lines.append(f"{error_info['error_type']}: {error_info['message']}")
            break
    
    return traceback_lines


def _analyze_code_line(line):
    """分析单行代码，确定操作类型和上下文"""
    line = line.strip().lower()
    
    # 文件操作
    if 'open(' in line or '.read(' in line or '.write(' in line:
        return {
            'operation': 'file_io',
            'function': 'load_data' if 'read' in line else 'save_data',
            'should_error': True
        }
    
    # JSON操作
    elif 'json.load' in line or 'json.dump' in line or '.json(' in line:
        return {
            'operation': 'json',
            'function': 'parse_config' if 'load' in line else 'save_config',
            'should_error': True
        }
    
    # 函数定义
    elif line.startswith('def '):
        func_name = line.split('(')[0].replace('def ', '').strip()
        return {
            'operation': 'function_def',
            'function': '<module>',
            'should_error': False
        }
    
    # 类定义
    elif line.startswith('class '):
        return {
            'operation': 'class_def',
            'function': '<module>',
            'should_error': False
        }
    
    # 导入语句
    elif 'import ' in line:
        return {
            'operation': 'import',
            'function': '<module>',
            'should_error': True
        }
    
    # 属性访问
    elif '.' in line and '=' in line:
        return {
            'operation': 'attribute',
            'function': '__init__' if 'self.' in line else 'setup',
            'should_error': random.choice([True, False])
        }
    
    # 函数调用
    elif '(' in line and ')' in line and not line.startswith(('def ', 'class ')):
        return {
            'operation': 'function_call',
            'function': 'process_data',
            'should_error': random.choice([True, False])
        }
    
    # 默认
    else:
        return {
            'operation': 'general',
            'function': 'execute',
            'should_error': False
        }


def _select_contextual_path(operation):
    """根据操作类型选择相关的文件路径"""
    path_map = {
        'file_io': [
            "/usr/lib/python3.11/pathlib.py",
            "/usr/lib/python3.11/io.py",
            "/home/user/app/file_handler.py"
        ],
        'json': [
            "/usr/lib/python3.11/json/decoder.py",
            "/usr/lib/python3.11/json/encoder.py",
            "/home/user/app/config_parser.py"
        ],
        'import': [
            "/usr/lib/python3.11/importlib/__init__.py",
            "/usr/lib/python3.11/importlib/_bootstrap.py"
        ],
        'function_def': [
            "/home/user/app/main.py",
            "/home/user/app/core.py"
        ],
        'class_def': [
            "/home/user/app/models.py",
            "/home/user/app/base.py"
        ],
        'attribute': [
            "/home/user/app/config.py",
            "/usr/lib/python3.11/types.py"
        ],
        'function_call': [
            "/home/user/app/utils.py",
            "/usr/lib/python3.11/functools.py"
        ]
    }
    
    paths = path_map.get(operation, ["/home/user/app/main.py"])
    return random.choice(paths)


def _select_contextual_error(operation, line):
    """根据操作类型和具体代码选择相关错误"""
    line_lower = line.lower()
    
    if operation == 'file_io':
        if 'open(' in line_lower:
            return {
                'error_type': 'FileNotFoundError',
                'message': "[Errno 2] No such file or directory"
            }
        else:
            return {
                'error_type': 'PermissionError', 
                'message': "[Errno 13] Permission denied"
            }
    
    elif operation == 'json':
        return {
            'error_type': 'JSONDecodeError',
            'message': "Expecting value: line 1 column 1 (char 0)"
        }
    
    elif operation == 'import':
        # 从代码中提取模块名
        if 'import ' in line_lower:
            try:
                module = line_lower.split('import ')[1].split()[0].split('.')[0]
                return {
                    'error_type': 'ModuleNotFoundError',
                    'message': f"No module named '{module}'"
                }
            except:
                pass
        return {
            'error_type': 'ImportError',
            'message': "cannot import name"
        }
    
    elif operation == 'attribute':
        if 'self.' in line_lower:
            return {
                'error_type': 'AttributeError',
                'message': "'NoneType' object has no attribute"
            }
        else:
            return {
                'error_type': 'NameError',
                'message': "name 'config' is not defined"
            }
    
    elif operation == 'function_call':
        return {
            'error_type': 'TypeError',
            'message': "missing 1 required positional argument"
        }
    
    else:
        return {
            'error_type': 'ValueError',
            'message': "invalid value"
        }


def _select_error_for_code(code_segment):
    """根据代码内容选择相关的错误类型"""
    code_text = ' '.join(code_segment).lower()
    
    # 根据代码内容智能选择错误
    if 'def ' in code_text:
        errors = [
            {"error_type": "NameError", "message": "name 'self' is not defined", "context": "__init__"},
            {"error_type": "AttributeError", "message": "'NoneType' object has no attribute 'config'", "context": "setup"},
            {"error_type": "TypeError", "message": "missing 1 required positional argument", "context": "<module>"}
        ]
    elif 'import ' in code_text:
        errors = [
            {"error_type": "ModuleNotFoundError", "message": "No module named 'config'", "context": "<module>"},
            {"error_type": "ImportError", "message": "cannot import name 'logger'", "context": "<module>"}
        ]
    elif 'open(' in code_text or 'file' in code_text:
        errors = [
            {"error_type": "FileNotFoundError", "message": "[Errno 2] No such file or directory", "context": "load_data"},
            {"error_type": "PermissionError", "message": "[Errno 13] Permission denied", "context": "save_config"}
        ]
    elif 'json' in code_text:
        errors = [
            {"error_type": "JSONDecodeError", "message": "Expecting value: line 1 column 1 (char 0)", "context": "parse"},
            {"error_type": "ValueError", "message": "Invalid JSON format", "context": "load_config"}
        ]
    else:
        errors = [
            {"error_type": "ValueError", "message": "invalid literal for int()", "context": "process"},
            {"error_type": "TypeError", "message": "unsupported operand type(s)", "context": "calculate"},
            {"error_type": "AttributeError", "message": "object has no attribute", "context": "execute"}
        ]
    
    return random.choice(errors)


def _select_realistic_path():
    """选择真实的文件路径"""
    paths = [
        "/usr/lib/python3.11/site-packages/requests/models.py",
        "/usr/lib/python3.11/site-packages/urllib3/response.py",
        "/usr/lib/python3.11/json/decoder.py",
        "/home/user/app/config.py",
        "/opt/venv/lib/python3.11/site-packages/flask/app.py",
        "/usr/lib/python3.11/logging/__init__.py"
    ]
    return random.choice(paths)




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
    
    for line in lines:
        line = line.rstrip()
        
        # 跳过各种伪装内容
        if _is_disguise_line(line):
            continue
        
        # 检测并提取真实代码行（以4个空格开头的缩进行）
        if line.startswith('    ') and line.strip():
            # 确保这不是伪装的traceback输出
            if not _is_fake_traceback_code(line):
                code_line = line[4:]  # 去掉4个空格的缩进
                restored_line = _restore_code_line(code_line)
                code_lines.append(restored_line)
    
    return code_lines


def _is_disguise_line(line):
    """判断是否为伪装行（紧凑版）"""
    line = line.strip()
    
    # Traceback信息头部
    if line.startswith('Traceback (most recent call last):'):
        return True
    
    # 文件路径行（traceback中的文件引用）
    if line.startswith('File "') and ', line ' in line and ', in ' in line:
        return True
    
    # 错误类型行（traceback结尾的错误）
    error_patterns = [
        'Error:', 'SyntaxError:', 'NameError:', 'TypeError:', 'ValueError:',
        'AttributeError:', 'ImportError:', 'KeyError:', 'RuntimeError:',
        'IndentationError:', 'ModuleNotFoundError:', 'FileNotFoundError:',
        'PermissionError:', 'JSONDecodeError:'
    ]
    if any(line.startswith(pattern) for pattern in error_patterns):
        return True
    
    # 空行
    if not line:
        return True
    
    return False


def _is_fake_traceback_code(line):
    """判断是否为伪造的traceback中的代码行"""
    # 在新的多层traceback结构中，几乎所有4空格缩进的行都是真实代码
    # 只过滤明显的系统调用
    code_content = line.strip()
    
    fake_patterns = [
        'return _bootstrap._gcd_import(',
        '_bootstrap._gcd_import(name[level:], package, level)',
        'sys.exit(main())',
        'server.run()',
        'return self._process()',
        # 添加一些明显的系统内部调用
        '__import__(',
        '_find_and_load(',
        '_find_spec('
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
