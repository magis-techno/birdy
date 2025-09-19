import argparse
import os
import textwrap

def encrypt_code_as_tracebacks(code: str, parts: int = 6):
    """拆分代码 -> 多份伪装成 traceback"""
    lines = code.splitlines()
    chunk_size = len(lines) // parts + 1
    chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
    
    outputs = []
    for idx, chunk in enumerate(chunks, start=1):
        snippet = "\n    ".join(chunk)
        fake_tb = textwrap.dedent(f"""
        Traceback (most recent call last):
          File "part{idx}.py", line 1, in <module>
            {snippet}
        SyntaxError: invalid syntax
        """).strip()
        outputs.append(fake_tb)
    return outputs


def decrypt_tracebacks(files):
    """从多份 traceback 文件还原源码"""
    restored = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            tb = fh.read()
        lines = tb.splitlines()
        snippet_lines = lines[2:-1]  # 去掉头尾
        snippet = "\n".join([l.strip() for l in snippet_lines if l.strip()])
        restored.append(snippet)
    return "\n".join(restored)


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
