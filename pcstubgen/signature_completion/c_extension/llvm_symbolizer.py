from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import msgspec


LLVM_SYMBOLIZER = "llvm-symbolizer"


class SymbolizerError(msgspec.Struct):
    Message: str
    Code: int | None = None
    Type: str | None = None


class SymbolizerSymbol(msgspec.Struct):
    FunctionName: str      # 当前 frame 的函数名

    FileName: str          # 当前查询地址命中的源码文件
    Line: int              # 当前查询地址命中的源码行
    Column: int            # 当前查询地址命中的源码列
    Discriminator: int     # 同一行内区分不同代码位置的编号

    StartFileName: str     # 该函数 / inline frame 的起始文件
    StartLine: int         # 该函数 / inline frame 的起始行
    StartAddress: str      # 该函数 / inline frame 的起始机器码地址


class SymbolizerEntry(msgspec.Struct):
    Address: str
    ModuleName: str
    Symbol: list[SymbolizerSymbol] | None = None
    Error: SymbolizerError | None = None


def find_llvm_symbolizer() -> str | None:
    return shutil.which(LLVM_SYMBOLIZER)


def require_llvm_symbolizer() -> str:
    symbolizer_path = find_llvm_symbolizer()
    if symbolizer_path is None:
        raise RuntimeError(
            "C扩展源码补全依赖 llvm-symbolizer，但当前 PATH 中未找到该可执行文件。"
        )
    return symbolizer_path


def run(binary_path: Path, relative_address: int) -> tuple[SymbolizerEntry, ...]:
    symbolizer_path = require_llvm_symbolizer()
    completed = subprocess.run(
        [
            symbolizer_path,
            "--output-style=JSON",
            "--relative-address",
            f"--obj={binary_path}",
            f"0x{relative_address:x}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or "<empty>"
        raise RuntimeError(
            f"llvm-symbolizer 执行失败: exit_code={completed.returncode}, detail={detail}"
        )

    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError("llvm-symbolizer 未返回任何输出。")

    try:
        return msgspec.json.decode(stdout, type=tuple[SymbolizerEntry, ...])
    except (msgspec.DecodeError, msgspec.ValidationError) as ex:
        raise RuntimeError("llvm-symbolizer 返回了非预期JSON结果。") from ex
