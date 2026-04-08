from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess


LLVM_SYMBOLIZER = "llvm-symbolizer"


@dataclass(frozen=True)
class SymbolizedAddressLocation:
    binary_path: Path
    relative_address: int
    function_name: str | None = None
    resolved_path: Path | None = None
    resolved_line: int | None = None
    function_start_path: Path | None = None
    function_start_line: int | None = None


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


_dladdr = ctypes.CDLL(None).dladdr
_dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
_dladdr.restype = ctypes.c_int


def find_llvm_symbolizer() -> str | None:
    return shutil.which(LLVM_SYMBOLIZER)


def require_llvm_symbolizer() -> str:
    symbolizer_path = find_llvm_symbolizer()
    if symbolizer_path is None:
        raise RuntimeError(
            "C扩展源码补全依赖 llvm-symbolizer，但当前 PATH 中未找到该可执行文件。"
        )
    return symbolizer_path


def resolve_symbolized_address(address: int) -> SymbolizedAddressLocation:
    """将运行时函数地址解析为共享库内相对地址和源码位置。"""
    binary_path, relative_address = _resolve_binary_address(address)
    symbolizer_path = require_llvm_symbolizer()
    payload = _run_llvm_symbolizer(
        symbolizer_path=symbolizer_path,
        binary_path=binary_path,
        relative_address=relative_address,
    )
    return _parse_symbolizer_payload(
        payload=payload,
        binary_path=binary_path,
        relative_address=relative_address,
    )


def _resolve_binary_address(address: int) -> tuple[Path, int]:
    dl_info = _DlInfo()
    if _dladdr(ctypes.c_void_p(address), ctypes.byref(dl_info)) != 1:
        raise RuntimeError(f"无法定位函数地址所属共享库: 0x{address:x}")
    if dl_info.dli_fname is None or dl_info.dli_fbase is None:
        raise RuntimeError(f"共享库位置信息不完整: 0x{address:x}")

    binary_path = Path(dl_info.dli_fname.decode("utf-8", errors="replace")).resolve()
    base_address = int(dl_info.dli_fbase)
    return binary_path, address - base_address


def _run_llvm_symbolizer(
    *,
    symbolizer_path: str,
    binary_path: Path,
    relative_address: int,
) -> object:
    completed = subprocess.run(
        [
            symbolizer_path,
            "--output-style=JSON",
            "--relative-address",
            f"--obj={binary_path}",
            "--no-demangle",
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
        return json.loads(stdout)
    except json.JSONDecodeError as ex:
        raise RuntimeError("llvm-symbolizer 返回了无效JSON。") from ex


def _parse_symbolizer_payload(
    *,
    payload: object,
    binary_path: Path,
    relative_address: int,
) -> SymbolizedAddressLocation:
    if isinstance(payload, dict):
        raise RuntimeError(_format_symbolizer_error(payload))
    if not isinstance(payload, list):
        raise RuntimeError("llvm-symbolizer 返回了非预期JSON结构。")
    if len(payload) != 1:
        raise RuntimeError("llvm-symbolizer 返回了非单地址结果。")

    entry = payload[0]
    if not isinstance(entry, dict):
        raise RuntimeError("llvm-symbolizer 返回了非预期结果条目。")
    if "Error" in entry:
        raise RuntimeError(_format_symbolizer_error(entry))

    symbols = entry.get("Symbol")
    if not isinstance(symbols, list) or len(symbols) == 0:
        raise RuntimeError("llvm-symbolizer 未返回任何符号信息。")

    first_symbol = symbols[0]
    if not isinstance(first_symbol, dict):
        raise RuntimeError("llvm-symbolizer 返回了非预期符号信息。")

    return SymbolizedAddressLocation(
        binary_path=binary_path,
        relative_address=relative_address,
        function_name=_read_optional_text(first_symbol.get("FunctionName")),
        resolved_path=_read_optional_path(first_symbol.get("FileName")),
        resolved_line=_read_optional_line(first_symbol.get("Line")),
        function_start_path=_read_optional_path(first_symbol.get("StartFileName")),
        function_start_line=_read_optional_line(first_symbol.get("StartLine")),
    )


def _format_symbolizer_error(payload: dict[str, object]) -> str:
    error = payload.get("Error")
    if isinstance(error, dict):
        message = error.get("Message")
        if isinstance(message, str) and message:
            return f"llvm-symbolizer 解析地址失败: {message}"
    return "llvm-symbolizer 解析地址失败: 返回了错误结果。"


def _read_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if value == "":
        return None
    return value


def _read_optional_path(value: object) -> Path | None:
    path_text = _read_optional_text(value)
    if path_text is None:
        return None
    return Path(path_text).resolve()


def _read_optional_line(value: object) -> int | None:
    if not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return value
