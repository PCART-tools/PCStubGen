from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

from . import llvm_symbolizer


@dataclass(frozen=True)
class SymbolizedAddressLocation:
    binary_path: Path
    relative_address: int
    function_name: str
    resolved_path: Path
    resolved_line: int
    function_start_path: Path
    function_start_line: int


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


def resolve_symbolized_address(address: int) -> SymbolizedAddressLocation:
    """将运行时函数地址解析为共享库内相对地址和源码位置。"""
    binary_path, relative_address = _get_binary_and_ra(address)
    payload = llvm_symbolizer.run(binary_path, relative_address)
    return _parse_symbolizer_payload(
        payload=payload,
        binary_path=binary_path,
        relative_address=relative_address,
    )


def _get_binary_and_ra(address: int) -> tuple[Path, int]:
    dl_info = _DlInfo()
    if _dladdr(ctypes.c_void_p(address), ctypes.byref(dl_info)) != 1:
        raise RuntimeError(f"无法定位函数地址所属共享库: 0x{address:x}")
    if dl_info.dli_fname is None or dl_info.dli_fbase is None:
        raise RuntimeError(f"共享库位置信息不完整: 0x{address:x}")

    binary_path = Path(dl_info.dli_fname.decode("utf-8", errors="replace")).resolve()
    base_address = int(dl_info.dli_fbase)
    return binary_path, address - base_address


def _parse_symbolizer_payload(
    *,
    payload: tuple[llvm_symbolizer.SymbolizerEntry, ...],
    binary_path: Path,
    relative_address: int,
) -> SymbolizedAddressLocation:
    if len(payload) != 1:
        raise RuntimeError("llvm-symbolizer 返回了非单地址结果。")

    entry = payload[0]
    if entry.Error is not None:
        raise RuntimeError(f"llvm-symbolizer 解析地址失败: {entry.Error.Message}")

    if entry.Symbol is None or len(entry.Symbol) == 0:
        raise RuntimeError("llvm-symbolizer 未返回任何符号信息。")

    symbol = entry.Symbol[0]
    return SymbolizedAddressLocation(
        binary_path=binary_path,
        relative_address=relative_address,
        function_name=symbol.FunctionName,
        resolved_path=Path(symbol.FileName).resolve(),
        resolved_line=symbol.Line,
        function_start_path=Path(symbol.StartFileName).resolve(),
        function_start_line=symbol.StartLine,
    )
