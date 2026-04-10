from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

try:
    from . import _dwarfdump_llvm
except ImportError as ex:
    _LOOKUP_IMPORT_ERROR: ImportError | None = ex
    _lookup_raw: Callable[[str, int], tuple[str, str, str | None]] | None = None
else:
    _LOOKUP_IMPORT_ERROR = None
    _lookup_raw = _dwarfdump_llvm.lookup_raw


@dataclass(frozen=True)
class LookupResult:
    compilation_unit_path: Path
    function_name: str
    linkage_name: str | None = None


def lookup(binary_path: Path, relative_address: int) -> LookupResult:
    """按共享库内相对地址查询 LLVM 识别到的编译单元与函数身份。"""
    lookup_raw = _lookup_raw
    if lookup_raw is None:
        raise RuntimeError("无法导入 LLVM dwarfdump 扩展。") from _LOOKUP_IMPORT_ERROR

    compilation_unit_path, function_name, linkage_name = lookup_raw(
        str(binary_path),
        relative_address,
    )
    return LookupResult(
        compilation_unit_path=Path(compilation_unit_path).resolve(),
        function_name=function_name,
        linkage_name=linkage_name,
    )
