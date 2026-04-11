from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import _dwarfdump


@dataclass(frozen=True)
class LookupResult:
    compilation_unit_path: Path
    function_name: str
    linkage_name: str | None = None


def lookup(binary_path: Path, relative_address: int) -> LookupResult:
    """按共享库内相对地址查询 LLVM 识别到的编译单元与函数身份。"""
    compilation_unit_path, function_name, linkage_name = _dwarfdump.lookup(
        str(binary_path),
        relative_address,
    )
    return LookupResult(
        compilation_unit_path=Path(compilation_unit_path).resolve(),
        function_name=function_name,
        linkage_name=linkage_name,
    )
