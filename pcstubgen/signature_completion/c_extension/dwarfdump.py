from __future__ import annotations

from dataclasses import dataclass
import functools
from pathlib import Path
from typing import Callable

from . import _dwarfdump


@dataclass(frozen=True)
class LookupResult:
    compilation_unit_path: Path
    function_name: str
    linkage_name: str | None = None


class DWARFFile:
    """持有单个共享库的 DWARF 查询上下文。"""

    def __init__(self, binary_path: Path) -> None:
        """打开共享库并创建可复用的 native DWARF 查询对象。"""
        self._file = _dwarfdump.DWARFFile(str(binary_path.resolve()))

    def lookup(self, relative_address: int) -> LookupResult:
        """按共享库内相对地址查询 LLVM 识别到的编译单元与函数身份。"""
        compilation_unit_path, function_name, linkage_name = self._file.lookup(
            relative_address,
        )
        return LookupResult(
            compilation_unit_path=Path(compilation_unit_path).resolve(),
            function_name=function_name,
            linkage_name=linkage_name,
        )


class DWARFManager:
    """按共享库路径缓存 DWARFFile，并提供地址查询入口。"""

    def __init__(self, maxsize: int = 8) -> None:
        """创建实例级 DWARFFile LRU 缓存。"""
        self._get_file: Callable[[Path], DWARFFile] = functools.lru_cache(maxsize=maxsize)(
            self._open_file,
        )

    def lookup(self, binary_path: Path, relative_address: int) -> LookupResult:
        """复用已打开的 DWARFFile 查询共享库内相对地址。"""
        return self._get_file(binary_path.resolve()).lookup(relative_address)

    def _open_file(self, binary_path: Path) -> DWARFFile:
        """打开共享库对应的 DWARFFile。"""
        return DWARFFile(binary_path)
