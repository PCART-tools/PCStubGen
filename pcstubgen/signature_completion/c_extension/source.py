from __future__ import annotations

from pathlib import Path

from ...ir_modules import IRFunction, IRModule, IRSignature
from .address_resolver import (
    get_func_file_location,
)
from .clang.cursor_utils import get_func_cursor, source_range_get_text
from .clang.parser import ClangParser
from .runtime import read_builtin_function_runtime_info
from .signatures import inference


class CExtensionSource:
    def __init__(
        self,
        compilation_database: Path,
    ) -> None:
        self._clang_parser = ClangParser(compilation_database)

    def infer_function_signatures(
        self,
        irmodule: IRModule,
        irfunction: IRFunction,
    ) -> tuple[list[IRSignature], str | None]:
        """按函数懒解析 builtin function 的 C 扩展签名。"""
        runtime_info = read_builtin_function_runtime_info(irfunction.runtime_handle)
        location = get_func_file_location(runtime_info.address)
        tu = self._clang_parser.get_translation_unit(location.compilation_unit_path)
        func_cursor = get_func_cursor(tu, location.function_name, location.linkage_name)
        source_comment = source_range_get_text(func_cursor.extent)
        signatures = inference.infer_signature(
            func_cursor,
            ml_flags=runtime_info.flags,
        )

        if not signatures:
            raise RuntimeError(f"C函数 {irmodule.full_name}.{irfunction.name} 没有可用签名")

        return signatures, source_comment
