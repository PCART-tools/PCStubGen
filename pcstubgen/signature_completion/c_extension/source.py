from __future__ import annotations

from pathlib import Path

from loguru import logger

from ...ir_modules import IRFunction, IRModule, IRSignature
from .address_resolver import (
    get_func_file_location,
)
from .clang.ast_utils import cursor_get_text, get_func_cursor
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

        signatures = inference.infer_signature(
            func_cursor,
            ml_flags=runtime_info.flags,
        )
        source_comment = None
        try:
            source_comment = cursor_get_text(func_cursor)
        except RuntimeError as ex:
            logger.warning(
                "读取函数源码注释失败, module: {}, func: {}, reason: {}",
                irmodule.full_name,
                irfunction.name,
                ex,
            )

        if not signatures:
            raise RuntimeError(f"C函数 {irmodule.full_name}.{irfunction.name} 没有可用签名")

        return signatures, source_comment
