from __future__ import annotations

from pathlib import Path

from loguru import logger

from ...models import Function, Module, Signature
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
        module_node: Module,
        function_node: Function,
    ) -> tuple[list[Signature], str | None]:
        """按函数懒解析 builtin function 的 C 扩展签名。"""
        runtime_info = read_builtin_function_runtime_info(function_node.runtime_handle)
        location = get_func_file_location(runtime_info.address)
        tu = self._clang_parser.get_translation_unit(location.compilation_unit_path)
        func_cursor = get_func_cursor(tu, location.function_name, location.linkage_name)

        signatures = inference.infer_signature(
            func_cursor,
            ml_flags=runtime_info.flags,
        )
        comment = None
        try:
            location_text = str(func_cursor.location)
            source_text = cursor_get_text(func_cursor)
            comment = f"{location_text}\n{source_text}"
        except RuntimeError as ex:
            logger.warning(
                "读取函数源码注释失败, module: {}, func: {}, reason: {}",
                module_node.full_name,
                function_node.name,
                ex,
            )

        if not signatures:
            raise RuntimeError(f"C函数 {module_node.full_name}.{function_node.name} 没有可用签名")

        return signatures, comment
