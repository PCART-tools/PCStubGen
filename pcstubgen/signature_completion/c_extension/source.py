from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...models import Function, Module, Signature
from .address_resolver import (
    get_func_file_location,
)
from .clang.ast_utils import get_cursor_text, get_func_cursor
from .clang.parser import ClangParser
from pcstubgen.runtime import read_builtin_function_runtime_info
from .signatures import inference


@dataclass(frozen=True)
class CInferenceResult:
    """C 扩展函数签名推断结果。"""

    signatures: list[Signature]
    comment: str


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
    ) -> CInferenceResult:
        """按函数懒解析 builtin function 的 C 扩展签名。"""
        runtime_info = read_builtin_function_runtime_info(function_node.runtime_handle)
        location = get_func_file_location(runtime_info.address)
        tu = self._clang_parser.get_translation_unit(location.compilation_unit_path)
        func_cursor = get_func_cursor(tu, location.function_name, location.linkage_name)
        location_text = str(func_cursor.location)
        signatures = inference.infer_signature(
            func_cursor,
            flags=runtime_info.flags,
        )
        source_text = get_cursor_text(func_cursor)
        comment = f"{location_text}\n{source_text}"

        if not signatures:
            raise RuntimeError(
                f"C函数 {module_node.full_name}.{function_node.name} 没有可用签名, cursor: {location_text}"
            )

        return CInferenceResult(signatures=signatures, comment=comment)
