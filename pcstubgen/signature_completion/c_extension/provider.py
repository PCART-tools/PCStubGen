from __future__ import annotations

from pathlib import Path

from clang.cindex import Cursor

from pcstubgen import runtime

from .libclang import ast_utils
from .libclang.parser import ClangParser
from ..completion_models import SignatureCompletionResult
from ...models import Function
from . import dladdr, dwarfdump
from .signatures.inferencer import Inferencer

class CExtensionProvider:
    """通过 C 扩展运行时信息补全签名。"""

    def __init__(self, compilation_database: Path) -> None:
        self._clang_parser = ClangParser(compilation_database)

    @staticmethod
    def support(handle: object) -> bool:
        """判断运行时对象是否属于 CPython C 扩展。"""
        return runtime.is_cpython_builtin(handle)

    def get_func_cursor_and_flags(self, func: Function) -> tuple[Cursor, int]:
        """根据运行时句柄反查函数 cursor 和调用 flags。"""
        runtime_info = runtime.read_cpython_function_runtime_info(func.handle)
        binary_path, ra = dladdr.get_binary_and_ra(runtime_info.address)
        lookup_result = dwarfdump.lookup(binary_path, ra)
        tu = self._clang_parser.get_translation_unit(lookup_result.compilation_unit_path)
        func_cursor = ast_utils.get_func_cursor(
            tu,
            lookup_result.function_name,
            lookup_result.linkage_name,
        )
        return func_cursor, runtime_info.flags

    def get(
        self,
        func: Function,
        is_method: bool,
    ) -> SignatureCompletionResult:
        """为单个函数执行 C 源码签名推断。"""
        func_cursor, flags = self.get_func_cursor_and_flags(func)
        signatures = Inferencer(func_cursor, flags, is_method).run()
        source_text = ast_utils.get_cursor_source_text(func_cursor)
        comment = f"{func_cursor.location}\n{source_text}"
        return SignatureCompletionResult(signatures, comment)
