from __future__ import annotations

import typing
from pathlib import Path

from clang.cindex import Cursor
from pcstubgen import runtime
from . import dladdr, dwarfdump
from .libclang import ast_utils
from .libclang.parser import ClangFunctionLocator
from .signatures.inferencer import Inferencer
from ..completion_models import (
    SignatureCompletionContext,
    SignatureCompletionResult,
)
from ...models import Decorator


class CExtensionProvider:
    """通过 C 扩展运行时信息补全签名。"""

    def __init__(self, compilation_database: Path) -> None:
        self._clang_function_locator = ClangFunctionLocator(compilation_database)

    @staticmethod
    def support(member: object, is_method: bool) -> bool:
        """判断运行时对象是否属于 CPython C 扩展函数或方法。"""
        if is_method:
            return (
                runtime.is_c_extension_instance_method(member)
                or runtime.is_c_extension_static_method(member)
                or runtime.is_c_extension_class_method(member)
            )
        return runtime.is_c_extension_module_function(member)

    def get_func_cursor_and_flags(self, handle: object) -> tuple[Cursor, int]:
        """根据运行时句柄反查函数 cursor 和调用 flags。"""
        runtime_info = runtime.read_c_extension_function_runtime_info(handle)
        binary_path, ra = dladdr.get_binary_and_ra(runtime_info.address)
        lookup_result = dwarfdump.lookup(binary_path, ra)
        func_cursor = self._clang_function_locator.get_function_cursor(
            lookup_result.compilation_unit_path,
            lookup_result.function_name,
            lookup_result.linkage_name,
        )
        return func_cursor, runtime_info.flags

    def get(self, context: SignatureCompletionContext) -> SignatureCompletionResult:
        """为单个函数执行 C 源码签名推断。"""
        runtime_handle, decorator, doc = self._analyze_member(context.member)
        func_cursor, flags = self.get_func_cursor_and_flags(runtime_handle)
        source_text = ast_utils.get_cursor_source_text(func_cursor)
        comment = f"{func_cursor.location}\n{source_text}"
        signatures = Inferencer(func_cursor, flags, context.is_method).run()

        return SignatureCompletionResult(
            signatures=signatures,
            doc=doc,
            decorator=decorator,
            comment=comment,
        )

    def _analyze_member(
        self,
        member: typing.Any,
    ) -> tuple[object, Decorator, str | None]:
        if runtime.is_c_extension_module_function(member):
            return member, None, _get_doc(member)

        if runtime.is_c_extension_instance_method(member):
            return member, None, _get_doc(member)

        if runtime.is_c_extension_class_method(member):
            return member, "classmethod", _get_doc(member)

        if runtime.is_c_extension_static_method(member):
            return member.__func__, "staticmethod", _get_doc(member.__func__)

        raise RuntimeError(f"不支持的 C 扩展成员: {type(member).__name__}")


def _get_doc(obj: object) -> str | None:
    doc = getattr(obj, "__doc__", None)
    if isinstance(doc, str) and doc and not doc.isspace():
        return doc
    return None
