from __future__ import annotations

import typing
from pathlib import Path

from clang.cindex import Cursor
from pcstubgen import runtime
from . import dladdr, dwarfdump
from .libclang import ast_utils
from .libclang.function_cursor_locator import FunctionCursorLocator
from .signatures.inferencer import Inferencer
from ..completion_models import (
    SignatureCompletionContext,
    SignatureCompletionResult,
    UnsupportedSignatureCompletion,
)
from ...models import Decorator


class CExtensionProvider:
    """通过 C 扩展运行时信息补全签名。"""

    def __init__(self, compilation_database: Path) -> None:
        self._function_cursor_locator = FunctionCursorLocator(compilation_database)
        self._dwarf_manager = dwarfdump.DWARFManager()

    @staticmethod
    def match(
        member: object,
        owner_class: type | None = None,
    ) -> bool:
        """判断运行时对象是否匹配 CPython C 扩展 provider。"""
        if owner_class is not None:
            if (
                runtime.is_c_extension_instance_method(member)
                or runtime.is_c_extension_class_method(member)
            ):
                return not (
                    _is_foreign_method_descriptor(member, owner_class)
                    or _is_cython_pickle_method_descriptor(member, owner_class)
                )
            if runtime.is_c_extension_static_method(member):
                return True
            return False
        return runtime.is_c_extension_module_function(member)

    def get_func_cursor_and_flags(self, handle: object) -> tuple[Cursor, int]:
        """根据运行时句柄反查函数 cursor 和调用 flags。"""
        runtime_info = runtime.read_c_extension_function_runtime_info(handle)
        binary_path, ra = dladdr.get_binary_and_ra(runtime_info.address)
        lookup_result = self._dwarf_manager.lookup(binary_path, ra)
        func_cursor = self._function_cursor_locator.get_function_cursor(
            lookup_result.compilation_unit_path,
            lookup_result.function_name,
            lookup_result.linkage_name,
        )
        return func_cursor, runtime_info.flags

    def get(self, context: SignatureCompletionContext) -> SignatureCompletionResult:
        """为单个函数执行 C 源码签名推断。"""
        runtime_handle, decorator, doc = self._analyze_member(context.member)
        func_cursor, flags = self.get_func_cursor_and_flags(runtime_handle)
        if _is_pythran_wrapall_cursor(func_cursor):
            raise UnsupportedSignatureCompletion(
                f"跳过 Pythran wrapall 分派函数: {func_cursor.spelling}"
            )
        source_text = ast_utils.get_cursor_source_text(func_cursor)
        comment = f"{func_cursor.location}\n{source_text}"
        signatures = Inferencer(func_cursor, flags, context.owner_class).run()

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


def _is_foreign_method_descriptor(member: object, owner_class: type) -> bool:
    """
    判断已确认的 C method descriptor 是否归属于其他类。

    过滤这个噪音是为了避免 `EnumMeta` 等元类把外部 descriptor 复制到
    类字典后被误认为当前扩展类方法，例如 `IntEnum` 子类中的 `int.__format__
    IntEnum子类会被注入int的__format__，实现在Python内，后续DWARF没有调试符号`。
    """
    return getattr(member, "__objclass__", None) is not owner_class


def _is_cython_pickle_method_descriptor(member: object, owner_class: type) -> bool:
    """判断方法描述符是否为 Cython 自动生成的 pickle 辅助方法。"""
    return (
        getattr(member, "__objclass__", None) is owner_class
        and getattr(member, "__name__", None)
        in {"__reduce_cython__", "__setstate_cython__"}
    )


def _is_pythran_wrapall_cursor(func_cursor: Cursor) -> bool:
    """判断函数 cursor 是否为 Pythran 生成的 wrapall 分派入口。"""
    return func_cursor.spelling.startswith("__pythran_wrapall")
