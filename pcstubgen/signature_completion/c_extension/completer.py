from __future__ import annotations

import sys
import typing
from pathlib import Path

from clang.cindex import Cursor
from pcstubgen import runtime
from . import dladdr, dwarfdump
from .inferencer import Inferencer
from .libclang import ast_utils
from .libclang.function_cursor_locator import FunctionCursorLocator
from ..completion_models import (
    PartialSignatureCompletionError,
    SignatureCompletionContext,
    SignatureCompletionResult,
    UnsupportedSignatureCompletion,
)
from ...models import Decorator


class CExtensionCompleter:
    """通过 C 扩展运行时信息补全签名。"""

    def __init__(self, compilation_database: Path) -> None:
        self._function_cursor_locator = FunctionCursorLocator(compilation_database)
        self._dwarf_manager = dwarfdump.DWARFManager()

    @staticmethod
    def match(
        member: object,
        owner_class: type | None = None,
    ) -> bool:
        """判断运行时对象是否匹配 CPython C 扩展 completer。"""
        if owner_class is not None:
            if (
                runtime.is_c_extension_instance_method(member)
                or runtime.is_c_extension_class_method(member)
            ):
                return not _is_cython_pickle_method_descriptor(member, owner_class)
            if runtime.is_c_extension_static_method(member):
                return True
            return False
        return runtime.is_c_extension_module_function(member)

    def get_func_cursor_and_flags(self, handle: object) -> tuple[Cursor, int]:
        """根据运行时句柄反查函数 cursor 和调用 flags。"""
        runtime_info = runtime.read_c_extension_function_runtime_info(handle)
        binary_path, ra = dladdr.get_binary_and_ra(runtime_info.address)

        if binary_path.samefile(Path(sys.executable)):
            """代码在Python内实现，没有调试符号，跳过"""
            raise UnsupportedSignatureCompletion(
                f"跳过实现落在当前解释器中的方法描述符: {handle.__name__}"
            )

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
            """不是我们关心的，属于噪音"""
            raise UnsupportedSignatureCompletion(
                f"跳过 Pythran wrapall 分派函数: {func_cursor.spelling}"
        )
        source_text = ast_utils.get_cursor_source_text(func_cursor)
        source_location = str(func_cursor.location)
        inferencer = Inferencer(func_cursor, flags, context.owner_class)
        try:
            signatures = inferencer.run()
        except Exception as ex:
            raise PartialSignatureCompletionError(
                f"{ex!r}",
                provider="c_extension",
                source_location=source_location,
                source_text=source_text,
            ) from ex

        return SignatureCompletionResult(
            signatures=signatures,
            doc=doc,
            decorator=decorator,
            provider="c_extension",
            mapping_status="success",
            parameter_inference_status=_status_from_bool(inferencer.parameter_inference_success),
            return_inference_status=_status_from_bool(inferencer.return_inference_success),
            source_location=source_location,
            source_text=source_text,
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


def _status_from_bool(success: bool) -> str:
    """将布尔推断结果转换为实验状态文本。"""
    return "success" if success else "failed"


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
