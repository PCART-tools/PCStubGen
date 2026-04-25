from __future__ import annotations

import typing

from pcstubgen import runtime

from .completion_models import (
    SignatureCompletionContext,
    SignatureCompletionResult,
)
from .docstring_source import parse_docstring_signature_text
from ..models import Decorator
from loguru import logger


class Pybind11Provider:
    """从 pybind11 docstring 生产最终可导出的结果。"""

    @staticmethod
    def support(member: object, is_method: bool) -> bool:
        if is_method:
            is_pybind11 = (
                runtime.is_pybind11_instance_method(member)
                or runtime.is_pybind11_static_method(member)
            )
        else:
            is_pybind11 = runtime.is_pybind11_module_function(member)
        return is_pybind11 and not _is_internal_pybind11_member(member)

    def get(self, context: SignatureCompletionContext) -> SignatureCompletionResult:
        runtime_handle, decorator, doc = self._analyze_member(context.member)
        signatures = parse_docstring_signature_text(context.func_name, doc)

        _ = runtime_handle
        return SignatureCompletionResult(
            signatures=signatures,
            doc=doc,
            decorator=decorator,
        )

    def _analyze_member(
        self,
        member: typing.Any,
    ) -> tuple[object, Decorator, str | None]:
        if runtime.is_pybind11_module_function(member):
            return member, None, _get_doc(member)

        if runtime.is_pybind11_instance_method(member):
            return member, None, _get_doc(member)

        if runtime.is_pybind11_static_method(member):
            return member.__func__, "staticmethod", _get_doc(member.__func__)

        raise RuntimeError(f"不支持的 pybind11 成员: {type(member).__name__}")


def _get_doc(obj: object) -> str | None:
    doc = getattr(obj, "__doc__", None)
    if isinstance(doc, str) and doc and not doc.isspace():
        return doc
    return None


def _is_internal_pybind11_member(member: object) -> bool:
    """判断成员是否为不应导出的 pybind11 内部符号。不是用户代码导出的，是pybind11的内部互操作函数"""
    try:
        return getattr(member, "__name__", None) == "_pybind11_conduit_v1_"
    except Exception as ex:
        logger.exception(ex)
        return False
