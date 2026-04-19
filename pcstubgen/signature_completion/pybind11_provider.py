from __future__ import annotations

from pcstubgen import runtime

from .completion_models import (
    SignatureCompletionContext,
    SignatureCompletionResult,
)
from .docstring_source import parse_docstring_signature_text
from . import producers
from ..models import Decorator


class Pybind11Provider:
    """从 pybind11 docstring 生产最终可导出的结果。"""

    @staticmethod
    def support(member: object, is_method: bool) -> bool:
        if is_method:
            return (
                runtime.is_pybind11_instance_method(member)
                or runtime.is_pybind11_static_method(member)
            )
        return runtime.is_pybind11_module_function(member)

    def get(self, context: SignatureCompletionContext) -> SignatureCompletionResult:
        runtime_handle, decorator, doc = self._analyze_member(context.member)
        signatures = parse_docstring_signature_text(context.func_name, doc)
        signatures = producers._finalize_signatures(
            signatures,
            is_method=context.is_method,
            decorator=decorator,
        )

        _ = runtime_handle
        return SignatureCompletionResult(
            success=True,
            message="",
            provider="pybind11",
            signatures=signatures,
            doc=doc,
            decorator=decorator,
        )

    def _analyze_member(
        self,
        member: object,
    ) -> tuple[object, Decorator, str | None]:
        if runtime.is_pybind11_module_function(member):
            return member, None, _get_doc(member)

        if runtime.is_pybind11_instance_method(member):
            return member, None, _get_doc(member)

        if isinstance(member, staticmethod) and runtime.is_pybind11_bound(member.__func__):
            return member.__func__, "staticmethod", _get_doc(member.__func__)

        raise RuntimeError(f"不支持的 pybind11 成员: {type(member).__name__}")


def _get_doc(obj: object) -> str | None:
    doc = getattr(obj, "__doc__", None)
    if isinstance(doc, str) and doc and not doc.isspace():
        return doc
    return None
