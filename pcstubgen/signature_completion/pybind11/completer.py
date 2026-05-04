from __future__ import annotations

import typing

from pcstubgen import runtime
from loguru import logger

from ..completion_models import (
    SignatureCompletionContext,
    SignatureCompletionResult,
)
from .inferencer import parse_pybind11_signature
from .runtime_introspection import extract_pybind11_signatures
from ...models import Decorator, Signature


class Pybind11Completer:
    """从 pybind11 runtime overload chain 生产最终可导出的结果。"""

    @staticmethod
    def match(member: object, owner_class: type | None = None) -> bool:
        """判断运行时对象是否匹配 pybind11 completer。"""
        if owner_class is not None:
            is_pybind11 = (
                runtime.is_pybind11_instance_method(member)
                or runtime.is_pybind11_static_method(member)
            )
        else:
            is_pybind11 = runtime.is_pybind11_module_function(member)
        return is_pybind11 and not _is_internal_pybind11_member(member)

    def get(self, context: SignatureCompletionContext) -> SignatureCompletionResult:
        runtime_handle, decorator, doc = self._analyze_member(context.member)
        raw_signatures = extract_pybind11_signatures(runtime_handle)

        signatures = self._parse_signatures(raw_signatures)
        if not signatures:
            raise RuntimeError("pybind11 单签名解析全部失败。")

        return SignatureCompletionResult(
            signatures=signatures,
            doc=doc,
            decorator=decorator,
        )

    def _parse_signatures(self, raw_signatures: list[str]) -> list[Signature]:
        signatures: list[Signature] = []
        for index, raw_signature in enumerate(raw_signatures, start=1):
            try:
                signature = parse_pybind11_signature(raw_signature)
            except Exception as ex:
                logger.warning(
                    "pybind11 单签名解析失败, overload: {}, raw_signature: {!r}, reason: {!r}",
                    index,
                    raw_signature,
                    ex,
                )
                continue

            signature.comment = _build_signature_comment(raw_signature)
            signatures.append(signature)

        return signatures

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


def _build_signature_comment(raw_signature: str) -> str:
    """为单条 pybind11 overload 构建调试注释。"""
    return f"pybind11\n{raw_signature}"
