from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from loguru import logger

from . import producers
from .c_extension.provider import CExtensionProvider
from .completion_models import (
    SignatureCompletionContext,
    SignatureCompletionResult,
    SignatureCompletionSummary,
)
from .minimal_provider import MinimalProvider
from .pybind11_provider import Pybind11Provider
from ..models import Function


class SignatureCompleter:
    """按单函数上下文补全签名并累计统计信息。"""

    def __init__(self, compilation_database: Path) -> None:
        self._c_extension_provider = CExtensionProvider(compilation_database)
        self._pybind11_provider = Pybind11Provider()
        self._minimal_provider = MinimalProvider()
        self._summary = SignatureCompletionSummary()

    def support(self, handle: object) -> bool:
        """判断运行时对象是否属于受支持的补全来源。"""
        return CExtensionProvider.support(handle) or Pybind11Provider.support(handle)

    @property
    def summary(self) -> SignatureCompletionSummary:
        """返回当前累计的签名补全统计。"""
        return replace(self._summary)

    def complete(
        self,
        context: SignatureCompletionContext,
    ) -> SignatureCompletionResult:
        """按单函数上下文补全签名。"""
        self._summary.total_functions += 1
        func = Function(
            name=context.name,
            handle=context.handle,
            doc=context.doc,
            decorator=context.decorator,
        )

        logger.info(
            "开始补全, module: {}, func: {}, is_method: {}",
            context.module_name,
            context.name,
            context.is_method,
        )
        try:
            if CExtensionProvider.support(context.handle):
                result = self._c_extension_provider.get(
                    func,
                    context.is_method,
                )
                self._summary.c_extension_completed += 1
                logger.info(
                    "补全成功, branch: c_extension, module: {}, func: {}, is_method: {}",
                    context.module_name,
                    context.name,
                    context.is_method,
                )
                return self._finalize_result(context, result)

            if Pybind11Provider.support(context.handle):
                result = self._pybind11_provider.get(
                    func,
                    context.is_method,
                )
                self._summary.pybind11_completed += 1
                logger.info(
                    "补全成功, branch: pybind11, module: {}, func: {}, is_method: {}",
                    context.module_name,
                    context.name,
                    context.is_method,
                )
                return self._finalize_result(context, result)

            raise RuntimeError("函数不属于受支持的签名补全来源。")
        except Exception as ex:
            reason = f"{type(ex).__name__}: {ex}"

        result = self._minimal_provider.get(
            func,
            context.is_method,
        )
        self._summary.failed += 1
        logger.warning(
            "补全失败, branch: minimal, module: {}, func: {}, is_method: {}, reason: {}",
            context.module_name,
            context.name,
            context.is_method,
            reason,
        )
        return self._finalize_result(context, result)

    def _finalize_result(
        self,
        context: SignatureCompletionContext,
        result: SignatureCompletionResult,
    ) -> SignatureCompletionResult:
        """应用方法接收者整形后返回最终结果。"""
        return SignatureCompletionResult(
            signatures=producers._finalize_signatures(
                result.signatures,
                is_method=context.is_method,
                decorator=context.decorator,
            ),
            comment=result.comment,
        )
