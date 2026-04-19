from __future__ import annotations

from pathlib import Path

from loguru import logger

from .c_extension.provider import CExtensionProvider
from .completion_models import (
    SignatureCompletionContext,
    SignatureCompletionResult,
    SignatureCompletionSummary,
)
from .minimal_provider import MinimalProvider
from .pybind11_provider import Pybind11Provider


class SignatureCompleter:
    """编排 provider 执行并在失败时回退到最小签名。"""

    def __init__(self, compilation_database: Path) -> None:
        self._c_extension_provider = CExtensionProvider(compilation_database)
        self._pybind11_provider = Pybind11Provider()
        self._minimal_provider = MinimalProvider()
        self.summary = SignatureCompletionSummary()

    def reset_summary(self) -> None:
        """重置本轮补全统计。"""
        self.summary = SignatureCompletionSummary()

    def support(self, member: object, is_method: bool) -> bool:
        """判断运行时对象是否属于受支持的补全来源。"""
        return (
            self._c_extension_provider.support(member, is_method)
            or self._pybind11_provider.support(member, is_method)
        )

    def complete(self, context: SignatureCompletionContext) -> SignatureCompletionResult:
        """补全单个 callable。"""
        message = "函数不属于受支持的签名补全来源。"
        provider = "minimal"
        try:
            if self._c_extension_provider.support(context.member, context.is_method):
                provider = "c_extension"
                result = self._c_extension_provider.get(context)
                self._record_result(context, result)
                return result

            if self._pybind11_provider.support(context.member, context.is_method):
                provider = "pybind11"
                result = self._pybind11_provider.get(context)
                self._record_result(context, result)
                return result

        except Exception as ex:
            message = f"{type(ex).__name__}: {ex}"

        result = SignatureCompletionResult(
            success=False,
            message=message,
            provider=provider,
            signatures=self._minimal_provider.get(context),
        )
        self._record_result(context, result)
        return result

    def _record_result(
        self,
        context: SignatureCompletionContext,
        result: SignatureCompletionResult,
    ) -> None:
        self.summary.total += 1
        if result.provider == "c_extension":
            self.summary.c_extension += 1
        elif result.provider == "pybind11":
            self.summary.pybind11 += 1

        if result.success:
            logger.info(
                "补全成功, provider: {}, module: {}, func: {}, is_method: {}",
                result.provider,
                context.module_name,
                context.func_name,
                context.is_method,
            )
            return

        self.summary.failed += 1
        logger.warning(
            "补全失败, provider: {}, module: {}, func: {}, is_method: {}, message: {}",
            result.provider,
            context.module_name,
            context.func_name,
            context.is_method,
            result.message,
        )
