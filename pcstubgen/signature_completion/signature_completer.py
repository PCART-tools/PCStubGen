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

    def support(
        self,
        member: object,
        owner_class: type | None = None,
    ) -> bool:
        """判断运行时对象是否属于受支持的补全来源。"""
        return (
            self._c_extension_provider.support(member, owner_class)
            or self._pybind11_provider.support(member, owner_class)
        )

    def complete(self, context: SignatureCompletionContext) -> SignatureCompletionResult:
        """补全单个 callable。"""
        self.summary.total += 1
        provider = "minimal"
        reason = "函数不属于受支持的签名补全来源。"
        try:
            if self._c_extension_provider.support(context.member, context.owner_class):
                provider = "c_extension"
                result = self._c_extension_provider.get(context)
                self.summary.c_extension += 1
                _log_success(context, provider)
                return result

            if self._pybind11_provider.support(context.member, context.owner_class):
                provider = "pybind11"
                result = self._pybind11_provider.get(context)
                self.summary.pybind11 += 1
                _log_success(context, provider)
                return result

        except Exception as ex:
            reason = f"{ex!r}"

        self.summary.failed += 1
        result = self._minimal_provider.get(context)
        _log_failure(context, provider, reason)
        return result

def _log_success(context: SignatureCompletionContext, provider: str) -> None:
    logger.info(
        "补全成功, provider: {}, module: {}, func: {}, owner_class: {}",
        provider,
        context.module_name,
        context.func_name,
        context.owner_class,
    )

def _log_failure(context: SignatureCompletionContext, provider: str, reason: str) -> None:
    logger.warning(
        "补全失败, provider: {}, module: {}, func: {}, owner_class: {}, reason: {}",
        provider,
        context.module_name,
        context.func_name,
        context.owner_class,
        reason,
    )
