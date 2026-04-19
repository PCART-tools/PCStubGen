from __future__ import annotations

from pathlib import Path

from loguru import logger

from .c_extension.provider import CExtensionProvider
from .completion_models import (
    SignatureCompletionContext,
    SignatureCompletionResult,
    SignatureCompletionSummary,
    SignatureProviderError,
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

    def support(self, member: object) -> bool:
        """判断运行时对象是否属于受支持的补全来源。"""
        return self._pick_provider(member) is not None

    def complete(self, context: SignatureCompletionContext) -> SignatureCompletionResult:
        """补全单个 callable。"""
        provider_name, provider = self._pick_provider(context.member)
        if provider is None:
            result = SignatureCompletionResult(
                success=False,
                message="函数不属于受支持的签名补全来源。",
                provider="minimal",
                signatures=self._minimal_provider.get(context),
            )
            self._record_result(context, result)
            return result

        try:
            result = provider.get(context)
        except SignatureProviderError as ex:
            cause = ex.__cause__ if ex.__cause__ is not None else ex
            result = SignatureCompletionResult(
                success=False,
                message=f"{type(cause).__name__}: {cause}",
                provider=provider_name,
                signatures=self._minimal_provider.get(
                    context,
                    decorator=ex.decorator,
                ),
                doc=ex.doc,
                decorator=ex.decorator,
                comment=ex.comment,
            )
        except Exception as ex:
            result = SignatureCompletionResult(
                success=False,
                message=f"{type(ex).__name__}: {ex}",
                provider=provider_name,
                signatures=self._minimal_provider.get(context),
            )

        self._record_result(context, result)
        return result

    def _pick_provider(self, member: object) -> tuple[str, object] | tuple[None, None]:
        if self._c_extension_provider.support(member):
            return "c_extension", self._c_extension_provider

        if self._pybind11_provider.support(member):
            return "pybind11", self._pybind11_provider

        return None, None

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
