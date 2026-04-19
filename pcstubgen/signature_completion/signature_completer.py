from __future__ import annotations

from pathlib import Path

from .c_extension.provider import CExtensionProvider
from .completion_models import (
    SignatureCompletionContext,
    SignatureCompletionResult,
)
from .minimal_provider import MinimalProvider
from .pybind11_provider import Pybind11Provider


class SignatureCompleter:

    def __init__(self, compilation_database: Path) -> None:
        self._c_extension_provider = CExtensionProvider(compilation_database)
        self._pybind11_provider = Pybind11Provider()
        self._minimal_provider = MinimalProvider()

    @staticmethod
    def support(handle: object) -> bool:
        """判断运行时对象是否属于受支持的补全来源。"""
        return CExtensionProvider.support(handle) or Pybind11Provider.support(handle)

    def complete(self, context: SignatureCompletionContext) -> SignatureCompletionResult:
        """补全签名。"""
        message = "函数不属于受支持的签名补全来源。"
        try:
            if CExtensionProvider.support(context.handle):
                signatures, comment = self._c_extension_provider.get(context)
                return SignatureCompletionResult(True, "", "c_extension", signatures, comment)

            if Pybind11Provider.support(context.handle):
                signatures, comment = self._pybind11_provider.get(context)
                return SignatureCompletionResult(True, "", "pybind11", signatures, comment)
        except Exception as ex:
            message = f"{type(ex).__name__}: {ex}"

        signatures, comment = self._minimal_provider.get(context)

        return SignatureCompletionResult(False, message, "minimal", signatures, comment)