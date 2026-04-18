from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..models import Argument, ArgumentKind, Decorator, Function, Module, Signature
from ..runtime import is_cpython_builtin, read_cpython_function_runtime_info
from .c_extension.signatures.inference import infer_argument_lists_from_flags
from .c_extension.source import CExtensionSource
from .docstring_source import parse_docstring_signatures


@dataclass(frozen=True)
class SignatureProductionResult:
    """单个 producer 的签名生产结果。"""

    signatures: list[Signature]
    comment: str | None = None


class DocstringSignatureProducer:
    """从 docstring 直接生产最终可导出的签名。"""

    def produce(
        self,
        module: Module,
        func: Function,
        *,
        is_method: bool,
    ) -> SignatureProductionResult:
        signatures = parse_docstring_signatures(module, func)
        finalized = _finalize_signatures(signatures, is_method=is_method, decorator=func.decorator)
        return SignatureProductionResult(signatures=finalized)


class CExtensionSignatureProducer:
    """从 CPython C 扩展信息生产最终可导出的签名。"""

    def __init__(self, compilation_database: Path) -> None:
        self._source = CExtensionSource(compilation_database)

    def produce(
        self,
        module: Module,
        func: Function,
        *,
        is_method: bool,
    ) -> SignatureProductionResult:
        result = self._source.infer_function_signatures(module, func)
        finalized = _finalize_signatures(
            result.signatures,
            is_method=is_method,
            decorator=func.decorator,
        )
        return SignatureProductionResult(signatures=finalized, comment=result.comment)


class MinimalSignatureProducer:
    """在正式 producer 失败时生成最小可导出签名。"""

    def produce(
        self,
        module: Module,
        func: Function,
        *,
        is_method: bool,
    ) -> SignatureProductionResult:
        _ = module
        raw_signatures = self._build_raw_signatures(func)
        finalized = _finalize_signatures(raw_signatures, is_method=is_method, decorator=func.decorator)
        return SignatureProductionResult(signatures=finalized)

    def _build_raw_signatures(self, func: Function) -> list[Signature]:
        if is_cpython_builtin(func.runtime_handle):
            runtime_info = read_cpython_function_runtime_info(func.runtime_handle)
            argument_lists = infer_argument_lists_from_flags(runtime_info.flags)
            if argument_lists:
                return [Signature(args=arguments) for arguments in argument_lists]

        return [
            Signature(
                args=[
                    Argument(name="args", kind=ArgumentKind.VAR_POSITIONAL),
                    Argument(name="kwargs", kind=ArgumentKind.VAR_KEYWORD),
                ]
            )
        ]


def _finalize_signatures(
    signatures: list[Signature],
    *,
    is_method: bool,
    decorator: Decorator,
) -> list[Signature]:
    if not signatures:
        raise RuntimeError("producer 没有返回可用签名。")
    return [
        _finalize_signature(signature, is_method=is_method, decorator=decorator)
        for signature in signatures
    ]


def _finalize_signature(
    signature: Signature,
    *,
    is_method: bool,
    decorator: Decorator,
) -> Signature:
    if not is_method:
        return signature

    receiver_name: str | None = None
    if decorator is None:
        receiver_name = "self"
    elif decorator == "classmethod":
        receiver_name = "cls"

    if receiver_name is None:
        return _strip_receiver(signature)

    return _ensure_receiver(signature, receiver_name)


def _strip_receiver(signature: Signature) -> Signature:
    if signature.args and signature.args[0].name in {"self", "cls"}:
        return Signature(args=signature.args[1:], return_type=signature.return_type)
    return signature


def _ensure_receiver(signature: Signature, receiver_name: str) -> Signature:
    receiver = Argument(name=receiver_name)
    if not signature.args:
        return Signature(args=[receiver], return_type=signature.return_type)

    first_arg = signature.args[0]
    if first_arg.name == receiver_name:
        if (
            first_arg.type is None
            and first_arg.default_value is None
            and first_arg.kind is ArgumentKind.POSITIONAL_OR_KEYWORD
        ):
            return signature
        return Signature(args=[receiver, *signature.args[1:]], return_type=signature.return_type)

    if first_arg.name in {"self", "cls"}:
        return Signature(args=[receiver, *signature.args[1:]], return_type=signature.return_type)

    return Signature(args=[receiver, *signature.args], return_type=signature.return_type)
