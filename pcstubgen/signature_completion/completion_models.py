from __future__ import annotations

from dataclasses import dataclass

from ..models import Decorator, QualifiedName, Signature


class UnsupportedSignatureCompletion(Exception):
    """表示候选对象不应产出签名补全结果。"""


class PartialSignatureCompletionError(Exception):
    """表示已取得签名来源，但从来源推断签名失败。"""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        source_location: str | None = None,
        source_text: str | None = None,
    ) -> None:
        """保存失败原因和已取得的来源证据。"""
        super().__init__(message)
        self.provider = provider
        self.source_location = source_location
        self.source_text = source_text


@dataclass(frozen=True)
class SignatureCompletionContext:
    """单函数签名补全所需的收集上下文。"""

    module_name: QualifiedName
    func_name: str
    member: object
    owner_class: type | None = None


@dataclass(frozen=True)
class SignatureCompletionResult:
    """单个 callable 的补全结果。"""
    signatures: list[Signature]
    doc: str | None = None
    decorator: Decorator = None
    provider: str | None = None
    mapping_status: str = "unknown"
    parameter_inference_status: str = "unknown"
    return_inference_status: str = "unknown"
    failure_reason: str | None = None
    source_location: str | None = None
    source_text: str | None = None


@dataclass
class SignatureCompletionSummary:
    """一次收集流程内的签名补全统计。"""

    total: int = 0
    c_extension: int = 0
    pybind11: int = 0
    failed: int = 0

    def __str__(self) -> str:
        return (
            "签名补全结果: "
            f"函数总数={self.total}, "
            f"C扩展补全={self.c_extension}, "
            f"pybind11补全={self.pybind11}, "
            f"失败={self.failed}"
        )
