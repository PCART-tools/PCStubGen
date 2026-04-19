from __future__ import annotations

from dataclasses import dataclass

from ..models import Decorator, QualifiedName, Signature


@dataclass(frozen=True)
class SignatureCompletionContext:
    """单函数签名补全所需的收集上下文。"""

    module_name: QualifiedName
    func_name: str
    member: object
    is_method: bool = False


@dataclass(frozen=True)
class SignatureCompletionResult:
    """单个 callable 的补全结果。"""

    success: bool
    message: str
    provider: str
    signatures: list[Signature]
    doc: str | None = None
    decorator: Decorator = None
    comment: str | None = None


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
            f"C扩展命中={self.c_extension}, "
            f"pybind11命中={self.pybind11}, "
            f"失败={self.failed}"
        )


class SignatureProviderError(RuntimeError):
    """provider 处理失败时携带已完成产物。"""

    def __init__(
        self,
        *,
        doc: str | None = None,
        decorator: Decorator = None,
        comment: str | None = None,
    ) -> None:
        super().__init__("provider 处理失败")
        self.doc = doc
        self.decorator = decorator
        self.comment = comment
