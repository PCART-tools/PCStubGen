from __future__ import annotations

from dataclasses import dataclass

from ..models import Decorator, QualifiedName, Signature


@dataclass(frozen=True)
class SignatureCompletionContext:
    """单函数签名补全所需的收集上下文。"""

    path: QualifiedName
    handle: object
    doc: str | None = None
    decorator: Decorator = None
    is_method: bool = False

    @property
    def name(self) -> str:
        """返回函数短名。"""
        return self.path.name

    @property
    def module_name(self) -> QualifiedName:
        """返回函数所属模块名。"""
        return self.path.parent


@dataclass(frozen=True)
class SignatureCompletionResult:
    """单个 provider 的签名生产结果。"""

    success: bool
    signatures: list[Signature]
    comment: str | None = None


@dataclass
class SignatureCompletionSummary:
    """一次收集流程内的签名补全统计。"""

    total_functions: int = 0
    c_extension_completed: int = 0
    pybind11_completed: int = 0
    failed: int = 0

    def __str__(self) -> str:
        return (
            "签名补全结果: "
            f"函数总数={self.total_functions}, "
            f"C扩展补全={self.c_extension_completed}, "
            f"pybind11补全={self.pybind11_completed}, "
            f"失败={self.failed}"
        )
