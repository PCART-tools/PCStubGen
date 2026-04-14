from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal, Tuple, TypeAlias

from .type_models import Type


class QualifiedName(Tuple[str, ...]):
    """完全限定名称"""

    @classmethod
    def from_str(cls, name: str) -> QualifiedName:
        return QualifiedName(name.split("."))

    def __str__(self) -> str:
        return ".".join(self)

    def concat(self, name: str) -> QualifiedName:
        return QualifiedName((*self, name))

    @property
    def parent(self) -> QualifiedName:
        return QualifiedName(self[:-1])

    @property
    def name(self) -> str:
        return self[-1]


MethodDecorator: TypeAlias = Literal["staticmethod", "classmethod"] | None


class ArgumentKind(Enum):
    POSITIONAL_ONLY = auto()  # 仅限位置参数
    POSITIONAL_OR_KEYWORD = auto()  # 位置或关键字参数
    VAR_POSITIONAL = auto()  # *args 可变位置参数
    KEYWORD_ONLY = auto()  # 仅限关键字参数
    VAR_KEYWORD = auto()  # **kwargs 可变关键字参数


@dataclass
class Argument:
    name: str
    type: Type | None = None
    default_value: str | None = None
    kind: ArgumentKind = ArgumentKind.POSITIONAL_OR_KEYWORD


@dataclass
class Signature:
    """模型中的单条函数签名。"""

    args: list[Argument] = field(default_factory=list)
    return_type: Type | None = None


@dataclass
class Function:
    """模型中的函数节点。"""

    name: str
    runtime_handle: Any = field(repr=False, compare=False)
    signatures: list[Signature] = field(default_factory=list)
    doc: str | None = None
    comment: str | None = None


@dataclass
class Method:
    function: Function
    decorator: MethodDecorator
    runtime_owner: type | None = field(default=None, repr=False, compare=False)


@dataclass
class Class:
    name: str
    doc: str | None = None
    bases: list[QualifiedName] = field(default_factory=list)
    classes: list[Class] = field(default_factory=list)
    methods: list[Method] = field(default_factory=list)


@dataclass
class Module:
    full_name: QualifiedName

    # 文档
    doc: str | None = None

    # 类
    classes: list[Class] = field(default_factory=list)

    # 函数
    functions: list[Function] = field(default_factory=list)

    # 子模块
    sub_modules: list[Module] = field(default_factory=list)

    # 是否是包
    is_package: bool = False


__all__ = [
    "Argument",
    "ArgumentKind",
    "Class",
    "Function",
    "Method",
    "MethodDecorator",
    "Module",
    "Signature",
    "QualifiedName",
]
