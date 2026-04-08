from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal, Tuple, TypeAlias

from .types import Type


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


IRMethodDecorator: TypeAlias = Literal["staticmethod", "classmethod"] | None


class IRArgumentKind(Enum):
    POSITIONAL_ONLY = auto()  # 仅限位置参数
    POSITIONAL_OR_KEYWORD = auto()  # 位置或关键字参数
    VAR_POSITIONAL = auto()  # *args 可变位置参数
    KEYWORD_ONLY = auto()  # 仅限关键字参数
    VAR_KEYWORD = auto()  # **kwargs 可变关键字参数


@dataclass
class IRArgument:
    name: str | None
    type: Type | None = field(default=None)
    default_value: str | None = field(default=None)
    has_default: bool = field(default=False)
    kind: IRArgumentKind = field(default=IRArgumentKind.POSITIONAL_OR_KEYWORD)


@dataclass
class IRSignature:
    """IR 中的单条函数签名。"""

    args: list[IRArgument] = field(default_factory=list)
    return_type: Type | None = field(default=None)


@dataclass
class IRFunction:
    """IR 中的函数节点。"""

    name: str
    signatures: list[IRSignature] = field(default_factory=list)
    doc: str | None = field(default=None)
    c_inferred_source_comment: str | None = field(default=None)
    runtime_handle: Any | None = field(default=None, repr=False, compare=False)


@dataclass
class IRMethod:
    function: IRFunction
    decorator: IRMethodDecorator
    runtime_owner: type | None = field(default=None, repr=False, compare=False)


@dataclass
class IRClass:
    name: str
    doc: str | None = field(default=None)
    bases: list[QualifiedName] = field(default_factory=list)
    classes: list[IRClass] = field(default_factory=list)
    methods: list[IRMethod] = field(default_factory=list)


class IRModuleType(Enum):
    UNKNOWN = "unknown"
    PYTHON = "python"
    BUILTIN = "builtin"
    EXTENSION = "extension"


@dataclass
class IRModule:
    full_name: QualifiedName

    # 文档
    doc: str | None = field(default=None)

    # 模块实现类型
    module_type: IRModuleType = field(default=IRModuleType.UNKNOWN)

    # 类
    classes: list[IRClass] = field(default_factory=list)

    # 函数
    functions: list[IRFunction] = field(default_factory=list)

    # 子模块
    sub_modules: list[IRModule] = field(default_factory=list)

    # 是否是包
    is_package: bool = field(default=False)


__all__ = [
    "IRArgument",
    "IRArgumentKind",
    "IRClass",
    "IRFunction",
    "IRMethod",
    "IRMethodDecorator",
    "IRModule",
    "IRModuleType",
    "IRSignature",
    "QualifiedName",
]
