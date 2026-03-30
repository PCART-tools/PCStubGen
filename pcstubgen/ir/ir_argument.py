from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from ..c_signature.types import Type


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

    def __str__(self) -> str:
        result = []
        if self.kind is IRArgumentKind.VAR_POSITIONAL:
            result.append("*")
        elif self.kind is IRArgumentKind.VAR_KEYWORD:
            result.append("**")

        result.append(f"{self.name}")
        if self.type is not None:
            result.append(f": {self.type.render()}")
        if self.default_value is not None:
            result.append(f" = {self.default_value}")
        elif self.has_default:
            result.append(" = ...")

        return "".join(result)
