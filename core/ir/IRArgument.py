from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .InvalidExpression import InvalidExpression
    from .IRValue import IRValue


class IRArgumentKind(Enum):
    POSITIONAL_ONLY = auto()  # 仅限位置参数
    POSITIONAL_OR_KEYWORD = auto()  # 位置或关键字参数
    VAR_POSITIONAL = auto()  # *args 可变位置参数
    KEYWORD_ONLY = auto()  # 仅限关键字参数
    VAR_KEYWORD = auto()  # **kwargs 可变关键字参数


@dataclass
class IRArgument:
    name: str | None
    kind: IRArgumentKind = field(default=IRArgumentKind.POSITIONAL_OR_KEYWORD)
    default: IRValue | InvalidExpression | None = field(default=None)
    annotation: str | None = field(default=None)

    def __str__(self) -> str:
        result = []
        if self.kind is IRArgumentKind.VAR_POSITIONAL:
            result.append("*")
        elif self.kind is IRArgumentKind.VAR_KEYWORD:
            result.append("**")

        result.append(f"{self.name}")
        if self.annotation:
            result.append(f": {self.annotation}")
        if self.default:
            result.append(f" = {self.default}")

        return "".join(result)
