from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .QualifiedName import QualifiedName

if TYPE_CHECKING:
    from .InvalidExpression import InvalidExpression
    from .IRValue import IRValue


@dataclass
class ResolvedType:
    name: QualifiedName
    parameters: list[ResolvedType | IRValue | InvalidExpression] | None = field(
        default=None
    )

    def __str__(self) -> str:
        if self.parameters:
            param_str = "[" + ", ".join(str(p) for p in self.parameters) + "]"
        else:
            param_str = ""
        return f"{self.name}{param_str}"
