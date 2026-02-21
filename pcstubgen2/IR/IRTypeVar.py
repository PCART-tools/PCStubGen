from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .IRAnnotation import IRAnnotation


@dataclass
class IRTypeVar:
    name: str
    constraints: list[IRAnnotation] = field(default_factory=list)
    bound: IRAnnotation | None = field(default=None)
    covariant: bool = field(default=False)
    contravariant: bool = field(default=False)

    def __str__(self) -> str:
        sb = [f'{self.name} = typing.TypeVar("{self.name}"']

        for constraint in self.constraints:
            sb.append(f", {constraint}")
        
        if self.bound is not None:
            sb.append(f", bound={self.bound}")
        
        if self.covariant:
            sb.append(f", covariant={self.covariant}")
        
        if self.contravariant:
            sb.append(f", contravariant={self.contravariant}")
        
        sb.append(")")
        return "".join(sb)
