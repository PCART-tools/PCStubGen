from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .ParserError import ParserError

if TYPE_CHECKING:
    from ..ir import QualifiedName


@dataclass
class NameResolutionError(ParserError):
    """名称解析失败。"""

    name: QualifiedName

    def __str__(self) -> str:
        return f"Can't resolve `{self.name}`"
