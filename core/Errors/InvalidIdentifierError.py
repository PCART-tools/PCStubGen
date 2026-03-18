from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .ParserError import ParserError

if TYPE_CHECKING:
    from ..IR import QualifiedName


@dataclass
class InvalidIdentifierError(ParserError):
    """出现了无效的 Python 标识符。"""

    name: str
    path: QualifiedName

    def __str__(self) -> str:
        return f"Invalid identifier `{self.name}` at `{self.path}`"
