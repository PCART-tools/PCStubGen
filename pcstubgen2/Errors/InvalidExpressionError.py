from __future__ import annotations

from dataclasses import dataclass

from .ParserError import ParserError


@dataclass
class InvalidExpressionError(ParserError):
    """注解或值中出现了无效的 Python 表达式。"""

    expression: str

    def __str__(self) -> str:
        return f"Invalid expression `{self.expression}`"
