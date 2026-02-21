from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InvalidExpression:
    text: str

    def __str__(self) -> str:
        return f"Invalid python expression `{self.text}`"
