from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IRValue:
    repr: str
    is_print_safe: bool = False  # `self.repr` 是有效的 Python 代码，可按原样安全打印

    def __str__(self) -> str:
        return self.repr
