from __future__ import annotations

from dataclasses import dataclass, field

from .IRArgument import IRArgument


@dataclass
class IRSignature:
    """IR 中的单条函数签名。"""

    args: list[IRArgument] = field(default_factory=list)
    return_type_name: str | None = field(default=None)
    doc: str | None = field(default=None)
