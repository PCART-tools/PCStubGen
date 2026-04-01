from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Type
from .argument import IRArgument


@dataclass
class IRSignature:
    """IR 中的单条函数签名。"""

    args: list[IRArgument] = field(default_factory=list)
    return_type: Type | None = field(default=None)
