from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .IRFunction import IRFunction
    from .IRModifier import IRModifier


@dataclass
class IRProperty:
    name: str
    modifier: IRModifier
    doc: str | None = field(default=None)
    getter: IRFunction | None = field(default=None)
    setter: IRFunction | None = field(default=None)
