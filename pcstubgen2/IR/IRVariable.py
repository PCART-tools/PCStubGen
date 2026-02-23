from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .IRAnnotation import IRAnnotation
    from .IRValue import IRValue


@dataclass
class IRVariable:
    name: str
    value: IRValue | None
    annotation: IRAnnotation | None = field(default=None)
    runtime_value: object | None = field(default=None, repr=False, compare=False)
