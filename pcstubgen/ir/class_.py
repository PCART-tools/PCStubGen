from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .method import IRMethod
    from .qualified_name import QualifiedName


@dataclass
class IRClass:
    name: str
    doc: str | None = field(default=None)
    bases: list[QualifiedName] = field(default_factory=list)
    classes: list[IRClass] = field(default_factory=list)
    methods: list[IRMethod] = field(default_factory=list)
