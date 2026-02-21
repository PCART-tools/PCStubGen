from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .IRAlias import IRAlias
    from .IRField import IRField
    from .IRMethod import IRMethod
    from .IRProperty import IRProperty
    from .QualifiedName import QualifiedName


@dataclass
class IRClass:
    name: str
    doc: str | None = field(default=None)
    bases: list[QualifiedName] = field(default_factory=list)
    classes: list[IRClass] = field(default_factory=list)
    fields: list[IRField] = field(default_factory=list)
    methods: list[IRMethod] = field(default_factory=list)
    properties: list[IRProperty] = field(default_factory=list)
    aliases: list[IRAlias] = field(default_factory=list)
