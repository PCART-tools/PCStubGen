from __future__ import annotations

from dataclasses import dataclass

from .QualifiedName import QualifiedName


@dataclass
class IRAlias:
    name: str
    origin: QualifiedName
