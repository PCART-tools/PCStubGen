from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .QualifiedName import QualifiedName


@dataclass(eq=True, frozen=True)
class IRImport:
    name: str | None
    origin: QualifiedName
