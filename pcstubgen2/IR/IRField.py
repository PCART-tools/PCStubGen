from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .IRVariable import IRVariable
    from .IRModifier import IRModifier


@dataclass
class IRField:
    variable: IRVariable
    modifier: IRModifier
