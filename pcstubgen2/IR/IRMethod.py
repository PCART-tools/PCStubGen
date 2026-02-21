from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .IRFunction import IRFunction
    from .IRModifier import IRModifier


@dataclass
class IRMethod:
    function: IRFunction
    modifier: IRModifier
