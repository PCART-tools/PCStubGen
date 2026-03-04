from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .IRFunction import IRFunction

@dataclass
class IRMethod:
    function: IRFunction
    decorator: str | None