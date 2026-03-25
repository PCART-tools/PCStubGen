from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ir_function import IRFunction
    from .ir_method_decorator import IRMethodDecorator

@dataclass
class IRMethod:
    function: IRFunction
    decorator: IRMethodDecorator
