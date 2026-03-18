from __future__ import annotations

from typing import Literal, TypeAlias

IRMethodDecorator: TypeAlias = Literal["staticmethod", "classmethod"] | None
