from __future__ import annotations

import sys

if sys.version_info >= (3, 8):
    from typing import Literal

    IRModifier = Literal["static", "class", None]
else:
    from typing import Optional

    IRModifier = Optional[str]  # type: ignore

__all__ = ["IRModifier"]
