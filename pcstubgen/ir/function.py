from __future__ import annotations

from dataclasses import dataclass, field

from .signature import IRSignature


@dataclass
class IRFunction:
    """IR 中的函数节点。"""

    name: str
    signatures: list[IRSignature] = field(default_factory=list)
    doc: str | None = field(default=None)
    c_inferred_source_comment: str | None = field(default=None)
