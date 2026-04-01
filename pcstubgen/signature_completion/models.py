from __future__ import annotations

from dataclasses import dataclass, field

from ..ir import IRSignature


@dataclass
class ResolvedFunctionSignatures:
    signatures: list[IRSignature] = field(default_factory=list)
    c_inferred_source_comment: str | None = None
