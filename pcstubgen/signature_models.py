from __future__ import annotations

from dataclasses import dataclass, field

from .c_signature.types import Type
from .ir import IRArgumentKind


@dataclass
class ResolvedArgument:
    name: str
    type: Type | None = None
    default_value: str | None = None
    has_default: bool = False
    kind: IRArgumentKind = IRArgumentKind.POSITIONAL_OR_KEYWORD


@dataclass
class ResolvedSignature:
    arguments: list[ResolvedArgument] = field(default_factory=list)
    return_type: Type | None = None
    doc: str | None = None


@dataclass
class ResolvedFunctionSignatures:
    signatures: list[ResolvedSignature] = field(default_factory=list)
    c_inferred_source_comment: str | None = None
