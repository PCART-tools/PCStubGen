from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .c_signature_extraction import extract_c_signature_modules
    from .models import (
        ExtractedArgument,
        ExtractedFunction,
        ExtractedModule,
        ExtractedSignature,
    )
    from .resolver import CSignatureResolver


__all__ = [
    "extract_c_signature_modules",
    "CSignatureResolver",
    "ExtractedArgument",
    "ExtractedFunction",
    "ExtractedModule",
    "ExtractedSignature",
]


def __getattr__(name: str) -> Any:
    if name == "extract_c_signature_modules":
        return getattr(import_module(".c_signature_extraction", __name__), name)
    if name == "CSignatureResolver":
        return getattr(import_module(".resolver", __name__), name)
    if name in {
        "ExtractedArgument",
        "ExtractedFunction",
        "ExtractedModule",
        "ExtractedSignature",
    }:
        return getattr(import_module(".models", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
