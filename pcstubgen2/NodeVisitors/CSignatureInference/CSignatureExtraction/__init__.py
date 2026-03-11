from __future__ import annotations

"""C 签名提取子包对外导出入口。"""

from .CSignatureExtractor import CSignatureExtractor
from .Models import (
    ExtractedArgument,
    ExtractedArgumentKind,
    ExtractedFunction,
    ExtractedSignature,
)

__all__ = [
    "CSignatureExtractor",
    "ExtractedArgument",
    "ExtractedArgumentKind",
    "ExtractedFunction",
    "ExtractedSignature",
]

