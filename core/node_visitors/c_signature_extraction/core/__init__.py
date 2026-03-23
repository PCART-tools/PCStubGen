from __future__ import annotations

"""C 签名提取子包对外导出入口。"""

from .c_signature_extraction import extract_c_signature_modules
from .models import (
    ExtractedArgument,
    ExtractedFunction,
    ExtractedModule,
    ExtractedSignature,
)

__all__ = [
    "extract_c_signature_modules",
    "ExtractedArgument",
    "ExtractedFunction",
    "ExtractedModule",
    "ExtractedSignature",
]

