from __future__ import annotations

from .models import ExtractedArgument, ExtractedFunction, ExtractedSignature

def inference_signature(function: ExtractedFunction) -> None:
    func_cursor = function.function_cursor