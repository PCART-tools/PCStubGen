from __future__ import annotations

from .completion_models import SignatureCompletionResult
from ..models import Argument, ArgumentKind, Function, Signature


class MinimalProvider:
    """生成最小签名。"""

    @staticmethod
    def get(func: Function, is_method: bool) -> SignatureCompletionResult:
        _ = func, is_method
        signatures = [
            Signature(
                args=[
                    Argument(name="args", kind=ArgumentKind.VAR_POSITIONAL),
                    Argument(name="kwargs", kind=ArgumentKind.VAR_KEYWORD),
                ]
            )
        ]
        return SignatureCompletionResult(signatures)
