from __future__ import annotations

from .completion_models import SignatureCompletionContext, SignatureCompletionResult
from ..models import Argument, ArgumentKind, Signature


class MinimalCompleter:
    """生成最小签名。"""

    @staticmethod
    def get(context: SignatureCompletionContext) -> SignatureCompletionResult:
        signatures = [
            Signature(
                args=[
                    Argument(name="args", kind=ArgumentKind.VAR_POSITIONAL),
                    Argument(name="kwargs", kind=ArgumentKind.VAR_KEYWORD),
                ]
            )
        ]
        return SignatureCompletionResult(signatures=signatures)
