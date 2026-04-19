from __future__ import annotations

from .completion_models import SignatureCompletionResult, SignatureCompletionContext
from ..models import Argument, ArgumentKind, Function, Signature


class MinimalProvider:
    """生成最小签名。"""

    @staticmethod
    def get(context: SignatureCompletionContext) -> tuple[list[Signature], str]:
        signatures = [
            Signature(
                args=[
                    Argument(name="args", kind=ArgumentKind.VAR_POSITIONAL),
                    Argument(name="kwargs", kind=ArgumentKind.VAR_KEYWORD),
                ]
            )
        ]
        return signatures, ""
