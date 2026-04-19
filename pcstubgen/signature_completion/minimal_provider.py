from __future__ import annotations

import types

from . import producers
from .completion_models import SignatureCompletionContext
from ..models import Argument, ArgumentKind, Decorator, Signature


class MinimalProvider:
    """生成最小签名。"""

    @staticmethod
    def get(
        context: SignatureCompletionContext,
        *,
        decorator: Decorator = None,
    ) -> list[Signature]:
        effective_decorator = decorator
        if effective_decorator is None:
            effective_decorator = _infer_decorator(context.member)

        signatures = [
            Signature(
                args=[
                    Argument(name="args", kind=ArgumentKind.VAR_POSITIONAL),
                    Argument(name="kwargs", kind=ArgumentKind.VAR_KEYWORD),
                ]
            )
        ]
        return producers._finalize_signatures(
            signatures,
            is_method=context.is_method,
            decorator=effective_decorator,
        )


def _infer_decorator(member: object) -> Decorator:
    if isinstance(member, types.ClassMethodDescriptorType):
        return "classmethod"
    if isinstance(member, staticmethod):
        return "staticmethod"
    return None
