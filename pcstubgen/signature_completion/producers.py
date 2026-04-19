from __future__ import annotations

from ..models import Argument, ArgumentKind, Decorator, Signature


def _finalize_signatures(
    signatures: list[Signature],
    *,
    is_method: bool,
    decorator: Decorator,
) -> list[Signature]:
    if not signatures:
        raise RuntimeError("producer 没有返回可用签名。")
    return [
        _finalize_signature(signature, is_method=is_method, decorator=decorator)
        for signature in signatures
    ]

def _finalize_signature(
    signature: Signature,
    *,
    is_method: bool,
    decorator: Decorator,
) -> Signature:
    if not is_method:
        return signature

    receiver_name: str | None = None
    if decorator is None:
        receiver_name = "self"
    elif decorator == "classmethod":
        receiver_name = "cls"

    if receiver_name is None:
        return _strip_receiver(signature)

    return _ensure_receiver(signature, receiver_name)


def _strip_receiver(signature: Signature) -> Signature:
    if signature.args and signature.args[0].name in {"self", "cls"}:
        return Signature(args=signature.args[1:], return_type=signature.return_type)
    return signature


def _ensure_receiver(signature: Signature, receiver_name: str) -> Signature:
    receiver = Argument(name=receiver_name)
    if not signature.args:
        return Signature(args=[receiver], return_type=signature.return_type)

    first_arg = signature.args[0]
    if first_arg.name == receiver_name:
        if (
            first_arg.type is None
            and first_arg.default_value is None
            and first_arg.kind is ArgumentKind.POSITIONAL_OR_KEYWORD
        ):
            return signature
        return Signature(args=[receiver, *signature.args[1:]], return_type=signature.return_type)

    if first_arg.name in {"self", "cls"}:
        return Signature(args=[receiver, *signature.args[1:]], return_type=signature.return_type)

    return Signature(args=[receiver, *signature.args], return_type=signature.return_type)
