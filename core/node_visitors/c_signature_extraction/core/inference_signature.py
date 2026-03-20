from __future__ import annotations

from .models import ExtractedArgument, ExtractedFunction, ExtractedSignature


def extract_signatures_from_function(function_cursor: object) -> list[ExtractedSignature]:
    _ = function_cursor
    return []


def infer_return_type_from_function(function_cursor: object) -> str | None:
    _ = function_cursor
    return None


def signature_from_param_decls(function_cursor: object) -> ExtractedSignature | None:
    _ = function_cursor
    return None


def infer_function_signature(function: ExtractedFunction) -> None:
    function_cursor = function.function_cursor
    if function_cursor is None:
        return

    signatures = extract_signatures_from_function(function_cursor)
    if not signatures:
        fallback_signature = signature_from_param_decls(function_cursor)
        if fallback_signature is not None:
            signatures = [fallback_signature]
    if not signatures:
        return

    return_type_name = infer_return_type_from_function(function_cursor)
    normalized_signatures: list[ExtractedSignature] = []
    for signature in signatures:
        arguments = list(signature.arguments)
        if not arguments or arguments[0].name not in {"self", "cls"}:
            arguments.insert(0, ExtractedArgument(name="self", type_name="object"))
        normalized_signatures.append(
            ExtractedSignature(
                arguments=arguments,
                return_type_name=return_type_name
                if return_type_name is not None
                else signature.return_type_name,
            )
        )
    function.signatures = normalized_signatures


def inference_signature(function: ExtractedFunction) -> None:
    infer_function_signature(function)
