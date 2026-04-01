from __future__ import annotations

import inspect
from typing import Any

from loguru import logger

from ..type_system.types import RawType, Type
from ..ir import IRArgument, IRArgumentKind, IRModuleType, IRSignature, QualifiedName
from ..module_build.reflection import get_module_name
from .models import ResolvedFunctionSignatures


def resolve_inspect_signatures(
    func: Any,
    *,
    module_type: IRModuleType = IRModuleType.UNKNOWN,
) -> ResolvedFunctionSignatures | None:
    if func is None:
        return None

    try:
        sig = inspect.signature(_get_signature_target(func))
    except Exception as ex:
        if module_type is IRModuleType.EXTENSION:
            logger.warning(
                "EXTENSION 模块 inspect 签名获取失败, function: {}, error_type: {}, error: {}",
                _describe_function(func),
                type(ex).__name__,
                ex,
            )
        return None

    kind_map = {
        inspect.Parameter.POSITIONAL_ONLY: IRArgumentKind.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD: IRArgumentKind.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL: IRArgumentKind.VAR_POSITIONAL,
        inspect.Parameter.KEYWORD_ONLY: IRArgumentKind.KEYWORD_ONLY,
        inspect.Parameter.VAR_KEYWORD: IRArgumentKind.VAR_KEYWORD,
    }

    args: list[IRArgument] = []
    for param in sig.parameters.values():
        arg = IRArgument(name=param.name, kind=kind_map[param.kind])
        if param.default is not inspect.Signature.empty:
            arg.default_value = _build_value(param.default)
            arg.has_default = True
        if param.annotation is not inspect.Signature.empty:
            arg.type = _build_annotation(param.annotation)
        args.append(arg)

    return_type: Type | None = None
    if sig.return_annotation is not inspect.Signature.empty:
        return_type = _build_annotation(sig.return_annotation)

    if module_type is IRModuleType.EXTENSION:
        logger.info("EXTENSION 模块 inspect 签名获取成功, function: {}", _describe_function(func))

    return ResolvedFunctionSignatures(
        signatures=[IRSignature(args=args, return_type=return_type)]
    )


def _get_signature_target(func: Any) -> Any:
    if inspect.ismethod(func) and inspect.isclass(getattr(func, "__self__", None)):
        return func.__func__
    return func


def _describe_function(func: Any) -> str:
    module_name = get_module_name(func)
    qual_name = getattr(func, "__qualname__", None)
    if isinstance(module_name, str) and isinstance(qual_name, str):
        return f"{module_name}.{qual_name}"
    if hasattr(func, "__name__"):
        return str(func.__name__)
    return repr(func)


def _build_annotation(annotation: Any) -> Type | None:
    if isinstance(annotation, str):
        return _build_raw_annotation(annotation)
    if isinstance(annotation, type):
        qualified_name = _get_type_fullname(annotation)
        imports: tuple[str, ...] = ()
        if annotation.__module__ != "builtins":
            imports = (annotation.__module__,)
        return RawType(str(qualified_name), imports=imports)
    return _build_raw_annotation(_build_value(annotation))


def _build_raw_annotation(annotation_text: str | None) -> RawType | None:
    if annotation_text is None:
        return None
    text = annotation_text.strip()
    if not text:
        return None
    return RawType(text)


def _build_value(value: Any) -> str:
    value_type = type(value)
    if value is Ellipsis:
        return "..."
    if value is None or value_type in (bool, int, str):
        return repr(value)
    if value_type in (float, complex):
        try:
            repr_str = repr(value)
            eval(repr_str)
            return repr_str
        except (SyntaxError, NameError):
            pass
    if value_type in (list, tuple, set):
        if len(value) == 0:
            return f"{value_type.__name__}()"
        elements = [_build_value(el) for el in value]
        left, right = {
            list: ("[", "]"),
            tuple: ("(", ")"),
            set: ("{", "}"),
        }[value_type]
        return "".join([left, ", ".join(elements), right])
    if value_type is dict:
        parts = []
        for key, item in value.items():
            key_value = _build_value(key)
            item_value = _build_value(item)
            parts.append(f"{key_value}: {item_value}")
        return "".join(["{", ", ".join(parts), "}"])
    if inspect.isroutine(value):
        module_name = get_module_name(value)
        qual_name = getattr(value, "__qualname__", None)
        if (
            module_name is not None
            and "<" not in module_name
            and isinstance(qual_name, str)
            and "<" not in qual_name
        ):
            if module_name == "builtins":
                return qual_name
            return f"{module_name}.{qual_name}"
    if inspect.isclass(value):
        return str(_get_type_fullname(value))
    if inspect.ismodule(value):
        return value.__name__
    return repr(value)


def _get_type_fullname(type_: type) -> QualifiedName:
    module = type_.__module__
    qualname = type_.__qualname__
    if module == "builtins":
        return QualifiedName.from_str(qualname)
    return QualifiedName.from_str(f"{module}.{qualname}")
