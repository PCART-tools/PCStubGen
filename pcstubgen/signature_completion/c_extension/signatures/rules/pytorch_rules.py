from __future__ import annotations

"""PyTorch `PythonArgParser` 签名规则。"""

from collections.abc import Callable
from dataclasses import dataclass
import re

from clang.cindex import Cursor, CursorKind, Type as ClangType

from .....models import Argument, ArgumentKind
from .....type_models import RawType, Type, UnionType
from ...libclang.ast_utils import get_string_literal, unwrap_transparent, walk
from .lookup import get_name_to_type


class PyTorchSignatureParserError(ValueError):
    """表示 PyTorch `PythonArgParser` 签名字符串无法被当前解析器接受。"""


@dataclass(frozen=True)
class _ParsedParameter:
    """保存单个 PyTorch 参数的字符串结构。"""

    type_text: str
    name: str
    default_value: str | None


_SIGNATURE_RE = re.compile(r"^(?P<name>[^()|]+)\((?P<params>.*)\)(?:\|(?P<suffix>[^|]+))?$")
_PARAMETER_RE = re.compile(r"^(?P<type>\S+)\s+(?P<name>[^=\s]+)(?:=(?P<default>.+))?$")
_TYPE_RE = re.compile(r"^(?P<name>[^\[\]?]+)(?:\[(?P<size>\d+)])?(?P<optional>\?)?$")

_TYPE_NAME_TO_TYPE: dict[str, Type] = {
    "Tensor": RawType("torch.Tensor", imports=("torch",)),
    "Scalar": RawType("torch.types.Number", imports=("torch.types",)),
    "int64_t": RawType.int_,
    "DeviceIndex": RawType.int_,
    "SymInt": RawType("torch.SymInt", imports=("torch",)),
    "double": RawType.float_,
    "complex": RawType.complex_,
    "TensorList": RawType(
        "collections.abc.Sequence[torch.Tensor]",
        imports=("collections.abc", "torch"),
    ),
    "c10::List<::std::optional<Tensor>>": RawType(
        "collections.abc.Sequence[torch.Tensor | None]",
        imports=("collections.abc", "torch"),
    ),
    "IntArrayRef": RawType("torch.types._size", imports=("torch.types",)),
    "SymIntArrayRef": RawType("torch.types._symsize", imports=("torch.types",)),
    "ArrayRef<double>": RawType(
        "collections.abc.Sequence[float]",
        imports=("collections.abc",),
    ),
    "Generator": RawType("torch.Generator", imports=("torch",)),
    "bool": RawType.bool_,
    "Storage": RawType("torch.Storage", imports=("torch",)),
    "PyObject*": RawType.object_,
    "ScalarType": RawType("torch.dtype", imports=("torch",)),
    "Layout": RawType("torch.layout", imports=("torch",)),
    "MemoryFormat": RawType("torch.memory_format", imports=("torch",)),
    "QScheme": RawType("torch.qscheme", imports=("torch",)),
    "Device": RawType("torch.device", imports=("torch",)),
    "Stream": RawType("torch.Stream", imports=("torch",)),
    "std::string": RawType.str_,
    "c10::string_view": RawType.str_,
    "std::string_view": RawType.str_,
    "::std::string_view": RawType.str_,
    "Dimname": RawType.str_,
    "DimnameList": RawType("collections.abc.Sequence[str]", imports=("collections.abc",)),
    "ScalarList": RawType(
        "collections.abc.Sequence[torch.types.Number]",
        imports=("collections.abc", "torch.types"),
    ),
    "DispatchKeySet": RawType("torch.DispatchKeySet", imports=("torch",)),
}

_TENSOR_TYPE = RawType("torch.Tensor", imports=("torch",))
_TENSOR_TUPLE_TYPE = RawType("tuple[torch.Tensor, ...]", imports=("torch",))
_INT_TUPLE_TYPE = RawType("tuple[int, ...]")
_STREAM_TYPE = RawType("torch.Stream", imports=("torch",))
_DTYPE_TYPE = RawType("torch.dtype", imports=("torch",))
_LAYOUT_TYPE = RawType("torch.layout", imports=("torch",))
_QSCHEME_TYPE = RawType("torch.qscheme", imports=("torch",))
_NONE_OR_TENSOR_TYPE = UnionType((RawType.none_, _TENSOR_TYPE)).canonicalize()

_NORMALIZED_CPP_TYPE_NAME_TO_TYPE: dict[str, Type] = {
    "bool": RawType.bool_,
    "c10::DeviceIndex": RawType.int_,
    "int64_t": RawType.int_,
    "double": RawType.float_,
    "at::Tensor": _TENSOR_TYPE,
    "at::TensorList": _TENSOR_TUPLE_TYPE,
    "at::Stream": _STREAM_TYPE,
    "c10::Stream": _STREAM_TYPE,
    "at::ScalarType": _DTYPE_TYPE,
    "c10::ScalarType": _DTYPE_TYPE,
    "THPDtype*": _DTYPE_TYPE,
    "at::Layout": _LAYOUT_TYPE,
    "c10::Layout": _LAYOUT_TYPE,
    "THPLayout*": _LAYOUT_TYPE,
    "at::QScheme": _QSCHEME_TYPE,
    "c10::QScheme": _QSCHEME_TYPE,
}

def infer_python_arg_parser_arguments(func_cursor: Cursor) -> list[list[Argument]]:
    """从函数体内的 PyTorch `PythonArgParser` 直接字符串初始化式推断参数列表。"""
    signatures: list[str] = []
    for cursor in walk(func_cursor):
        if cursor.kind != CursorKind.VAR_DECL:
            continue
        if not _is_python_arg_parser_decl(cursor):
            continue
        extracted = _extract_python_arg_parser_signature_strings(cursor)
        if not extracted:
            continue
        signatures.extend(extracted)

    return parse_python_arg_parser_signatures(signatures)


def parse_python_arg_parser_signatures(signatures: list[str]) -> list[list[Argument]]:
    """解析一组 PyTorch `PythonArgParser` 签名字符串。"""
    arguments_list: list[list[Argument]] = []
    for signature in signatures:
        arguments = parse_python_arg_parser_signature(signature)
        if arguments is None:
            continue
        arguments_list.append(arguments)
    return arguments_list


def parse_python_arg_parser_signature(signature: str) -> list[Argument] | None:
    """解析单条 PyTorch `PythonArgParser` 签名字符串；隐藏签名返回 `None`。"""
    params_text = _extract_params_text(signature)
    if params_text is None:
        return None

    keyword_only = False
    arguments: list[Argument] = []
    for param_text in _split_parameters(params_text):
        if param_text == "*":
            keyword_only = True
            continue

        arguments.append(_build_argument(_parse_parameter(param_text), keyword_only))
    return arguments


def _is_python_arg_parser_decl(cursor: Cursor) -> bool:
    """判断变量声明是否是 `PythonArgParser` 实例。"""
    return any(
        child.kind == CursorKind.TYPE_REF and child.spelling == "struct torch::PythonArgParser"
        for child in cursor.get_children()
    )


def _extract_python_arg_parser_signature_strings(cursor: Cursor) -> list[str]:
    """从 `PythonArgParser` 变量声明的直接初始化列表中提取字符串字面量。"""
    init_list = _find_direct_init_list(cursor)
    if init_list is None:
        return []

    signatures: list[str] = []
    for child in init_list.get_children():
        entry = unwrap_transparent(child)
        if entry.kind != CursorKind.STRING_LITERAL:
            return []
        signatures.append(get_string_literal(entry))
    return signatures


def _find_direct_init_list(cursor: Cursor) -> Cursor | None:
    """查找变量声明初始化器里的第一层 `INIT_LIST_EXPR`。"""
    for child in walk(cursor):
        if child.kind == CursorKind.INIT_LIST_EXPR:
            return child
    return None


def _extract_params_text(signature: str) -> str | None:
    """提取 PyTorch 签名括号内的参数文本；隐藏签名返回 `None`。"""
    match = _SIGNATURE_RE.fullmatch(signature)
    if match is None:
        raise PyTorchSignatureParserError(f"PyTorch 签名格式非法: {signature!r}。")

    suffix = match.group("suffix")
    if suffix is None:
        return match.group("params")
    if suffix in {"hidden", "deprecated"}:
        return None
    raise PyTorchSignatureParserError(f"不支持的 PyTorch 签名后缀: {suffix!r}。")


def _split_parameters(params_text: str) -> list[str]:
    """按 PyTorch parser 的 `, ` 分隔规则切分参数。"""
    if params_text == "":
        return []
    params = params_text.split(", ")
    if any(param == "" for param in params):
        raise PyTorchSignatureParserError(f"PyTorch 参数列表包含空参数: {params_text!r}。")
    return params


def _parse_parameter(param_text: str) -> _ParsedParameter:
    """解析 `Type name=default` 形式的 PyTorch 参数文本。"""
    match = _PARAMETER_RE.fullmatch(param_text)
    if match is None:
        raise PyTorchSignatureParserError(f"PyTorch 参数格式非法: {param_text!r}。")
    return _ParsedParameter(
        type_text=match.group("type"),
        name=match.group("name"),
        default_value=match.group("default"),
    )


def _build_argument(parsed: _ParsedParameter, keyword_only: bool) -> Argument:
    """将已解析的 PyTorch 参数转换为模型参数。"""
    kind = ArgumentKind.KEYWORD_ONLY if keyword_only else ArgumentKind.POSITIONAL_OR_KEYWORD
    return Argument(
        name=parsed.name,
        type=_build_argument_type(parsed.type_text, parsed.default_value),
        default_value=parsed.default_value,
        kind=kind,
    )


def _build_argument_type(type_text: str, default_value: str | None) -> Type:
    """将 PyTorch 参数类型文本转换为 stub 类型。"""
    type_text, allow_none = _parse_type_text(type_text)

    arg_type = get_name_to_type(_TYPE_NAME_TO_TYPE, type_text, None) or RawType(type_text)
    if allow_none or default_value == "None":
        return UnionType((arg_type, RawType.none_)).canonicalize()
    return arg_type


def _parse_type_text(type_text: str) -> tuple[str, bool]:
    """解析 `Type?` 与 `Type[2]` 这类 PyTorch 类型修饰。"""
    match = _TYPE_RE.fullmatch(type_text)
    if match is None:
        raise PyTorchSignatureParserError(f"PyTorch 参数类型格式非法: {type_text!r}。")
    return match.group("name"), match.group("optional") is not None


def infer_wrap_call_type(call_expr: Cursor) -> Type:
    """按 `wrap(...)` 首个参数的静态 C++ 类型推断 Python 返回类型。"""
    args = list(call_expr.get_children())[1:]
    if len(args) != 1:
        raise RuntimeError(f"wrap 调用参数个数不是 1: {len(args)}")
    value_expr = unwrap_transparent(args[0])
    return infer_cpp_type(value_expr.type.get_canonical())


def infer_cpp_type(call_type: ClangType) -> Type:
    """将有限集合的 PyTorch C++ 类型映射成 Python 返回类型。"""
    type_name = _normalize_cpp_type_name(call_type.spelling)
    mapped = get_name_to_type(_NORMALIZED_CPP_TYPE_NAME_TO_TYPE, type_name, None)
    if mapped is not None:
        return mapped

    if type_name in {"at::IntArrayRef", "c10::ArrayRef<int64_t>"}:
        return _INT_TUPLE_TYPE

    if type_name.startswith("c10::ArrayRef<") and type_name.endswith(">"):
        element_type = type_name.removeprefix("c10::ArrayRef<").removesuffix(">")
        if element_type == "at::Tensor":
            return _TENSOR_TUPLE_TYPE
        if element_type in {"int64_t", "long", "long int", "long long", "long long int"}:
            return _INT_TUPLE_TYPE

    raise RuntimeError(f"无法识别的 PyTorch C++ 返回类型: {call_type.spelling}")


def _normalize_cpp_type_name(type_name: str) -> str:
    """归一化 libclang 类型文本，去掉限定符与无关空白。"""
    normalized = re.sub(r"\s+", " ", type_name).strip()
    normalized = re.sub(r"\bconst\s+", "", normalized)
    normalized = re.sub(r"\s*([<>,*&])\s*", r"\1", normalized)
    normalized = normalized.removesuffix("&")
    return normalized.removesuffix("&")


CALL_NAME_TO_TYPE: dict[str, Type | Callable[[Cursor], Type]] = {
    "THPVariable_Wrap": _TENSOR_TYPE,
    "THPVariable_WrapWithType": _NONE_OR_TENSOR_TYPE,
    "THPVariable_is_nonzero": RawType.bool_,
    "THPUtils_packInt64": RawType.int_,
    "THPUtils_packDoubleAsInt": RawType.int_,
    "wrap": infer_wrap_call_type,
}
