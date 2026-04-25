from __future__ import annotations

"""PyTorch `PythonArgParser` 签名规则。"""

from dataclasses import dataclass
import re

from clang.cindex import Cursor, CursorKind

from .....models import Argument, ArgumentKind
from .....type_models import RawType, Type, UnionType
from ...libclang.ast_utils import get_string_literal, unwrap_transparent, walk


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

    arg_type = _TYPE_NAME_TO_TYPE.get(type_text, RawType(type_text))
    if allow_none or default_value == "None":
        return UnionType((arg_type, RawType.none_)).canonicalize()
    return arg_type


def _parse_type_text(type_text: str) -> tuple[str, bool]:
    """解析 `Type?` 与 `Type[2]` 这类 PyTorch 类型修饰。"""
    match = _TYPE_RE.fullmatch(type_text)
    if match is None:
        raise PyTorchSignatureParserError(f"PyTorch 参数类型格式非法: {type_text!r}。")
    return match.group("name"), match.group("optional") is not None
