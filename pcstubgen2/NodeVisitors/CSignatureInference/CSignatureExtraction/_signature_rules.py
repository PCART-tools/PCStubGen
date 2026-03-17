from __future__ import annotations

import re
from typing import TypeAlias

from clang.cindex import Cursor, CursorKind, TokenKind

from .Constants import (
    FORMAT_TYPE_MAP,
    METH_TYPE_LITERAL_MAP,
    PY_BUILDVALUE_SINGLE_MARKER_TYPE_MAP,
    RETURN_CALL_PREFIX_TYPE_MAP,
    RETURN_MACRO_TYPE_MAP,
    RETURN_TOKEN_TYPE_MAP,
    UNRELATED_TOKENS,
)
from .Models import ExtractedArgument, ExtractedSignature
from ._cursor_utils import (
    collect_identifier_literal_tokens,
    find_first_call_name,
    looks_like_identifier,
    split_top_level,
    strip_string_literal_quotes,
    unique_keep_order,
    walk_cursor,
)

SignatureArgumentKey: TypeAlias = tuple[str, str | None, str | None, str]
SignatureKey: TypeAlias = tuple[str | None, tuple[SignatureArgumentKey, ...]]


def merge_signature_return_type(
    signature: ExtractedSignature,
    return_type_name: str | None,
) -> ExtractedSignature:
    """在签名未显式设置返回值时填充函数级推断结果。"""
    if return_type_name is None or signature.return_type_name is not None:
        return signature
    return ExtractedSignature(
        arguments=list(signature.arguments),
        return_type_name=return_type_name,
    )


def explode_py_buildvalue_format_string(format_text: str) -> list[str]:
    """
    拆分 `Py_BuildValue` 格式串为返回值 marker。

    与 `PyArg_*` 的解析不同，这里不会把 `s#`/`z#` 等拆成两个参数位。
    """
    result: list[str] = []
    idx = 0
    while idx < len(format_text):
        current = format_text[idx]
        if current in {" ", ","}:
            idx += 1
            continue
        if current in {"(", ")", "[", "]", "{", "}"}:
            result.append(current)
            idx += 1
            continue
        if current in {":", ";"}:
            break
        if idx + 1 < len(format_text) and format_text[idx + 1] in {"!", "#", "&", "*"}:
            result.append(current + format_text[idx + 1])
            idx += 2
            continue
        if current not in {"!", "#", "&"}:
            result.append(current)
        idx += 1
    return result


def infer_return_type_from_py_buildvalue_format(format_text: str) -> str:
    """根据 `Py_BuildValue` 格式串估算返回类型。"""
    markers = explode_py_buildvalue_format_string(format_text)
    if not markers:
        return "None"

    if "{" in markers:
        return "dict"
    if "[" in markers:
        return "list"
    if "(" in markers:
        return "tuple"

    value_markers = [m for m in markers if m not in {"(", ")", "[", "]", "{", "}"}]
    if not value_markers:
        return "None"
    if len(value_markers) > 1:
        return "tuple"
    return PY_BUILDVALUE_SINGLE_MARKER_TYPE_MAP.get(value_markers[0], "object")


def is_parameter_parser_call(call_name: str) -> bool:
    """判断调用名是否属于可提取参数签名的 `PyArg_*` 解析 API。"""
    if call_name.startswith("PyArg_Parse"):
        return True
    return call_name in {"PyArg_UnpackTuple"}


def resolve_format_index(call_name: str, meth_flags: list[str]) -> tuple[int, int]:
    """
    按 `METH_*` 约定推导格式串索引与参数偏移量。

    不同 `PyArg_*` API 的实参布局不同，这里统一归一化为
    `(format_idx, offset)` 供后续解析复用。
    """
    has_keywords = "METH_KEYWORDS" in meth_flags
    has_varargs = "METH_VARARGS" in meth_flags or "METH_FASTCALL" in meth_flags
    if has_keywords and has_varargs:
        if call_name == "PyArg_ParseTupleAndKeywords":
            return 3, 2
        if call_name == "PyArg_ParseTuple":
            return 2, 1
        if call_name == "PyArg_NoKeywords":
            return -1, 0
        return 3, 2
    if "METH_VARARGS" in meth_flags or "METH_O" in meth_flags or "METH_FASTCALL" in meth_flags:
        return 2, 1
    if has_keywords:
        return 3, 2
    if "METH_NOARGS" in meth_flags:
        return -1, 0
    return 2, 1


def explode_format_string(format_text: str) -> list[str]:
    """
    拆分 CPython 格式串为逐参数 marker。

    对 `s#` / `z#` / `y#` / `O!` 这类“一个标记对应多个 C 参数”的情况，
    会展开成 `xxx1`/`xxx2` 以便后续一一对齐参数名。
    """
    result: list[str] = []
    idx = 0
    while idx < len(format_text):
        current = format_text[idx]
        if current in {" ", ",", "(", ")", "[", "]"}:
            idx += 1
            continue
        if current in {"|", "*", "$", ":"}:
            result.append(current)
            idx += 1
            if current == ":":
                break
            continue
        if idx + 1 < len(format_text) and format_text[idx + 1] in {"!", "#", "&", "*"}:
            duo = current + format_text[idx + 1]
            if duo in {"s#", "z#", "y#", "O!"}:
                result.extend([f"{duo}1", f"{duo}2"])
                idx += 2
                continue
            if duo in {"y*", "z*", "s*", "O&"}:
                result.append(duo)
                idx += 2
                continue
        if current not in {"!", "#", "&"}:
            result.append(current)
        idx += 1
    return result


def split_required_optional(markers: list[str]) -> tuple[list[str], list[str]]:
    """按 `|` 分隔必填/可选参数，并在 `:` 处截断函数名后缀。"""
    required: list[str] = []
    optional: list[str] = []
    target = required
    for marker in markers:
        if marker == ":":
            break
        if marker == "|":
            target = optional
            continue
        target.append(marker)
    return required, optional


def collect_call_tokens(call_node: Cursor) -> list[str]:
    """
    从调用表达式提取与参数解析相关的 token。

    该步骤会过滤大量无关转换器/宏标识，降低误判概率。
    """
    result: list[str] = []
    started = False
    for token in call_node.get_tokens():
        if token.kind not in {TokenKind.IDENTIFIER, TokenKind.LITERAL}:
            continue
        spelling = str(token.spelling)
        if not started:
            if spelling.startswith("PyArg_") or spelling == "Py_BuildValue":
                result.append(spelling)
                started = True
            continue

        if spelling in {"NULL", "return"}:
            break
        if spelling in UNRELATED_TOKENS or spelling in {"if"}:
            continue
        result.append(spelling)
    return result


def apply_method_flags(signature: ExtractedSignature, meth_flags: list[str]) -> ExtractedSignature:
    """
    根据 `METH_*` 规则修正首参语义。

    该步骤负责统一 `self`/`cls` 行为，避免来源差异导致的方法签名不一致。
    """
    args = list(signature.arguments)
    return_type_name = signature.return_type_name
    if "METH_STATIC" in meth_flags:
        while args and args[0].name in {"self", "cls"}:
            args.pop(0)
        if "METH_NOARGS" in meth_flags:
            return ExtractedSignature(arguments=[], return_type_name=return_type_name)
        return ExtractedSignature(arguments=args, return_type_name=return_type_name)
    if "METH_CLASS" in meth_flags:
        if not args or args[0].name not in {"cls", "self"}:
            args.insert(0, ExtractedArgument(name="cls", type_name="type"))
        else:
            args[0].name = "cls"
            args[0].type_name = "type"
        return ExtractedSignature(arguments=args, return_type_name=return_type_name)
    if not args or args[0].name not in {"self", "cls"}:
        args.insert(0, ExtractedArgument(name="self", type_name="object"))
    else:
        args[0].name = "self"
        args[0].type_name = "object"
    return ExtractedSignature(arguments=args, return_type_name=return_type_name)


def get_init_value(name: str, func_cursor: Cursor) -> str | None:
    """从局部变量定义中提取默认值字面表达式。"""
    for node in walk_cursor(func_cursor):
        if node.kind != CursorKind.VAR_DECL or node.spelling != name:
            continue
        tokens = [str(token.spelling) for token in node.get_tokens()]
        if "=" not in tokens:
            continue
        eq_idx = tokens.index("=")
        value = "".join(tokens[eq_idx + 1:]).strip()
        if value:
            return value
    return None


def find_format_string(func_cursor: Cursor, format_var_name: str) -> str | None:
    """回溯查找格式串变量对应的字符串字面量。"""
    for node in walk_cursor(func_cursor):
        if node.kind != CursorKind.VAR_DECL or node.spelling != format_var_name:
            continue
        for child in walk_cursor(node):
            if child.kind == CursorKind.STRING_LITERAL:
                return strip_string_literal_quotes(str(child.spelling))
    return None


def normalize_parser_type(raw_type: str) -> str:
    """将 parser 文本中的类型描述归一化为 Python 基础类型名。"""
    normalized = raw_type.replace("const", "").replace("&", "").replace("*", "").strip()
    normalized = normalized.replace("std::", "")
    lower = normalized.lower()
    if any(token in lower for token in ("str", "string", "unicode")):
        return "str"
    if "bool" in lower:
        return "bool"
    if any(token in lower for token in ("double", "float")):
        return "float"
    if any(token in lower for token in ("int", "long", "size_t", "ssize")):
        return "int"
    return "object"


def normalize_c_type(raw_type: str) -> str:
    """将 C 类型拼写映射到简化 Python 类型。"""
    lower = raw_type.lower()
    if "char" in lower and "*" in lower:
        return "str"
    if "bool" in lower:
        return "bool"
    if "float" in lower or "double" in lower:
        return "float"
    if any(token in lower for token in ("int", "long", "short", "size_t", "ssize")):
        return "int"
    return "object"


def signature_key(signature: ExtractedSignature) -> SignatureKey:
    """构建签名的可哈希键，供去重使用。"""
    return (
        signature.return_type_name,
        tuple(
            (arg.name, arg.type_name, arg.default_value, arg.kind)
            for arg in signature.arguments
        ),
    )


def resolve_py_buildvalue_format_token(*, func_cursor: Cursor, token: str) -> str | None:
    """解析 `Py_BuildValue` 的格式串 token。"""
    if '"' in token:
        return strip_string_literal_quotes(token)
    if looks_like_identifier(token):
        return find_format_string(func_cursor=func_cursor, format_var_name=token)
    return None


def parse_parser_args(args_text: str) -> list[ExtractedArgument]:
    """
    解析 parser 风格的参数文本。

    支持默认值、`*`/`$`、`*args`/`**kwargs`，并在信息不足时生成占位参数名。
    """
    parts = split_top_level(args_text, ",")
    result: list[ExtractedArgument] = []
    kw_only = False
    for raw_part in parts:
        part = raw_part.strip()
        if not part:
            continue
        if part in {"*", "$"}:
            kw_only = True
            continue

        default: str | None = None
        if "=" in part:
            before_default, default = part.split("=", 1)
        else:
            before_default = part
        before_default = before_default.strip()

        type_name = "object"
        if " " in before_default:
            type_candidate, name_candidate = before_default.rsplit(" ", 1)
            if looks_like_identifier(name_candidate):
                name = name_candidate
                type_name = normalize_parser_type(type_candidate)
            else:
                name = f"arg{len(result) + 1}"
                type_name = normalize_parser_type(before_default)
        else:
            name = before_default if looks_like_identifier(before_default) else f"arg{len(result) + 1}"
            if name != before_default:
                type_name = normalize_parser_type(before_default)

        kind = "keyword_only" if kw_only else "positional_or_keyword"
        if name.startswith("**"):
            kind = "var_keyword"
            name = name[2:]
        elif name.startswith("*"):
            kind = "var_positional"
            name = name[1:]
            kw_only = True

        result.append(
            ExtractedArgument(
                name=name,
                type_name=type_name,
                default_value=default.strip() if default else None,
                kind=kind,
            )
        )
    return result


def parse_format_token(func_cursor: Cursor, token: str) -> list[str] | None:
    """将格式串 token 解析为 marker 列表，支持字面量与变量两种来源。"""
    if token == "F_INT_PYFMT":
        return ["F_INT_PYFMT"]
    if '"' in token:
        text = strip_string_literal_quotes(token)
        return explode_format_string(text)
    if looks_like_identifier(token):
        text = find_format_string(func_cursor=func_cursor, format_var_name=token)
        if text:
            return explode_format_string(text)
    return None


def infer_return_type_from_py_buildvalue(return_stmt: Cursor, func_cursor: Cursor) -> str:
    """从 `Py_BuildValue` 的格式串推断返回类型。"""
    tokens = collect_identifier_literal_tokens(return_stmt)
    if "Py_BuildValue" not in tokens:
        return "object"

    call_idx = tokens.index("Py_BuildValue")
    for token in tokens[call_idx + 1:]:
        format_text = resolve_py_buildvalue_format_token(func_cursor=func_cursor, token=token)
        if format_text is None:
            continue
        return infer_return_type_from_py_buildvalue_format(format_text)
    return "object"


def infer_return_type_from_call(
    *,
    call_name: str,
    return_stmt: Cursor,
    func_cursor: Cursor,
) -> str | None:
    """
    根据返回调用名推断 Python 返回类型。

    优先处理 `Py_BuildValue`、`Py_NewRef` 这类需要额外上下文的调用，
    其余再按前缀表或兜底规则映射。
    """
    if call_name == "Py_BuildValue":
        return infer_return_type_from_py_buildvalue(return_stmt=return_stmt, func_cursor=func_cursor)

    if call_name == "Py_NewRef":
        token_set = set(collect_identifier_literal_tokens(return_stmt))
        if "Py_None" in token_set:
            return "None"
        if "Py_True" in token_set or "Py_False" in token_set:
            return "bool"
        return "object"

    for prefix, type_name in RETURN_CALL_PREFIX_TYPE_MAP:
        if call_name.startswith(prefix):
            return type_name

    if call_name.startswith("PyObject_Call"):
        return "object"
    return None


def extract_parser_signatures(node: Cursor) -> list[ExtractedSignature]:
    """从声明语句中的 `\"func(type name, ...)\"` 文本签名提取参数。"""
    signatures: list[ExtractedSignature] = []
    literals: list[str] = []
    for token in node.get_tokens():
        if token.kind != TokenKind.LITERAL:
            continue
        text = strip_string_literal_quotes(str(token.spelling))
        if "(" in text and ")" in text:
            literals.append(text)
    for text in literals:
        args_part = text[text.find("(") + 1 : text.rfind(")")]
        args = parse_parser_args(args_part)
        if args:
            signatures.append(ExtractedSignature(arguments=args))
    return signatures


def set_token_params(
    func_cursor: Cursor,
    meth_flags: list[str],
    token_list: list[str],
) -> list[ExtractedArgument] | None:
    """
    基于调用 token 与方法标志推断参数名、类型和默认值。

    返回 `None` 表示无法可靠解析该调用；返回空列表表示明确无参数。
    """
    if not token_list:
        return None

    call_name = token_list[0]
    if not is_parameter_parser_call(call_name):
        return None

    format_idx, offset = resolve_format_index(call_name=call_name, meth_flags=meth_flags)
    if format_idx < 0 or format_idx >= len(token_list):
        return None

    format_markers: list[str] | None = None
    format_token_index = format_idx
    for idx in range(format_idx, len(token_list)):
        parsed = parse_format_token(func_cursor, token_list[idx])
        if parsed is None:
            continue
        format_markers = parsed
        format_token_index = idx
        break

    if format_markers is None:
        return None

    required, optional = split_required_optional(format_markers)
    param_cursor = format_token_index + offset
    if param_cursor < len(token_list) and token_list[param_cursor] == "kwlist":
        param_cursor += 1

    result: list[ExtractedArgument] = []
    kw_only = False
    for marker, is_optional in ([(m, False) for m in required] + [(m, True) for m in optional]):
        if marker in {"*", "$"}:
            kw_only = True
            continue
        if marker == ":":
            break
        if param_cursor >= len(token_list):
            break

        name = token_list[param_cursor].strip('"')
        param_cursor += 1
        if not looks_like_identifier(name):
            continue

        type_name = FORMAT_TYPE_MAP.get(marker, "object")
        default_value = get_init_value(name=name, func_cursor=func_cursor) if is_optional else None
        kind = "keyword_only" if kw_only else "positional_or_keyword"
        result.append(
            ExtractedArgument(
                name=name,
                type_name=type_name,
                default_value=default_value,
                kind=kind,
            )
        )

    return result


def infer_return_type_from_return_stmt(return_stmt: Cursor, func_cursor: Cursor) -> str | None:
    """从单条 `return` 语句中提取可识别的返回类型。"""
    tokens = collect_identifier_literal_tokens(return_stmt)
    if not tokens:
        return None
    token_set = set(tokens)

    for macro_name, type_name in RETURN_MACRO_TYPE_MAP.items():
        if macro_name in token_set:
            return type_name
    for token_name, type_name in RETURN_TOKEN_TYPE_MAP.items():
        if token_name in token_set:
            return type_name

    call_name = find_first_call_name(return_stmt)
    if call_name is None:
        return None
    return infer_return_type_from_call(
        call_name=call_name,
        return_stmt=return_stmt,
        func_cursor=func_cursor,
    )


def decode_meth_literal_flags(literal: str) -> list[str]:
    """把数字字面量（含位掩码组合）解析为 `METH_*` 列表。"""
    text = literal.strip()
    if not text:
        return []
    text = re.sub(r"[uUlL]+$", "", text)
    try:
        mask = int(text, 0)
    except ValueError:
        return []
    if mask <= 0:
        return []

    result: list[str] = []
    for raw_bit, flag_name in METH_TYPE_LITERAL_MAP.items():
        try:
            bit = int(raw_bit, 0)
        except ValueError:
            continue
        if bit and (mask & bit) == bit:
            result.append(flag_name)
    return unique_keep_order(result)


def signature_from_param_decls(func_cursor: Cursor) -> ExtractedSignature:
    """回退方案：直接从 C 形参声明推断签名。"""
    args: list[ExtractedArgument] = []
    for node in func_cursor.get_children():
        if node.kind != CursorKind.PARM_DECL or not node.spelling:
            continue
        args.append(
            ExtractedArgument(
                name=node.spelling,
                type_name=normalize_c_type(node.type.spelling),
                default_value=None,
            )
        )
    return ExtractedSignature(arguments=args)


def deduplicate_signatures(signatures: list[ExtractedSignature]) -> list[ExtractedSignature]:
    """按参数四元组键去重签名列表。"""
    seen: set[SignatureKey] = set()
    result: list[ExtractedSignature] = []
    for signature in signatures:
        key = signature_key(signature)
        if key in seen:
            continue
        seen.add(key)
        result.append(signature)
    return result


def infer_return_type_from_function(func_cursor: Cursor) -> str | None:
    """
    从函数体中推断 Python 返回类型。

    规则：
    - 优先收集 `return` 语句中的显式工厂函数与宏。
    - 若出现多个已识别返回类型，回退为 `object`。
    """
    inferred_types: set[str] = set()

    all_tokens = set(collect_identifier_literal_tokens(func_cursor))
    for macro_name, type_name in RETURN_MACRO_TYPE_MAP.items():
        if macro_name in all_tokens:
            inferred_types.add(type_name)

    for node in walk_cursor(func_cursor):
        if node.kind != CursorKind.RETURN_STMT:
            continue
        inferred = infer_return_type_from_return_stmt(return_stmt=node, func_cursor=func_cursor)
        if inferred is not None:
            inferred_types.add(inferred)

    if not inferred_types:
        return None
    if len(inferred_types) == 1:
        return sorted(inferred_types)[0]
    return "object"


def collect_pyarg_token_lists(node: Cursor) -> list[list[str]]:
    """
    递归收集参数解析调用（`PyArg_*`）的 token 序列。

    `IF_STMT` 与 `UNEXPOSED_EXPR` 在不同编译单元下结构可能不同，
    因此这里采用保守递归策略统一处理。
    """
    result: list[list[str]] = []

    for child in node.get_children():
        token_list: list[str] | None = None
        if child.kind == CursorKind.CALL_EXPR:
            token_list = collect_call_tokens(child)
        elif child.kind == CursorKind.IF_STMT:
            first_child = next(child.get_children(), None)
            if first_child is not None and first_child.kind == CursorKind.UNEXPOSED_EXPR:
                token_list = collect_call_tokens(first_child)
            else:
                token_list = collect_call_tokens(child)

        if token_list and is_parameter_parser_call(token_list[0]):
            result.append(token_list)
            continue

        result.extend(collect_pyarg_token_lists(child))
    return result


def extract_signatures_from_function(func_cursor: Cursor, meth_flags: list[str]) -> list[ExtractedSignature]:
    """从函数体中提取候选签名，并在末尾做去重。"""
    signatures: list[ExtractedSignature] = []
    for token_list in collect_pyarg_token_lists(func_cursor):
        args = set_token_params(func_cursor, meth_flags, token_list)
        if args is not None:
            signatures.append(ExtractedSignature(arguments=args))

    for node in walk_cursor(func_cursor):
        if node.kind == CursorKind.DECL_STMT:
            signatures.extend(extract_parser_signatures(node))
    return deduplicate_signatures(signatures)
