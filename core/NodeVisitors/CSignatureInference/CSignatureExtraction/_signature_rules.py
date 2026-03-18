from __future__ import annotations

import re
from typing import TypeAlias

from clang.cindex import Cursor, CursorKind

from . import ClangEval
from .Constants import (
    FORMAT_TYPE_MAP,
    METH_TYPE_LITERAL_MAP,
    PY_BUILDVALUE_SINGLE_MARKER_TYPE_MAP,
    RETURN_CALL_PREFIX_TYPE_MAP,
    RETURN_MACRO_TYPE_MAP,
    RETURN_TOKEN_TYPE_MAP,
    UNRELATED_TOKENS,
)
from .Models import (
    ExtractedArgument,
    ExtractedFunction,
    ExtractedSignature,
)
from ._cursor_utils import (
    is_nullptr_or_zero,
    looks_like_identifier,
    split_top_level,
    unique_keep_order,
    unwrap_transparent,
    var_decl_to_init_list_expr,
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


def iter_effective_children(cursor: Cursor) -> list[Cursor]:
    """返回剥离透明节点后的直接子节点。"""
    return list(unwrap_transparent(cursor).get_children())


def find_var_decl(func_cursor: Cursor, name: str) -> Cursor | None:
    """按名字在当前函数内回溯变量声明。"""
    for node in walk_cursor(func_cursor):
        if node.kind == CursorKind.VAR_DECL and node.spelling == name:
            return node
    return None


def get_initializer_cursor(var_decl: Cursor) -> Cursor | None:
    """提取变量声明的初始化表达式节点。"""
    init_candidate: Cursor | None = None
    for child in var_decl.get_children():
        if child.kind == CursorKind.TYPE_REF:
            continue
        init_candidate = child
    if init_candidate is None:
        return None
    return unwrap_transparent(init_candidate)


def resolve_decl_ref_target(func_cursor: Cursor, cursor: Cursor) -> Cursor | None:
    """从引用表达式解析出被引用的声明节点。"""
    node = unwrap_transparent(cursor)
    if node.kind != CursorKind.DECL_REF_EXPR:
        return None
    referenced = node.referenced
    if referenced is not None:
        return referenced
    if not node.spelling:
        return None
    return find_var_decl(func_cursor, node.spelling)


def extract_string_literal(cursor: Cursor) -> str | None:
    """从表达式节点提取字符串字面量文本。"""
    node = unwrap_transparent(cursor)
    if node.kind != CursorKind.STRING_LITERAL:
        return None
    return str(node.spelling).strip('"')


def resolve_string_argument(func_cursor: Cursor, arg_cursor: Cursor) -> str | None:
    """解析字符串形态的调用实参，支持字面量与变量引用。"""
    literal = extract_string_literal(arg_cursor)
    if literal is not None:
        return literal

    node = unwrap_transparent(arg_cursor)
    if node.kind != CursorKind.DECL_REF_EXPR:
        return None
    if node.spelling == "F_INT_PYFMT":
        return "F_INT_PYFMT"

    target = resolve_decl_ref_target(func_cursor, node)
    if target is None or target.kind != CursorKind.VAR_DECL:
        return None
    init_cursor = get_initializer_cursor(target)
    if init_cursor is None:
        return None
    return extract_string_literal(init_cursor)


def stringify_literal_cursor(cursor: Cursor) -> str | None:
    """尽量从 AST 节点稳定还原简单字面量文本。"""
    node = unwrap_transparent(cursor)
    if node.kind == CursorKind.STRING_LITERAL:
        return str(node.spelling)
    if node.kind == CursorKind.DECL_REF_EXPR and node.spelling:
        return str(node.spelling)
    if is_nullptr_or_zero(node):
        return "0"
    if node.kind == CursorKind.INTEGER_LITERAL:
        value = ClangEval.eval_int(node)
        if value is not None:
            return str(value)
    if node.kind == CursorKind.UNARY_OPERATOR:
        children = iter_effective_children(node)
        if len(children) != 1:
            return None
        inner_text = stringify_literal_cursor(children[0])
        if inner_text is None:
            return None
        inner_value = ClangEval.eval_int(node)
        if inner_value is not None:
            return str(inner_value)
        return inner_text
    return None


def get_init_value(name: str, func_cursor: Cursor) -> str | None:
    """从局部变量定义中提取默认值字面表达式。"""
    var_decl = find_var_decl(func_cursor, name)
    if var_decl is None:
        return None
    init_cursor = get_initializer_cursor(var_decl)
    if init_cursor is None:
        return None
    return stringify_literal_cursor(init_cursor)


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


def parse_format_arg(func_cursor: Cursor, arg_cursor: Cursor) -> list[str] | None:
    """将格式串实参解析为 marker 列表，支持字面量与变量两种来源。"""
    format_text = resolve_string_argument(func_cursor, arg_cursor)
    if format_text is None:
        return None
    if format_text == "F_INT_PYFMT":
        return ["F_INT_PYFMT"]
    return explode_format_string(format_text)


def extract_call_name(call_cursor: Cursor) -> str | None:
    """从 `CALL_EXPR` 提取调用名。"""
    if call_cursor.kind != CursorKind.CALL_EXPR:
        return None
    if call_cursor.spelling:
        return str(call_cursor.spelling)
    children = list(call_cursor.get_children())
    if not children:
        return None
    callee = unwrap_transparent(children[0])
    if callee.spelling:
        return str(callee.spelling)
    return None


def extract_call_arguments(call_cursor: Cursor) -> list[Cursor]:
    """按调用顺序返回 `CALL_EXPR` 的实参列表。"""
    children = list(call_cursor.get_children())
    if not children:
        return []
    return [unwrap_transparent(child) for child in children[1:]]


def extract_output_argument_name(arg_cursor: Cursor) -> str | None:
    """从输出参数实参中提取目标变量名。"""
    node = unwrap_transparent(arg_cursor)
    if node.kind == CursorKind.DECL_REF_EXPR and node.spelling:
        name = str(node.spelling)
        if name in UNRELATED_TOKENS:
            return None
        return name
    if node.kind != CursorKind.UNARY_OPERATOR:
        return None
    children = iter_effective_children(node)
    if len(children) != 1:
        return None
    target = unwrap_transparent(children[0])
    if target.kind != CursorKind.DECL_REF_EXPR or not target.spelling:
        return None
    name = str(target.spelling)
    if name in UNRELATED_TOKENS:
        return None
    return name


def extract_keyword_names(func_cursor: Cursor, arg_cursor: Cursor) -> list[str] | None:
    """从 `kwlist` 风格字符串数组变量提取关键字参数名。"""
    target = resolve_decl_ref_target(func_cursor, arg_cursor)
    if target is None or target.kind != CursorKind.VAR_DECL:
        return None
    init_list_expr = var_decl_to_init_list_expr(target)
    if init_list_expr is None:
        return None

    names: list[str] = []
    for entry in init_list_expr.get_children():
        value_cursor = unwrap_transparent(entry)
        if is_nullptr_or_zero(value_cursor):
            break
        literal = extract_string_literal(value_cursor)
        if literal is None or not looks_like_identifier(literal):
            return None
        names.append(literal)
    return names


def find_py_buildvalue_call(return_stmt: Cursor) -> Cursor | None:
    """在 `return` 子树中定位 `Py_BuildValue` 调用。"""
    for node in walk_cursor(return_stmt):
        if node.kind != CursorKind.CALL_EXPR:
            continue
        call_name = extract_call_name(node)
        if call_name == "Py_BuildValue":
            return node
    return None


def infer_return_type_from_py_buildvalue(return_stmt: Cursor, func_cursor: Cursor) -> str:
    """从 `Py_BuildValue` 的格式串推断返回类型。"""
    call_cursor = find_py_buildvalue_call(return_stmt)
    if call_cursor is None:
        return "object"
    call_args = extract_call_arguments(call_cursor)
    if not call_args:
        return "object"
    format_text = resolve_string_argument(func_cursor, call_args[0])
    if format_text is None:
        return "object"
    return infer_return_type_from_py_buildvalue_format(format_text)


def has_named_reference(node: Cursor, names: set[str]) -> bool:
    """判断子树中是否存在指定名字的引用或调用。"""
    for cursor in walk_cursor(node):
        if cursor.spelling in names:
            return True
    return False


def find_first_call_expr(node: Cursor) -> Cursor | None:
    """在子树中查找首个 `CALL_EXPR`。"""
    for child in walk_cursor(node):
        if child.kind == CursorKind.CALL_EXPR:
            return child
    return None


def infer_return_type_from_call(
    *,
    call_cursor: Cursor,
    return_stmt: Cursor,
    func_cursor: Cursor,
) -> str | None:
    """
    根据返回调用名推断 Python 返回类型。

    优先处理 `Py_BuildValue`、`Py_NewRef` 这类需要额外上下文的调用，
    其余再按前缀表或兜底规则映射。
    """
    call_name = extract_call_name(call_cursor)
    if call_name is None:
        return None

    if call_name == "Py_BuildValue":
        return infer_return_type_from_py_buildvalue(return_stmt=return_stmt, func_cursor=func_cursor)

    if call_name == "Py_NewRef":
        call_args = extract_call_arguments(call_cursor)
        if not call_args:
            return "object"
        target = unwrap_transparent(call_args[0])
        if target.spelling == "Py_None":
            return "None"
        if target.spelling in {"Py_True", "Py_False"}:
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
    for child in walk_cursor(node):
        if child.kind != CursorKind.STRING_LITERAL:
            continue
        text = str(child.spelling).strip('"')
        if "(" not in text or ")" not in text:
            continue
        args_part = text[text.find("(") + 1 : text.rfind(")")]
        args = parse_parser_args(args_part)
        if args:
            signatures.append(ExtractedSignature(arguments=args))
    return signatures


def set_call_params(
    func_cursor: Cursor,
    meth_flags: list[str],
    call_cursor: Cursor,
) -> list[ExtractedArgument] | None:
    """
    基于调用 AST 与方法标志推断参数名、类型和默认值。

    返回 `None` 表示无法可靠解析该调用；返回空列表表示明确无参数。
    """
    call_name = extract_call_name(call_cursor)
    if call_name is None or not is_parameter_parser_call(call_name):
        return None

    format_idx, offset = resolve_format_index(call_name=call_name, meth_flags=meth_flags)
    if format_idx <= 0:
        return None

    call_args = extract_call_arguments(call_cursor)
    format_arg_index = format_idx - 1
    if format_arg_index >= len(call_args):
        return None

    format_markers: list[str] | None = None
    format_token_index = format_arg_index
    for index in range(format_arg_index, len(call_args)):
        parsed = parse_format_arg(func_cursor, call_args[index])
        if parsed is None:
            continue
        format_markers = parsed
        format_token_index = index
        break
    if format_markers is None:
        return None

    required, optional = split_required_optional(format_markers)
    param_cursor = format_token_index + offset
    if param_cursor < len(call_args):
        if extract_keyword_names(func_cursor, call_args[param_cursor]) is not None:
            param_cursor += 1

    result: list[ExtractedArgument] = []
    kw_only = False
    for marker, is_optional in ([(m, False) for m in required] + [(m, True) for m in optional]):
        if marker in {"*", "$"}:
            kw_only = True
            continue
        if marker == ":":
            break
        if param_cursor >= len(call_args):
            break

        name = extract_output_argument_name(call_args[param_cursor])
        param_cursor += 1
        if name is None or not looks_like_identifier(name):
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
    macro_names = set(RETURN_MACRO_TYPE_MAP)
    for macro_name, type_name in RETURN_MACRO_TYPE_MAP.items():
        if has_named_reference(return_stmt, {macro_name}):
            return type_name

    token_names = set(RETURN_TOKEN_TYPE_MAP)
    for token_name, type_name in RETURN_TOKEN_TYPE_MAP.items():
        if has_named_reference(return_stmt, {token_name}):
            return type_name

    if not has_named_reference(return_stmt, macro_names | token_names):
        call_cursor = find_first_call_expr(return_stmt)
        if call_cursor is None:
            return None
        return infer_return_type_from_call(
            call_cursor=call_cursor,
            return_stmt=return_stmt,
            func_cursor=func_cursor,
        )
    return None


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


def infer_function_signature(function: ExtractedFunction) -> None:
    """基于已收集的函数骨架原地补全签名。"""
    func_cursor = function.function_cursor
    if func_cursor is None:
        return

    signatures = extract_signatures_from_function(func_cursor, function.ml_flags)
    return_type_name = infer_return_type_from_function(func_cursor)
    if not signatures:
        fallback = signature_from_param_decls(func_cursor)
        if fallback.arguments:
            signatures = [fallback]

    if not signatures:
        signatures = [ExtractedSignature(arguments=[], return_type_name=return_type_name)]

    signatures = [merge_signature_return_type(sig, return_type_name) for sig in signatures]
    signatures = [apply_method_flags(sig, function.ml_flags) for sig in signatures]
    function.signatures = deduplicate_signatures(signatures)


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

    for macro_name, type_name in RETURN_MACRO_TYPE_MAP.items():
        if has_named_reference(func_cursor, {macro_name}):
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


def collect_pyarg_calls(node: Cursor) -> list[Cursor]:
    """递归收集参数解析调用（`PyArg_*`）的 `CALL_EXPR`。"""
    result: list[Cursor] = []
    for child in walk_cursor(node):
        if child.kind != CursorKind.CALL_EXPR:
            continue
        call_name = extract_call_name(child)
        if call_name is None or not is_parameter_parser_call(call_name):
            continue
        result.append(child)
    return result


def extract_signatures_from_function(func_cursor: Cursor, meth_flags: list[str]) -> list[ExtractedSignature]:
    """从函数体中提取候选签名，并在末尾做去重。"""
    signatures: list[ExtractedSignature] = []
    for call_cursor in collect_pyarg_calls(func_cursor):
        args = set_call_params(func_cursor, meth_flags, call_cursor)
        if args is not None:
            signatures.append(ExtractedSignature(arguments=args))

    for node in walk_cursor(func_cursor):
        if node.kind == CursorKind.DECL_STMT:
            signatures.extend(extract_parser_signatures(node))
    return deduplicate_signatures(signatures)
