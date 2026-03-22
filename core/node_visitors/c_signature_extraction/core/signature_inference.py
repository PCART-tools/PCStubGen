from __future__ import annotations

import ast
from collections.abc import Iterable

from clang.cindex import Cursor, CursorKind

from .cursor_utils import unwrap_transparent, walk_cursor
from .models import ExtractedFunction, ExtractedSignature
from .py_buildvalue_type_nodes import AnyTypeNode, NamedTypeNode, TypeNode, UnionTypeNode
from .py_buildvalue_type_parser import PyBuildValueTypeParser, PyBuildValueTypeParserError

_OBJECT_NAME_TO_TYPE: dict[str, str] = {
    "Py_None": "None",
    "Py_True": "bool",
    "Py_False": "bool",
}

_RETURN_MACRO_TO_TYPE: dict[str, str] = {
    "Py_RETURN_NONE": "None",
    "Py_RETURN_TRUE": "bool",
    "Py_RETURN_FALSE": "bool",
    "Py_RETURN_NAN": "float",
    "Py_RETURN_INF": "float",
}

_FUNCTION_NAME_TO_TYPE: dict[str, str] = {
    # bool
    "PyBool_FromLong": "bool",

    # int
    "PyLong_FromLong": "int",
    "PyLong_FromUnsignedLong": "int",
    "PyLong_FromSsize_t": "int",
    "PyLong_FromSize_t": "int",
    "PyLong_FromLongLong": "int",
    "PyLong_FromUnsignedLongLong": "int",
    "PyLong_FromDouble": "int",
    "PyLong_FromString": "int",
    "PyLong_FromUnicodeObject": "int",
    "PyLong_FromVoidPtr": "int",

    # float
    "PyFloat_FromString": "float",
    "PyFloat_FromDouble": "float",

    # complex
    "PyComplex_FromCComplex": "complex",
    "PyComplex_FromDoubles": "complex",

    # str
    "PyUnicode_New": "str",
    "PyUnicode_FromKindAndData": "str",
    "PyUnicode_FromString": "str",
    "PyUnicode_FromStringAndSize": "str",
    "PyUnicode_FromFormat": "str",
    "PyUnicode_FromFormatV": "str",
    "PyUnicode_FromObject": "str",
    "PyUnicode_FromEncodedObject": "str",
    "PyUnicode_FromWideChar": "str",
    "PyUnicode_Decode": "str",
    "PyUnicode_DecodeUTF8": "str",
    "PyUnicode_DecodeUTF8Stateful": "str",
    "PyUnicode_DecodeUTF32": "str",
    "PyUnicode_DecodeUTF32Stateful": "str",
    "PyUnicode_DecodeUTF16": "str",
    "PyUnicode_DecodeUTF16Stateful": "str",
    "PyUnicode_DecodeUTF7": "str",
    "PyUnicode_DecodeUTF7Stateful": "str",
    "PyUnicode_DecodeUnicodeEscape": "str",
    "PyUnicode_DecodeRawUnicodeEscape": "str",
    "PyUnicode_DecodeLatin1": "str",
    "PyUnicode_DecodeASCII": "str",
    "PyUnicode_DecodeCharmap": "str",
    "PyUnicode_DecodeLocaleAndSize": "str",
    "PyUnicode_DecodeLocale": "str",
    "PyUnicode_DecodeFSDefaultAndSize": "str",
    "PyUnicode_DecodeFSDefault": "str",
    "PyUnicode_Translate": "str",
    "PyUnicode_DecodeMBCS": "str",
    "PyUnicode_DecodeMBCSStateful": "str",
    "PyUnicode_DecodeCodePageStateful": "str",
    "PyUnicode_Substring": "str",
    "PyUnicode_Concat": "str",
    "PyUnicode_Join": "str",
    "PyUnicode_Replace": "str",
    "PyUnicode_Format": "str",
    "PyUnicode_InternFromString": "str",

    # bytes
    "PyBytes_FromString": "bytes",
    "PyBytes_FromStringAndSize": "bytes",
    "PyBytes_FromFormat": "bytes",
    "PyBytes_FromFormatV": "bytes",
    "PyBytes_FromObject": "bytes",
    "PyUnicode_AsEncodedString": "bytes",
    "PyUnicode_AsUTF8String": "bytes",
    "PyUnicode_AsUTF32String": "bytes",
    "PyUnicode_AsUTF16String": "bytes",
    "PyUnicode_AsUnicodeEscapeString": "bytes",
    "PyUnicode_AsRawUnicodeEscapeString": "bytes",
    "PyUnicode_AsLatin1String": "bytes",
    "PyUnicode_AsASCIIString": "bytes",
    "PyUnicode_AsCharmapString": "bytes",
    "PyUnicode_EncodeLocale": "bytes",
    "PyUnicode_EncodeFSDefault": "bytes",
    "PyUnicode_AsMBCSString": "bytes",
    "PyUnicode_EncodeCodePage": "bytes",

    # bytearray
    "PyByteArray_FromObject": "bytearray",
    "PyByteArray_FromStringAndSize": "bytearray",
    "PyByteArray_Concat": "bytearray",

    # slice
    "PySlice_New": "slice",

    # memoryview
    "PyMemoryView_FromObject": "memoryview",
    "PyMemoryView_FromMemory": "memoryview",
    "PyMemoryView_FromBuffer": "memoryview",
    "PyMemoryView_GetContiguous": "memoryview",

    # tuple
    "PyTuple_New": "tuple",
    "PyTuple_Pack": "tuple",
    "PyTuple_GetSlice": "tuple",
    "PyList_AsTuple": "tuple",
    "PyUnicode_Partition": "tuple",
    "PyUnicode_RPartition": "tuple",

    # list
    "PyList_New": "list",
    "PyList_GetSlice": "list",
    "PyUnicode_Split": "list",
    "PyUnicode_RSplit": "list",
    "PyUnicode_Splitlines": "list",
    "PyDict_Items": "list",
    "PyDict_Keys": "list",
    "PyDict_Values": "list",

    # dict
    "PyDict_New": "dict",
    "PyDict_Copy": "dict",

    # set
    "PySet_New": "set",

    # frozenset
    "PyFrozenSet_New": "frozenset",
}

def infer_signature(function: ExtractedFunction) -> None:
    """将可识别的返回值类型接入函数签名骨架。"""
    inferred_return_type = infer_return_type(function.function_cursor)
    if inferred_return_type is None:
        return

    if not function.signatures:
        function.signatures.append(
            ExtractedSignature(
                arguments=[],
                return_type_name=inferred_return_type,
            )
        )
        return

    for signature in function.signatures:
        if signature.return_type_name is None:
            signature.return_type_name = inferred_return_type


def infer_return_type(func_cursor: Cursor) -> str | None:
    """遍历函数子树中的 return 语句并汇总可识别的返回类型。"""
    inferred_types: list[TypeNode] = []

    for return_stmt in _iter_return_statements(func_cursor):
        inferred_type = _infer_type_from_return_stmt(return_stmt)
        if inferred_type is None:
            continue
        inferred_types.append(inferred_type)

    merged_type = _merge_inferred_type_nodes(inferred_types)
    if merged_type is None:
        return None
    return merged_type.render()


def _iter_return_statements(func_cursor: Cursor) -> Iterable[Cursor]:
    """按前序遍历函数子树中的所有 return 语句。"""
    for cursor in walk_cursor(func_cursor):
        if cursor.kind == CursorKind.RETURN_STMT:
            yield cursor


def _infer_type_from_return_stmt(return_stmt: Cursor) -> TypeNode | None:
    """从单条 return 语句推断返回类型。"""
    return_expr = _get_return_expression(return_stmt)
    if return_expr is None:
        return None
    return infer_expr_type(return_expr)


def infer_expr_type(expr_cursor: Cursor) -> TypeNode | None:
    """对单个表达式做 Python 类型推断。"""
    normalized_expr = unwrap_transparent(expr_cursor)

    if normalized_expr.kind == CursorKind.CONDITIONAL_OPERATOR:
        return _infer_conditional_operator_type(normalized_expr)

    if normalized_expr.kind == CursorKind.CALL_EXPR:
        return _infer_call_expr_type(normalized_expr)

    if normalized_expr.kind == CursorKind.DECL_REF_EXPR:
        direct_type = _infer_decl_ref_expr_type(normalized_expr)
        if direct_type is not None:
            return direct_type

    return _infer_macro_expr_type(normalized_expr)


def _infer_conditional_operator_type(expr_cursor: Cursor) -> TypeNode | None:
    """推断标准三元表达式 `cond ? a : b` 的结果类型。"""
    children = list(expr_cursor.get_children())
    if len(children) != 3:
        return None

    branch_types: list[TypeNode] = []
    for branch in children[1:]:
        inferred = infer_expr_type(branch)
        if inferred is None:
            continue
        branch_types.append(inferred)
    return _merge_inferred_type_nodes(branch_types)


def _get_return_expression(return_stmt: Cursor) -> Cursor | None:
    """获取 return 语句中的返回表达式。"""
    children = list(return_stmt.get_children())
    if not children:
        return None
    return children[0]


def _infer_decl_ref_expr_type(expr_cursor: Cursor) -> TypeNode | None:
    """识别 `DECL_REF_EXPR` 形式的直接对象类型。"""
    identifier_name = _get_cursor_name(expr_cursor)
    mapped = _OBJECT_NAME_TO_TYPE.get(identifier_name)
    if mapped is not None:
        return NamedTypeNode(mapped)
    return None


def _infer_macro_expr_type(expr_cursor: Cursor) -> TypeNode | None:
    """识别 AST 子树中可见名称对应的返回宏类型。"""
    for name in _iter_subtree_names(expr_cursor):
        mapped = _RETURN_MACRO_TO_TYPE.get(name)
        if mapped is not None:
            return NamedTypeNode(mapped)
    return None


def _iter_subtree_names(node: Cursor) -> Iterable[str]:
    """遍历子树中可由 AST 直接读取到的名称。"""
    for cursor in walk_cursor(node):
        name = _get_cursor_name(cursor)
        if name is not None:
            yield name


def _infer_call_expr_type(call_expr_cursor: Cursor) -> TypeNode | None:
    """从调用表达式推断返回类型。"""
    assert call_expr_cursor.kind == CursorKind.CALL_EXPR
    call_name = call_expr_cursor.spelling
    if call_name is None:
        return None

    if call_name == "Py_BuildValue":
        return _infer_py_buildvalue_type(call_expr_cursor)
    mapped = _FUNCTION_NAME_TO_TYPE.get(call_name)
    if mapped is None:
        return None
    return NamedTypeNode(mapped)


def _get_cursor_name(cursor: Cursor) -> str | None:
    """从 AST 节点直接提取名称，优先使用 spelling，再回退到 referenced.spelling。"""
    if cursor.spelling:
        return str(cursor.spelling)

    referenced = getattr(cursor, "referenced", None)
    if referenced is not None and getattr(referenced, "spelling", ""):
        return str(referenced.spelling)
    return None


def _extract_call_arguments(call_cursor: Cursor) -> list[Cursor]:
    """提取调用表达式的实参游标列表。"""
    children = list(call_cursor.get_children())
    if len(children) <= 1:
        return []
    return children[1:]


def _infer_py_buildvalue_type(call_cursor: Cursor) -> TypeNode | None:
    """解析 `Py_BuildValue` 的格式串并返回 parser 推断结果。"""
    args = _extract_call_arguments(call_cursor)
    if not args:
        return None

    format_string = _extract_string_literal(args[0])
    if format_string is None:
        return None

    try:
        parsed_type = PyBuildValueTypeParser(
            format_string,
            args[1:],
            resolve_object_type_func=_resolve_expr_python_type_for_buildvalue,
        ).parse()
        return parsed_type.canonicalize()
    except PyBuildValueTypeParserError:
        return None


def _resolve_expr_python_type_for_buildvalue(cursor: Cursor) -> TypeNode | None:
    """给 `Py_BuildValue` 的对象位提供表达式类型解析。"""
    return infer_expr_type(cursor)


def _merge_inferred_type_nodes(type_nodes: Iterable[TypeNode]) -> TypeNode | None:
    """合并推断结果，并统一复用联合类型的规范化语义。"""
    members = tuple(type_nodes)
    if not members:
        return None
    return UnionTypeNode(members).canonicalize()


def _extract_string_literal(node: Cursor) -> str | None:
    """从子树中提取首个字符串字面量的实际内容。"""
    for cursor in walk_cursor(node):
        if cursor.kind != CursorKind.STRING_LITERAL:
            continue

        if not cursor.spelling:
            continue

        decoded = _decode_string_literal(str(cursor.spelling))
        if decoded is not None:
            return decoded
    return None


def _decode_string_literal(text: str) -> str | None:
    """将 C 风格字符串字面量近似解码为 Python 字符串。"""
    quote_positions = [index for index, char in enumerate(text) if char in {'"', "'"}]
    if not quote_positions:
        return None

    literal = text[min(quote_positions):]
    try:
        value = ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        stripped = literal.strip("\"'")
        return stripped
    if isinstance(value, str):
        return value
    return None
