from __future__ import annotations

import ast
from collections.abc import Iterable

from clang.cindex import Cursor, CursorKind

from .cursor_utils import (
    looks_like_identifier,
    unwrap_transparent,
    var_decl_to_init_list_expr,
    walk_cursor,
)
from .models import ExtractedArgument, ExtractedFunction, ExtractedSignature
from .py_arg_parse_tuple_and_keywords_type_parser import (
    PyArgParseTupleAndKeywordsTypeParser,
    PyArgParseTupleAndKeywordsTypeParserError,
)
from .py_arg_parse_tuple_type_parser import (
    PyArgParseTupleTypeParser,
    PyArgParseTupleTypeParserError,
)
from .py_build_value_type_nodes import AnyTypeNode, NamedTypeNode, TypeNode, UnionTypeNode
from .py_build_value_type_parser import PyBuildValueTypeParser, PyBuildValueTypeParserError

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

_PYARG_OBJECT_NAME_TO_TYPE: dict[str, str] = {
    "PyList_Type": "list",
    "PyTuple_Type": "tuple",
    "PyDict_Type": "dict",
    "PyUnicode_Type": "str",
    "PyLong_Type": "int",
    "PyFloat_Type": "float",
    "PyBool_Type": "bool",
    "PyBytes_Type": "bytes",
    "PyByteArray_Type": "bytearray",
    "PySet_Type": "set",
    "PyFrozenSet_Type": "frozenset",
    "PyType_Type": "type",
    "PyBaseObject_Type": "object",
}

_DEFAULT_IDENTIFIER_TO_VALUE: dict[str, str] = {
    "Py_None": "None",
    "Py_True": "True",
    "Py_False": "False",
}

_SUPPORTED_PYARG_CALLS = {
    "PyArg_ParseTuple",
    "PyArg_ParseTupleAndKeywords",
}

_DECL_CURSOR_KINDS = {
    CursorKind.VAR_DECL,
    CursorKind.PARM_DECL,
    CursorKind.FIELD_DECL,
}


def infer_signature(function: ExtractedFunction) -> None:
    """汇合参数推断与返回值推断结果，生成函数签名。"""
    inferred_argument_signatures = infer_argument_signatures(function.function_cursor)
    if inferred_argument_signatures:
        function.signatures = inferred_argument_signatures

    inferred_return_type = infer_return_type(function.function_cursor)
    if inferred_return_type is None:
        return

    if not function.signatures:
        function.signatures.append(ExtractedSignature())

    for signature in function.signatures:
        if signature.return_type_name is None:
            signature.return_type_name = inferred_return_type


def infer_argument_signatures(func_cursor: Cursor) -> list[ExtractedSignature]:
    """遍历函数体内支持的 `PyArg_*` 调用并收集参数签名。"""
    inferred_signatures: list[ExtractedSignature] = []
    seen_keys: set[tuple[tuple[str, str | None, str | None, bool, object], ...]] = set()

    for call_expr in _iter_call_exprs(func_cursor):
        signature = _infer_signature_from_pyarg_call(call_expr)
        if signature is None:
            continue

        signature_key = _make_signature_argument_key(signature.arguments)
        if signature_key in seen_keys:
            continue
        seen_keys.add(signature_key)
        inferred_signatures.append(signature)

    return inferred_signatures


def infer_return_type(func_cursor: Cursor) -> str | None:
    """遍历函数子树中的 return 语句并汇总可识别的返回类型。"""
    inferred_types: list[TypeNode] = []

    for cursor in walk_cursor(func_cursor):
        if cursor.kind != CursorKind.RETURN_STMT:
            continue

        inferred_type = _infer_type_from_return_stmt(cursor)
        if inferred_type is None:
            continue
        inferred_types.append(inferred_type)

    merged_type = _merge_inferred_type_nodes(inferred_types)
    if merged_type is None:
        return None
    return merged_type.render()


def _iter_call_exprs(func_cursor: Cursor) -> Iterable[Cursor]:
    """按前序遍历函数子树中的所有调用表达式。"""
    for cursor in walk_cursor(func_cursor):
        if cursor.kind == CursorKind.CALL_EXPR:
            yield cursor


def _infer_signature_from_pyarg_call(call_expr: Cursor) -> ExtractedSignature | None:
    """从单个支持的 `PyArg_*` 调用解析参数签名。"""
    assert call_expr.kind == CursorKind.CALL_EXPR
    call_name = call_expr.spelling
    if call_name not in _SUPPORTED_PYARG_CALLS:
        return None

    args = _extract_call_arguments(call_expr)
    if call_name == "PyArg_ParseTuple":
        return _infer_pyarg_parsetuple_signature(args)
    if call_name == "PyArg_ParseTupleAndKeywords":
        return _infer_pyarg_parsetuple_and_keywords_signature(args)
    return None


def _infer_pyarg_parsetuple_signature(args: list[Cursor]) -> ExtractedSignature | None:
    """调用 `PyArg_ParseTuple` parser 解析参数签名。"""
    if len(args) < 2:
        return None

    format_string = _extract_string_literal(args[1])
    if format_string is None:
        return None

    try:
        arguments = PyArgParseTupleTypeParser(
            format_string,
            args[2:],
            resolve_name_func=_resolve_argument_name,
            resolve_object_type_func=_resolve_object_type_for_pyarg,
            resolve_default_value_func=_resolve_default_value_for_pyarg,
        ).parse()
    except PyArgParseTupleTypeParserError:
        return None

    return ExtractedSignature(arguments=arguments)


def _infer_pyarg_parsetuple_and_keywords_signature(args: list[Cursor]) -> ExtractedSignature | None:
    """调用 `PyArg_ParseTupleAndKeywords` parser 解析参数签名。"""
    if len(args) < 4:
        return None

    format_string = _extract_string_literal(args[2])
    if format_string is None:
        return None

    kwlist = _extract_kwlist(args[3])
    if kwlist is None:
        return None

    try:
        arguments = PyArgParseTupleAndKeywordsTypeParser(
            format_string,
            kwlist,
            args[4:],
            resolve_object_type_func=_resolve_object_type_for_pyarg,
            resolve_default_value_func=_resolve_default_value_for_pyarg,
        ).parse()
    except PyArgParseTupleAndKeywordsTypeParserError:
        return None

    return ExtractedSignature(arguments=arguments)


def _make_signature_argument_key(
    arguments: list[ExtractedArgument],
) -> tuple[tuple[str, str | None, str | None, bool, object], ...]:
    """构造仅基于参数内容的稳定签名去重键。"""
    return tuple(
        (
            argument.name,
            argument.type_name,
            argument.default_value,
            argument.has_default,
            argument.kind,
        )
        for argument in arguments
    )


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

    referenced = cursor.referenced
    if referenced is not None and referenced.spelling:
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


def _resolve_argument_name(c_args: list[Cursor]) -> str | None:
    """从 parser 提供的 C 槽位中挑选最像 Python 参数名的输出变量。"""
    for c_arg in c_args:
        candidate = _resolve_decl_candidate(c_arg)
        if candidate is None:
            continue

        name = _get_cursor_name(candidate)
        if name is None or not looks_like_identifier(name):
            continue
        if _looks_like_type_or_converter_name(name):
            continue
        return name

    return None


def _resolve_object_type_for_pyarg(cursor: Cursor) -> str | None:
    """解析 `PyArg_*` 中对象槽位对应的 Python 类型名。"""
    target = _unwrap_pointer_target(cursor)
    target_name = _get_cursor_name(target)
    if target_name is not None:
        mapped = _PYARG_OBJECT_NAME_TO_TYPE.get(target_name)
        if mapped is not None:
            return mapped

    target_decl = _resolve_decl_candidate(cursor)
    if target_decl is None or target_decl.kind != CursorKind.VAR_DECL:
        return None

    return _extract_python_type_name_from_type_object(target_decl)


def _resolve_default_value_for_pyarg(cursor: Cursor) -> str | None:
    """从参数接收变量的声明初始化式中解析默认值文本。"""
    target_decl = _resolve_decl_candidate(cursor)
    if target_decl is None:
        return None

    initializer = _extract_decl_initializer(target_decl)
    if initializer is None:
        return None
    return _render_default_expr(initializer)


def _resolve_decl_candidate(cursor: Cursor) -> Cursor | None:
    """将实参槽位解析为被写入的目标声明节点。"""
    target = _unwrap_pointer_target(cursor)
    if target.kind in _DECL_CURSOR_KINDS:
        return target

    if target.kind == CursorKind.DECL_REF_EXPR:
        referenced = target.referenced
        if referenced is not None and referenced.kind in _DECL_CURSOR_KINDS:
            return referenced
    return None


def _unwrap_pointer_target(cursor: Cursor) -> Cursor:
    """剥离透明包装和一层取地址节点，定位到底层目标。"""
    normalized = unwrap_transparent(cursor)
    while normalized.kind == CursorKind.UNARY_OPERATOR:
        children = list(normalized.get_children())
        if len(children) != 1:
            break
        normalized = unwrap_transparent(children[0])
    return normalized


def _looks_like_type_or_converter_name(name: str) -> bool:
    """过滤明显不是 Python 参数名的类型对象与转换器符号。"""
    if name.endswith("_Type"):
        return True
    if name.endswith("Type"):
        return True
    if "Converter" in name:
        return True
    return False


def _extract_kwlist(node: Cursor) -> list[str] | None:
    """解析 `PyArg_ParseTupleAndKeywords` 的静态关键字名数组。"""
    kwlist_decl = _resolve_decl_candidate(node)
    if kwlist_decl is None or kwlist_decl.kind != CursorKind.VAR_DECL:
        return None

    init_list_expr = var_decl_to_init_list_expr(kwlist_decl)
    if init_list_expr is None:
        return None

    result: list[str] = []
    for child in init_list_expr.get_children():
        entry = unwrap_transparent(child)
        if _is_null_like_expr(entry):
            break

        keyword_name = _extract_string_literal(entry)
        if keyword_name is None:
            return None
        result.append(keyword_name)

    return result


def _extract_python_type_name_from_type_object(type_decl: Cursor) -> str | None:
    """从 `PyTypeObject` 变量初始化中提取 Python 侧类型名。"""
    init_list_expr = var_decl_to_init_list_expr(type_decl)
    if init_list_expr is None:
        return None

    tp_name = _extract_tp_name_from_init_list(init_list_expr)
    if tp_name is None or tp_name == "":
        return None
    return tp_name.rsplit(".", 1)[-1]


def _extract_tp_name_from_init_list(init_list_expr: Cursor) -> str | None:
    """从类型对象初始化列表里读取 `.tp_name` 字符串。"""
    for entry in init_list_expr.get_children():
        entry_children = list(entry.get_children())
        if len(entry_children) < 2:
            continue

        member_ref = unwrap_transparent(entry_children[0])
        if member_ref.kind != CursorKind.MEMBER_REF:
            continue
        if _get_cursor_name(member_ref) != "tp_name":
            continue

        return _extract_string_literal(entry_children[1])
    return None


def _extract_decl_initializer(decl_cursor: Cursor) -> Cursor | None:
    """提取声明节点的初始化表达式。"""
    children = list(decl_cursor.get_children())
    if not children:
        return None
    return unwrap_transparent(children[-1])


def _render_default_expr(expr_cursor: Cursor) -> str | None:
    """将有限集合内的 C 初始化式渲染为 Python 默认值文本。"""
    normalized_expr = unwrap_transparent(expr_cursor)

    if _is_null_like_expr(normalized_expr):
        return "None"

    if normalized_expr.kind == CursorKind.DECL_REF_EXPR:
        name = _get_cursor_name(normalized_expr)
        if name is None:
            return None
        return _DEFAULT_IDENTIFIER_TO_VALUE.get(name)

    if normalized_expr.kind == CursorKind.STRING_LITERAL:
        decoded = _extract_string_literal(normalized_expr)
        if decoded is None:
            return None
        return repr(decoded)

    literal_text = _render_numeric_literal(normalized_expr)
    if literal_text is not None:
        return literal_text

    if normalized_expr.kind == CursorKind.UNARY_OPERATOR:
        return _render_unary_numeric_literal(normalized_expr)

    return None


def _render_numeric_literal(expr_cursor: Cursor) -> str | None:
    """渲染整数字面量或浮点字面量。"""
    if expr_cursor.kind not in {CursorKind.INTEGER_LITERAL, CursorKind.FLOATING_LITERAL}:
        return None

    tokens = list(expr_cursor.get_tokens())
    if not tokens:
        return None
    return str(tokens[0].spelling)


def _render_unary_numeric_literal(expr_cursor: Cursor) -> str | None:
    """渲染一层正负号包裹的数字字面量。"""
    children = list(expr_cursor.get_children())
    if len(children) != 1:
        return None

    value_text = _render_numeric_literal(unwrap_transparent(children[0]))
    if value_text is None:
        return None

    tokens = list(expr_cursor.get_tokens())
    for token in tokens:
        spelling = str(token.spelling)
        if spelling in {"+", "-"}:
            return f"{spelling}{value_text}"
    return None


def _is_null_like_expr(expr_cursor: Cursor) -> bool:
    """识别空指针、零值和保留为标识符的 `NULL`。"""
    if expr_cursor.kind in {
        CursorKind.CXX_NULL_PTR_LITERAL_EXPR,
        CursorKind.GNU_NULL_EXPR,
    }:
        return True

    if _is_zero_integer_literal(expr_cursor):
        return True

    for name in _iter_subtree_names(expr_cursor):
        if name in {"NULL", "nullptr"}:
            return True
    return False


def _is_zero_integer_literal(expr_cursor: Cursor) -> bool:
    """在不依赖 libclang 常量求值的前提下识别字面量 `0`。"""
    if expr_cursor.kind != CursorKind.INTEGER_LITERAL:
        return False

    tokens = list(expr_cursor.get_tokens())
    if not tokens:
        return False

    try:
        return int(str(tokens[0].spelling), 0) == 0
    except ValueError:
        return False


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
