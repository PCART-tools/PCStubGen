from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from clang.cindex import Cursor, CursorKind
from loguru import logger

from .clang_eval import eval_int
from .cursor_utils import (
    unwrap_transparent,
    var_decl_to_init_list_expr,
    walk_cursor,
    is_nullptr_or_zero,
    extract_string_literal,
    IDENTIFIER_RE,
    DECL_CURSOR_KINDS
)
from .models import ExtractedArgument, ExtractedFunction, ExtractedSignature
from .name_to_type import *
from .py_arg_parse_tuple_and_keywords_type_parser import (
    PyArgParseTupleAndKeywordsTypeParser,
    PyArgParseTupleAndKeywordsTypeParserError,
)
from .py_arg_parse_tuple_type_parser import (
    PyArgParseTupleTypeParser,
    PyArgParseTupleTypeParserError,
)
from .py_build_value_type_nodes import NamedTypeNode, TypeNode, UnionTypeNode
from .py_build_value_type_parser import PyBuildValueTypeParser, PyBuildValueTypeParserError

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

    for call_expr in walk_cursor(func_cursor):
        if call_expr.kind != CursorKind.CALL_EXPR:
            continue

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
    if isinstance(merged_type, UnionTypeNode):
        if len(merged_type.members) > 1:
            logger.warning("返回值Union多个, func_name: {}", func_cursor.spelling)
    return merged_type.render()

def _infer_signature_from_pyarg_call(call_expr: Cursor) -> ExtractedSignature | None:
    """从单个支持的 `PyArg_*` 调用解析参数签名。"""
    assert call_expr.kind == CursorKind.CALL_EXPR
    call_name = call_expr.spelling

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

    format_string = extract_string_literal(args[1])

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

    format_string = extract_string_literal(args[2])

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
    mapped = OBJECT_NAME_TO_TYPE.get(identifier_name)
    if mapped is not None:
        return NamedTypeNode(mapped)
    return None


def _infer_macro_expr_type(expr_cursor: Cursor) -> TypeNode | None:
    """识别 AST 子树中可见名称对应的返回宏类型。"""
    for name in _iter_subtree_names(expr_cursor):
        mapped = RETURN_MACRO_TO_TYPE.get(name)
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

    if call_name == "Py_BuildValue":
        return _infer_py_buildvalue_type(call_expr_cursor)
    mapped = FUNCTION_NAME_TO_TYPE.get(call_name)
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

    format_string = extract_string_literal(args[0])

    try:
        parsed_type = PyBuildValueTypeParser(
            format_string,
            args[1:],
            resolve_object_type_func=infer_expr_type,
        ).parse()
        return parsed_type.canonicalize()
    except PyBuildValueTypeParserError:
        return None


def _resolve_argument_name(c_args: list[Cursor]) -> str | None:
    """将 parser 提供的 decl-ref 槽位变量名按顺序拼接为参数名。"""
    names: list[str] = []
    for c_arg in c_args:
        candidate = _resolve_decl_candidate(c_arg)
        if candidate is None:
            return None

        name = candidate.spelling
        names.append(name)

    if not names:
        return None
    return "_".join(names)


def _resolve_object_type_for_pyarg(cursor: Cursor) -> str | None:
    """解析 `PyArg_*` 中对象槽位对应的 Python 类型名。"""
    source_text = _extract_cursor_source_text(cursor)
    if source_text is None:
        return None

    match = IDENTIFIER_RE.search(source_text)
    if match is None:
        return None
    return PY_TYPE_OBJECT_NAME_TO_TYPE.get(match.group(0))


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
    if target.kind in DECL_CURSOR_KINDS:
        return target

    if target.kind == CursorKind.DECL_REF_EXPR:
        referenced = target.referenced
        if referenced is not None and referenced.kind in DECL_CURSOR_KINDS:
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
        if is_nullptr_or_zero(entry):
            break

        keyword_name = extract_string_literal(entry)
        result.append(keyword_name)

    return result


def _extract_decl_initializer(decl_cursor: Cursor) -> Cursor | None:
    """提取声明节点的初始化表达式。"""
    children = list(decl_cursor.get_children())
    if not children:
        return None
    return unwrap_transparent(children[-1])


def _render_default_expr(expr_cursor: Cursor) -> str | None:
    """将有限集合内的 C 初始化式渲染为 Python 默认值文本。"""
    expr_cursor = unwrap_transparent(expr_cursor)

    # todo)) logic error
    if is_nullptr_or_zero(expr_cursor):
        return "None"

    if expr_cursor.kind == CursorKind.DECL_REF_EXPR:
        return DEFAULT_IDENTIFIER_TO_VALUE.get(expr_cursor.spelling)

    if expr_cursor.kind == CursorKind.STRING_LITERAL:
        decoded = extract_string_literal(expr_cursor)
        return decoded

    if expr_cursor.kind == CursorKind.INTEGER_LITERAL:
        return str(eval_int(expr_cursor))

    # todo)) float...

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


def _merge_inferred_type_nodes(type_nodes: Iterable[TypeNode]) -> TypeNode | None:
    """合并推断结果，并统一复用联合类型的规范化语义。"""
    members = tuple(type_nodes)
    if not members:
        return None
    return UnionTypeNode(members).canonicalize()


def _extract_cursor_source_text(cursor: Cursor) -> str | None:
    """按 cursor extent 从源文件中截取原始源码文本。"""
    extent = cursor.extent
    if extent is None:
        return None

    start = extent.start
    end = extent.end
    if start.file is None or end.file is None:
        return None

    start_file = Path(start.file.name)
    end_file = Path(end.file.name)
    if start_file != end_file:
        return None

    try:
        source_bytes = start_file.read_bytes()
    except OSError:
        return None

    if start.offset < 0 or end.offset < start.offset or end.offset > len(source_bytes):
        return None
    return source_bytes[start.offset:end.offset].decode("utf-8", errors="ignore")