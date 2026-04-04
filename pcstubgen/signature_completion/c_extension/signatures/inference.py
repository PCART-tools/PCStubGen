from __future__ import annotations

from clang.cindex import Cursor, CursorKind
from loguru import logger

from ....checks import check
from ....ir_modules import IRArgumentKind
from ....types import RawType, Type, UnionType
from ..clang.constant_eval import eval_int
from ..clang.cursor_utils import (
    unwrap_transparent,
    var_decl_to_init_list_expr,
    walk_cursor,
    is_nullptr_or_zero,
    extract_string_literal,
    source_range_get_text,
    IDENTIFIER_RE,
    DECL_CURSOR_KINDS
)
from ..models import CArgument, CFunction, CSignature
from ..modules.method_flags import METH_NOARGS, METH_O
from .object_type_maps import OBJECT_NAME_TO_TYPE
from .return_type_maps import FUNCTION_NAME_TO_TYPE
from .py_arg_parse.parser_maps import (
    DEFAULT_IDENTIFIER_TO_VALUE,
    PY_TYPE_OBJECT_NAME_TO_TYPE,
)
from .py_arg_parse.tuple_and_keywords_parser import (
    PyArgParseTupleAndKeywordsTypeParser,
    PyArgParseTupleAndKeywordsTypeParserError,
)
from .py_arg_parse.tuple_parser import (
    PyArgParseTupleTypeParser,
    PyArgParseTupleTypeParserError,
)
from .py_build_value.parser import PyBuildValueTypeParser, PyBuildValueTypeParserError

_PYARG_PARSETUPLE_CALL_NAMES = {
    "PyArg_ParseTuple",
    "_PyArg_ParseTuple_SizeT",
}

_PYARG_PARSETUPLE_AND_KEYWORDS_CALL_NAMES = {
    "PyArg_ParseTupleAndKeywords",
    "_PyArg_ParseTupleAndKeywords_SizeT",
}


def infer_signature(c_function: CFunction) -> list[CSignature]:
    """汇合参数推断与返回值推断结果，生成函数签名列表。"""
    inferred_argument_lists = infer_argument_lists_from_flags(c_function)
    if inferred_argument_lists is None:
        inferred_argument_lists = infer_argument_lists(c_function.function_cursor)

    inferred_return_type = infer_return_type(c_function.function_cursor)

    if inferred_argument_lists:
        return [
            CSignature(
                arguments=arguments,
                return_type=inferred_return_type,
            )
            for arguments in inferred_argument_lists
        ]

    if inferred_return_type is None:
        return []
    return [CSignature(return_type=inferred_return_type)]


def infer_argument_lists_from_flags(
    c_function: CFunction,
) -> list[list[CArgument]] | None:
    """根据 `PyMethodDef.ml_flags` 推断最小参数形状。"""
    ml_flags = c_function.ml_flags

    if ml_flags & METH_NOARGS:
        return [[]]

    if ml_flags & METH_O:
        return [[
            CArgument(
                name="arg",
                type=RawType("object"),
                kind=IRArgumentKind.POSITIONAL_ONLY,
            )
        ]]

    return None


def infer_argument_lists(func_cursor: Cursor) -> list[list[CArgument]]:
    """遍历函数体内支持的 `PyArg_*` 调用并收集参数列表。"""
    inferred_argument_lists: list[list[CArgument]] = []

    for call_expr in walk_cursor(func_cursor):
        if call_expr.kind != CursorKind.CALL_EXPR:
            continue

        call_name = call_expr.spelling
        args = list(call_expr.get_children())[1:]
        if call_name in _PYARG_PARSETUPLE_CALL_NAMES:
            inferred_argument_lists.append(_infer_pyarg_parsetuple_arguments(args))
        elif call_name in _PYARG_PARSETUPLE_AND_KEYWORDS_CALL_NAMES:
            inferred_argument_lists.append(
                _infer_pyarg_parsetuple_and_keywords_arguments(args)
            )
        else:
            continue

    return inferred_argument_lists


def infer_return_type(func_cursor: Cursor) -> Type | None:
    """遍历函数子树中的 return 语句并汇总可识别的返回类型。"""
    inferred_types: list[Type] = []

    for cursor in walk_cursor(func_cursor):
        if cursor.kind != CursorKind.RETURN_STMT:
            continue

        return_children = list(cursor.get_children())
        if not return_children:
            continue
        return_expr = return_children[0]

        inferred_type = infer_expr_type(return_expr)
        if inferred_type is None:
            continue
        inferred_types.append(inferred_type)

    if len(inferred_types) == 0:
        return None

    merged_type = UnionType(tuple(inferred_types)).canonicalize()
    if isinstance(merged_type, UnionType):
        if len(merged_type.members) > 1:
            logger.warning("返回值Union多个, func_name: {}", func_cursor.spelling)
    return merged_type


def _infer_pyarg_parsetuple_arguments(args: list[Cursor]) -> list[CArgument]:
    """调用 `PyArg_ParseTuple` parser 解析参数列表。"""
    format_string = extract_string_literal(args[1])
    check(format_string is not None, "PyArg_ParseTuple format string 必须是字符串字面量。")

    try:
        return PyArgParseTupleTypeParser(
            format_string,
            args[2:],
            resolve_name_func=_resolve_argument_name,
            resolve_object_type_func=_resolve_object_type_for_pyarg,
            resolve_default_value_func=_resolve_default_value_for_pyarg,
        ).parse()
    except PyArgParseTupleTypeParserError as ex:
        raise RuntimeError("解析 PyArg_ParseTuple 参数失败。") from ex


def _infer_pyarg_parsetuple_and_keywords_arguments(
    args: list[Cursor],
) -> list[CArgument]:
    """调用 `PyArg_ParseTupleAndKeywords` parser 解析参数列表。"""
    format_string = extract_string_literal(args[2])
    check(
        format_string is not None,
        "PyArg_ParseTupleAndKeywords format string 必须是字符串字面量。",
    )
    kwlist = _extract_kwlist(args[3])

    try:
        return PyArgParseTupleAndKeywordsTypeParser(
            format_string,
            kwlist,
            args[4:],
            resolve_object_type_func=_resolve_object_type_for_pyarg,
            resolve_default_value_func=_resolve_default_value_for_pyarg,
        ).parse()
    except PyArgParseTupleAndKeywordsTypeParserError as ex:
        raise RuntimeError("解析 PyArg_ParseTupleAndKeywords 参数失败。") from ex


def infer_expr_type(expr: Cursor) -> Type | None:
    """对单个表达式做 Python 类型推断。"""
    expr = unwrap_transparent(expr)

    if expr.kind == CursorKind.CONDITIONAL_OPERATOR:
        return _infer_conditional_operator_type(expr)

    if expr.kind == CursorKind.CALL_EXPR:
        return _infer_call_expr_type(expr)

    if expr.kind == CursorKind.UNARY_OPERATOR:
        # 可能为&Obj
        child = next(expr.get_children())
        child = unwrap_transparent(child)
        if child.kind == CursorKind.DECL_REF_EXPR:
            return _infer_decl_ref_expr_type(child)

    return None


def _infer_conditional_operator_type(expr_cursor: Cursor) -> Type | None:
    """推断标准三元表达式 `cond ? a : b` 的结果类型。"""
    assert expr_cursor.kind == CursorKind.CONDITIONAL_OPERATOR
    children = list(expr_cursor.get_children())
    if len(children) != 3:
        raise RuntimeError(
            f"CONDITIONAL_OPERATOR 子节点数量非法: expected 3, got {len(children)}"
        )

    branch_types: list[Type] = []
    for branch in children[1:]:
        inferred = infer_expr_type(branch)
        if inferred is None:
            continue
        branch_types.append(inferred)
    if len(branch_types) == 0:
        return None
    return UnionType(tuple(branch_types))


def _infer_decl_ref_expr_type(expr_cursor: Cursor) -> Type | None:
    """识别 `DECL_REF_EXPR` 形式的直接对象类型。"""
    assert expr_cursor.kind == CursorKind.DECL_REF_EXPR

    identifier_name = _get_cursor_name(expr_cursor)
    mapped = OBJECT_NAME_TO_TYPE.get(identifier_name)
    if mapped is not None:
        return mapped
    return None


def _infer_call_expr_type(cursor: Cursor) -> Type | None:
    """
    从调用表达式推断返回类型。
    不一定能从spelling获取，可能是宏 #define PyArray_Return (*(PyObject *(*)(PyArrayObject *)) PyArray_API[76])
    选择从源代码获取
    """
    assert cursor.kind == CursorKind.CALL_EXPR
    children = list(cursor.get_children())
    func_cursor = children[0]
    call_name = None
    if getattr(func_cursor, "extent", None) is not None:
        call_name = source_range_get_text(func_cursor.extent)
    if call_name is None:
        call_name = _get_cursor_name(func_cursor)
    if call_name is None:
        return None

    if call_name == "Py_BuildValue":
        return _infer_py_buildvalue_type(cursor)
    mapped = FUNCTION_NAME_TO_TYPE.get(call_name)
    if mapped is None:
        return None
    return mapped


def _get_cursor_name(cursor: Cursor) -> str | None:
    """从 AST 节点直接提取名称，优先使用 spelling，再回退到 referenced.spelling。"""
    if cursor.spelling:
        return str(cursor.spelling)

    referenced = cursor.referenced
    if referenced is not None and referenced.spelling:
        return str(referenced.spelling)
    return None

def _infer_py_buildvalue_type(call_cursor: Cursor) -> Type | None:
    """解析 `Py_BuildValue` 的格式串并返回 parser 推断结果。"""
    args = list(call_cursor.get_children())[1:]
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


def _resolve_argument_name(c_args: list[Cursor]) -> str:
    """将 parser 提供的 decl-ref 槽位变量名按顺序拼接为参数名。"""
    names: list[str] = []
    for c_arg in c_args:
        candidate = _resolve_decl_candidate(c_arg)
        check(candidate is not None, "无法将 C 参数槽位解析为声明节点。")

        name = candidate.spelling
        names.append(name)

    check(bool(names), "参数槽位列表为空，无法生成参数名。")
    return "_".join(names)


def _resolve_object_type_for_pyarg(cursor: Cursor) -> Type | None:
    """
    解析 `PyArg_*` 中对象槽位对应的 Python 类型名。
    可能名字是宏
    """
    source_text = source_range_get_text(cursor.extent)
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


def _extract_kwlist(node: Cursor) -> list[str]:
    """解析 `PyArg_ParseTupleAndKeywords` 的静态关键字名数组。"""
    kwlist_decl = _resolve_decl_candidate(node)
    check(
        kwlist_decl is not None and kwlist_decl.kind == CursorKind.VAR_DECL,
        "kwlist 必须引用 VAR_DECL。",
    )

    init_list_expr = var_decl_to_init_list_expr(kwlist_decl)
    check(init_list_expr is not None, "kwlist 必须使用初始化列表定义。")

    result: list[str] = []
    for child in init_list_expr.get_children():
        entry = unwrap_transparent(child)
        if is_nullptr_or_zero(entry):
            break

        keyword_name = extract_string_literal(entry)
        check(keyword_name is not None, "kwlist 中的关键字名必须是字符串字面量。")
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
