from __future__ import annotations

from clang.cindex import Cursor, CursorKind
from loguru import logger

from ....models import Argument, ArgumentKind, Signature
from ....types import AnyType, RawType, Type, UnionType
from ..clang.constant_eval import eval_int
from ..clang.ast_utils import (
    DECL_CURSOR_KINDS,
    IDENTIFIER_RE,
    cursor_get_text,
    extract_string_literal,
    is_nullptr_or_zero,
    unwrap_transparent,
    var_decl_to_init_list_expr,
    walk_cursor,
)
from ..method_flags import (
    METH_FASTCALL,
    METH_KEYWORDS,
    METH_NOARGS,
    METH_O,
    METH_VARARGS,
)
from .object_type_maps import OBJECT_NAME_TO_TYPE
from .py_arg_parse.parser_maps import DEFAULT_IDENTIFIER_TO_VALUE, PY_TYPE_OBJECT_NAME_TO_TYPE
from .py_arg_parse.tuple_and_keywords_parser import (
    PyArgParseTupleAndKeywordsTypeParser,
    PyArgParseTupleAndKeywordsTypeParserError,
)
from .py_arg_parse.tuple_parser import (
    PyArgParseTupleTypeParser,
    PyArgParseTupleTypeParserError,
)
from .py_build_value.parser import PyBuildValueTypeParser, PyBuildValueTypeParserError
from .return_type_maps import FUNCTION_NAME_TO_TYPE

_PYARG_PARSETUPLE_CALL_NAMES = {
    "PyArg_ParseTuple",
    "_PyArg_ParseTuple_SizeT",
}

_PYARG_PARSETUPLE_AND_KEYWORDS_CALL_NAMES = {
    "PyArg_ParseTupleAndKeywords",
    "_PyArg_ParseTupleAndKeywords_SizeT",
}


def infer_signature(
    function_cursor: Cursor,
    *,
    ml_flags: int = 0,
) -> list[Signature]:
    """汇合参数推断与返回值推断结果，直接生成签名。"""
    inferred_return_type = infer_return_type(function_cursor)
    inferred_argument_lists = infer_argument_lists(function_cursor)

    if inferred_argument_lists:
        return [
            Signature(
                args=arguments,
                return_type=_default_return_type(inferred_return_type),
            )
            for arguments in inferred_argument_lists
        ]

    minimal_signatures = infer_minimal_signatures(
        ml_flags,
        return_type=inferred_return_type,
    )
    if minimal_signatures:
        return minimal_signatures

    if inferred_return_type is None:
        return []
    return [Signature(return_type=inferred_return_type)]


def infer_minimal_signatures(
    ml_flags: int,
    *,
    return_type: Type | None = None,
) -> list[Signature]:
    """根据 `PyMethodDef.ml_flags` 推断最小签名。"""
    argument_lists = infer_argument_lists_from_flags(ml_flags)
    if argument_lists is None:
        return []

    effective_return_type = _default_return_type(return_type)
    return [
        Signature(
            args=arguments,
            return_type=effective_return_type,
        )
        for arguments in argument_lists
    ]


def infer_argument_lists_from_flags(
    ml_flags: int,
) -> list[list[Argument]] | None:
    """根据 `PyMethodDef.ml_flags` 推断最小参数形状。"""
    if ml_flags & METH_NOARGS:
        return [[]]

    if ml_flags & METH_O:
        return [[
            Argument(
                name="arg",
                type=RawType("object"),
                kind=ArgumentKind.POSITIONAL_ONLY,
            )
        ]]

    if ml_flags & (METH_VARARGS | METH_FASTCALL):
        arguments = [
            Argument(
                name="args",
                type=RawType("object"),
                kind=ArgumentKind.VAR_POSITIONAL,
            )
        ]
        if ml_flags & METH_KEYWORDS:
            arguments.append(
                Argument(
                    name="kwargs",
                    type=RawType("object"),
                    kind=ArgumentKind.VAR_KEYWORD,
                )
            )
        return [arguments]

    return None


def infer_argument_lists(func_cursor: Cursor) -> list[list[Argument]]:
    """遍历函数体内支持的 `PyArg_*` 调用并收集参数列表。"""
    inferred_argument_lists: list[list[Argument]] = []

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
    if isinstance(merged_type, UnionType) and len(merged_type.members) > 1:
        logger.warning("返回值Union多个, func_name: {}", func_cursor.spelling)
    return merged_type


def _infer_pyarg_parsetuple_arguments(args: list[Cursor]) -> list[Argument]:
    """调用 `PyArg_ParseTuple` parser 解析参数列表。"""
    try:
        format_string = extract_string_literal(args[1])
    except RuntimeError as ex:
        raise RuntimeError("PyArg_ParseTuple format string 必须是字符串字面量。") from ex

    try:
        return PyArgParseTupleTypeParser(
            format_string,
            args[2:],
            infer_name_func=_infer_argument_name,
            infer_object_type_func=_infer_object_type_for_pyarg,
            infer_default_value_func=_infer_default_value_for_pyarg,
        ).parse()
    except PyArgParseTupleTypeParserError as ex:
        raise RuntimeError("解析 PyArg_ParseTuple 参数失败。") from ex


def _infer_pyarg_parsetuple_and_keywords_arguments(
    args: list[Cursor],
) -> list[Argument]:
    """调用 `PyArg_ParseTupleAndKeywords` parser 解析参数列表。"""
    try:
        format_string = extract_string_literal(args[2])
    except RuntimeError as ex:
        raise RuntimeError(
            "PyArg_ParseTupleAndKeywords format string 必须是字符串字面量。"
        ) from ex
    kwlist = _extract_kwlist(args[3])

    try:
        return PyArgParseTupleAndKeywordsTypeParser(
            format_string,
            kwlist,
            args[4:],
            infer_object_type_func=_infer_object_type_for_pyarg,
            infer_default_value_func=_infer_default_value_for_pyarg,
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
    不一定能从 spelling 获取，可能是宏展开后的函数指针调用，因此优先回读源码。
    """
    assert cursor.kind == CursorKind.CALL_EXPR
    children = list(cursor.get_children())
    func_cursor = children[0]
    try:
        call_name = cursor_get_text(func_cursor)
    except RuntimeError:
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

    try:
        format_string = extract_string_literal(args[0])
    except RuntimeError as ex:
        raise RuntimeError("Py_BuildValue format string 必须是字符串字面量。") from ex

    try:
        parsed_type = PyBuildValueTypeParser(
            format_string,
            args[1:],
            infer_object_type_func=infer_expr_type,
        ).parse()
        return parsed_type.canonicalize()
    except PyBuildValueTypeParserError as ex:
        raise RuntimeError("解析 Py_BuildValue 返回类型失败。") from ex


def _infer_argument_name(c_args: list[Cursor]) -> str:
    """将 parser 提供的 decl-ref 槽位变量名按顺序拼接为参数名。"""
    names: list[str] = []
    for c_arg in c_args:
        candidate = _find_decl_candidate(c_arg)
        if candidate is None:
            raise RuntimeError("无法将 C 参数槽位解析为声明节点。")
        names.append(candidate.spelling)

    return "_".join(names)


def _infer_object_type_for_pyarg(cursor: Cursor) -> Type | None:
    """解析 `PyArg_*` 中对象槽位对应的 Python 类型名。"""
    if getattr(cursor, "extent", None) is None:
        return None

    try:
        source_text = cursor_get_text(cursor)
    except RuntimeError:
        return None

    match = IDENTIFIER_RE.search(source_text)
    if match is None:
        return None
    return PY_TYPE_OBJECT_NAME_TO_TYPE.get(match.group(0))


def _infer_default_value_for_pyarg(cursor: Cursor) -> str | None:
    """从参数接收变量的声明初始化式中解析默认值文本。"""
    target_decl = _find_decl_candidate(cursor)
    if target_decl is None:
        return None

    initializer = _extract_decl_initializer(target_decl)
    if initializer is None:
        return None
    return _render_default_expr(initializer)


def _find_decl_candidate(cursor: Cursor) -> Cursor | None:
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
    kwlist_decl = _find_decl_candidate(node)
    if kwlist_decl is None or kwlist_decl.kind != CursorKind.VAR_DECL:
        raise RuntimeError("kwlist 必须引用 VAR_DECL。")

    try:
        init_list_expr = var_decl_to_init_list_expr(kwlist_decl)
    except RuntimeError as ex:
        raise RuntimeError("kwlist 必须使用初始化列表定义。") from ex

    result: list[str] = []
    for child in init_list_expr.get_children():
        entry = unwrap_transparent(child)
        if is_nullptr_or_zero(entry):
            break

        try:
            keyword_name = extract_string_literal(entry)
        except RuntimeError as ex:
            raise RuntimeError("kwlist 中的关键字名必须是字符串字面量。") from ex
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

    if is_nullptr_or_zero(expr_cursor):
        return "None"

    if expr_cursor.kind == CursorKind.DECL_REF_EXPR:
        return DEFAULT_IDENTIFIER_TO_VALUE.get(expr_cursor.spelling)

    if expr_cursor.kind == CursorKind.STRING_LITERAL:
        decoded = extract_string_literal(expr_cursor)
        if decoded is None:
            return None
        return repr(decoded)

    numeric_literal = _render_numeric_literal(expr_cursor)
    if numeric_literal is not None:
        return numeric_literal

    if expr_cursor.kind == CursorKind.UNARY_OPERATOR:
        return _render_unary_numeric_literal(expr_cursor)

    return None


def _render_numeric_literal(expr_cursor: Cursor) -> str | None:
    """渲染整数字面量或浮点字面量。"""
    if expr_cursor.kind == CursorKind.INTEGER_LITERAL:
        value = eval_int(expr_cursor)
        if value is not None:
            return str(value)

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


def _default_return_type(return_type: Type | None) -> Type:
    if return_type is None:
        return AnyType()
    return return_type
