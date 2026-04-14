from __future__ import annotations

from clang.cindex import Cursor, CursorKind, StorageClass, TypeKind
from loguru import logger

from ....models import Argument, ArgumentKind, Signature
from ....type_models import AnyType, RawType, Type, UnionType
from ..clang.ast_utils import (
    DECL_CURSOR_KINDS,
    IDENTIFIER_RE,
    get_cursor_text,
    get_string_literal,
    is_nullptr_or_zero,
    unwrap_transparent,
    var_decl_to_init_list_expr,
    walk_cursor,
)
from ..clang.libclang_wrap import evaluate_cursor
from ..method_flags import (
    METH_FASTCALL,
    METH_KEYWORDS,
    METH_NOARGS,
    METH_O,
    METH_VARARGS,
)
from .py_arg_parse.tuple_and_keywords_parser import (
    PyArgParseTupleAndKeywordsTypeParser,
)
from .py_arg_parse.tuple_parser import (
    PyArgParseTupleTypeParser,
)
from .py_build_value.parser import PyBuildValueTypeParser
from .rules import (
    FUNCTION_NAME_TO_TYPE,
    OBJECT_NAME_TO_TYPE,
    PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE,
    PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE,
)

_PYARG_PARSETUPLE_CALL_NAMES = {
    "PyArg_ParseTuple",
    "_PyArg_ParseTuple_SizeT",
}

_PYARG_PARSETUPLE_AND_KEYWORDS_CALL_NAMES = {
    "PyArg_ParseTupleAndKeywords",
    "_PyArg_ParseTupleAndKeywords_SizeT",
}


def infer_signature(
    func_cursor: Cursor,
    *,
    flags: int = 0,
) -> list[Signature]:
    """汇合参数推断与返回值推断结果，直接生成签名。"""
    inferred_argument_lists = infer_argument_lists(func_cursor)
    inferred_return_type = infer_return_type(func_cursor)

    if inferred_argument_lists:
        return [
            Signature(
                args=arguments,
                return_type=inferred_return_type,
            )
            for arguments in inferred_argument_lists
        ]

    minimal_signatures = infer_minimal_signatures(
        flags,
        return_type=inferred_return_type,
    )
    if minimal_signatures:
        return minimal_signatures
    return [Signature(return_type=inferred_return_type)]


def infer_minimal_signatures(
    flags: int,
    *,
    return_type: Type,
) -> list[Signature]:
    """根据来自 `PyMethodDef.ml_flags` 的 flags 值推断最小签名。"""
    argument_lists = infer_argument_lists_from_flags(flags)
    if not argument_lists:
        return []
    return [
        Signature(
            args=arguments,
            return_type=return_type,
        )
        for arguments in argument_lists
    ]


def infer_argument_lists_from_flags(
    flags: int,
) -> list[list[Argument]]:
    """根据来自 `PyMethodDef.ml_flags` 的 flags 值推断最小参数形状。"""
    if flags & METH_NOARGS:
        return [[]]

    if flags & METH_O:
        return [[
            Argument(
                name="arg",
                type=RawType("object"),
                kind=ArgumentKind.POSITIONAL_ONLY,
            )
        ]]

    if flags & (METH_VARARGS | METH_FASTCALL):
        arguments = [
            Argument(
                name="args",
                type=RawType("object"),
                kind=ArgumentKind.VAR_POSITIONAL,
            )
        ]
        if flags & METH_KEYWORDS:
            arguments.append(
                Argument(
                    name="kwargs",
                    type=RawType("object"),
                    kind=ArgumentKind.VAR_KEYWORD,
                )
            )
        return [arguments]

    return []


def infer_argument_lists(func_cursor: Cursor) -> list[list[Argument]]:
    """遍历函数体内支持的 `PyArg_*` 调用并收集参数列表。"""
    arguments_list: list[list[Argument]] = []

    for call_expr in walk_cursor(func_cursor):
        if call_expr.kind != CursorKind.CALL_EXPR:
            continue

        call_name = call_expr.spelling
        if call_name in _PYARG_PARSETUPLE_CALL_NAMES:
            arguments_list.append(_infer_pyarg_parsetuple_arguments(call_expr))
        elif call_name in _PYARG_PARSETUPLE_AND_KEYWORDS_CALL_NAMES:
            arguments_list.append(_infer_pyarg_parsetuple_and_keywords_arguments(call_expr))

    return arguments_list


def infer_return_type(func_cursor: Cursor) -> Type:
    """遍历函数子树中的 return 语句并汇总返回类型。"""
    inferred_types: list[Type] = []

    for cursor in walk_cursor(func_cursor):
        if cursor.kind != CursorKind.RETURN_STMT:
            continue

        try:
            return_expr = list(cursor.get_children())[0]
            inferred_types.append(infer_expr_type(return_expr))
        except Exception as ex:
            logger.warning(
                "跳过无法推断的 return 表达式, func_name: {}, reason: {}: {}",
                func_cursor.spelling,
                type(ex).__name__,
                ex,
            )

    merged_type = UnionType(tuple(inferred_types)).canonicalize()
    if isinstance(merged_type, UnionType) and len(merged_type.members) == 0:
        return AnyType()
    if isinstance(merged_type, UnionType) and len(merged_type.members) > 1:
        logger.warning("返回值Union多个, func_name: {}", func_cursor.spelling)
    return merged_type


def _infer_pyarg_parsetuple_arguments(call_expr: Cursor) -> list[Argument]:
    """调用 `PyArg_ParseTuple` parser 解析参数列表。"""
    args = list(call_expr.get_children())[1:]
    format_string = get_string_literal(args[1])

    return PyArgParseTupleTypeParser(
        format_string,
        args[2:],
        infer_name_func=_infer_argument_name,
        infer_type_object_func=_infer_type_object_type_for_pyarg,
        infer_converter_type_func=_infer_converter_type_for_pyarg,
        infer_default_value_func=_infer_default_value_for_pyarg,
    ).parse()


def _infer_pyarg_parsetuple_and_keywords_arguments(call_expr: Cursor) -> list[Argument]:
    """调用 `PyArg_ParseTupleAndKeywords` parser 解析参数列表。"""
    args = list(call_expr.get_children())[1:]
    format_string = get_string_literal(args[2])
    kwlist = _extract_kwlist(args[3])

    return PyArgParseTupleAndKeywordsTypeParser(
        format_string,
        kwlist,
        args[4:],
        infer_type_object_func=_infer_type_object_type_for_pyarg,
        infer_converter_type_func=_infer_converter_type_for_pyarg,
        infer_default_value_func=_infer_default_value_for_pyarg,
    ).parse()


def infer_expr_type(expr: Cursor) -> Type:
    """对单个表达式做 Python 类型推断。"""
    expr = unwrap_transparent(expr)

    if expr.kind == CursorKind.CONDITIONAL_OPERATOR:
        return _infer_conditional_operator_type(expr)

    if expr.kind == CursorKind.CALL_EXPR:
        return _infer_call_expr_type(expr)

    if expr.kind == CursorKind.DECL_REF_EXPR:
        return _infer_decl_ref_expr_type(expr)

    if expr.kind == CursorKind.UNARY_OPERATOR:
        child = next(expr.get_children())
        child = unwrap_transparent(child)
        if child.kind == CursorKind.DECL_REF_EXPR:
            return _infer_decl_ref_expr_type(child)

    if is_nullptr_or_zero(expr):
        return UnionType(())

    raise RuntimeError(f"不支持的表达式类型: {expr.kind.name}, cursor: {expr.location}")


def _infer_conditional_operator_type(expr_cursor: Cursor) -> Type:
    """推断标准三元表达式 `cond ? a : b` 的结果类型。"""
    assert expr_cursor.kind == CursorKind.CONDITIONAL_OPERATOR
    children = list(expr_cursor.get_children())

    branch_types: list[Type] = []
    for branch in children[1:]:
        try:
            branch_types.append(infer_expr_type(branch))
        except Exception as ex:
            logger.warning(
                "跳过无法推断的条件分支表达式, reason: {}: {}",
                type(ex).__name__,
                ex,
            )
    return UnionType(tuple(branch_types))


def _infer_decl_ref_expr_type(expr_cursor: Cursor) -> Type:
    """识别 `DECL_REF_EXPR` 形式的直接对象类型。"""
    assert expr_cursor.kind == CursorKind.DECL_REF_EXPR

    identifier_name = _get_cursor_name(expr_cursor)
    mapped = OBJECT_NAME_TO_TYPE.get(identifier_name)
    if mapped is not None:
        return mapped
    try:
        return _infer_local_decl_ref_expr_type(expr_cursor)
    except RuntimeError as ex:
        raise RuntimeError(
            f"无法识别的对象返回标识符: {identifier_name}, cursor: {expr_cursor.location}"
        ) from ex


def _infer_local_decl_ref_expr_type(expr_cursor: Cursor) -> Type:
    """从函数内局部变量的定值表达式中推断 `DECL_REF_EXPR` 类型。"""
    target_decl = expr_cursor.referenced
    if target_decl is None or target_decl.kind != CursorKind.VAR_DECL:
        raise RuntimeError(f"引用节点未指向局部变量声明, cursor: {expr_cursor.location}")
    if target_decl.storage_class == StorageClass.STATIC:
        raise RuntimeError(f"不追溯 static 局部变量, cursor: {target_decl.location}")

    function_cursor = _find_local_decl_function_parent(target_decl)
    candidate_types: list[Type] = []
    for candidate_expr in _iter_local_decl_assignment_exprs(function_cursor, target_decl):
        candidate_expr = unwrap_transparent(candidate_expr)
        if is_nullptr_or_zero(candidate_expr):
            continue
        candidate_types.append(infer_expr_type(candidate_expr).canonicalize())

    if not candidate_types:
        raise RuntimeError(f"局部变量没有可用定值表达式: {target_decl.spelling}")

    inferred_type = candidate_types[0]
    for candidate_type in candidate_types[1:]:
        if candidate_type != inferred_type:
            raise RuntimeError(
                f"局部变量定值表达式类型不收敛: {target_decl.spelling}, "
                f"left: {inferred_type.render()}, right: {candidate_type.render()}"
            )
    return inferred_type


def _find_local_decl_function_parent(decl_cursor: Cursor) -> Cursor:
    """从声明节点的语义父节点中定位所在函数。"""
    parent = decl_cursor.semantic_parent
    while parent is not None:
        if parent.kind == CursorKind.FUNCTION_DECL:
            return parent
        parent = parent.semantic_parent
    raise RuntimeError(
        f"局部变量声明不在函数内: {decl_cursor.spelling}, cursor: {decl_cursor.location}"
    )


def _iter_local_decl_assignment_exprs(function_cursor: Cursor, target_decl: Cursor) -> list[Cursor]:
    """收集函数内目标局部变量的声明初始化和直接赋值右值表达式。"""
    candidates: list[Cursor] = []
    initializer = _extract_optional_decl_initializer(target_decl)
    if initializer is not None:
        candidates.append(initializer)

    for cursor in walk_cursor(function_cursor):
        if cursor.kind != CursorKind.BINARY_OPERATOR:
            continue
        assignment_value = _extract_direct_assignment_value(cursor, target_decl)
        if assignment_value is not None:
            candidates.append(assignment_value)
    return candidates


def _extract_optional_decl_initializer(decl_cursor: Cursor) -> Cursor | None:
    """提取声明初始化表达式；无初始化式时返回 `None`。"""
    children = list(decl_cursor.get_children())
    if not children:
        return None

    initializer = unwrap_transparent(children[-1])
    if initializer.kind == CursorKind.TYPE_REF:
        return None
    return initializer


def _extract_direct_assignment_value(assignment_cursor: Cursor, target_decl: Cursor) -> Cursor | None:
    """在 `x = expr` 中提取目标局部变量对应的右值表达式。"""
    token_spellings = [token.spelling for token in assignment_cursor.get_tokens()]
    if token_spellings.count("=") != 1:
        return None

    children = list(assignment_cursor.get_children())
    if len(children) != 2:
        return None

    target_expr = unwrap_transparent(children[0])
    if not _is_decl_ref_to_decl(target_expr, target_decl):
        return None
    return children[1]


def _is_decl_ref_to_decl(expr_cursor: Cursor, target_decl: Cursor) -> bool:
    """判断表达式是否直接引用目标声明节点。"""
    expr_cursor = unwrap_transparent(expr_cursor)
    if expr_cursor.kind != CursorKind.DECL_REF_EXPR:
        return False

    referenced = expr_cursor.referenced
    if referenced is None:
        return False
    if referenced == target_decl:
        return True

    referenced_usr = referenced.get_usr()
    target_usr = target_decl.get_usr()
    return bool(referenced_usr and target_usr and referenced_usr == target_usr)


def _infer_call_expr_type(cursor: Cursor) -> Type:
    """
    从调用表达式推断返回类型。
    不一定能从 spelling 获取，可能是宏展开后的函数指针调用，因此优先回读源码。
    """
    assert cursor.kind == CursorKind.CALL_EXPR
    children = list(cursor.get_children())
    func_cursor = children[0]
    call_name = get_cursor_text(func_cursor)

    if call_name == "Py_BuildValue":
        return _infer_py_buildvalue_type(cursor)
    mapped = FUNCTION_NAME_TO_TYPE.get(call_name)
    if mapped is None:
        raise RuntimeError(
            f"无法识别的返回值工厂调用: {call_name}, cursor: {cursor.location}"
        )
    return mapped


def _get_cursor_name(cursor: Cursor) -> str:
    """从 AST 节点直接提取名称，优先使用 spelling，再回退到 referenced.spelling。"""
    if cursor.spelling:
        return str(cursor.spelling)

    referenced = cursor.referenced
    if referenced is not None and referenced.spelling:
        return str(referenced.spelling)
    raise RuntimeError(f"无法从 AST 节点提取名称, cursor: {cursor.location}")


def _infer_py_buildvalue_type(call_cursor: Cursor) -> Type:
    """解析 `Py_BuildValue` 的格式串并返回 parser 推断结果。"""
    args = list(call_cursor.get_children())[1:]
    format_string = get_string_literal(args[0])

    return PyBuildValueTypeParser(
        format_string,
        args[1:],
        infer_object_type_func=infer_expr_type,
    ).parse()


def _infer_argument_name(c_args: list[Cursor]) -> str:
    """将 parser 提供的 decl-ref 槽位变量名按顺序拼接为参数名。"""
    names: list[str] = []
    for c_arg in c_args:
        candidate = _find_decl_candidate(c_arg)
        names.append(candidate.spelling)

    return "_".join(names)


def _infer_type_object_type_for_pyarg(cursor: Cursor) -> Type:
    """解析 `PyArg_*` 中 `O!` 类型对象槽位对应的 Python 类型名。"""
    source_text = get_cursor_text(cursor)
    match = IDENTIFIER_RE.search(source_text)
    if match is None:
        raise RuntimeError(
            f"类型对象槽位源码中未找到标识符, source_text: {source_text!r}, cursor: {cursor.location}"
        )
    type_name = match.group(0)
    mapped = PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE.get(type_name)
    if mapped is None:
        raise RuntimeError(
            f"无法识别的类型对象标识符: {type_name}, source_text: {source_text!r}, cursor: {cursor.location}"
        )
    return mapped


def _infer_converter_type_for_pyarg(cursor: Cursor) -> Type:
    """解析 `PyArg_*` 中 `O&` converter 槽位对应的 Python 类型名。"""
    source_text = get_cursor_text(cursor)
    match = IDENTIFIER_RE.search(source_text)
    if match is None:
        raise RuntimeError(
            f"converter 槽位源码中未找到标识符, source_text: {source_text!r}, cursor: {cursor.location}"
        )
    converter_name = match.group(0)
    mapped = PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE.get(converter_name)
    if mapped is None:
        raise RuntimeError(
            f"无法识别的 converter 标识符: {converter_name}, source_text: {source_text!r}, cursor: {cursor.location}"
        )
    return mapped


def _infer_default_value_for_pyarg(cursor: Cursor) -> str:
    """从参数接收变量的声明初始化式中解析默认值文本。"""
    target_decl = _find_decl_candidate(cursor)

    initializer = _extract_decl_initializer(target_decl)
    return _render_default_expr(initializer, target_decl)


def _find_decl_candidate(cursor: Cursor) -> Cursor:
    """将实参槽位解析为被写入的目标声明节点。"""
    target = _unwrap_pointer_target(cursor)
    if target.kind in DECL_CURSOR_KINDS:
        return target

    if target.kind == CursorKind.DECL_REF_EXPR:
        referenced = target.referenced
        if referenced is not None and referenced.kind in DECL_CURSOR_KINDS:
            return referenced
    raise RuntimeError(f"无法将 C 参数槽位解析为声明节点, cursor: {cursor.location}")


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
    if kwlist_decl.kind != CursorKind.VAR_DECL:
        raise RuntimeError(f"kwlist 必须引用 VAR_DECL, cursor: {node.location}")

    try:
        init_list_expr = var_decl_to_init_list_expr(kwlist_decl)
    except RuntimeError as ex:
        raise RuntimeError(
            f"kwlist 必须使用初始化列表定义, cursor: {kwlist_decl.location}"
        ) from ex

    result: list[str] = []
    for child in init_list_expr.get_children():
        entry = unwrap_transparent(child)
        if is_nullptr_or_zero(entry):
            break

        try:
            keyword_name = get_string_literal(entry)
        except RuntimeError as ex:
            raise RuntimeError(
                f"kwlist 中的关键字名必须是字符串字面量, cursor: {entry.location}"
            ) from ex
        result.append(keyword_name)

    return result


def _extract_decl_initializer(decl_cursor: Cursor) -> Cursor:
    """提取声明节点的初始化表达式。"""
    children = list(decl_cursor.get_children())
    if not children:
        raise RuntimeError(
            f"声明节点缺少初始化表达式: {decl_cursor.spelling}, cursor: {decl_cursor.location}"
        )
    return unwrap_transparent(children[-1])


def _render_default_expr(expr: Cursor, target_decl: Cursor) -> str:
    """结合目标声明节点，将有限集合内的 C 初始化式渲染为 Python 默认值文本。"""
    expr = unwrap_transparent(expr)

    if expr.kind == CursorKind.STRING_LITERAL:
        return get_string_literal(expr)

    if expr.kind == CursorKind.FLOATING_LITERAL:
        return str(evaluate_cursor(expr))

    if expr.kind == CursorKind.INTEGER_LITERAL and target_decl.type.get_canonical().kind != TypeKind.POINTER:
        return str(evaluate_cursor(expr))

    raise RuntimeError(
        f"不支持的默认值表达式类型: {expr.kind.name}, cursor: {expr.location}"
    )
