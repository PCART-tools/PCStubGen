from __future__ import annotations

from typing import cast

from clang.cindex import Cursor, CursorKind, StorageClass, TypeKind
from loguru import logger

from ....models import Argument, ArgumentKind, Signature
from ....type_models import AnyType, RawType, Type, UnionType
from ..clang.ast_utils import (
    DECL_CURSOR_KINDS,
    IDENTIFIER_RE,
    get_call_expr_source_name,
    get_cursor_text,
    get_string_literal,
    is_nullptr_or_zero,
    unwrap_transparent,
    var_decl_to_init_list_expr,
    walk_cursor,
)
from ..clang.libclang_wrap import (
    CX_BINARY_OPERATOR_ASSIGN,
    evaluate_cursor,
    get_cursor_binary_operator_kind,
)
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

_PYTHON_SINGLETON_DEFAULT_NAME_TO_VALUE = {
    "_Py_NoneStruct": "None",
    "_Py_TrueStruct": "True",
    "_Py_FalseStruct": "False",
}
_BOOL_TYPE = RawType("bool")
_FLOAT_TYPE = RawType("float")


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
        try:
            if call_name in _PYARG_PARSETUPLE_CALL_NAMES:
                arguments_list.append(_infer_pyarg_parsetuple_arguments(call_expr))
            elif call_name in _PYARG_PARSETUPLE_AND_KEYWORDS_CALL_NAMES:
                arguments_list.append(_infer_pyarg_parsetuple_and_keywords_arguments(call_expr))
        except Exception as ex:
            logger.warning(
                "跳过无法推断的 PyArg 参数列表, func_name: {}, call_name: {}, reason: {}: {}",
                func_cursor.spelling,
                call_name,
                type(ex).__name__,
                ex,
            )

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
        infer_default_value_func=lambda cursor, expected_type: _infer_default_value_for_pyarg(
            cursor,
            expected_type,
            before_cursor=call_expr,
        ),
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
        infer_default_value_func=lambda cursor, expected_type: _infer_default_value_for_pyarg(
            cursor,
            expected_type,
            before_cursor=call_expr,
        ),
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
        '''return NULL异常返回分支 union后就不存在了'''
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
        candidate_expr = _unwrap_assignment_chain_value(candidate_expr)
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


def _unwrap_assignment_chain_value(expr_cursor: Cursor) -> Cursor:
    """剥离链式赋值表达式，定位到最终右值。"""
    value_expr = unwrap_transparent(expr_cursor)
    while value_expr.kind == CursorKind.BINARY_OPERATOR:
        if get_cursor_binary_operator_kind(value_expr) != CX_BINARY_OPERATOR_ASSIGN:
            break
        children = list(value_expr.get_children())
        assert len(children) == 2
        value_expr = unwrap_transparent(children[1])
    return value_expr


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
    if get_cursor_binary_operator_kind(assignment_cursor) != CX_BINARY_OPERATOR_ASSIGN:
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
    return _is_same_decl(referenced, target_decl)


def _is_same_decl(left_decl: Cursor, right_decl: Cursor) -> bool:
    """判断两个声明节点是否指向同一个 C 声明。"""
    if left_decl == right_decl:
        return True

    left_usr = left_decl.get_usr()
    right_usr = right_decl.get_usr()
    return bool(left_usr and right_usr and left_usr == right_usr)


def _infer_call_expr_type(cursor: Cursor) -> Type:
    """
    从调用表达式推断返回类型。
    调用名按源码调用表达式起点 token 提取，避免函数式宏的 callee source range
    扩成整段调用文本。
    """
    assert cursor.kind == CursorKind.CALL_EXPR
    call_name = get_call_expr_source_name(cursor)

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
    """将 parser 提供的槽位变量名按首次出现顺序拼接为参数名。"""
    names: list[str] = []
    seen_names: set[str] = set()
    for c_arg in c_args:
        candidate = _find_decl_candidate(c_arg)
        if candidate.spelling in seen_names:
            continue
        names.append(candidate.spelling)
        seen_names.add(candidate.spelling)

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


def _infer_default_value_for_pyarg(
    cursor: Cursor,
    expected_type: Type,
    *,
    before_cursor: Cursor | None = None,
) -> str:
    """从参数接收变量的声明初始化式中解析默认值文本。"""
    array_slot = _extract_array_subscript_slot(cursor)
    if array_slot is not None:
        array_decl, index = array_slot
        expr = _find_array_element_assignment_value(array_decl, index, before_cursor)
        return _render_default_value_expr(expr, array_decl, expected_type)

    target_decl = _find_decl_candidate(cursor)
    expr = _find_decl_assignment_value(target_decl, before_cursor)
    if expr is None:
        expr = _extract_decl_initializer(target_decl)
    return _render_default_value_expr(expr, target_decl, expected_type)


def _render_default_value_expr(expr: Cursor, target_decl: Cursor, expected_type: Type) -> str:
    """将 C 默认值表达式渲染为 Python 字面量。"""
    expr = unwrap_transparent(expr)
    expected_type = expected_type.canonicalize()

    if (
        target_decl.type.get_canonical().kind == TypeKind.POINTER
        and is_nullptr_or_zero(expr)
    ):
        return "..."

    if expr.kind == CursorKind.STRING_LITERAL:
        return repr(get_string_literal(expr))

    if expr.kind == CursorKind.FLOATING_LITERAL:
        return _render_number_default(_evaluate_number_cursor(expr), expected_type)

    if expr.kind == CursorKind.CXX_BOOL_LITERAL_EXPR:
        evaluated = evaluate_cursor(expr)
        if type(evaluated) is not int:
            raise RuntimeError(
                f"C++ bool 字面量求值结果不是整数: {evaluated!r}, cursor: {expr.location}"
            )
        if expected_type == _BOOL_TYPE:
            if evaluated == 0:
                return "False"
            if evaluated == 1:
                return "True"
            raise RuntimeError(
                f"C++ bool 字面量求值结果不是 0 或 1: {evaluated!r}, cursor: {expr.location}"
            )
        return _render_number_default(cast(int, evaluated), expected_type)

    if (
        expr.kind == CursorKind.INTEGER_LITERAL
        and target_decl.type.get_canonical().kind != TypeKind.POINTER
    ):
        evaluated = evaluate_cursor(expr)
        if type(evaluated) is not int:
            raise RuntimeError(
                f"整数默认值求值结果不是整数: {evaluated!r}, cursor: {expr.location}"
            )
        if expected_type == _BOOL_TYPE:
            if evaluated == 0:
                return "False"
            if evaluated == 1:
                return "True"
            raise RuntimeError(
                f"bool 默认值整数不是 0 或 1: {evaluated!r}, cursor: {expr.location}"
            )
        return _render_number_default(cast(int, evaluated), expected_type)

    if expr.kind == CursorKind.UNARY_OPERATOR:
        children = list(expr.get_children())
        assert len(children) == 1
        child = unwrap_transparent(children[0])
        if child.kind == CursorKind.DECL_REF_EXPR:
            rendered = _PYTHON_SINGLETON_DEFAULT_NAME_TO_VALUE.get(child.spelling)
            if rendered is not None:
                return rendered
        if target_decl.type.get_canonical().kind != TypeKind.POINTER:
            return _render_number_default(_evaluate_number_cursor(expr), expected_type)

    raise RuntimeError(
        f"不支持的默认值表达式类型: {expr.kind.name}, cursor: {expr.location}"
    )


def _evaluate_number_cursor(expr: Cursor) -> int | float:
    """求值 C 数字表达式，并拒绝非数字求值结果。"""
    evaluated = evaluate_cursor(expr)
    if type(evaluated) in (int, float):
        return cast(int | float, evaluated)
    raise RuntimeError(f"数字默认值求值结果不是数字: {evaluated!r}, cursor: {expr.location}")


def _render_number_default(value: int | float, expected_type: Type) -> str:
    """将已求值的 C 数字默认值渲染为 Python 字面量。"""
    if expected_type == _FLOAT_TYPE:
        return str(float(value))
    return str(value)


def _find_decl_candidate(cursor: Cursor) -> Cursor:
    """将实参槽位解析为被写入的目标声明节点。"""
    target = _unwrap_pointer_target(cursor)
    if target.kind in DECL_CURSOR_KINDS:
        return target

    if target.kind == CursorKind.DECL_REF_EXPR:
        referenced = target.referenced
        if referenced is not None and referenced.kind in DECL_CURSOR_KINDS:
            return referenced
    if target.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
        referenced = _find_array_subscript_base_decl(target)
        if referenced is not None:
            return referenced
    raise RuntimeError(f"无法将 C 参数槽位解析为声明节点, cursor: {cursor.location}")


def _extract_array_subscript_slot(cursor: Cursor) -> tuple[Cursor, int] | None:
    """从 `array[index]` 槽位中提取数组声明和固定下标。"""
    target = _unwrap_pointer_target(cursor)
    if target.kind != CursorKind.ARRAY_SUBSCRIPT_EXPR:
        return None

    array_decl = _find_array_subscript_base_decl(target)
    if array_decl is None:
        return None

    children = list(target.get_children())
    if len(children) != 2:
        raise RuntimeError(f"数组下标表达式结构不受支持, cursor: {target.location}")

    index_expr = unwrap_transparent(children[1])
    evaluated = evaluate_cursor(index_expr)
    if type(evaluated) is not int:
        raise RuntimeError(
            f"数组下标表达式求值结果不是整数: {evaluated!r}, cursor: {index_expr.location}"
        )
    return array_decl, int(evaluated)


def _find_array_subscript_base_decl(cursor: Cursor) -> Cursor | None:
    """从数组下标表达式中提取数组变量声明。"""
    if cursor.kind != CursorKind.ARRAY_SUBSCRIPT_EXPR:
        return None

    children = list(cursor.get_children())
    if len(children) != 2:
        return None

    array_expr = unwrap_transparent(children[0])
    if array_expr.kind != CursorKind.DECL_REF_EXPR:
        return None

    referenced = array_expr.referenced
    if referenced is not None and referenced.kind in DECL_CURSOR_KINDS:
        return referenced
    return None


def _find_array_element_assignment_value(
    array_decl: Cursor,
    index: int,
    before_cursor: Cursor | None,
) -> Cursor:
    """查找函数内指定数组元素在目标调用前的最后一个直接赋值右值。"""
    function_cursor = _find_local_decl_function_parent(array_decl)
    candidates: list[Cursor] = []

    for cursor in walk_cursor(function_cursor):
        if before_cursor is not None and cursor == before_cursor:
            break
        if cursor.kind != CursorKind.BINARY_OPERATOR:
            continue
        assignment_value = _extract_array_element_assignment_value(cursor, array_decl, index)
        if assignment_value is not None:
            candidates.append(assignment_value)

    if not candidates:
        raise RuntimeError(
            f"数组元素没有可用定值表达式: {array_decl.spelling}[{index}]"
        )
    return candidates[-1]


def _find_decl_assignment_value(
    target_decl: Cursor,
    before_cursor: Cursor | None,
) -> Cursor | None:
    """查找函数内目标声明在目标调用前的最后一个直接赋值右值。"""
    try:
        function_cursor = _find_local_decl_function_parent(target_decl)
    except RuntimeError:
        return None

    candidates: list[Cursor] = []

    for cursor in walk_cursor(function_cursor):
        if before_cursor is not None and cursor == before_cursor:
            break
        if cursor.kind != CursorKind.BINARY_OPERATOR:
            continue
        candidates.extend(_extract_decl_assignment_values(cursor, target_decl))

    if not candidates:
        return None
    return candidates[-1]


def _extract_decl_assignment_values(
    assignment_cursor: Cursor,
    target_decl: Cursor,
) -> list[Cursor]:
    """递归提取链式赋值中目标声明对应的最终右值表达式。"""
    if get_cursor_binary_operator_kind(assignment_cursor) != CX_BINARY_OPERATOR_ASSIGN:
        return []

    children = list(assignment_cursor.get_children())
    if len(children) != 2:
        return []

    target_expr = unwrap_transparent(children[0])
    value_expr = children[1]
    values: list[Cursor] = []
    if _is_decl_ref_to_decl(target_expr, target_decl):
        values.append(_unwrap_assignment_chain_value(value_expr))

    nested_value_expr = unwrap_transparent(value_expr)
    if nested_value_expr.kind == CursorKind.BINARY_OPERATOR:
        values.extend(_extract_decl_assignment_values(nested_value_expr, target_decl))
    return values


def _extract_array_element_assignment_value(
    assignment_cursor: Cursor,
    array_decl: Cursor,
    index: int,
) -> Cursor | None:
    """在 `array[index] = expr` 中提取指定数组元素的右值表达式。"""
    if get_cursor_binary_operator_kind(assignment_cursor) != CX_BINARY_OPERATOR_ASSIGN:
        return None

    children = list(assignment_cursor.get_children())
    if len(children) != 2:
        return None

    try:
        array_slot = _extract_array_subscript_slot(children[0])
    except RuntimeError:
        return None
    if array_slot is None:
        return None

    candidate_decl, candidate_index = array_slot
    if not _is_same_decl(candidate_decl, array_decl):
        return None
    if candidate_index != index:
        return None
    return children[1]


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
