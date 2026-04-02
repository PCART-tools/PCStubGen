from __future__ import annotations

from clang.cindex import Cursor, CursorKind, TokenKind, TypeKind
from loguru import logger

from ..clang import constant_eval
from ..clang.cursor_utils import (
    is_nullptr_or_zero,
    unwrap_transparent,
    var_decl_to_init_list_expr,
    walk_cursor,
)
from ..definition_index import DefinitionIndex
from ..models import CFunction, CModule

_ARRAY_TYPE_KINDS = {
    TypeKind.CONSTANTARRAY,
    TypeKind.INCOMPLETEARRAY,
    TypeKind.VARIABLEARRAY,
    TypeKind.DEPENDENTSIZEDARRAY,
}

_PY_METHOD_DEF_TYPE_NAMES = {"PyMethodDef", "struct PyMethodDef"}

_PY_MODULE_DEF_FIELD_NAMES = (
    "m_base",
    "m_name",
    "m_doc",
    "m_size",
    "m_methods",
    "m_slots",
    "m_traverse",
    "m_clear",
    "m_free",
)

_PY_METHOD_DEF_FIELD_NAMES = (
    "ml_name",
    "ml_meth",
    "ml_flags",
    "ml_doc",
)


def is_PyMethodDef_array_definition(cursor: Cursor) -> bool:
    """判断节点是否为 `PyMethodDef[]`。"""
    if (cursor.kind == CursorKind.VAR_DECL
        and cursor.type.kind in _ARRAY_TYPE_KINDS
        and cursor.is_definition()
    ):
        elem_type = cursor.type.get_array_element_type()
        canonical_type = elem_type.get_canonical()
        if (
            elem_type.spelling in _PY_METHOD_DEF_TYPE_NAMES
            or canonical_type.spelling in _PY_METHOD_DEF_TYPE_NAMES
        ):
            return True
    return False


def resolve_init_list_expr(
    cursor: Cursor,
    field_names: tuple[str, ...] | list[str],
) -> dict[str, Cursor]:
    """解析顶层初始化列表，支持位置初始化与 designated initializer 混用。"""
    assert cursor.kind == CursorKind.INIT_LIST_EXPR

    field_name_to_index = {
        field_name: index
        for index, field_name in enumerate(field_names)
    }
    values: dict[str, Cursor] = {}
    positional_index = 0

    for entry in cursor.get_children():
        entry_children = list(entry.get_children())

        if len(entry_children) >= 2 and entry_children[0].kind == CursorKind.MEMBER_REF:
            # 指定初始化
            field_name = entry_children[0].spelling
            if field_name not in field_name_to_index:
                continue
            value_cursor = unwrap_transparent(entry_children[1])
            positional_index = field_name_to_index[field_name] + 1
        else:
            # 位置初始化
            if positional_index >= len(field_names):
                continue
            field_name = field_names[positional_index]
            value_cursor = unwrap_transparent(entry)
            positional_index += 1

        values[field_name] = value_cursor

    return values


def _is_null_identifier(cursor: Cursor) -> bool:
    """识别 `NULL` 标识符，作为旧式 C 哨兵兼容路径。"""
    tokens = list(cursor.get_tokens())
    if len(tokens) != 1:
        return False
    token = tokens[0]
    return token.kind == TokenKind.IDENTIFIER and str(token.spelling) == "NULL"


def _is_null_like_cursor(cursor: Cursor) -> bool:
    """识别可作为 `PyMethodDef` 哨兵字段的空值表达式。"""
    unwrapped = unwrap_transparent(cursor)
    return is_nullptr_or_zero(unwrapped) or _is_null_identifier(unwrapped)


def collect_pymethoddef_init_list_expr(
    init_list_expr: Cursor,
    *,
    definition_index: DefinitionIndex,
) -> tuple[bool, CFunction | None]:
    """
    从 `PyMethodDef` 的单个初始化项提取函数骨架数据。

    返回 `(is_sentinel, extracted)`：
    若首字段是哨兵，则返回 `(True, None)` 供外层停止遍历；
    若不是哨兵但当前项无法提取，则返回 `(False, None)`。
    """
    assert init_list_expr.kind == CursorKind.INIT_LIST_EXPR

    values = resolve_init_list_expr(init_list_expr, _PY_METHOD_DEF_FIELD_NAMES)

    ml_name_cursor = values.get("ml_name")
    if ml_name_cursor is None or is_nullptr_or_zero(ml_name_cursor):
        # 判断哨兵
        return True, None
    assert ml_name_cursor.kind == CursorKind.STRING_LITERAL
    ml_name = ml_name_cursor.spelling.strip('"')

    ml_meth_cursor = values.get("ml_meth")
    assert ml_meth_cursor is not None
    assert ml_meth_cursor.kind == CursorKind.DECL_REF_EXPR

    ml_flags_cursor = values.get("ml_flags")
    assert ml_flags_cursor is not None
    ml_flags = constant_eval.eval_int(ml_flags_cursor)
    if ml_flags is None:
        ml_flags = 0

    func_def_cursor = definition_index.get_definition(ml_meth_cursor)
    if (
        func_def_cursor is None
        or func_def_cursor.kind != CursorKind.FUNCTION_DECL
    ):
        logger.warning(
            "找不到 function definition, ml_name: {}, 位置: {}",
            ml_name,
            ml_meth_cursor.location,
        )
        return False, None

    return False, CFunction(
        ml_name=ml_name,
        ml_flags=ml_flags,
        function_cursor=func_def_cursor,
    )


def collect_method_table(
    cursor: Cursor,
    *,
    module_name: str,
    definition_index: DefinitionIndex,
) -> dict[str, CFunction]:
    """解析 `PyMethodDef[]` 变量。"""

    init_expr_node = var_decl_to_init_list_expr(cursor)
    assert init_expr_node is not None

    result: dict[str, CFunction] = {}

    for element in init_expr_node.get_children():
        is_sentinel, extracted = collect_pymethoddef_init_list_expr(
            init_list_expr=element,
            definition_index=definition_index,
        )
        if is_sentinel:
            break
        if extracted is None:
            continue
        if extracted.ml_name in result:
            logger.warning(
                "模块重复函数, 丢弃新函数, module_name: {}, ml_name: {}",
                module_name,
                extracted.ml_name,
            )
            continue
        result[extracted.ml_name] = extracted
    return result


def collect_module_from_pymoduledef(
    module_def_cursor: Cursor,
    *,
    definition_index: DefinitionIndex,
) -> CModule | None:
    """
    从单个 `PyModuleDef` 变量中提取模块定义与模块方法。

    模块名认 `m_name`，方法认 `m_methods`。
    """
    init_list_expr = var_decl_to_init_list_expr(module_def_cursor)
    assert init_list_expr is not None

    values = resolve_init_list_expr(init_list_expr, _PY_MODULE_DEF_FIELD_NAMES)

    # 名字
    m_name_cursor = values.get("m_name")
    assert m_name_cursor is not None
    assert m_name_cursor.kind == CursorKind.STRING_LITERAL
    m_name = str(m_name_cursor.spelling).strip('"')

    module = CModule(name=m_name)

    # 方法表
    m_methods_cursor = values.get("m_methods")
    if m_methods_cursor is None:
        return module
    if _is_null_like_cursor(m_methods_cursor):
        return module

    assert m_methods_cursor.kind == CursorKind.DECL_REF_EXPR
    method_list_cursor = definition_index.get_definition(m_methods_cursor)
    if method_list_cursor is None or not is_PyMethodDef_array_definition(method_list_cursor):
        logger.warning(
            "找不到 method table definition, module_name: {}, 位置: {}",
            m_name,
            m_methods_cursor.location,
        )
        return module

    module.functions = collect_method_table(
        method_list_cursor,
        module_name=m_name,
        definition_index=definition_index,
    )
    return module


def collect_modules_from_translation_unit(
    cursor: Cursor,
    *,
    definition_index: DefinitionIndex,
) -> list[CModule]:
    """从单个 translation unit 的 `PyModuleDef` 变量定义提取模块。"""
    modules: list[CModule] = []
    # PyModuleDef可能定义为局部变量
    for node in walk_cursor(cursor):
        if (
            node.kind == CursorKind.VAR_DECL
            and node.is_definition()
            and node.type.spelling in {"PyModuleDef", "struct PyModuleDef"}
        ):
            extracted = collect_module_from_pymoduledef(
                node,
                definition_index=definition_index,
            )
            if extracted is not None:
                modules.append(extracted)
    return modules
