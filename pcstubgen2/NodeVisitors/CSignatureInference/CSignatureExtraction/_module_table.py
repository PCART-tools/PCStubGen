from __future__ import annotations

import logging

from clang.cindex import Cursor, CursorKind, TokenKind, TypeKind

from .Models import ExtractedFunction, ExtractedModule, ExtractedSignature
from ._cursor_utils import (
    is_nullptr_or_zero,
    strip_string_literal_quotes,
    unique_keep_order,
    unwrap_transparent,
    var_decl_to_init_list_expr,
    walk_cursor,
)
from ._signature_rules import (
    apply_method_flags,
    decode_meth_literal_flags,
    deduplicate_signatures,
    extract_signatures_from_function,
    infer_return_type_from_function,
    merge_signature_return_type,
    signature_from_param_decls,
)

logger = logging.getLogger(__name__)

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


def is_pymethoddef_array_definition(cursor: Cursor) -> bool:
    """判断节点是否为 `PyMethodDef[]`。"""
    if (cursor.kind == CursorKind.VAR_DECL
        and cursor.type.kind in _ARRAY_TYPE_KINDS
        and cursor.is_definition()
    ):
        elem_type = cursor.type.get_array_element_type()
        if elem_type.spelling in _PY_METHOD_DEF_TYPE_NAMES:
            return True
    return False


def build_module_lookup_names(module_name: str) -> set[str]:
    """构建模块别名集合，仅保留完整名与叶子名。"""
    lookup_names = {module_name}
    lookup_names.add(module_name.rsplit(".", 1)[-1])
    return lookup_names


def add_discovered_function(
    target: dict[str, ExtractedFunction],
    function: ExtractedFunction,
    *,
    module_name: str,
) -> None:
    """向模块函数映射写入函数；若 Python 名重复则告警并保留旧值。"""
    existing = target.get(function.py_name)
    if existing is None:
        target[function.py_name] = function
        return
    logger.warning(
        "Discarded duplicate extracted function in module %s for Python name %s: kept existing function, discarded incoming function",
        module_name,
        function.py_name,
    )


def extract_pymethoddef_ml_flags(field_cursor: Cursor) -> list[str]:
    """从 `ml_flags` 字段 AST 子树中提取 `METH_*` 列表。"""
    flags: list[str] = []
    for node in walk_cursor(field_cursor):
        for token in node.get_tokens():
            spelling = str(token.spelling)
            if token.kind == TokenKind.IDENTIFIER and spelling.startswith("METH_"):
                flags.append(spelling)
                continue
            if token.kind == TokenKind.LITERAL:
                flags.extend(decode_meth_literal_flags(spelling))
    return unique_keep_order(flags)


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
            field_name = entry_children[0].spelling
            value_cursor = unwrap_transparent(entry_children[1])
            positional_index = field_name_to_index[field_name] + 1
        else:
            if positional_index >= len(field_names):
                continue
            field_name = field_names[positional_index]
            value_cursor = unwrap_transparent(entry)
            positional_index += 1

        values[field_name] = value_cursor

    return values


def extract_pymethoddef_init_list_expr(init_list_expr: Cursor) -> ExtractedFunction | None:
    """
    从 `PyMethodDef` 的单个初始化项提取函数元数据和签名。

    若关键字段（Python 名、C 函数名）缺失则返回 `None`，
    保持提取过程对异常样本的容错性。
    """
    assert init_list_expr.kind == CursorKind.INIT_LIST_EXPR

    fields = resolve_init_list_expr(init_list_expr, _PY_METHOD_DEF_FIELD_NAMES)

    ml_name_cursor = fields.get("ml_name")
    if ml_name_cursor is None or is_nullptr_or_zero(ml_name_cursor):
        # 判断哨兵
        return None
    assert ml_name_cursor.kind == CursorKind.STRING_LITERAL
    ml_name = strip_string_literal_quotes(str(ml_name_cursor.spelling))

    ml_meth_cursor = fields.get("ml_meth")
    assert ml_meth_cursor is not None
    assert ml_meth_cursor.kind == CursorKind.DECL_REF_EXPR

    ml_flags_cursor = fields.get("ml_flags")
    assert ml_flags_cursor is not None
    ml_flags = extract_pymethoddef_ml_flags(ml_flags_cursor)

    function_cursor = ml_meth_cursor.referenced
    if function_cursor is None:
        logger.warning("cant find function cursor, location: %s", ml_meth_cursor.location)
        return None

    signatures = extract_signatures_from_function(function_cursor, ml_flags)
    return_type_name = infer_return_type_from_function(function_cursor)
    if not signatures:
        fallback = signature_from_param_decls(function_cursor)
        if fallback.arguments:
            signatures = [fallback]

    if not signatures:
        signatures = [ExtractedSignature(arguments=[], return_type_name=return_type_name)]

    signatures = [merge_signature_return_type(sig, return_type_name) for sig in signatures]
    signatures = [apply_method_flags(sig, ml_flags) for sig in signatures]
    signatures = deduplicate_signatures(signatures)
    return ExtractedFunction(
        py_name=ml_name,
        ml_flags=ml_flags,
        signatures=signatures,
    )


def extract_method_table(
    cursor: Cursor,
    *,
    module_name: str,
) -> dict[str, ExtractedFunction]:
    """解析 `PyMethodDef[]` 变量。"""
    assert is_pymethoddef_array_definition(cursor)

    init_expr_node = var_decl_to_init_list_expr(cursor)
    assert init_expr_node is not None

    grouped: dict[str, ExtractedFunction] = {}

    for element in init_expr_node.get_children():
        extracted = extract_pymethoddef_init_list_expr(init_list_expr=element)
        if extracted is None:
            continue
        add_discovered_function(grouped, extracted, module_name=module_name)
    return grouped


def extract_module_from_pymoduledef(module_def_cursor: Cursor) -> ExtractedModule | None:
    """
    从单个 `PyModuleDef` 变量中提取模块定义与模块方法。

    模块名认 `m_name`，方法认 `m_methods`。
    """
    init_list_expr = var_decl_to_init_list_expr(module_def_cursor)
    assert init_list_expr is not None

    values = resolve_init_list_expr(init_list_expr, _PY_MODULE_DEF_FIELD_NAMES)

    m_name_cursor = values.get("m_name")
    assert m_name_cursor is not None
    assert m_name_cursor.kind == CursorKind.STRING_LITERAL
    m_name = strip_string_literal_quotes(str(m_name_cursor.spelling))

    module = ExtractedModule(name=m_name)
    module.lookup_names.update(build_module_lookup_names(m_name))

    m_methods_cursor = values.get("m_methods")
    if m_methods_cursor is None:
        return module

    assert m_methods_cursor.kind == CursorKind.DECL_REF_EXPR
    method_list_cursor = m_methods_cursor.referenced
    assert is_pymethoddef_array_definition(method_list_cursor)

    module.functions = extract_method_table(method_list_cursor, module_name=m_name)
    return module


def process_translation_unit(cursor: Cursor) -> list[ExtractedModule]:
    """从单个 translation unit 的 `PyModuleDef` 变量定义提取模块。"""
    modules: list[ExtractedModule] = []
    for node in walk_cursor(cursor):
        if (
            node.kind == CursorKind.VAR_DECL
            and node.is_definition()
            and node.type.spelling in {"PyModuleDef", "struct PyModuleDef"}
        ):
            extracted = extract_module_from_pymoduledef(node)
            if extracted is not None:
                modules.append(extracted)
    return modules
