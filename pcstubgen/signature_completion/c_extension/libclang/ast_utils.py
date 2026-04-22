from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from clang.cindex import Cursor, CursorKind, SourceLocation, SourceRange, TranslationUnit

from .libclang_wrap import evaluate_cursor, get_file_contents, get_file_location

SINGLE_TRANSPARENT_CURSOR_KINDS = {
    # python libclang 未暴露的若干表达式会落到 UNEXPOSED_EXPR。
    CursorKind.UNEXPOSED_EXPR,
    CursorKind.PAREN_EXPR,
}

CAST_CURSOR_KINDS = {
    CursorKind.CSTYLE_CAST_EXPR,
    CursorKind.CXX_STATIC_CAST_EXPR,
    CursorKind.CXX_REINTERPRET_CAST_EXPR,
    CursorKind.CXX_CONST_CAST_EXPR,
    CursorKind.CXX_FUNCTIONAL_CAST_EXPR,
}

TRANSPARENT_CURSOR_KINDS = SINGLE_TRANSPARENT_CURSOR_KINDS | CAST_CURSOR_KINDS

NULLPTR_CURSOR_KINDS = {
    CursorKind.CXX_NULL_PTR_LITERAL_EXPR,
    CursorKind.GNU_NULL_EXPR,
}

DECL_CURSOR_KINDS = {
    CursorKind.VAR_DECL,
    CursorKind.PARM_DECL,
    CursorKind.FIELD_DECL,
}

FUNCTION_DECL_CONTEXT_KINDS = {
    CursorKind.TRANSLATION_UNIT,
    CursorKind.NAMESPACE,
    CursorKind.LINKAGE_SPEC,
}

def to_str(cursor: Cursor) -> str:
    return f"kind={cursor.kind}, spelling={cursor.spelling}, first_token={get_first_token_str(cursor)}, location={cursor.location}"


def unwrap_transparent(cursor: Cursor) -> Cursor:
    """剥离透明包装节点，定位到更有语义价值的底层表达式。"""
    while cursor.kind in TRANSPARENT_CURSOR_KINDS:
        children = list(cursor.get_children())
        if not children:
            break
        if cursor.kind in SINGLE_TRANSPARENT_CURSOR_KINDS:
            cursor = children[0]
        else:
            cursor = children[-1]
    return cursor


def walk(cursor: Cursor) -> Iterable[Cursor]:
    """生成器，前序遍历 cursor 子树。"""
    yield cursor
    for child in cursor.get_children():
        yield from walk(child)


def is_nullptr_or_zero(cursor: Cursor) -> bool:
    """识别 `0` / `NULL` 展开后的空指针 / `nullptr`。"""
    if cursor.kind in NULLPTR_CURSOR_KINDS:
        return True
    return cursor.kind == CursorKind.INTEGER_LITERAL and evaluate_cursor(cursor) == 0

def extract_array_subscript(cursor: Cursor) -> tuple[Cursor, int]:
    """从 `array[index]` 槽位中提取数组声明和固定下标。"""
    assert cursor.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR

    children = list(cursor.get_children())

    array_decl = unwrap_transparent(children[0]).referenced

    index_expr = unwrap_transparent(children[1])
    evaluated = evaluate_cursor(index_expr)
    if type(evaluated) is not int:
        raise RuntimeError(
            f"数组下标表达式求值结果不是整数: {evaluated!r}, cursor: {index_expr.location}"
        )
    return array_decl, int(evaluated)


def var_decl_to_init_list_expr(cursor: Cursor) -> Cursor:
    """从变量声明直接找出其初始化列表节点。"""
    assert cursor.kind == CursorKind.VAR_DECL
    for child in walk(cursor):
        if child.kind == CursorKind.INIT_LIST_EXPR:
            return child
    raise RuntimeError(f"变量声明未包含初始化列表, cursor: {cursor.location}")


def try_get_decl_initializer(cursor: Cursor) -> Cursor | None:
    """提取声明初始化表达式；无初始化式时返回 `None`。"""
    children = list(cursor.get_children())
    if not children:
        return None

    initializer = unwrap_transparent(children[-1])
    if initializer.kind == CursorKind.TYPE_REF:
        return None
    return initializer


def unwrap_single_unary_op(cursor: Cursor) -> Cursor:
    """剥离透明包装和一层UNARY_OPERATOR节点，定位到底层目标。"""
    cursor = unwrap_transparent(cursor)
    if cursor.kind == CursorKind.UNARY_OPERATOR:
        children = list(cursor.get_children())
        cursor = unwrap_transparent(children[0])
    return cursor

def get_cursor_source_text(cursor: Cursor) -> str:
    """从 cursor 对应的源码范围中提取原始文本。"""
    extent = cursor.extent
    start_file, _, _, start_offset = get_file_location(extent.start)
    end_file, _, _, end_offset = get_file_location(extent.end)
    if start_file is None or end_file is None:
        raise RuntimeError(f"源码范围缺少起止文件信息, cursor: {cursor.location}")

    if start_file.name != end_file.name:
        raise RuntimeError(f"源码范围跨越多个文件，无法提取文本, cursor: {cursor.location}")

    source_bytes = get_file_contents(cursor.translation_unit, start_file)
    read_length = end_offset - start_offset
    if read_length < 0:
        raise RuntimeError(f"源码范围终点位于起点之前，无法提取文本, cursor: {cursor.location}")
    source_bytes = source_bytes[start_offset:end_offset]
    return source_bytes.decode("utf-8", errors="ignore")


def get_first_token_str(cursor: Cursor) -> str:
    start_file, _, _, start_offset = get_file_location(cursor.extent.start)
    if start_file is None:
        raise RuntimeError(f"起点缺少文件信息, cursor: {cursor.location}")

    token_range = SourceRange.from_locations(
        SourceLocation.from_offset(cursor.translation_unit, start_file, start_offset),
        SourceLocation.from_offset(cursor.translation_unit, start_file, start_offset + 1),
    )
    tokens = list(cursor.translation_unit.get_tokens(extent=token_range))
    if not tokens:
        raise RuntimeError(f"起点缺少 token, cursor: {cursor.location}")
    return tokens[0].spelling


IDENTIFIER_RE = re.compile(r"\b[_A-Za-z]\w*\b")


def get_string_literal(node: Cursor) -> str:
    """从子树中提取首个字符串字面量的实际内容。"""
    node = unwrap_transparent(node)
    if node.kind == CursorKind.STRING_LITERAL:
        return node.spelling.strip('"')
    raise RuntimeError(f"节点不是字符串字面量, cursor: {node.location}")


def get_func_cursor(
    translation_unit: TranslationUnit,
    function_name: str,
    linkage_name: str | None,
) -> Cursor:
    """按函数名和 linkage name 定位函数定义节点。"""
    for cursor in _iter_function_definition_candidates(translation_unit.cursor):
        if linkage_name is not None:
            if cursor.mangled_name == linkage_name:
                return cursor
            continue
        if cursor.spelling == function_name:
            return cursor

    raise RuntimeError(
        "未在 translation unit 中定位到函数定义, "
        f"translation_unit: {translation_unit.cursor.location}, "
        f"function_name: {function_name}, "
        f"linkage_name: {linkage_name}"
    )


def _iter_function_definition_candidates(node: Cursor) -> Iterator[Cursor]:
    """仅在声明上下文中递归收集函数定义节点。"""
    for child in node.get_children():
        if child.kind == CursorKind.FUNCTION_DECL:
            if child.is_definition():
                yield child
            continue
        if child.kind in FUNCTION_DECL_CONTEXT_KINDS:
            yield from _iter_function_definition_candidates(child)
