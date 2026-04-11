from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from clang.cindex import Cursor, CursorKind, SourceRange, TranslationUnit
from loguru import logger

from . import constant_eval

_SINGLE_TRANSPARENT_CURSOR_KINDS = {
    # python libclang 未暴露的若干表达式会落到 UNEXPOSED_EXPR。
    CursorKind.UNEXPOSED_EXPR,
    CursorKind.PAREN_EXPR,
}

_CAST_CURSOR_KINDS = {
    CursorKind.CSTYLE_CAST_EXPR,
    CursorKind.CXX_STATIC_CAST_EXPR,
    CursorKind.CXX_REINTERPRET_CAST_EXPR,
    CursorKind.CXX_CONST_CAST_EXPR,
    CursorKind.CXX_FUNCTIONAL_CAST_EXPR,
}

_TRANSPARENT_CURSOR_KINDS = _SINGLE_TRANSPARENT_CURSOR_KINDS | _CAST_CURSOR_KINDS

_NULLPTR_CURSOR_KINDS = {
    CursorKind.CXX_NULL_PTR_LITERAL_EXPR,
    CursorKind.GNU_NULL_EXPR,
}

DECL_CURSOR_KINDS = {
    CursorKind.VAR_DECL,
    CursorKind.PARM_DECL,
    CursorKind.FIELD_DECL,
}

_FUNCTION_DECL_CONTEXT_KINDS = {
    CursorKind.TRANSLATION_UNIT,
    CursorKind.NAMESPACE,
    CursorKind.LINKAGE_SPEC,
}


def unwrap_transparent(cursor: Cursor) -> Cursor:
    """剥离透明包装节点，定位到更有语义价值的底层表达式。"""
    while cursor.kind in _TRANSPARENT_CURSOR_KINDS:
        children = list(cursor.get_children())
        if not children:
            break
        if cursor.kind in _SINGLE_TRANSPARENT_CURSOR_KINDS:
            cursor = children[0]
        else:
            cursor = children[-1]
    return cursor


def walk_cursor(node: Cursor) -> Iterable[Cursor]:
    """生成器，前序遍历 cursor 子树。"""
    yield node
    for child in node.get_children():
        yield from walk_cursor(child)


def is_integer_literal_zero(cursor: Cursor) -> bool:
    """判断是否为值为 0 的整数字面量。"""
    if cursor.kind != CursorKind.INTEGER_LITERAL:
        return False
    value = constant_eval.eval_int(cursor)
    if value is None:
        return False
    return value == 0


def is_nullptr_or_zero(node: Cursor) -> bool:
    """识别 `0` / `NULL` 展开后的空指针 / `nullptr`。"""
    if node.kind in _NULLPTR_CURSOR_KINDS:
        return True
    return is_integer_literal_zero(node)


def var_decl_to_init_list_expr(cursor: Cursor) -> Cursor:
    """从变量声明直接找出其初始化列表节点。"""
    if cursor.kind != CursorKind.VAR_DECL:
        raise RuntimeError("只能从 VAR_DECL 提取初始化列表。")
    for child in walk_cursor(cursor):
        if child.kind == CursorKind.INIT_LIST_EXPR:
            return child
    raise RuntimeError("变量声明未包含初始化列表。")


def source_range_get_text(extent: SourceRange) -> str:
    """源文件中截取原始源码文本。"""
    start = extent.start
    end = extent.end
    if start.file is None or end.file is None:
        raise RuntimeError("源码范围缺少起止文件信息。")

    start_file = Path(start.file.name)
    end_file = Path(end.file.name)
    if start_file != end_file:
        raise RuntimeError("源码范围跨越多个文件，无法提取文本。")

    read_length = end.offset - start.offset
    try:
        with start_file.open("rb") as source_file:
            source_file.seek(start.offset)
            source_bytes = source_file.read(read_length)
    except OSError as ex:
        raise RuntimeError(
            f"源码范围读取失败: {start_file}, offset={start.offset}, length={read_length}"
        ) from ex
    return source_bytes.decode("utf-8", errors="ignore")


IDENTIFIER_RE = re.compile(r"\b[_A-Za-z]\w*\b")


def extract_string_literal(node: Cursor) -> str:
    """从子树中提取首个字符串字面量的实际内容。"""
    node = unwrap_transparent(node)
    if node.kind == CursorKind.STRING_LITERAL:
        return node.spelling.strip('"')
    raise RuntimeError("节点不是字符串字面量。")


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
        if child.kind in _FUNCTION_DECL_CONTEXT_KINDS:
            yield from _iter_function_definition_candidates(child)
