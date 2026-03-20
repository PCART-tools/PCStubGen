from __future__ import annotations

import re
from collections.abc import Iterable

from clang.cindex import Cursor, CursorKind, TokenKind

from . import clang_eval

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
    value = clang_eval.eval_int(cursor)
    if value is None:
        return False
    return value == 0


def is_nullptr_or_zero(node: Cursor) -> bool:
    """识别 `0` / `NULL` 展开后的空指针 / `nullptr`。"""
    if node.kind in _NULLPTR_CURSOR_KINDS:
        return True
    return is_integer_literal_zero(node)


def var_decl_to_init_list_expr(cursor: Cursor) -> Cursor | None:
    """从变量声明直接找出其初始化列表节点。"""
    assert cursor.kind == CursorKind.VAR_DECL
    for child in cursor.get_children():
        if child.kind == CursorKind.INIT_LIST_EXPR:
            return child
    return None


def collect_identifier_literal_tokens(node: Cursor) -> list[str]:
    """收集 cursor 子树中的 `IDENTIFIER` / `LITERAL` token。"""
    return [
        str(token.spelling)
        for token in node.get_tokens()
        if token.kind in {TokenKind.IDENTIFIER, TokenKind.LITERAL}
    ]


def find_first_call_name(node: Cursor) -> str | None:
    """在子树中查找首个 `CALL_EXPR` 的函数名。"""
    for child in walk_cursor(node):
        if child.kind == CursorKind.CALL_EXPR and child.spelling:
            return str(child.spelling)
    return None


def find_string_end(text: str, start: int) -> int | None:
    """在包含转义字符的场景下查找字符串结束位置。"""
    quote = text[start]
    idx = start + 1
    while idx < len(text):
        if text[idx] == "\\":
            idx += 2
            continue
        if text[idx] == quote:
            return idx
        idx += 1
    return None


def split_top_level(text: str, delim: str) -> list[str]:
    """
    仅在顶层分隔文本，忽略括号/引号内部的分隔符。

    该工具用于安全拆分形如 `a, dict[str, int], "x,y"` 的参数串。
    """
    closing = {"(": ")", "[": "]", "{": "}", "<": ">"}
    stack: list[str] = []
    parts: list[str] = []
    start = 0
    idx = 0
    while idx < len(text):
        char = text[idx]
        if char in "\"'":
            end = find_string_end(text, idx)
            if end is None:
                break
            idx = end + 1
            continue
        if char in closing:
            stack.append(closing[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif not stack and char == delim:
            parts.append(text[start:idx])
            start = idx + 1
        idx += 1
    parts.append(text[start:])
    return parts


def looks_like_identifier(value: str) -> bool:
    """判断文本是否符合标识符命名规则。"""
    return bool(re.match(r"^[_A-Za-z]\w*$", value))


def unique_keep_order(values: list[str]) -> list[str]:
    """在保持原顺序的前提下去重。"""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
