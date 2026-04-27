from __future__ import annotations

from collections.abc import Callable

from clang.cindex import Cursor

from .....type_models import Type


def get_name_to_type(
    name_to_type: dict[str, Type | Callable[[Cursor], Type]],
    name: str,
    cursor: Cursor | None,
) -> Type | None:
    """从规则表中获取名称对应的返回类型。"""
    value = name_to_type.get(name)
    if value is None:
        return None
    if isinstance(value, Type):
        return value
    if cursor is None:
        raise RuntimeError(f"规则 {name!r} 需要 Cursor 上下文。")
    return value(cursor)
