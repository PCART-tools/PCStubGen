from __future__ import annotations

"""`Py_BuildValue` 格式串到类型树的解析器。"""

from typing import Callable

from clang.cindex import Cursor

from .py_build_value_format_units import _FORMAT_UNIT_SPECS, _FormatUnitSpec
from .types import (
    AnyType,
    DictType,
    ListType,
    RawType,
    TupleType,
    Type,
    UnionType,
)


class PyBuildValueTypeParserError(ValueError):
    """表示 `Py_BuildValue` 格式串无法被当前解析器接受。"""


class PyBuildValueTypeParser:
    """将 `Py_BuildValue` 的格式串解析为类型树。"""

    def __init__(
        self,
        fmt: str,
        args: list[Cursor],
        resolve_object_type_func: Callable[[Cursor], Type | None] | None = None,
    ) -> None:
        """初始化格式串解析器。"""
        self._format = fmt
        self._args = args
        self._resolve_object_type_func = resolve_object_type_func
        self._char_index = 0
        self._arg_index = 0

    def parse(self) -> Type:
        """解析格式串并返回未经规范化的类型树。"""
        self._char_index = 0
        self._arg_index = 0

        top_level_types = self._parse_items(stop_char=None)
        self._skip_ignored_chars()
        if self._arg_index != len(self._args):
            raise PyBuildValueTypeParserError(
                f"期望 {self._arg_index} 个 arguments，实际找到 {len(self._args)} 个。"
            )

        if not top_level_types:
            return RawType("None")
        if len(top_level_types) == 1:
            return top_level_types[0]
        return TupleType(tuple(top_level_types))

    def _parse_items(self, stop_char: str | None) -> list[Type]:
        """解析直到终止字符为止的一组类型项。"""
        items: list[Type] = []

        while True:
            self._skip_ignored_chars()
            current = self._peek_char()

            if current is None:
                if stop_char is None:
                    return items
                raise PyBuildValueTypeParserError(
                    f"在 format string 结束前应找到 '{stop_char}'。"
                )

            if stop_char is not None and current == stop_char:
                return items

            items.append(self._parse_value())

    def _parse_value(self) -> Type:
        """解析单个类型值。"""
        current = self._peek_char_required()

        if current == "(":
            return self._parse_tuple()
        if current == "[":
            return self._parse_list()
        if current == "{":
            return self._parse_dict()
        return self._parse_scalar()

    def _parse_tuple(self) -> Type:
        """解析 tuple 结构。"""
        self._consume_char("(")
        items = self._parse_items(stop_char=")")
        self._consume_char(")")
        return TupleType(tuple(items))

    def _parse_list(self) -> Type:
        """解析 list 结构。"""
        self._consume_char("[")
        items = self._parse_items(stop_char="]")
        self._consume_char("]")
        return ListType(UnionType(tuple(items)))

    def _parse_dict(self) -> Type:
        """解析 dict 结构。"""
        self._consume_char("{")
        items = self._parse_items(stop_char="}")
        self._consume_char("}")

        if len(items) % 2 != 0:
            raise PyBuildValueTypeParserError("Dictionary format 必须包含成对的 key/value。")

        key_types = items[0::2]
        value_types = items[1::2]
        return DictType(
            UnionType(tuple(key_types)),
            UnionType(tuple(value_types)),
        )

    def _parse_scalar(self) -> Type:
        """解析标量格式单元。"""
        spec = self._advance_format_unit_required()
        c_args = self._advance_args_required(spec.c_arg_count)
        if spec.object_type_arg_offset is not None:
            return self._resolve_object_type(c_args[spec.object_type_arg_offset])
        return spec.value_type

    def _advance_format_unit_required(self) -> _FormatUnitSpec:
        """按最长匹配规则消费一个标量格式单元。"""
        current = self._peek_char_required()
        for spec in _FORMAT_UNIT_SPECS:
            if self._format.startswith(spec.unit, self._char_index):
                self._char_index += len(spec.unit)
                return spec

        raise PyBuildValueTypeParserError(
            f"索引 {self._char_index} 处的 format unit '{current}' 不受支持。"
        )

    def _peek_char(self) -> str | None:
        """查看当前位置字符而不推进游标。"""
        if self._char_index >= len(self._format):
            return None
        return self._format[self._char_index]

    def _peek_char_required(self) -> str:
        """查看当前位置字符；若已结束则抛错。"""
        if self._char_index >= len(self._format):
            raise PyBuildValueTypeParserError("已到达 format string 末尾。")
        return self._format[self._char_index]

    def _advance_char(self) -> str | None:
        """推进一个格式串字符。"""
        if self._char_index >= len(self._format):
            return None
        current = self._format[self._char_index]
        self._char_index += 1
        return current

    def _advance_char_required(self) -> str:
        """推进一个格式串字符；若已结束则抛错。"""
        if self._char_index >= len(self._format):
            raise PyBuildValueTypeParserError("已到达 format string 末尾。")
        current = self._format[self._char_index]
        self._char_index += 1
        return current

    def _consume_char(self, expected: str) -> None:
        """消费一个预期字符并在不匹配时抛错。"""
        current = self._advance_char()
        if current != expected:
            found = "end of format string" if current is None else repr(current)
            raise PyBuildValueTypeParserError(
                f"期望在索引 {self._char_index - 1} 处找到 '{expected}'，实际为 {found}。"
            )

    def _skip_ignored_chars(self) -> None:
        """跳过格式串中的空白和分隔符。"""
        while True:
            current = self._peek_char()
            if current is None or current not in " \t,:":
                return
            self._char_index += 1

    def _advance_arg_required(self) -> Cursor:
        """消费一个实参游标；若已无剩余则抛错。"""
        if self._arg_index >= len(self._args):
            raise PyBuildValueTypeParserError(
                f"期望在索引 {self._arg_index} 处取得 argument，但已没有剩余参数。"
            )

        value = self._args[self._arg_index]
        self._arg_index += 1
        return value

    def _advance_args_required(self, count: int) -> list[Cursor]:
        """连续消费指定数量的实参游标。"""
        return [self._advance_arg_required() for _ in range(count)]

    def _resolve_object_type(self, cursor: Cursor) -> Type:
        """解析对象槽位的类型，未知时保留为显式 `Any` 节点。"""
        if self._resolve_object_type_func is not None:
            resolved_type = self._resolve_object_type_func(cursor)
            if resolved_type is not None:
                return resolved_type
        return AnyType()
