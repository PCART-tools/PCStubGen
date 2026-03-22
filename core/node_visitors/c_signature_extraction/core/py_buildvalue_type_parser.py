from __future__ import annotations

"""`Py_BuildValue` 格式串到类型树的解析器。"""

from typing import Callable

from clang.cindex import Cursor

from .py_buildvalue_type_nodes import (
    AnyTypeNode,
    DictTypeNode,
    ListTypeNode,
    NamedTypeNode,
    TupleTypeNode,
    TypeNode,
    UnionTypeNode,
)


class PyBuildValueTypeParserError(ValueError):
    """表示 `Py_BuildValue` 格式串无法被当前解析器接受。"""


class PyBuildValueTypeParser:
    """将 `Py_BuildValue` 的格式串解析为类型树。"""

    def __init__(
        self,
        fmt: str,
        args: list[Cursor],
        resolve_object_type_func: Callable[[Cursor], TypeNode | None] | None = None,
    ) -> None:
        """初始化格式串解析器。"""
        self._format = fmt
        self._args = args
        self._char_index = 0
        self._arg_index = 0
        self._resolve_object_type_func: Callable[[Cursor], TypeNode | None] | None = (
            resolve_object_type_func
        )

    def parse(self) -> TypeNode:
        """解析格式串并返回未经规范化的类型树。"""
        top_level_types = self._parse_items(stop_char=None)
        self._skip_ignored_chars()
        if self._arg_index != len(self._args):
            raise PyBuildValueTypeParserError(
                f"Expected {self._arg_index} arguments, found {len(self._args)}."
            )

        if not top_level_types:
            return NamedTypeNode("None")
        if len(top_level_types) == 1:
            return top_level_types[0]
        return TupleTypeNode(tuple(top_level_types))

    def _parse_items(self, stop_char: str | None) -> list[TypeNode]:
        """解析直到终止字符为止的一组类型项。"""
        items: list[TypeNode] = []

        while True:
            self._skip_ignored_chars()
            current = self._peek_char()

            if current is None:
                if stop_char is None:
                    return items
                raise PyBuildValueTypeParserError(
                    f"Expected '{stop_char}' before end of format string."
                )

            if stop_char is not None and current == stop_char:
                return items

            items.append(self._parse_value())

    def _parse_value(self) -> TypeNode:
        """解析单个类型值。"""
        current = self._peek_char_required()

        if current == "(":
            return self._parse_tuple()
        if current == "[":
            return self._parse_list()
        if current == "{":
            return self._parse_dict()
        return self._parse_scalar()

    def _parse_tuple(self) -> TypeNode:
        """解析 tuple 结构。"""
        self._consume_char("(")
        items = self._parse_items(stop_char=")")
        self._consume_char(")")
        return TupleTypeNode(tuple(items))

    def _parse_list(self) -> TypeNode:
        """解析 list 结构。"""
        self._consume_char("[")
        items = self._parse_items(stop_char="]")
        self._consume_char("]")
        return ListTypeNode(UnionTypeNode(tuple(items)))

    def _parse_dict(self) -> TypeNode:
        """解析 dict 结构。"""
        self._consume_char("{")
        items = self._parse_items(stop_char="}")
        self._consume_char("}")

        if len(items) % 2 != 0:
            raise PyBuildValueTypeParserError("Dictionary format must contain key/value pairs.")

        key_types = items[0::2]
        value_types = items[1::2]
        return DictTypeNode(
            UnionTypeNode(tuple(key_types)),
            UnionTypeNode(tuple(value_types)),
        )

    def _parse_scalar(self) -> TypeNode:
        """解析标量格式单元。"""
        unit = self._peek_char_required()

        if unit in "bBhHiIlkLKn":
            self._advance_char_arg_required()
            return NamedTypeNode("int")

        if unit in "fd":
            self._advance_char_arg_required()
            return NamedTypeNode("float")

        if unit == "D":
            self._advance_char_arg_required()
            return NamedTypeNode("complex")

        if unit == "p":
            self._advance_char_arg_required()
            return NamedTypeNode("bool")

        if unit == "C":
            self._advance_char_arg_required()
            return NamedTypeNode("str")

        if unit in "szuU":
            self._advance_char_arg_required()
            if self._peek_char() == "#":
                self._advance_char_arg_required()
            return UnionTypeNode((NamedTypeNode("str"), NamedTypeNode("None")))

        if unit == "c":
            self._advance_char_arg_required()
            return NamedTypeNode("bytes")

        if unit == "y":
            self._advance_char_arg_required()
            if self._peek_char() == "#":
                self._advance_char_arg_required()
            return UnionTypeNode((NamedTypeNode("bytes"), NamedTypeNode("None")))

        if unit == "O":
            _, arg = self._advance_char_arg_required()
            if self._peek_char() == "&":
                self._advance_char_arg_required()
                return self._resolve_object_type(arg)
            return self._resolve_object_type(arg)

        if unit in "SN":
            _, arg = self._advance_char_arg_required()
            return self._resolve_object_type(arg)

        raise PyBuildValueTypeParserError(
            f"Unsupported format unit '{unit}' at index {self._char_index}."
        )

    def _peek_char(self) -> str | None:
        """查看当前位置字符而不推进游标。"""
        if self._char_index >= len(self._format):
            return None
        return self._format[self._char_index]

    def _peek_char_required(self) -> str:
        """查看当前位置字符；若已结束则抛错。"""
        if self._char_index >= len(self._format):
            raise PyBuildValueTypeParserError("Found end of format string.")
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
            raise PyBuildValueTypeParserError("Found end of format string.")
        current = self._format[self._char_index]
        self._char_index += 1
        return current

    def _consume_char(self, expected: str) -> None:
        """消费一个预期字符并在不匹配时抛错。"""
        current = self._advance_char()
        if current != expected:
            found = "end of format string" if current is None else repr(current)
            raise PyBuildValueTypeParserError(
                f"Expected '{expected}' at index {self._char_index - 1}, found {found}."
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
                f"Expected argument at index {self._arg_index}, but none remained."
            )

        value = self._args[self._arg_index]
        self._arg_index += 1
        return value

    def _advance_char_arg_required(self) -> tuple[str, Cursor]:
        """同时消费一个格式字符和对应的一个实参。"""
        s = self._advance_char_required()
        cursor = self._advance_arg_required()
        return s, cursor

    def _resolve_object_type(self, cursor: Cursor) -> TypeNode:
        """解析对象槽位的类型，未知时保留为显式 `Any` 节点。"""
        if self._resolve_object_type_func is not None:
            resolved_type = self._resolve_object_type_func(cursor)
            if resolved_type is not None:
                return resolved_type
        return AnyTypeNode()
