from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from clang.cindex import Cursor
from typing import Callable


class PyBuildValueTypeParserError(ValueError):
    pass


class TypeNode(ABC):
    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class NamedTypeNode(TypeNode):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class UnionTypeNode(TypeNode):
    members: tuple[TypeNode, ...]

    def __str__(self) -> str:
        if not self.members:
            return "Any"
        if len(self.members) == 1:
            return str(self.members[0])
        return " | ".join(str(member) for member in self.members)


@dataclass(frozen=True)
class TupleTypeNode(TypeNode):
    items: tuple[TypeNode, ...]

    def __str__(self) -> str:
        if not self.items:
            return "tuple[()]"
        if len(self.items) == 1:
            return f"tuple[{self.items[0]},]"
        return f"tuple[{', '.join(str(item) for item in self.items)}]"


@dataclass(frozen=True)
class ListTypeNode(TypeNode):
    element: UnionTypeNode

    def __str__(self) -> str:
        return f"list[{self.element}]"


@dataclass(frozen=True)
class DictTypeNode(TypeNode):
    key: UnionTypeNode
    value: UnionTypeNode

    def __str__(self) -> str:
        return f"dict[{self.key}, {self.value}]"


class PyBuildValueTypeParser:
    def __init__(self, fmt: str, args: list[Cursor],
                 resolve_object_type_func: Callable[[Cursor], str] | None = None) -> None:
        self._format = fmt
        self._args = args
        self._char_index = 0
        self._arg_index = 0
        self._resolve_object_type_func: Callable[[Cursor], str] | None = resolve_object_type_func

    def parse(self) -> str:
        top_level_types = self._parse_items(stop_char=None)
        self._skip_ignored_chars()
        if self._arg_index != len(self._args):
            raise PyBuildValueTypeParserError(
                f"Expected {self._arg_index} arguments, found {len(self._args)}."
            )

        if not top_level_types:
            return "None"
        if len(top_level_types) == 1:
            return str(top_level_types[0])
        return str(TupleTypeNode(tuple(top_level_types)))

    def _parse_items(self, stop_char: str | None) -> list[TypeNode]:
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
        current = self._peek_char_required()

        if current == "(":
            return self._parse_tuple()
        if current == "[":
            return self._parse_list()
        if current == "{":
            return self._parse_dict()
        return self._parse_scalar()

    def _parse_tuple(self) -> TypeNode:
        self._consume_char("(")
        items = self._parse_items(stop_char=")")
        self._consume_char(")")
        return TupleTypeNode(tuple(items))

    def _parse_list(self) -> TypeNode:
        self._consume_char("[")
        items = self._parse_items(stop_char="]")
        self._consume_char("]")
        return ListTypeNode(self._make_union(items))

    def _parse_dict(self) -> TypeNode:
        self._consume_char("{")
        items = self._parse_items(stop_char="}")
        self._consume_char("}")

        if len(items) % 2 != 0:
            raise PyBuildValueTypeParserError("Dictionary format must contain key/value pairs.")

        key_types = items[0::2]
        value_types = items[1::2]
        return DictTypeNode(self._make_union(key_types), self._make_union(value_types))

    def _parse_scalar(self) -> TypeNode:
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
                # arg是converter
                self._advance_char_arg_required()
                return self._resolve_object_type(arg)
            # arg是obj
            return self._resolve_object_type(arg)

        if unit in "SN":
            _, arg = self._advance_char_arg_required()
            return self._resolve_object_type(arg)

        raise PyBuildValueTypeParserError(
            f"Unsupported format unit '{unit}' at index {self._char_index}."
        )

    def _peek_char(self) -> str | None:
        if self._char_index >= len(self._format):
            return None
        return self._format[self._char_index]

    def _peek_char_required(self) -> str:
        if self._char_index >= len(self._format):
            raise PyBuildValueTypeParserError(f"Found end of format string.")
        return self._format[self._char_index]

    def _advance_char(self) -> str | None:
        if self._char_index >= len(self._format):
            return None
        current = self._format[self._char_index]
        self._char_index += 1
        return current

    def _advance_char_required(self) -> str:
        if self._char_index >= len(self._format):
            raise PyBuildValueTypeParserError(f"Found end of format string.")
        current = self._format[self._char_index]
        self._char_index += 1
        return current

    def _consume_char(self, expected: str) -> None:
        current = self._advance_char()
        if current != expected:
            found = "end of format string" if current is None else repr(current)
            raise PyBuildValueTypeParserError(
                f"Expected '{expected}' at index {self._char_index - 1}, found {found}."
            )

    def _skip_ignored_chars(self) -> None:
        while True:
            current = self._peek_char()
            if current is None or current not in " \t,:":
                return
            self._char_index += 1

    def _advance_arg_required(self) -> Cursor:
        if self._arg_index >= len(self._args):
            raise PyBuildValueTypeParserError(
                f"Expected argument at index {self._arg_index}, but none remained."
            )

        value = self._args[self._arg_index]
        self._arg_index += 1
        return value

    def _advance_char_arg_required(self) -> tuple[str, Cursor]:
        s = self._advance_char_required()
        cursor = self._advance_arg_required()
        return s, cursor

    def _resolve_object_type(self, cursor: Cursor) -> TypeNode:
        if self._resolve_object_type_func is not None:
            s = self._resolve_object_type_func(cursor)
            if s == "Any":
                return UnionTypeNode(())
            return NamedTypeNode(s)
        return UnionTypeNode(())

    def _make_union(self, nodes: list[TypeNode]) -> UnionTypeNode:
        if not nodes:
            return UnionTypeNode(())

        members: list[TypeNode] = []
        seen: set[str] = set()

        for node in nodes:
            flattened_candidates = self._flatten_union_members(node)
            if flattened_candidates is None:
                return UnionTypeNode(())

            for candidate in flattened_candidates:
                key = str(candidate)
                if key in seen:
                    continue
                seen.add(key)
                members.append(candidate)

        return UnionTypeNode(tuple(members))

    def _flatten_union_members(self, node: TypeNode) -> tuple[TypeNode, ...] | None:
        if not isinstance(node, UnionTypeNode):
            return (node,)
        if not node.members:
            return None

        flattened_members: list[TypeNode] = []
        for member in node.members:
            nested_members = self._flatten_union_members(member)
            if nested_members is None:
                return None
            flattened_members.extend(nested_members)
        return tuple(flattened_members)
