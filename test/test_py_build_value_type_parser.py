from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import cast

import pytest
from clang.cindex import Cursor

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "core"
    / "NodeVisitors"
    / "CSignatureInference"
    / "CSignatureExtraction"
    / "PyBuildValueTypeParser.py"
)
MODULE_NAME = "test_support_py_build_value_type_parser"

MODULE_SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_NAME] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)

DictTypeNode = MODULE.DictTypeNode
ListTypeNode = MODULE.ListTypeNode
NamedTypeNode = MODULE.NamedTypeNode
PyBuildValueTypeParser = MODULE.PyBuildValueTypeParser
PyBuildValueTypeParserError = MODULE.PyBuildValueTypeParserError
UnionTypeNode = MODULE.UnionTypeNode


def _fake_args(count: int) -> list[Cursor]:
    return [cast(Cursor, object()) for _ in range(count)]


def _parse(format_string: str, arg_count: int) -> str:
    return PyBuildValueTypeParser(format_string, _fake_args(arg_count)).parse()


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (UnionTypeNode(()), "Any"),
        (UnionTypeNode((NamedTypeNode("int"),)), "int"),
    ],
)
def test_union_type_node_str(node: UnionTypeNode, expected: str) -> None:
    assert str(node) == expected


def test_make_union_returns_empty_union_for_empty_input() -> None:
    parser = PyBuildValueTypeParser("", [])

    empty_union = parser._make_union([])

    assert isinstance(empty_union, UnionTypeNode)
    assert empty_union.members == ()


def test_make_union_returns_any_when_any_member_is_any() -> None:
    parser = PyBuildValueTypeParser("", [])

    result = parser._make_union([UnionTypeNode(()), NamedTypeNode("int")])

    assert str(result) == "Any"


def test_make_union_deduplicates_none_members() -> None:
    parser = PyBuildValueTypeParser("", [])

    result = parser._make_union(
        [
            UnionTypeNode((NamedTypeNode("str"), NamedTypeNode("None"))),
            UnionTypeNode((NamedTypeNode("str"), NamedTypeNode("None"))),
        ]
    )

    assert str(result) == "str | None"


@pytest.mark.parametrize(
    ("format_string", "arg_count", "expected"),
    [
        ("", 0, "None"),
        (" \t , : ", 0, "None"),
        ("()", 0, "tuple[()]"),
        ("[]", 0, "list[Any]"),
        ("{}", 0, "dict[Any, Any]"),
        ("(i)", 1, "tuple[int,]"),
        ("[szuU]", 4, "list[str | None]"),
        ("[Oi]", 2, "list[Any]"),
        ("{Oi}", 2, "dict[Any, int]"),
        ("{iO}", 2, "dict[int, Any]"),
        ("{Oiis}", 4, "dict[Any, int | str | None]"),
        ("{syUy}", 4, "dict[str | None, bytes | None]"),
        ("b", 1, "int"),
        ("f", 1, "float"),
        ("D", 1, "complex"),
        ("p", 1, "bool"),
        ("C", 1, "str"),
        ("s", 1, "str | None"),
        ("s#", 2, "str | None"),
        ("z", 1, "str | None"),
        ("z#", 2, "str | None"),
        ("u", 1, "str | None"),
        ("u#", 2, "str | None"),
        ("U", 1, "str | None"),
        ("U#", 2, "str | None"),
        ("y", 1, "bytes | None"),
        ("y#", 2, "bytes | None"),
        ("c", 1, "bytes"),
        ("O", 1, "Any"),
        ("O&", 2, "Any"),
        ("S", 1, "Any"),
        ("N", 1, "Any"),
        (
            "(i, [sz], {s:i, s:[f]}, y#, O&)",
            11,
            "tuple[int, list[str | None], dict[str | None, int | list[float]], bytes | None, Any]",
        ),
        (
            "([i{sz}](s#y#){isfs})",
            11,
            "tuple[list[int | dict[str | None, str | None]], tuple[str | None, bytes | None], dict[int | float, str | None]]",
        ),
    ],
)
def test_parse_returns_expected_type_string(
    format_string: str,
    arg_count: int,
    expected: str,
) -> None:
    assert _parse(format_string, arg_count) == expected


def test_parse_value_returns_list_node_for_empty_list_format() -> None:
    parser = PyBuildValueTypeParser("[]", [])

    list_node = parser._parse_value()

    assert isinstance(list_node, ListTypeNode)
    assert isinstance(list_node.element, UnionTypeNode)


def test_parse_value_returns_dict_node_for_empty_dict_format() -> None:
    parser = PyBuildValueTypeParser("{}", [])

    dict_node = parser._parse_value()

    assert isinstance(dict_node, DictTypeNode)
    assert isinstance(dict_node.key, UnionTypeNode)
    assert isinstance(dict_node.value, UnionTypeNode)


def test_parse_uses_converter_cursor_for_o_ampersand_resolver() -> None:
    converter_cursor = cast(Cursor, object())
    data_cursor = cast(Cursor, object())
    seen: list[Cursor] = []

    def resolve_object_type(cursor: Cursor) -> str:
        seen.append(cursor)
        return "Converted"

    parser = PyBuildValueTypeParser(
        "O&",
        [converter_cursor, data_cursor],
        resolve_object_type_func=resolve_object_type,
    )

    assert parser.parse() == "Converted"
    assert seen == [converter_cursor]


def test_parse_uses_resolved_converter_type_in_nested_o_ampersand_structure() -> None:
    converter_cursor = cast(Cursor, object())
    data_cursor = cast(Cursor, object())
    seen: list[Cursor] = []

    def resolve_object_type(cursor: Cursor) -> str:
        seen.append(cursor)
        return "Converted"

    parser = PyBuildValueTypeParser(
        "([O&])",
        [converter_cursor, data_cursor],
        resolve_object_type_func=resolve_object_type,
    )

    assert parser.parse() == "tuple[list[Converted],]"
    assert seen == [converter_cursor]


@pytest.mark.parametrize(
    ("format_string", "arg_count"),
    [
        ("q", 0),
        ("(i", 1),
        ("{sis}", 3),
        ("s#", 1),
        ("i", 2),
        ("i#", 2),
        ("s&", 2),
        ("S&", 2),
        ("N&", 2),
    ],
)
def test_parse_raises_for_invalid_format(format_string: str, arg_count: int) -> None:
    with pytest.raises(PyBuildValueTypeParserError):
        _parse(format_string, arg_count)
