from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from clang.cindex import Cursor

from core.ir import IRArgumentKind
from core.node_visitors.c_signature_extraction.core.models import ExtractedArgument
from core.node_visitors.c_signature_extraction.core.py_arg_parse_tuple_and_keywords_type_parser import (
    PyArgParseTupleAndKeywordsTypeParser,
    PyArgParseTupleAndKeywordsTypeParserError,
)


@dataclass(frozen=True)
class _FakeCursor:
    name: str


def _cursor(name: str) -> Cursor:
    return cast(Cursor, _FakeCursor(name))


def _parse(
    format_string: str,
    kwlist: list[str],
    args: list[Cursor],
    *,
    resolve_object_type_func=None,
    resolve_default_value_func=None,
) -> list[ExtractedArgument]:
    return PyArgParseTupleAndKeywordsTypeParser(
        format_string,
        kwlist,
        args,
        resolve_object_type_func=resolve_object_type_func,
        resolve_default_value_func=resolve_default_value_func,
    ).parse()


def test_parse_returns_required_optional_and_keyword_only_arguments() -> None:
    count_cursor = _cursor("count")
    label_cursor = _cursor("label")
    target_cursor = _cursor("target")

    parsed = _parse(
        "i|z$O",
        ["count", "label", "target"],
        [count_cursor, label_cursor, target_cursor],
    )

    assert parsed == [
        ExtractedArgument(name="count", type_name="int"),
        ExtractedArgument(name="label", type_name="str | None", has_default=True),
        ExtractedArgument(
            name="target",
            type_name="object",
            has_default=True,
            kind=IRArgumentKind.KEYWORD_ONLY,
        ),
    ]


@pytest.mark.parametrize("trailer", [":func_name", ";custom message"])
def test_parse_ignores_trailer_and_separators(trailer: str) -> None:
    count_cursor = _cursor("count")
    payload_cursor = _cursor("payload")
    payload_len_cursor = _cursor("payload_len")

    parsed = _parse(
        f" \ti, s# {trailer}",
        ["count", "payload"],
        [count_cursor, payload_cursor, payload_len_cursor],
    )

    assert parsed == [
        ExtractedArgument(name="count", type_name="int"),
        ExtractedArgument(name="payload", type_name="str | collections.abc.Buffer"),
    ]


def test_parse_uses_object_and_default_resolvers_for_multi_slot_units() -> None:
    count_cursor = _cursor("count")
    encoding_cursor = _cursor("encoding")
    text_buffer_cursor = _cursor("text_buffer")
    text_len_cursor = _cursor("text_len")
    type_cursor = _cursor("type")
    typed_result_cursor = _cursor("typed_result")
    converter_cursor = _cursor("converter")
    converted_result_cursor = _cursor("converted_result")
    raw_buffer_cursor = _cursor("raw_buffer")
    raw_len_cursor = _cursor("raw_len")
    maybe_buffer_cursor = _cursor("maybe_buffer")

    seen_object_cursors: list[Cursor] = []
    seen_default_cursors: list[Cursor] = []

    resolved_types = {
        type_cursor: "Point",
        converter_cursor: "ConvertedValue",
    }
    resolved_defaults = {
        text_buffer_cursor: '"utf8"',
        typed_result_cursor: "None",
        converted_result_cursor: "factory_default()",
        raw_buffer_cursor: "b''",
        maybe_buffer_cursor: "None",
    }

    def resolve_object_type(cursor: Cursor) -> str:
        seen_object_cursors.append(cursor)
        return resolved_types[cursor]

    def resolve_default(cursor: Cursor) -> str:
        seen_default_cursors.append(cursor)
        return resolved_defaults[cursor]

    parsed = _parse(
        "i|et#O!O&s#$z*",
        ["count", "text", "typed", "converted", "raw", "maybe"],
        [
            count_cursor,
            encoding_cursor,
            text_buffer_cursor,
            text_len_cursor,
            type_cursor,
            typed_result_cursor,
            converter_cursor,
            converted_result_cursor,
            raw_buffer_cursor,
            raw_len_cursor,
            maybe_buffer_cursor,
        ],
        resolve_object_type_func=resolve_object_type,
        resolve_default_value_func=resolve_default,
    )

    assert seen_object_cursors == [type_cursor, converter_cursor]
    assert seen_default_cursors == [
        text_buffer_cursor,
        typed_result_cursor,
        converted_result_cursor,
        raw_buffer_cursor,
        maybe_buffer_cursor,
    ]
    assert parsed == [
        ExtractedArgument(name="count", type_name="int"),
        ExtractedArgument(
            name="text",
            type_name="str | bytes | bytearray",
            default_value='"utf8"',
            has_default=True,
        ),
        ExtractedArgument(
            name="typed",
            type_name="Point",
            default_value="None",
            has_default=True,
        ),
        ExtractedArgument(
            name="converted",
            type_name="ConvertedValue",
            default_value="factory_default()",
            has_default=True,
        ),
        ExtractedArgument(
            name="raw",
            type_name="str | collections.abc.Buffer",
            default_value="b''",
            has_default=True,
        ),
        ExtractedArgument(
            name="maybe",
            type_name="str | collections.abc.Buffer | None",
            default_value="None",
            has_default=True,
            kind=IRArgumentKind.KEYWORD_ONLY,
        ),
    ]


@pytest.mark.parametrize(
    ("format_string", "kwlist", "args"),
    [
        ("q", ["value"], []),
        ("e", ["value"], []),
        ("w", ["value"], []),
        ("$i", ["value"], [_cursor("value")]),
        ("(i)", ["value"], [_cursor("value")]),
        ("i(i)", ["left", "right"], [_cursor("left"), _cursor("right")]),
        ("O!", ["value"], [_cursor("type_only")]),
        ("O&", ["value"], [_cursor("converter_only")]),
    ],
)
def test_parse_raises_for_unsupported_units_or_structure(
    format_string: str,
    kwlist: list[str],
    args: list[Cursor],
) -> None:
    with pytest.raises(PyArgParseTupleAndKeywordsTypeParserError):
        _parse(format_string, kwlist, args)


@pytest.mark.parametrize("kwlist", [[""], ["value", "value"]])
def test_parse_raises_for_invalid_keyword_names(kwlist: list[str]) -> None:
    with pytest.raises(PyArgParseTupleAndKeywordsTypeParserError):
        _parse("i" * len(kwlist), kwlist, [_cursor(f"arg_{index}") for index in range(len(kwlist))])


def test_parse_raises_with_chinese_message_for_invalid_keyword_name() -> None:
    with pytest.raises(
        PyArgParseTupleAndKeywordsTypeParserError,
        match=r"无效的 keyword name: ''。",
    ):
        _parse("i", [""], [_cursor("arg_0")])


@pytest.mark.parametrize(
    ("format_string", "kwlist", "args"),
    [
        ("ii", ["left"], [_cursor("left"), _cursor("right")]),
        ("i", ["left", "right"], [_cursor("left")]),
        ("s#", ["payload"], [_cursor("payload")]),
        ("i", ["value"], [_cursor("value"), _cursor("extra")]),
    ],
)
def test_parse_raises_for_keyword_or_c_argument_count_mismatch(
    format_string: str,
    kwlist: list[str],
    args: list[Cursor],
) -> None:
    with pytest.raises(PyArgParseTupleAndKeywordsTypeParserError):
        _parse(format_string, kwlist, args)


def test_parse_marks_optional_arguments_when_default_text_resolution_fails() -> None:
    first_cursor = _cursor("first")
    second_cursor = _cursor("second")
    seen_default_cursors: list[Cursor] = []

    def resolve_default(cursor: Cursor) -> str | None:
        seen_default_cursors.append(cursor)
        return None

    parsed = _parse(
        "i|i",
        ["first", "second"],
        [first_cursor, second_cursor],
        resolve_default_value_func=resolve_default,
    )

    assert seen_default_cursors == [second_cursor]
    assert parsed == [
        ExtractedArgument(name="first", type_name="int"),
        ExtractedArgument(name="second", type_name="int", has_default=True),
    ]


def test_parse_allows_empty_optional_section_before_keyword_only_arguments() -> None:
    value_cursor = _cursor("value")

    parsed = _parse("|$i", ["value"], [value_cursor])

    assert parsed == [
        ExtractedArgument(
            name="value",
            type_name="int",
            has_default=True,
            kind=IRArgumentKind.KEYWORD_ONLY,
        )
    ]


@pytest.mark.parametrize(
    "resolve_object_type_func",
    [
        None,
        lambda cursor: None,
    ],
)
def test_parse_falls_back_to_object_for_unresolved_object_units(
    resolve_object_type_func,
) -> None:
    type_cursor = _cursor("type")
    typed_result_cursor = _cursor("typed_result")
    converter_cursor = _cursor("converter")
    converted_result_cursor = _cursor("converted_result")

    parsed = _parse(
        "O!O&",
        ["typed", "converted"],
        [type_cursor, typed_result_cursor, converter_cursor, converted_result_cursor],
        resolve_object_type_func=resolve_object_type_func,
    )

    assert parsed == [
        ExtractedArgument(name="typed", type_name="object"),
        ExtractedArgument(name="converted", type_name="object"),
    ]


def test_parse_maps_p_unit_to_object() -> None:
    predicate_cursor = _cursor("predicate")

    parsed = _parse("p", ["predicate"], [predicate_cursor])

    assert parsed == [ExtractedArgument(name="predicate", type_name="object")]


@pytest.mark.parametrize(
    ("format_string", "kwlist", "args"),
    [
        ("$i", ["value"], [_cursor("value")]),
        ("i||i", ["left", "right"], [_cursor("left"), _cursor("right")]),
        ("i|$i$i", ["left", "middle", "right"], [_cursor("left"), _cursor("middle"), _cursor("right")]),
    ],
)
def test_parse_raises_for_invalid_control_separator_usage(
    format_string: str,
    kwlist: list[str],
    args: list[Cursor],
) -> None:
    with pytest.raises(PyArgParseTupleAndKeywordsTypeParserError):
        _parse(format_string, kwlist, args)
