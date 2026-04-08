from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from clang.cindex import Cursor

from pcstubgen.types import RawType, Type, UnionType
from pcstubgen.ir_modules import IRArgument, IRArgumentKind
from pcstubgen.signature_completion.c_extension.signatures.py_arg_parse.tuple_and_keywords_parser import (
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
) -> list[IRArgument]:
    return PyArgParseTupleAndKeywordsTypeParser(
        format_string,
        kwlist,
        args,
        resolve_object_type_func=resolve_object_type_func,
        resolve_default_value_func=resolve_default_value_func,
    ).parse()


_BUFFER_TYPE = RawType("collections.abc.Buffer", imports=("collections.abc",))
_STR_OR_NONE_TYPE = UnionType((RawType("str"), RawType("None")))
_STR_OR_BUFFER_TYPE = UnionType((RawType("str"), _BUFFER_TYPE))
_STR_OR_BUFFER_OR_NONE_TYPE = UnionType((RawType("str"), _BUFFER_TYPE, RawType("None")))
_STR_OR_BYTES_OR_BYTEARRAY_TYPE = UnionType(
    (RawType("str"), RawType("bytes"), RawType("bytearray"))
)


def _arg(
    name: str,
    type_text: str | Type,
    *,
    imports: tuple[str, ...] = (),
    default_value: str | None = None,
    has_default: bool = False,
    kind: IRArgumentKind = IRArgumentKind.POSITIONAL_OR_KEYWORD,
) -> IRArgument:
    return IRArgument(
        name=name,
        type=type_text if isinstance(type_text, Type) else RawType(type_text, imports=imports),
        default_value=default_value,
        has_default=has_default,
        kind=kind,
    )


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
        _arg("count", "int"),
        _arg("label", _STR_OR_NONE_TYPE, has_default=True),
        _arg("target", "object", has_default=True, kind=IRArgumentKind.KEYWORD_ONLY),
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
        _arg("count", "int"),
        _arg("payload", _STR_OR_BUFFER_TYPE),
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
        _arg("count", "int"),
        _arg("text", _STR_OR_BYTES_OR_BYTEARRAY_TYPE, default_value='"utf8"', has_default=True),
        _arg("typed", "Point", default_value="None", has_default=True),
        _arg("converted", "ConvertedValue", default_value="factory_default()", has_default=True),
        _arg("raw", _STR_OR_BUFFER_TYPE, default_value="b''", has_default=True),
        _arg(
            "maybe",
            _STR_OR_BUFFER_OR_NONE_TYPE,
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


def test_parse_accepts_empty_invalid_and_duplicate_keyword_names() -> None:
    parsed = _parse(
        "iii",
        ["", "same", "same"],
        [_cursor("arg_0"), _cursor("arg_1"), _cursor("arg_2")],
    )

    assert parsed == [
        _arg("", "int"),
        _arg("same", "int"),
        _arg("same", "int"),
    ]


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
        _arg("first", "int"),
        _arg("second", "int", has_default=True),
    ]


def test_parse_allows_empty_optional_section_before_keyword_only_arguments() -> None:
    value_cursor = _cursor("value")

    parsed = _parse("|$i", ["value"], [value_cursor])

    assert parsed == [
        _arg("value", "int", has_default=True, kind=IRArgumentKind.KEYWORD_ONLY)
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
        _arg("typed", "object"),
        _arg("converted", "object"),
    ]


def test_parse_maps_p_unit_to_object() -> None:
    predicate_cursor = _cursor("predicate")

    parsed = _parse("p", ["predicate"], [predicate_cursor])

    assert parsed == [_arg("predicate", "object")]


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
