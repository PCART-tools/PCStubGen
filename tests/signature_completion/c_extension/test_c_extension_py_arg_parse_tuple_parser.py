from __future__ import annotations

from typing import cast

import pytest
from clang.cindex import Cursor

from pcstubgen.models import Argument
from pcstubgen.signature_completion.c_extension.signatures.py_arg_parse.tuple_parser import (
    PyArgParseTupleTypeParser,
    PyArgParseTupleTypeParserError,
)
from tests.signature_completion.c_extension._py_arg_parse_test_support import (
    _FakeCursor,
    _STR_OR_BUFFER_OR_NONE_TYPE,
    _STR_OR_BUFFER_TYPE,
    _STR_OR_BYTES_OR_BYTEARRAY_TYPE,
    _arg,
    _cursor,
)


def _default_infer_name(c_args: list[Cursor]) -> str:
    """用首个游标名构造稳定测试参数名。"""
    if not c_args:
        raise AssertionError("infer_name_func should not be called with empty c_args.")
    return cast(_FakeCursor, c_args[0]).name


def _parse(
    format_string: str,
    args: list[Cursor],
    *,
    infer_name_func=None,
    infer_object_type_func=None,
    infer_default_value_func=None,
) -> list[Argument]:
    return PyArgParseTupleTypeParser(
        format_string,
        args,
        infer_name_func=infer_name_func or _default_infer_name,
        infer_object_type_func=infer_object_type_func,
        infer_default_value_func=infer_default_value_func,
    ).parse()


def test_parse_returns_required_and_optional_scalars_with_trailer_and_separators() -> None:
    count_cursor = _cursor("count")
    payload_cursor = _cursor("payload")
    payload_len_cursor = _cursor("payload_len")
    maybe_cursor = _cursor("maybe")

    parsed = _parse(
        " \ti, s# | z* :ignored",
        [count_cursor, payload_cursor, payload_len_cursor, maybe_cursor],
    )

    assert parsed == [
        _arg("count", "int"),
        _arg("payload", _STR_OR_BUFFER_TYPE),
        _arg("maybe", _STR_OR_BUFFER_OR_NONE_TYPE, has_default=True),
    ]


def test_parse_uses_name_object_and_default_resolvers_for_multi_slot_units() -> None:
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

    def infer_name(c_args: list[Cursor]) -> str:
        names = [cast(_FakeCursor, cursor).name for cursor in c_args]
        return {
            ("count",): "count",
            ("text_buffer",): "text",
            ("typed_result",): "typed",
            ("converted_result",): "converted",
            ("raw_buffer",): "raw",
            ("maybe_buffer",): "maybe",
        }[tuple(names)]

    def infer_object_type(cursor: Cursor) -> str:
        return {
            type_cursor: "Point",
            converter_cursor: "ConvertedValue",
        }[cursor]

    def infer_default_value(cursor: Cursor) -> str:
        return {
            text_buffer_cursor: '"utf8"',
            typed_result_cursor: "None",
            converted_result_cursor: "factory_default()",
            raw_buffer_cursor: "b''",
            maybe_buffer_cursor: "None",
        }[cursor]

    parsed = _parse(
        "i|et#O!O&s#z*",
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
        infer_name_func=infer_name,
        infer_object_type_func=infer_object_type,
        infer_default_value_func=infer_default_value,
    )

    assert parsed == [
        _arg("count", "int"),
        _arg("text", _STR_OR_BYTES_OR_BYTEARRAY_TYPE, default_value='"utf8"', has_default=True),
        _arg("typed", "Point", default_value="None", has_default=True),
        _arg("converted", "ConvertedValue", default_value="factory_default()", has_default=True),
        _arg("raw", _STR_OR_BUFFER_TYPE, default_value="b''", has_default=True),
        _arg("maybe", _STR_OR_BUFFER_OR_NONE_TYPE, default_value="None", has_default=True),
    ]


def test_parse_falls_back_to_object_for_unresolved_object_units() -> None:
    type_cursor = _cursor("type")
    typed_result_cursor = _cursor("typed_result")
    converter_cursor = _cursor("converter")
    converted_result_cursor = _cursor("converted_result")

    def infer_name(c_args: list[Cursor]) -> str:
        first_name = cast(_FakeCursor, c_args[0]).name
        return {
            "typed_result": "typed",
            "converted_result": "converted",
        }[first_name]

    parsed = _parse(
        "O!O&",
        [type_cursor, typed_result_cursor, converter_cursor, converted_result_cursor],
        infer_name_func=infer_name,
        infer_object_type_func=lambda cursor: None,
    )

    assert parsed == [
        _arg("typed", "object"),
        _arg("converted", "object"),
    ]


def test_parse_keeps_top_level_tuple_units_as_single_arguments() -> None:
    one_cursor = _cursor("one")
    text_cursor = _cursor("text")
    text_len_cursor = _cursor("text_len")
    type_cursor = _cursor("type")
    value_cursor = _cursor("value")
    buffer_cursor = _cursor("buffer")

    def infer_name(c_args: list[Cursor]) -> str:
        names = [cast(_FakeCursor, cursor).name for cursor in c_args]
        return {
            ("one",): "single",
            ("text", "value", "buffer"): "nested",
        }[tuple(names)]

    parsed = _parse(
        "(i), (s#, (O!y))",
        [one_cursor, text_cursor, text_len_cursor, type_cursor, value_cursor, buffer_cursor],
        infer_name_func=infer_name,
        infer_object_type_func=lambda cursor: {type_cursor: "Point"}[cursor],
    )

    assert parsed == [
        _arg("single", "tuple[int,]"),
        _arg(
            "nested",
            "tuple[str | collections.abc.Buffer, tuple[Point, collections.abc.Buffer]]",
            imports=("collections.abc",),
        ),
    ]


def test_parse_builds_tuple_default_values_from_leaf_defaults() -> None:
    count_cursor = _cursor("count")
    label_cursor = _cursor("label")
    label_len_cursor = _cursor("label_len")

    def infer_default_value(cursor: Cursor) -> str | None:
        return {
            count_cursor: "1",
            label_cursor: "'abc'",
        }.get(cursor)

    parsed = _parse(
        "|(i, (s#))",
        [count_cursor, label_cursor, label_len_cursor],
        infer_name_func=lambda c_args: "payload",
        infer_default_value_func=infer_default_value,
    )

    assert parsed == [
        _arg(
            "payload",
            "tuple[int, tuple[str | collections.abc.Buffer,]]",
            imports=("collections.abc",),
            default_value="(1, ('abc',))",
            has_default=True,
        )
    ]


def test_parse_keeps_optional_tuple_argument_when_any_leaf_default_is_unknown() -> None:
    count_cursor = _cursor("count")
    label_cursor = _cursor("label")

    def infer_default_value(cursor: Cursor) -> str | None:
        return {
            count_cursor: "1",
            label_cursor: None,
        }[cursor]

    parsed = _parse(
        "|(is)",
        [count_cursor, label_cursor],
        infer_name_func=lambda c_args: "pair",
        infer_default_value_func=infer_default_value,
    )

    assert parsed == [
        _arg("pair", "tuple[int, str]", has_default=True)
    ]


def test_parse_marks_optional_scalar_without_default_text_when_resolution_fails() -> None:
    first_cursor = _cursor("first")
    second_cursor = _cursor("second")

    def infer_default_value(cursor: Cursor) -> str | None:
        return None

    parsed = _parse(
        "i|i",
        [first_cursor, second_cursor],
        infer_default_value_func=infer_default_value,
    )

    assert parsed == [
        _arg("first", "int"),
        _arg("second", "int", has_default=True),
    ]


@pytest.mark.parametrize(
    ("format_string", "args"),
    [
        ("q", []),
        ("e", []),
        ("w", []),
        ("$i", [_cursor("value")]),
        ("i||i", [_cursor("left"), _cursor("right")]),
        ("()", []),
        ("(i", [_cursor("value")]),
        ("(())", []),
        ("i)", [_cursor("value")]),
        ("(i, ())", [_cursor("value")]),
        ("(i|i)", [_cursor("left"), _cursor("right")]),
        ("(i$i)", [_cursor("left"), _cursor("right")]),
        ("|()", []),
        ("|(i, ())", [_cursor("value")]),
        ("[i]", [_cursor("value")]),
    ],
)
def test_parse_raises_for_unsupported_units_or_invalid_control_usage(
    format_string: str,
    args: list[Cursor],
) -> None:
    with pytest.raises(PyArgParseTupleTypeParserError):
        _parse(format_string, args)


@pytest.mark.parametrize(
    ("format_string", "args"),
    [
        ("s#", [_cursor("payload")]),
        ("(i)", []),
        ("i", [_cursor("value"), _cursor("extra")]),
    ],
)
def test_parse_raises_for_c_argument_count_mismatch(
    format_string: str,
    args: list[Cursor],
) -> None:
    with pytest.raises(PyArgParseTupleTypeParserError):
        _parse(format_string, args)


def test_parse_raises_when_resolved_name_is_none() -> None:
    with pytest.raises(PyArgParseTupleTypeParserError, match="无法解析 argument name。"):
        _parse("i", [_cursor("value")], infer_name_func=lambda c_args: None)


def test_parse_accepts_empty_invalid_and_duplicate_resolved_names() -> None:
    parsed = _parse(
        "iiii",
        [_cursor("first"), _cursor("second"), _cursor("third"), _cursor("fourth")],
        infer_name_func=lambda c_args: {
            "first": "",
            "second": "123bad",
            "third": "same",
            "fourth": "same",
        }[cast(_FakeCursor, c_args[0]).name],
    )

    assert parsed == [
        _arg("", "int"),
        _arg("123bad", "int"),
        _arg("same", "int"),
        _arg("same", "int"),
    ]
