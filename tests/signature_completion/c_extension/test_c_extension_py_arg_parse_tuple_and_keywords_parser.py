from __future__ import annotations

import pytest
from clang.cindex import Cursor

from pcstubgen.models import Argument, ArgumentKind
from pcstubgen.type_models import RawType, Type
from pcstubgen.signature_completion.c_extension.signatures.py_arg_parse.tuple_and_keywords_parser import (
    PyArgParseTupleAndKeywordsTypeParser,
    PyArgParseTupleAndKeywordsTypeParserError,
)
from tests.signature_completion.c_extension._py_arg_parse_test_support import (
    _STR_OR_BUFFER_OR_NONE_TYPE,
    _STR_OR_BUFFER_TYPE,
    _STR_OR_BYTES_OR_BYTEARRAY_TYPE,
    _STR_OR_NONE_TYPE,
    _arg,
    _cursor,
)


def _parse(
    format_string: str,
    kwlist: list[str],
    args: list[Cursor],
    *,
    infer_type_object_func=None,
    infer_converter_type_func=None,
    infer_default_value_func=None,
) -> list[Argument]:
    return PyArgParseTupleAndKeywordsTypeParser(
        format_string,
        kwlist,
        args,
        infer_type_object_func=infer_type_object_func or (lambda cursor: RawType("ResolvedTypeObject")),
        infer_converter_type_func=infer_converter_type_func or (lambda cursor: RawType("ResolvedConverter")),
        infer_default_value_func=infer_default_value_func or (lambda cursor, expected_type: "None"),
    ).parse()


def test_parse_returns_required_optional_and_keyword_only_arguments() -> None:
    count_cursor = _cursor("count")
    label_cursor = _cursor("label")
    target_cursor = _cursor("target")

    parsed = _parse(
        "i|z$O",
        ["count", "label", "target"],
        [count_cursor, label_cursor, target_cursor],
        infer_default_value_func=lambda cursor, expected_type: {
            label_cursor: "None",
            target_cursor: "None",
        }[cursor],
    )

    assert parsed == [
        _arg("count", "int"),
        _arg("label", _STR_OR_NONE_TYPE, default_value="None"),
        _arg(
            "target",
            "object",
            default_value="None",
            kind=ArgumentKind.KEYWORD_ONLY,
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
        _arg("count", "int"),
        _arg("payload", _STR_OR_BUFFER_TYPE),
    ]


def test_parse_uses_object_and_default_inference_for_multi_slot_units() -> None:
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

    resolved_types = {
        type_cursor: RawType("Point"),
        converter_cursor: RawType("ConvertedValue"),
    }
    resolved_defaults = {
        text_buffer_cursor: '"utf8"',
        typed_result_cursor: "None",
        converted_result_cursor: "factory_default()",
        raw_buffer_cursor: "b''",
        maybe_buffer_cursor: "None",
    }

    def infer_type_object(cursor: Cursor) -> RawType:
        return {type_cursor: resolved_types[type_cursor]}[cursor]

    def infer_converter(cursor: Cursor) -> RawType:
        return {converter_cursor: resolved_types[converter_cursor]}[cursor]

    def infer_default_value(cursor: Cursor, expected_type: Type) -> str:
        _ = expected_type
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
        infer_type_object_func=infer_type_object,
        infer_converter_type_func=infer_converter,
        infer_default_value_func=infer_default_value,
    )

    assert parsed == [
        _arg("count", "int"),
        _arg("text", _STR_OR_BYTES_OR_BYTEARRAY_TYPE, default_value='"utf8"'),
        _arg("typed", "Point", default_value="None"),
        _arg("converted", "ConvertedValue", default_value="factory_default()"),
        _arg("raw", _STR_OR_BUFFER_TYPE, default_value="b''"),
        _arg(
            "maybe",
            _STR_OR_BUFFER_OR_NONE_TYPE,
            default_value="None",
            kind=ArgumentKind.KEYWORD_ONLY,
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


def test_parse_allows_empty_optional_section_before_keyword_only_arguments() -> None:
    value_cursor = _cursor("value")

    parsed = _parse(
        "|$i",
        ["value"],
        [value_cursor],
        infer_default_value_func=lambda cursor, expected_type: {value_cursor: "0"}[cursor],
    )

    assert parsed == [
        _arg("value", "int", default_value="0", kind=ArgumentKind.KEYWORD_ONLY)
    ]


def test_parse_falls_back_to_object_when_type_object_or_converter_inference_raises() -> None:
    type_cursor = _cursor("type")
    typed_result_cursor = _cursor("typed_result")
    converter_cursor = _cursor("converter")
    converted_result_cursor = _cursor("converted_result")

    parsed = _parse(
        "O!O&",
        ["typed", "converted"],
        [type_cursor, typed_result_cursor, converter_cursor, converted_result_cursor],
        infer_type_object_func=lambda cursor: (_ for _ in ()).throw(RuntimeError("type object boom")),
        infer_converter_type_func=lambda cursor: (_ for _ in ()).throw(RuntimeError("converter boom")),
    )

    assert parsed == [
        _arg("typed", "object"),
        _arg("converted", "object"),
    ]


def test_parse_routes_o_bang_and_o_ampersand_to_different_resolvers() -> None:
    type_cursor = _cursor("type")
    typed_result_cursor = _cursor("typed_result")
    converter_cursor = _cursor("converter")
    converted_result_cursor = _cursor("converted_result")
    seen_type_objects: list[Cursor] = []
    seen_converters: list[Cursor] = []

    parsed = _parse(
        "O!O&",
        ["typed", "converted"],
        [type_cursor, typed_result_cursor, converter_cursor, converted_result_cursor],
        infer_type_object_func=lambda cursor: seen_type_objects.append(cursor) or RawType("Typed"),
        infer_converter_type_func=lambda cursor: seen_converters.append(cursor) or RawType("Converted"),
    )

    assert parsed == [
        _arg("typed", "Typed"),
        _arg("converted", "Converted"),
    ]
    assert seen_type_objects == [type_cursor]
    assert seen_converters == [converter_cursor]


def test_parse_falls_back_to_unknown_default_value_when_default_inference_raises() -> None:
    first_cursor = _cursor("first")
    second_cursor = _cursor("second")

    parsed = _parse(
        "i|i",
        ["first", "second"],
        [first_cursor, second_cursor],
        infer_default_value_func=lambda cursor, expected_type: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
    )

    assert parsed == [
        _arg("first", "int"),
        _arg("second", "int", default_value="..."),
    ]


def test_parse_maps_p_unit_to_bool() -> None:
    predicate_cursor = _cursor("predicate")

    parsed = _parse("p", ["predicate"], [predicate_cursor])

    assert parsed == [_arg("predicate", "bool")]


def test_parse_passes_p_unit_type_to_optional_and_keyword_only_default_inference() -> None:
    optional_cursor = _cursor("optional")
    keyword_only_cursor = _cursor("keyword_only")
    observed: list[tuple[Cursor, Type]] = []

    def infer_default_value(cursor: Cursor, expected_type: Type) -> str:
        observed.append((cursor, expected_type))
        return {
            optional_cursor: "False",
            keyword_only_cursor: "True",
        }[cursor]

    parsed = _parse(
        "|p$p",
        ["optional", "keyword_only"],
        [optional_cursor, keyword_only_cursor],
        infer_default_value_func=infer_default_value,
    )

    assert parsed == [
        _arg("optional", "bool", default_value="False"),
        _arg(
            "keyword_only",
            "bool",
            default_value="True",
            kind=ArgumentKind.KEYWORD_ONLY,
        ),
    ]
    assert observed == [
        (optional_cursor, RawType("bool")),
        (keyword_only_cursor, RawType("bool")),
    ]


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
