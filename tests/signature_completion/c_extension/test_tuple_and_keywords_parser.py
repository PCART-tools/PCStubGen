from __future__ import annotations

import pytest
from clang.cindex import Cursor

from pcstubgen.models import Argument, ArgumentKind
from pcstubgen.type_models import RawType, Type
from pcstubgen.signature_completion.c_extension.signatures.py_arg_parse.tuple_and_keywords_parser import (
    PyArgParseTupleAndKeywordsTypeParser,
    PyArgParseTupleAndKeywordsTypeParserError,
)
from tests.signature_completion.c_extension._py_arg_parse_support import (
    _STR_OR_BUFFER_TYPE,
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
    infer_refined_object_type_func=None,
    infer_default_value_func=None,
) -> list[Argument]:
    return PyArgParseTupleAndKeywordsTypeParser(
        format_string,
        kwlist,
        args,
        infer_type_object_func=infer_type_object_func or (lambda cursor: RawType("ResolvedTypeObject")),
        infer_converter_type_func=infer_converter_type_func or (lambda cursor: RawType("ResolvedConverter")),
        infer_refined_object_type_func=infer_refined_object_type_func or (lambda cursor: RawType.object_),
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
def test_parse_ignores_trailer(trailer: str) -> None:
    count_cursor = _cursor("count")
    payload_cursor = _cursor("payload")
    payload_len_cursor = _cursor("payload_len")

    parsed = _parse(
        f"is#{trailer}",
        ["count", "payload"],
        [count_cursor, payload_cursor, payload_len_cursor],
    )

    assert parsed == [
        _arg("count", "int"),
        _arg("payload", _STR_OR_BUFFER_TYPE),
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
        (optional_cursor, RawType.bool_),
        (keyword_only_cursor, RawType.bool_),
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
