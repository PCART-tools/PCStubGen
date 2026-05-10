from __future__ import annotations

import pytest

from pcstubgen.signature_completion.pybind11.inferencer import (
    parse_args_str,
    parse_pybind11_signature,
)
from pcstubgen.models import ArgumentKind


def _render_type(type_: object | None) -> str | None:
    if type_ is None:
        return None
    return type_.render()


def test_parse_pybind11_signature_parses_generic_function_signature() -> None:
    parsed = parse_pybind11_signature("(x: int, y: str) -> str")

    assert [arg.name for arg in parsed.args] == ["x", "y"]
    assert [_render_type(arg.type) for arg in parsed.args] == ["int", "str"]
    assert _render_type(parsed.return_type) == "str"


def test_parse_pybind11_signature_parses_defaults_and_return_type() -> None:
    parsed = parse_pybind11_signature(
        "(x: object, y: object = None, p: typing.SupportsFloat = 2.0) -> numpy.ndarray"
    )

    assert [arg.name for arg in parsed.args] == ["x", "y", "p"]
    assert [_render_type(arg.type) for arg in parsed.args] == [
        "object",
        "object",
        "typing.SupportsFloat",
    ]
    assert [arg.default_value for arg in parsed.args] == [
        None,
        "None",
        "2.0",
    ]
    assert _render_type(parsed.return_type) == "numpy.ndarray"


def test_parse_pybind11_signature_parses_cpp_type_names() -> None:
    parsed = parse_pybind11_signature(
        "(self: torch._C.AliasDb, arg0: torch::jit::Value, arg1: torch::jit::Value) -> bool"
    )

    assert [arg.name for arg in parsed.args] == ["self", "arg0", "arg1"]
    assert [_render_type(arg.type) for arg in parsed.args] == [
        "torch._C.AliasDb",
        "torch::jit::Value",
        "torch::jit::Value",
    ]
    assert _render_type(parsed.return_type) == "bool"


def test_parse_pybind11_signature_supports_nested_defaults_and_markers() -> None:
    parsed = parse_pybind11_signature(
        '(a: int, /, x: tuple[int, int] = (1, 2), *, '
        'mapping: dict[str, int] = {"a": 1, "b": 2}, flag: str = "x") -> None'
    )

    assert [arg.name for arg in parsed.args] == ["a", "x", "mapping", "flag"]
    assert [_render_type(arg.type) for arg in parsed.args] == [
        "int",
        "tuple[int, int]",
        "dict[str, int]",
        "str",
    ]
    assert [arg.default_value for arg in parsed.args] == [
        None,
        "(1, 2)",
        '{"a": 1, "b": 2}',
        '"x"',
    ]
    assert [arg.kind for arg in parsed.args] == [
        ArgumentKind.POSITIONAL_ONLY,
        ArgumentKind.POSITIONAL_OR_KEYWORD,
        ArgumentKind.KEYWORD_ONLY,
        ArgumentKind.KEYWORD_ONLY,
    ]


def test_parse_pybind11_signature_supports_var_args_and_var_kwargs() -> None:
    parsed = parse_pybind11_signature(
        "(value: typing.Optional[list[int]], *args: tuple[str, ...], **kwargs: object) -> object"
    )

    assert [arg.name for arg in parsed.args] == ["value", "args", "kwargs"]
    assert [_render_type(arg.type) for arg in parsed.args] == [
        "typing.Optional[list[int]]",
        "tuple[str, ...]",
        "object",
    ]
    assert [arg.kind for arg in parsed.args] == [
        ArgumentKind.POSITIONAL_OR_KEYWORD,
        ArgumentKind.VAR_POSITIONAL,
        ArgumentKind.VAR_KEYWORD,
    ]


def test_parse_pybind11_signature_extracts_outer_signature_with_arrow_in_default() -> None:
    parsed = parse_pybind11_signature(
        '(text: str = ") -> inside", value: tuple[int, int] = (1, 2)) -> str'
    )

    assert [arg.name for arg in parsed.args] == ["text", "value"]
    assert [_render_type(arg.type) for arg in parsed.args] == [
        "str",
        "tuple[int, int]",
    ]
    assert [arg.default_value for arg in parsed.args] == [
        '") -> inside"',
        "(1, 2)",
    ]
    assert _render_type(parsed.return_type) == "str"


def test_parse_args_str_supports_angle_brackets() -> None:
    parsed = parse_args_str(
        "value: std::vector<int>, mapping: dict[str, std::pair<int, int>]"
    )

    assert [_render_type(arg.type) for arg in parsed] == [
        "std::vector<int>",
        "dict[str, std::pair<int, int>]",
    ]


def test_parse_pybind11_signature_rejects_full_docstring_format() -> None:
    with pytest.raises(ValueError, match="pybind11 单签名格式非法"):
        parse_pybind11_signature(
            "foo(*args, **kwargs)\nOverloaded function.\n1. foo(value: int) -> int"
        )


def test_parse_pybind11_signature_rejects_missing_annotation_for_normal_argument() -> None:
    with pytest.raises(ValueError, match="普通参数缺少类型注解"):
        parse_pybind11_signature("(value, other: int) -> int")


def test_parse_pybind11_signature_rejects_missing_return_arrow() -> None:
    with pytest.raises(ValueError, match="pybind11 单签名格式非法"):
        parse_pybind11_signature("(value: int)")


def test_parse_pybind11_signature_rejects_missing_return_type() -> None:
    with pytest.raises(ValueError, match="pybind11 单签名格式非法"):
        parse_pybind11_signature("(value: int) ->")
