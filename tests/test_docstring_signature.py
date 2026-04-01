from __future__ import annotations

import pytest

from pcstubgen.ir import IRFunction, IRModule, IRModuleType, QualifiedName
from pcstubgen.signature_completion.docstring_source import (
    parse_args_str,
    resolve_docstring_signatures,
)
from pcstubgen.ir import IRArgumentKind


def _render_type(type_: object | None) -> str | None:
    if type_ is None:
        return None
    return type_.render()


def _resolve(
    function_name: str,
    doc: str | None,
):
    irmodule = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
    )
    irfunction = IRFunction(name=function_name, doc=doc)
    return resolve_docstring_signatures(irmodule, irfunction)


def test_docstring_parser_parses_generic_function_signature() -> None:
    parsed = _resolve("foo", "foo(x: int, y: str) -> str\n\nparsed from docstring")

    assert parsed is not None
    signature = parsed[0]
    assert [arg.name for arg in signature.args] == ["x", "y"]
    assert _render_type(signature.return_type) == "str"


def test_docstring_parser_parses_pybind11_style_signature_with_defaults() -> None:
    parsed = _resolve(
        "cdist_minkowski",
        (
            "cdist_minkowski(x: object, y: object, w: object = None, "
            "out: object = None, p: typing.SupportsFloat = 2.0) -> numpy.ndarray"
        ),
    )

    assert parsed is not None
    signature = parsed[0]
    assert [arg.name for arg in signature.args] == ["x", "y", "w", "out", "p"]
    assert [_render_type(arg.type) for arg in signature.args] == [
        "object",
        "object",
        "object",
        "object",
        "typing.SupportsFloat",
    ]
    assert [arg.default_value for arg in signature.args] == [
        None,
        None,
        "None",
        "None",
        "2.0",
    ]
    assert [arg.has_default for arg in signature.args] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert _render_type(signature.return_type) == "numpy.ndarray"


def test_docstring_parser_preserves_overload_docs() -> None:
    parsed = _resolve(
        "foo",
        (
            "foo(*args, **kwargs)\n"
            "Overloaded function.\n"
            "1. foo(value: int) -> str\n"
            "\n"
            "first overload\n"
            "2. foo(value: str) -> int\n"
            "\n"
            "second overload"
        ),
    )

    assert parsed is not None
    assert len(parsed) == 2
    assert [_render_type(sig.return_type) for sig in parsed] == ["str", "int"]


def test_docstring_parser_returns_none_without_doc() -> None:
    assert _resolve("foo", None) is None


def test_docstring_parser_returns_none_for_non_signature_first_line() -> None:
    parsed = _resolve("foo", "This is not a signature.\nstill docs")

    assert parsed is None


def test_docstring_parser_raises_on_invalid_signature_like_doc() -> None:
    with pytest.raises(ValueError):
        _resolve("foo", "foo(a: int,, b: int) -> int\n\nbroken")


def test_docstring_parser_parse_args_str_supports_nested_defaults_and_markers() -> None:
    parsed = parse_args_str(
        'a: int, /, x: tuple[int, int] = (1, 2), *, '
        'mapping: dict[str, int] = {"a": 1, "b": 2}, flag: str = "x"'
    )

    assert [arg.name for arg in parsed] == ["a", "x", "mapping", "flag"]
    assert [_render_type(arg.type) for arg in parsed] == [
        "int",
        "tuple[int, int]",
        "dict[str, int]",
        "str",
    ]
    assert [arg.default_value for arg in parsed] == [
        None,
        "(1, 2)",
        '{"a": 1, "b": 2}',
        '"x"',
    ]
    assert [arg.has_default for arg in parsed] == [False, True, True, True]
    assert [arg.kind for arg in parsed] == [
        IRArgumentKind.POSITIONAL_ONLY,
        IRArgumentKind.POSITIONAL_OR_KEYWORD,
        IRArgumentKind.KEYWORD_ONLY,
        IRArgumentKind.KEYWORD_ONLY,
    ]


def test_docstring_parser_parse_args_str_supports_var_args_and_var_kwargs() -> None:
    parsed = parse_args_str(
        "value: typing.Optional[list[int]], *args: tuple[str, ...], **kwargs: object"
    )

    assert [arg.name for arg in parsed] == ["value", "args", "kwargs"]
    assert [_render_type(arg.type) for arg in parsed] == [
        "typing.Optional[list[int]]",
        "tuple[str, ...]",
        "object",
    ]
    assert [arg.kind for arg in parsed] == [
        IRArgumentKind.POSITIONAL_OR_KEYWORD,
        IRArgumentKind.VAR_POSITIONAL,
        IRArgumentKind.VAR_KEYWORD,
    ]


@pytest.mark.parametrize(
    "args_str",
    [
        "value: tuple[int, str",
        'value: str = "unterminated',
        "x: int,, y: int",
        "x: int: str",
        "x = 1 = 2",
        "/, a: int",
        "a: int, /, /",
        "a: int, *, *",
        "a: int, *: int",
        "a: int, **kwargs: object, b: int",
        "*: int",
        "*args = ()",
        "**kwargs = {}",
        "a: int, *, /, b: int",
        "a: int, *args: tuple[int, ...], *, b: int",
    ],
)
def test_docstring_parser_parse_args_str_rejects_invalid_input(args_str: str) -> None:
    with pytest.raises(ValueError):
        parse_args_str(args_str)
