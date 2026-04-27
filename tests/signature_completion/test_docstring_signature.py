from __future__ import annotations

import pytest

from pcstubgen.signature_completion.pybind11_inferencer import (
    infer,
    parse_args_str,
)
from pcstubgen.models import ArgumentKind


def _render_type(type_: object | None) -> str | None:
    if type_ is None:
        return None
    return type_.render()


def _parse_docstring(
    function_name: str,
    doc: str | None,
):
    return infer(function_name, doc)


def test_docstring_parser_parses_generic_function_signature() -> None:
    parsed = _parse_docstring("foo", "foo(x: int, y: str) -> str\n\nparsed from docstring")

    signature = parsed[0]
    assert [arg.name for arg in signature.args] == ["x", "y"]
    assert _render_type(signature.return_type) == "str"


def test_docstring_parser_parses_pybind11_style_signature_with_defaults() -> None:
    parsed = _parse_docstring(
        "cdist_minkowski",
        (
            "cdist_minkowski(x: object, y: object, w: object = None, "
            "out: object = None, p: typing.SupportsFloat = 2.0) -> numpy.ndarray"
        ),
    )

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
    assert _render_type(signature.return_type) == "numpy.ndarray"


def test_docstring_parser_parses_pybind11_signature_with_cpp_type_names() -> None:
    parsed = _parse_docstring(
        "may_contain_alias",
        (
            "may_contain_alias(self: torch._C.AliasDb, arg0: torch::jit::Value, "
            "arg1: torch::jit::Value) -> bool"
        ),
    )

    signature = parsed[0]
    assert [arg.name for arg in signature.args] == ["self", "arg0", "arg1"]
    assert [_render_type(arg.type) for arg in signature.args] == [
        "torch._C.AliasDb",
        "torch::jit::Value",
        "torch::jit::Value",
    ]
    assert _render_type(signature.return_type) == "bool"


def test_docstring_parser_parses_overload_signatures() -> None:
    parsed = _parse_docstring(
        "foo",
        (
            "foo(*args, **kwargs)\n"
            "Overloaded function.\n"
            "1. foo(value: int) -> str\n"
            "2. foo(value: str) -> int\n"
        ),
    )

    assert len(parsed) == 2
    assert [_render_type(sig.return_type) for sig in parsed] == ["str", "int"]


def test_docstring_parser_parses_overload_signatures_with_interleaved_docs() -> None:
    parsed = _parse_docstring(
        "broadcast",
        (
            "broadcast(*args, **kwargs)\n"
            "Overloaded function.\n"
            "\n"
            "1. broadcast(self: torch._C._distributed_c10d.ProcessGroup, "
            "tensors: collections.abc.Sequence[torch.Tensor]) -> c10d::Work\n"
            "\n"
            "Broadcasts the tensor to all processes in the process group.\n"
            "\n"
            "2. broadcast(self: torch._C._distributed_c10d.ProcessGroup, "
            "tensor: torch.Tensor, root: typing.SupportsInt) -> c10d::Work\n"
            "\n"
            "Broadcasts the tensor to all processes in the process group.\n"
        ),
    )

    assert len(parsed) == 2
    assert [_render_type(sig.return_type) for sig in parsed] == [
        "c10d::Work",
        "c10d::Work",
    ]
    assert [[arg.name for arg in sig.args] for sig in parsed] == [
        ["self", "tensors"],
        ["self", "tensor", "root"],
    ]


def test_docstring_parser_parses_overload_signatures_with_multiple_doc_blocks() -> None:
    parsed = _parse_docstring(
        "run",
        (
            "run(*args, **kwargs)\n"
            "Overloaded function.\n"
            "\n"
            "1. run(self: torch._C.FileCheck, arg0: str) -> None\n"
            "\n"
            "2. run(self: torch._C.FileCheck, arg0: torch._C.Graph) -> None\n"
            "\n"
            "3. run(self: torch._C.FileCheck, checks_file: str, test_file: str) -> None\n"
            "\n"
            "Run\n"
            "\n"
            "4. run(self: torch._C.FileCheck, checks_file: str, graph: torch._C.Graph) -> None\n"
            "\n"
            "Run\n"
        ),
    )

    assert len(parsed) == 4
    assert [_render_type(sig.return_type) for sig in parsed] == [
        "None",
        "None",
        "None",
        "None",
    ]
    assert [[arg.name for arg in sig.args] for sig in parsed] == [
        ["self", "arg0"],
        ["self", "arg0"],
        ["self", "checks_file", "test_file"],
        ["self", "checks_file", "graph"],
    ]


def test_docstring_parser_parses_runtime_name_signature() -> None:
    parsed = _parse_docstring(
        "_mtia_exchangeDevice",
        "_mtia_exchangeDevice(arg0: typing.SupportsInt) -> int\n",
    )

    signature = parsed[0]
    assert [arg.name for arg in signature.args] == ["arg0"]
    assert [_render_type(arg.type) for arg in signature.args] == [
        "typing.SupportsInt"
    ]
    assert _render_type(signature.return_type) == "int"


def test_docstring_parser_raises_without_doc() -> None:
    with pytest.raises(RuntimeError):
        _parse_docstring("foo", None)


def test_docstring_parser_raises_for_non_signature_first_line() -> None:
    with pytest.raises(RuntimeError):
        _parse_docstring("foo", "This is not a signature.\nstill docs")


def test_docstring_parser_raises_on_invalid_signature_like_doc() -> None:
    with pytest.raises(RuntimeError):
        _parse_docstring("foo", "foo(a: int,, b: int) -> int\n\nbroken")


def test_docstring_parser_raises_for_overload_with_non_consecutive_numbers() -> None:
    with pytest.raises(RuntimeError):
        _parse_docstring(
            "foo",
            (
                "foo(*args, **kwargs)\n"
                "Overloaded function.\n"
                "1. foo(value: int) -> str\n"
                "3. foo(value: str) -> int\n"
            ),
        )


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
    assert [arg.kind for arg in parsed] == [
        ArgumentKind.POSITIONAL_ONLY,
        ArgumentKind.POSITIONAL_OR_KEYWORD,
        ArgumentKind.KEYWORD_ONLY,
        ArgumentKind.KEYWORD_ONLY,
    ]


def test_docstring_parser_parse_args_str_maps_explicit_ellipsis_default_to_unknown() -> None:
    parsed = parse_args_str("value: object = ...")

    assert [arg.default_value for arg in parsed] == ["..."]


def test_docstring_parser_parse_args_str_supports_cpp_namespace_type_names() -> None:
    parsed = parse_args_str(
        "arg0: torch::jit::Value, arg1: c10::Type, arg2: list[torch::jit::Value]"
    )

    assert [arg.name for arg in parsed] == ["arg0", "arg1", "arg2"]
    assert [_render_type(arg.type) for arg in parsed] == [
        "torch::jit::Value",
        "c10::Type",
        "list[torch::jit::Value]",
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
        ArgumentKind.POSITIONAL_OR_KEYWORD,
        ArgumentKind.VAR_POSITIONAL,
        ArgumentKind.VAR_KEYWORD,
    ]


@pytest.mark.parametrize(
    "args_str",
    [
        "value: tuple[int, str",
        'value: str = "unterminated',
        "x: int,, y: int",
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
