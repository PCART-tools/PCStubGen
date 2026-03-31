from __future__ import annotations

import typing

import pytest

import pcstubgen.signature_completion.inspect_source as inspect_signature_module
from pcstubgen.signature_completion.inspect_source import resolve_inspect_signatures


def _render_type(type_: object | None) -> str | None:
    if type_ is None:
        return None
    return type_.render()


def test_resolve_inspect_signatures_parses_annotations_and_defaults() -> None:
    def sample(
        a: int,
        values: list[int] = [1, 2],
    ) -> typing.Optional[int]:
        raise NotImplementedError

    resolved = resolve_inspect_signatures(sample)

    assert resolved is not None
    signature = resolved.signatures[0]
    assert [arg.name for arg in signature.arguments] == ["a", "values"]
    assert [_render_type(arg.type) for arg in signature.arguments] == ["int", "list[int]"]
    assert [arg.default_value for arg in signature.arguments] == [None, "[1, 2]"]
    assert [arg.has_default for arg in signature.arguments] == [False, True]
    assert _render_type(signature.return_type) == "typing.Optional[int]"


def test_resolve_inspect_signatures_preserves_tuple_default_text() -> None:
    def sample(values: tuple[int, int] = (1, 2)) -> None:
        raise NotImplementedError

    resolved = resolve_inspect_signatures(sample)

    assert resolved is not None
    signature = resolved.signatures[0]
    assert signature.arguments[0].default_value == "(1, 2)"
    assert signature.arguments[0].has_default is True


def test_resolve_inspect_signatures_returns_none_when_inspect_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sample() -> None:
        raise NotImplementedError

    def _raise_signature_error(obj: object) -> object:
        raise RuntimeError(f"boom: {obj!r}")

    monkeypatch.setattr(inspect_signature_module.inspect, "signature", _raise_signature_error)

    assert resolve_inspect_signatures(sample) is None


def test_resolve_inspect_signatures_normalizes_class_bound_method() -> None:
    class Builder:
        @classmethod
        def build(cls, value: int) -> str:
            raise NotImplementedError

    resolved = resolve_inspect_signatures(Builder.build)

    assert resolved is not None
    signature = resolved.signatures[0]
    assert [arg.name for arg in signature.arguments] == ["cls", "value"]
    assert [_render_type(arg.type) for arg in signature.arguments] == [None, "int"]
    assert _render_type(signature.return_type) == "str"
