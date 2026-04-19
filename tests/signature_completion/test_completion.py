from __future__ import annotations

from pathlib import Path

import pytest

from pcstubgen.models import QualifiedName
from pcstubgen.signature_completion import SignatureCompleter
from pcstubgen.signature_completion.completion_models import (
    SignatureCompletionContext,
    SignatureCompletionResult,
)
from pcstubgen.type_models import RawType
from tests._c_extension_test_support import _arg, _signature


def _patch_compilation_database_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.ClangParser",
        lambda compilation_database: object(),
    )


def _context(*, func_name: str, member: object, is_method: bool = False) -> SignatureCompletionContext:
    return SignatureCompletionContext(
        module_name=QualifiedName.from_str("pkg.mod"),
        func_name=func_name,
        member=member,
        is_method=is_method,
    )


def test_completer_returns_c_extension_signature_and_comment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.CExtensionProvider.support",
        staticmethod(lambda member, is_method: not is_method),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11_provider.Pybind11Provider.support",
        staticmethod(lambda member, is_method: False),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.CExtensionProvider.get",
        lambda self, context: SignatureCompletionResult(
            success=True,
            message="",
            provider="c_extension",
            signatures=[
                _signature(
                    args=[_arg("value", "int")],
                    return_type=RawType("bool"),
                )
            ],
            doc="foo(value: int) -> bool",
            comment="mock:pkg.mod.foo\nmocked source",
        ),
    )

    completer = SignatureCompleter(tmp_path / "compile_commands.json")
    result = completer.complete(_context(func_name="foo", member=object()))

    assert [arg.name for arg in result.signatures[0].args] == ["value"]
    assert result.signatures[0].args[0].type is not None
    assert result.signatures[0].args[0].type.render() == "int"
    assert result.signatures[0].return_type is not None
    assert result.signatures[0].return_type.render() == "bool"
    assert result.doc == "foo(value: int) -> bool"
    assert result.comment == "mock:pkg.mod.foo\nmocked source"
    assert completer.summary.total == 1
    assert completer.summary.c_extension == 1
    assert completer.summary.pybind11 == 0
    assert completer.summary.failed == 0


def test_completer_returns_pybind11_signature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.CExtensionProvider.support",
        staticmethod(lambda member, is_method: False),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11_provider.Pybind11Provider.support",
        staticmethod(lambda member, is_method: not is_method),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11_provider.Pybind11Provider.get",
        lambda self, context: SignatureCompletionResult(
            success=True,
            message="",
            provider="pybind11",
            signatures=[
                _signature(
                    args=[_arg("value", "str")],
                    return_type=RawType("bool"),
                )
            ],
            doc="fallback(value: str) -> bool\n\nparsed from docstring",
        ),
    )

    completer = SignatureCompleter(tmp_path / "compile_commands.json")
    result = completer.complete(_context(func_name="fallback", member=object()))

    assert [arg.name for arg in result.signatures[0].args] == ["value"]
    assert result.signatures[0].args[0].type is not None
    assert result.signatures[0].args[0].type.render() == "str"
    assert result.signatures[0].return_type is not None
    assert result.signatures[0].return_type.render() == "bool"
    assert result.doc == "fallback(value: str) -> bool\n\nparsed from docstring"
    assert completer.summary.c_extension == 0
    assert completer.summary.pybind11 == 1
    assert completer.summary.failed == 0


def test_completer_falls_back_to_minimal_signature_on_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.CExtensionProvider.support",
        staticmethod(lambda member, is_method: is_method),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11_provider.Pybind11Provider.support",
        staticmethod(lambda member, is_method: False),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.CExtensionProvider.get",
        lambda self, context: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    completer = SignatureCompleter(tmp_path / "compile_commands.json")
    result = completer.complete(
        _context(func_name="build", member=dict.__dict__["fromkeys"], is_method=True)
    )

    assert [arg.name for arg in result.signatures[0].args] == ["cls", "args", "kwargs"]
    assert result.provider == "c_extension"
    assert result.success is False
    assert result.doc is None
    assert result.decorator is None
    assert result.comment is None
    assert result.message == "RuntimeError: boom"
    assert completer.summary.c_extension == 1
    assert completer.summary.failed == 1


def test_completer_falls_back_to_minimal_signature_without_metadata_on_generic_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.CExtensionProvider.support",
        staticmethod(lambda member, is_method: is_method),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11_provider.Pybind11Provider.support",
        staticmethod(lambda member, is_method: False),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.CExtensionProvider.get",
        lambda self, context: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    completer = SignatureCompleter(tmp_path / "compile_commands.json")
    result = completer.complete(
        _context(func_name="build", member=staticmethod(object()), is_method=True)
    )

    assert [arg.name for arg in result.signatures[0].args] == ["args", "kwargs"]
    assert result.provider == "c_extension"
    assert result.success is False
    assert result.doc is None
    assert result.decorator is None
    assert result.comment is None
    assert result.message == "RuntimeError: boom"
    assert completer.summary.c_extension == 1
    assert completer.summary.failed == 1
