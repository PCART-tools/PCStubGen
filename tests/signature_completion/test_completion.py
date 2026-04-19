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


def test_completer_returns_c_extension_signature_and_comment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.CExtensionProvider.support",
        lambda handle: True,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11_provider.Pybind11Provider.support",
        lambda handle: False,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.CExtensionProvider.get",
        lambda self, func, is_method: SignatureCompletionResult(
            signatures=[
                _signature(
                    args=[_arg("value", "int")],
                    return_type=RawType("bool"),
                )
            ],
            comment="mock:pkg.mod.foo\nmocked source",
        ),
    )

    completer = SignatureCompleter(tmp_path / "compile_commands.json")
    result = completer.complete(
        SignatureCompletionContext(
            path=QualifiedName.from_str("pkg.mod.foo"),
            handle=object(),
            doc="foo(value: str) -> str\n\nparsed from docstring",
        )
    )

    assert [arg.name for arg in result.signatures[0].args] == ["value"]
    assert result.signatures[0].args[0].type is not None
    assert result.signatures[0].args[0].type.render() == "int"
    assert result.signatures[0].return_type is not None
    assert result.signatures[0].return_type.render() == "bool"
    assert result.comment == "mock:pkg.mod.foo\nmocked source"
    assert completer.summary.total_functions == 1
    assert completer.summary.c_extension_completed == 1
    assert completer.summary.pybind11_completed == 0
    assert completer.summary.failed == 0


def test_completer_normalizes_classmethod_receiver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.CExtensionProvider.support",
        lambda handle: True,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11_provider.Pybind11Provider.support",
        lambda handle: False,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.CExtensionProvider.get",
        lambda self, func, is_method: SignatureCompletionResult(
            signatures=[
                _signature(
                    args=[_arg("value", "int")],
                    return_type=RawType("bool"),
                )
            ]
        ),
    )

    completer = SignatureCompleter(tmp_path / "compile_commands.json")
    result = completer.complete(
        SignatureCompletionContext(
            path=QualifiedName.from_str("pkg.mod.Factory.build"),
            handle=object(),
            decorator="classmethod",
            is_method=True,
        )
    )

    assert [arg.name for arg in result.signatures[0].args] == ["cls", "value"]


def test_completer_returns_pybind11_signature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.CExtensionProvider.support",
        lambda handle: False,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11_provider.Pybind11Provider.support",
        lambda handle: True,
    )

    completer = SignatureCompleter(tmp_path / "compile_commands.json")
    result = completer.complete(
        SignatureCompletionContext(
            path=QualifiedName.from_str("pkg.mod.fallback"),
            handle=object(),
            doc="fallback(value: str) -> bool\n\nparsed from docstring",
        )
    )

    assert [arg.name for arg in result.signatures[0].args] == ["value"]
    assert result.signatures[0].args[0].type is not None
    assert result.signatures[0].args[0].type.render() == "str"
    assert result.signatures[0].return_type is not None
    assert result.signatures[0].return_type.render() == "bool"
    assert completer.summary.pybind11_completed == 1
    assert completer.summary.failed == 0


def test_completer_falls_back_to_minimal_signature_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.CExtensionProvider.support",
        lambda handle: True,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11_provider.Pybind11Provider.support",
        lambda handle: False,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.CExtensionProvider.get",
        lambda self, func, is_method: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    completer = SignatureCompleter(tmp_path / "compile_commands.json")
    result = completer.complete(
        SignatureCompletionContext(
            path=QualifiedName.from_str("pkg.mod.Factory.build"),
            handle=object(),
            decorator="classmethod",
            is_method=True,
        )
    )

    assert [arg.name for arg in result.signatures[0].args] == ["cls", "args", "kwargs"]
    assert result.comment is None
    assert completer.summary.c_extension_completed == 0
    assert completer.summary.failed == 1
