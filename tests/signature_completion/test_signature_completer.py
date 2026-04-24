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
        "pcstubgen.signature_completion.c_extension.provider.FunctionCursorLocator",
        lambda compilation_database: object(),
    )


def _context(*, func_name: str, member: object, is_method: bool = False) -> SignatureCompletionContext:
    return SignatureCompletionContext(
        module_name=QualifiedName.from_str("pkg.mod"),
        func_name=func_name,
        member=member,
        is_method=is_method,
    )


def test_signature_completer_returns_provider_result_and_updates_summary(
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
            signatures=[
                _signature(
                    args=[_arg("value", "int")],
                    return_type=RawType.bool_,
                )
            ],
            doc="foo(value: int) -> bool",
            comment="mock:pkg.mod.foo\nmocked source",
        ),
    )

    completer = SignatureCompleter(tmp_path / "compile_commands.json")
    result = completer.complete(_context(func_name="foo", member=object()))

    assert result == SignatureCompletionResult(
        signatures=[
            _signature(
                args=[_arg("value", "int")],
                return_type=RawType.bool_,
            )
        ],
        doc="foo(value: int) -> bool",
        comment="mock:pkg.mod.foo\nmocked source",
    )
    assert completer.summary.total == 1
    assert completer.summary.c_extension == 1
    assert completer.summary.pybind11 == 0
    assert completer.summary.failed == 0


def test_signature_completer_falls_back_to_minimal_signature_on_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.CExtensionProvider.support",
        staticmethod(lambda member, is_method: True),
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
    result = completer.complete(_context(func_name="build", member=object()))

    assert [arg.name for arg in result.signatures[0].args] == ["args", "kwargs"]
    assert result.doc is None
    assert result.decorator is None
    assert result.comment is None
    assert completer.summary.c_extension == 0
    assert completer.summary.pybind11 == 0
    assert completer.summary.failed == 1
