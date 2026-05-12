from __future__ import annotations

from pathlib import Path

import pytest

from pcstubgen.models import QualifiedName
from pcstubgen.signature_completion import SignatureCompleter
from pcstubgen.signature_completion.completion_models import (
    SignatureCompletionContext,
    SignatureCompletionResult,
    UnsupportedSignatureCompletion,
)
from pcstubgen.type_models import RawType
from tests._c_extension_test_support import _arg, _signature


def _patch_compilation_database_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.completer.FunctionCursorLocator",
        lambda compilation_database: object(),
    )


def _context(*, func_name: str, member: object) -> SignatureCompletionContext:
    return SignatureCompletionContext(
        module_name=QualifiedName.from_str("pkg.mod"),
        func_name=func_name,
        member=member,
    )


def test_signature_completer_returns_completer_result_and_updates_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.completer.CExtensionCompleter.match",
        staticmethod(lambda member, owner_class=None: owner_class is None),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11.completer.Pybind11Completer.match",
        staticmethod(lambda member, owner_class=None: False),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.completer.CExtensionCompleter.get",
        lambda self, context: SignatureCompletionResult(
            signatures=[
                _signature(
                    args=[_arg("value", "int")],
                    return_type=RawType.bool_,
                )
            ],
            doc="foo(value: int) -> bool",
            provider="c_extension",
            mapping_status="success",
            parameter_inference_status="success",
            return_inference_status="success",
            source_location="mock:pkg.mod.foo",
            source_text="mocked source",
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
        provider="c_extension",
        mapping_status="success",
        parameter_inference_status="success",
        return_inference_status="success",
        source_location="mock:pkg.mod.foo",
        source_text="mocked source",
    )
    assert completer.summary.total == 1
    assert completer.summary.c_extension == 1
    assert completer.summary.pybind11 == 0
    assert completer.summary.failed == 0


def test_signature_completer_falls_back_to_minimal_signature_on_completer_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.completer.CExtensionCompleter.match",
        staticmethod(lambda member, owner_class=None: True),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11.completer.Pybind11Completer.match",
        staticmethod(lambda member, owner_class=None: False),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.completer.CExtensionCompleter.get",
        lambda self, context: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    completer = SignatureCompleter(tmp_path / "compile_commands.json")
    result = completer.complete(_context(func_name="build", member=object()))

    assert [arg.name for arg in result.signatures[0].args] == ["args", "kwargs"]
    assert result.doc is None
    assert result.decorator is None
    assert result.mapping_status == "failed"
    assert result.parameter_inference_status == "failed"
    assert result.return_inference_status == "failed"
    assert completer.summary.c_extension == 0
    assert completer.summary.pybind11 == 0
    assert completer.summary.failed == 1


def test_signature_completer_does_not_fallback_for_unsupported_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.completer.CExtensionCompleter.match",
        staticmethod(lambda member, owner_class=None: True),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.pybind11.completer.Pybind11Completer.match",
        staticmethod(lambda member, owner_class=None: False),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.completer.CExtensionCompleter.get",
        lambda self, context: (_ for _ in ()).throw(
            UnsupportedSignatureCompletion("skip")
        ),
    )

    completer = SignatureCompleter(tmp_path / "compile_commands.json")

    with pytest.raises(UnsupportedSignatureCompletion):
        completer.complete(_context(func_name="skip_me", member=object()))

    assert completer.summary.total == 0
    assert completer.summary.c_extension == 0
    assert completer.summary.pybind11 == 0
    assert completer.summary.failed == 0
