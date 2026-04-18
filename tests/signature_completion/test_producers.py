from __future__ import annotations

from pathlib import Path

import pytest

from pcstubgen.models import Argument, Function, Module, QualifiedName, Signature
from pcstubgen.signature_completion.producers import (
    CExtensionSignatureProducer,
    DocstringSignatureProducer,
    MinimalSignatureProducer,
)
from pcstubgen.type_models import RawType


def _module() -> Module:
    return Module(full_name=QualifiedName.from_str("pkg.mod"))


def test_docstring_producer_adds_instance_receiver() -> None:
    result = DocstringSignatureProducer().produce(
        _module(),
        Function(
            name="append",
            runtime_handle=object(),
            doc="append(value: int) -> bool",
        ),
        is_method=True,
    )

    assert [arg.name for arg in result.signatures[0].args] == ["self", "value"]
    assert result.signatures[0].args[1].type is not None
    assert result.signatures[0].args[1].type.render() == "int"


def test_docstring_producer_rewrites_classmethod_receiver() -> None:
    result = DocstringSignatureProducer().produce(
        _module(),
        Function(
            name="build",
            runtime_handle=object(),
            decorator="classmethod",
            doc="build(self, value: int) -> bool",
        ),
        is_method=True,
    )

    assert [arg.name for arg in result.signatures[0].args] == ["cls", "value"]


def test_docstring_producer_strips_staticmethod_receiver() -> None:
    result = DocstringSignatureProducer().produce(
        _module(),
        Function(
            name="make",
            runtime_handle=object(),
            decorator="staticmethod",
            doc="make(self, value: int) -> bool",
        ),
        is_method=True,
    )

    assert [arg.name for arg in result.signatures[0].args] == ["value"]


def test_c_extension_producer_adds_method_receiver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.ClangParser",
        lambda compilation_database: object(),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.producers.CExtensionSource.infer_function_signatures",
        lambda self, module, func: type(
            "Result",
            (),
            {
                "signatures": [
                    Signature(
                        args=[Argument(name="value", type=RawType("int"))],
                        return_type=RawType("bool"),
                    )
                ],
                "comment": "mock-comment",
            },
        )(),
    )

    result = CExtensionSignatureProducer(tmp_path / "compile_commands.json").produce(
        _module(),
        Function(name="append", runtime_handle=object()),
        is_method=True,
    )

    assert [arg.name for arg in result.signatures[0].args] == ["self", "value"]
    assert result.comment == "mock-comment"


def test_minimal_signature_producer_uses_generic_pybind_shape() -> None:
    result = MinimalSignatureProducer().produce(
        _module(),
        Function(name="broken", runtime_handle=object()),
        is_method=True,
    )

    assert [arg.name for arg in result.signatures[0].args] == ["self", "args", "kwargs"]
