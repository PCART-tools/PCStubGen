from __future__ import annotations

from pcstubgen.models import Module, QualifiedName
from pcstubgen.type_models import RawType
from tests._c_extension_test_support import (
    _arg,
    _patch_c_signature_extractor,
    _signature,
    _unknown_function,
    ResolvedFunctionFixture,
)


def test_gen_stubs_orchestrates_collection_completion_and_writing(
    monkeypatch,
    tmp_path,
) -> None:
    import pcstubgen.api as stubgen_module

    module_node = Module(
        full_name=QualifiedName.from_str("math"),
        functions=[_unknown_function("foo", doc="foo(value: str) -> bool")],
    )

    monkeypatch.setattr(
        stubgen_module.ModuleCollector,
        "run",
        lambda self, module_name: module_node,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.ClangParser",
        lambda compilation_database: object(),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.completion.is_cpython_builtin",
        lambda handle: True,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.completion.is_pybind11_builtin",
        lambda handle: False,
    )
    _patch_c_signature_extractor(
        monkeypatch,
        functions={
            "foo": ResolvedFunctionFixture(
                signatures=[
                    _signature(
                        args=[_arg("value", "str")],
                        return_type=RawType("bool"),
                    )
                ]
            )
        },
    )

    stubgen_module.gen_stubs(
        "math",
        tmp_path,
        tmp_path / "compile_commands.json",
    )

    stub_path = tmp_path / "math.pyi"
    assert stub_path.exists()
    assert stub_path.read_text(encoding="utf-8") == "def foo(value: str) -> bool:\n    ...\n"


def test_gen_stubs_writes_minimal_stub_when_c_completion_fails(
    monkeypatch,
    tmp_path,
) -> None:
    import pcstubgen.api as stubgen_module

    module_node = Module(
        full_name=QualifiedName.from_str("math"),
        functions=[_unknown_function("foo")],
    )

    monkeypatch.setattr(
        stubgen_module.ModuleCollector,
        "run",
        lambda self, module_name: module_node,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.ClangParser",
        lambda compilation_database: object(),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.completion.is_cpython_builtin",
        lambda handle: True,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.completion.is_pybind11_builtin",
        lambda handle: False,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.producers.is_cpython_builtin",
        lambda handle: True,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.producers.read_cpython_function_runtime_info",
        lambda handle: type("RuntimeInfo", (), {"address": 1234, "flags": 0})(),
    )
    _patch_c_signature_extractor(monkeypatch, functions={})

    stubgen_module.gen_stubs(
        "math",
        tmp_path,
        tmp_path / "compile_commands.json",
    )

    stub_path = tmp_path / "math.pyi"
    assert stub_path.exists()
    assert stub_path.read_text(encoding="utf-8") == "def foo(\n    *args,\n    **kwargs,\n):\n    ...\n"
