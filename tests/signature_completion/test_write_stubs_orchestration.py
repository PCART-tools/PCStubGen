from __future__ import annotations

from types import ModuleType

from pcstubgen.ir_modules import IRModule, QualifiedName
from pcstubgen.types import RawType
from tests._c_extension_test_support import (
    _arg,
    _patch_c_signature_extractor,
    _signature,
    _unknown_function,
    ResolvedFunctionFixture,
)


def test_write_stubs_writes_rendered_stub_file(
    monkeypatch,
    tmp_path,
) -> None:
    import pcstubgen.api as stubgen_module

    ir_module = IRModule(
        full_name=QualifiedName.from_str("math"),
        functions=[_unknown_function("foo", doc="foo(value: str) -> bool")],
    )

    monkeypatch.setattr(
        stubgen_module.importlib,
        "import_module",
        lambda module_name: ModuleType(module_name),
    )
    monkeypatch.setattr(stubgen_module, "collect_module", lambda path, module: ir_module)
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.compilation_database_loader.load_compilation_database",
        lambda path: object(),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.completion.resolve_runtime_pymethoddef",
        lambda handle: object(),
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

    stubgen_module.write_stubs(
        "math",
        tmp_path,
        tmp_path / "compile_commands.json",
    )

    stub_path = tmp_path / "math.pyi"
    assert stub_path.exists()
    assert stub_path.read_text(encoding="utf-8") == "def foo(value: str) -> bool:\n    ...\n"
