from __future__ import annotations

from types import ModuleType

import pytest

from pcstubgen.ir_modules import IRModule, IRModuleType, QualifiedName
from pcstubgen.stub_generation_options import StubGenerationOptions
from tests._c_extension_test_support import (
    _patch_raising_c_signature_extractor,
    _unknown_function,
)


@pytest.mark.integration
def test_write_stubs_writes_rendered_stub_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import pcstubgen.api as stubgen_module

    ir_module = IRModule(
        full_name=QualifiedName.from_str("math"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo", doc="foo(value: str) -> bool")],
    )

    monkeypatch.setattr(
        stubgen_module.importlib,
        "import_module",
        lambda module_name: ModuleType(module_name),
    )
    monkeypatch.setattr(stubgen_module, "collect_module", lambda path, module: ir_module)
    _patch_raising_c_signature_extractor(monkeypatch, RuntimeError("boom"))

    stubgen_module.write_stubs(
        "math",
        tmp_path,
        options=StubGenerationOptions(
            compilation_database=tmp_path / "compile_commands.json",
        ),
    )

    stub_path = tmp_path / "math.pyi"
    assert stub_path.exists()
    assert stub_path.read_text(encoding="utf-8") == "def foo(value: str) -> bool:\n    ...\n"
