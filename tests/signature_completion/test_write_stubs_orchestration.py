from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from pcstubgen.ir_modules import IRModule, IRModuleType, QualifiedName
from pcstubgen.stub_generation_options import StubGenerationOptions
from tests._c_extension_test_support import (
    _patch_raising_c_signature_extractor,
    _unknown_function,
)


@pytest.mark.integration
def test_write_stubs_orchestrates_completion_renderer_and_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import pcstubgen.api as stubgen_module

    captured: dict[str, object] = {}
    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"))

    class FakeSummary:
        def __str__(self) -> str:
            return "summary"

    class FakeStubRenderer:
        def __init__(
            self,
            include_docstrings: bool = False,
            include_c_inferred_source_comment: bool = False,
        ) -> None:
            captured["renderer_include_docstrings"] = include_docstrings
            captured["renderer_include_c_inferred_source_comment"] = (
                include_c_inferred_source_comment
            )

    class FakeWriter:
        def write(
            self,
            module: IRModule,
            renderer: FakeStubRenderer,
            to,
        ) -> None:
            captured["written_module"] = module
            captured["written_renderer"] = renderer
            captured["written_to"] = to

    class FakeSignatureCompleter:
        def __init__(self, options: StubGenerationOptions) -> None:
            captured["completion_options"] = options

        def run(self, module: IRModule) -> FakeSummary:
            captured["completed_module"] = module
            return FakeSummary()

    monkeypatch.setattr(stubgen_module.importlib, "import_module", lambda module_name: ModuleType(module_name))
    monkeypatch.setattr(stubgen_module, "collect_module", lambda path, module: ir_module)
    monkeypatch.setattr(stubgen_module, "SignatureCompleter", FakeSignatureCompleter)
    monkeypatch.setattr(stubgen_module, "StubRenderer", FakeStubRenderer)
    monkeypatch.setattr(stubgen_module, "logger", SimpleNamespace(info=lambda *args: None))

    options = StubGenerationOptions(
        compilation_database=tmp_path / "compile_commands.json",
        include_docstrings=False,
        include_c_inferred_source_comment=True,
    )
    writer = FakeWriter()

    stubgen_module.write_stubs("math", tmp_path, options=options, writer=writer)

    assert captured["completed_module"] is ir_module
    assert captured["completion_options"] is options
    assert captured["renderer_include_docstrings"] is False
    assert captured["renderer_include_c_inferred_source_comment"] is True
    assert captured["written_module"] is ir_module
    assert isinstance(captured["written_renderer"], FakeStubRenderer)
    assert captured["written_to"] == tmp_path


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


@pytest.mark.integration
def test_write_stubs_runs_without_tool_precheck_for_python_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import pcstubgen.api as stubgen_module

    captured: dict[str, object] = {}
    ir_module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
    )

    class FakeSummary:
        def __str__(self) -> str:
            return "summary"

    class FakeSignatureCompleter:
        def __init__(self, options: StubGenerationOptions) -> None:
            captured["options"] = options

        def run(self, module: IRModule) -> FakeSummary:
            captured["module"] = module
            return FakeSummary()

    class FakeWriter:
        def write(self, module: IRModule, renderer, to) -> None:
            captured["written_module"] = module
            captured["written_to"] = to

    monkeypatch.setattr(
        stubgen_module.importlib,
        "import_module",
        lambda module_name: ModuleType(module_name),
    )
    monkeypatch.setattr(stubgen_module, "collect_module", lambda path, module: ir_module)
    monkeypatch.setattr(stubgen_module, "SignatureCompleter", FakeSignatureCompleter)
    monkeypatch.setattr(stubgen_module, "logger", SimpleNamespace(info=lambda *args: None))

    writer = FakeWriter()
    stubgen_module.write_stubs("pkg.mod", tmp_path, options=StubGenerationOptions(), writer=writer)

    assert captured["module"] is ir_module
    assert captured["written_module"] is ir_module
    assert captured["written_to"] == tmp_path
