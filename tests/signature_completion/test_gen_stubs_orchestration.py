from __future__ import annotations

from pathlib import Path

from pcstubgen.models import Module, QualifiedName
from pcstubgen.stub_output import StubRenderer


class _RecordingWriter:
    def __init__(self) -> None:
        self.module: Module | None = None
        self.renderer: StubRenderer | None = None
        self.output: Path | None = None

    def write(
        self,
        module: Module,
        renderer: StubRenderer,
        to: Path,
    ) -> None:
        self.module = module
        self.renderer = renderer
        self.output = to


def test_gen_stubs_uses_collected_module_and_injected_writer(
    monkeypatch,
    tmp_path,
) -> None:
    import pcstubgen.api as stubgen_module

    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.FunctionCursorLocator",
        lambda compilation_database: object(),
    )
    (tmp_path / "compile_commands.json").write_text("[]", encoding="utf-8")

    module_node = Module(
        full_name=QualifiedName.from_str("math"),
    )
    writer = _RecordingWriter()
    output_dir = tmp_path / "nested" / "stubs"

    monkeypatch.setattr(
        stubgen_module.ModuleCollector,
        "run",
        lambda self, module_name: module_node,
    )

    stubgen_module.gen_stubs(
        "math",
        output_dir,
        tmp_path / "compile_commands.json",
        include_docstrings=True,
        writer=writer,
    )

    assert output_dir.is_dir()
    assert writer.module is module_node
    assert writer.output == output_dir
    assert writer.renderer is not None
    assert writer.renderer.include_docstrings is True
