from __future__ import annotations

from pathlib import Path
import tomllib

from typer.testing import CliRunner

from pcstubgen.__main__ import app
from pcstubgen.models import Argument, Function, Module, QualifiedName
from pcstubgen.stub_output import StubRenderer
from pcstubgen.type_models import RawType
from tests._c_extension_test_support import _signature


def test_gen_command_writes_toml_instead_of_stub_when_toml_flag_is_enabled(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.FunctionCursorLocator",
        lambda compilation_database: object(),
    )
    (tmp_path / "compile_commands.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        "pcstubgen.api.ModuleCollector.run",
        lambda self, module_name: Module(
            full_name=QualifiedName.from_str("pkg.mod"),
            functions=[
                Function(
                    name="foo",
                    signatures=[
                        _signature(
                            args=[Argument(name="value", type=RawType.int_)],
                            return_type=RawType.bool_,
                        )
                    ],
                )
            ],
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "gen",
            "pkg.mod",
            "--compilation-database",
            str(tmp_path / "compile_commands.json"),
            "--output",
            str(tmp_path),
            "--toml",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "mod.toml").exists()
    assert not (tmp_path / "mod.pyi").exists()
    entries = tomllib.loads((tmp_path / "mod.toml").read_text(encoding="utf-8"))["entries"]
    assert [entry["function_name"] for entry in entries] == ["foo"]


def test_gen_command_keeps_stub_output_when_toml_flag_is_disabled(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.provider.FunctionCursorLocator",
        lambda compilation_database: object(),
    )
    (tmp_path / "compile_commands.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        "pcstubgen.api.ModuleCollector.run",
        lambda self, module_name: Module(
            full_name=QualifiedName.from_str("pkg.mod"),
            functions=[
                Function(
                    name="foo",
                    signatures=[
                        _signature(
                            args=[Argument(name="value", type=RawType.str_)],
                            return_type=RawType.bool_,
                        )
                    ],
                )
            ],
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "gen",
            "pkg.mod",
            "--compilation-database",
            str(tmp_path / "compile_commands.json"),
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "mod.pyi").exists()
    assert not (tmp_path / "mod.toml").exists()
    assert "def foo(" in (tmp_path / "mod.pyi").read_text(encoding="utf-8")


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
