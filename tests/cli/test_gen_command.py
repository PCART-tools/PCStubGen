from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import pcstubgen.__main__ as main_module
from pcstubgen.stub_generation_options import StubGenerationOptions


RUNNER = CliRunner()


def _read_single_log_file(log_dir: Path, leaf_module_name: str) -> str:
    log_files = list(log_dir.glob(f"pcstubgen_{leaf_module_name}_*.log"))
    assert len(log_files) == 1
    return log_files[0].read_text(encoding="utf-8")


def test_gen_command_passes_stub_generation_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_module_name: str | None = None
    captured_output: Path | None = None
    captured_options: StubGenerationOptions | None = None

    def fake_write_stubs(*, module_name: str, output: Path, options: StubGenerationOptions) -> None:
        nonlocal captured_module_name, captured_output, captured_options
        captured_module_name = module_name
        captured_output = output
        captured_options = options

    monkeypatch.setattr(main_module, "write_stubs", fake_write_stubs)

    result = RUNNER.invoke(
        main_module.app,
        [
            "gen",
            "math",
            "--output",
            str(tmp_path),
            "--compilation-database",
            str(tmp_path / "compile_commands.json"),
            "--include-docstrings",
            "--include-c-inferred-source-comment",
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert captured_module_name == "math"
    assert captured_output == tmp_path
    assert captured_options is not None
    assert captured_options.compilation_database == tmp_path / "compile_commands.json"
    assert captured_options.include_docstrings is True
    assert captured_options.include_c_inferred_source_comment is True


def test_gen_command_writes_leaf_log_file_and_records_key_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_write_stubs(*, module_name: str, output: Path, options: StubGenerationOptions) -> None:
        _ = (module_name, output, options)

    monkeypatch.setattr(main_module, "write_stubs", fake_write_stubs)
    compilation_database = tmp_path / "compile_commands.json"

    result = RUNNER.invoke(
        main_module.app,
        [
            "gen",
            "numpy.linalg",
            "--output",
            str(tmp_path),
            "--compilation-database",
            str(compilation_database),
            "--include-docstrings",
        ],
        prog_name="pcstubgen",
    )

    log_files = list(tmp_path.glob("pcstubgen_linalg_*.log"))

    assert result.exit_code == 0
    assert len(log_files) == 1
    assert re.fullmatch(r"pcstubgen_linalg_\d{8}_\d{6}\.log", log_files[0].name) is not None

    log_text = _read_single_log_file(tmp_path, "linalg")
    assert "CLI参数:" in log_text
    assert "module_name=numpy.linalg" in log_text
    assert f"compilation_database={compilation_database}" in log_text
    assert "include_docstrings=True" in log_text


def test_gen_command_rejects_removed_include_module_type_comment_option(
    tmp_path: Path,
) -> None:
    result = RUNNER.invoke(
        main_module.app,
        [
            "gen",
            "math",
            "--output",
            str(tmp_path),
            "--include-module-type-comment",
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code != 0
    assert "No such option" in result.output
    assert "--include-module-type-comment" in result.output
