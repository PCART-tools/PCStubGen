from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

import pcstubgen.__main__ as main_module
from pcstubgen.stub_generation_options import StubGenerationOptions

RUNNER = CliRunner()


def _read_single_log_file(log_dir: Path, leaf_module_name: str) -> str:
    log_files = list(log_dir.glob(f"pcstubgen_{leaf_module_name}_*.log"))
    assert len(log_files) == 1
    return log_files[0].read_text(encoding="utf-8")


def test_stub_generation_options_excludes_docstrings_by_default() -> None:
    assert StubGenerationOptions().include_docstrings is False


def test_cli_passes_stub_generation_options(
    monkeypatch,
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
            "math",
            "--output",
            str(tmp_path),
            "--compilation-database",
            str(tmp_path / "compile_commands.json"),
            "--include-c-inferred-source-comment",
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert captured_module_name == "math"
    assert captured_output == tmp_path
    assert captured_options is not None
    assert captured_options.compilation_database == tmp_path / "compile_commands.json"
    assert captured_options.include_docstrings is False
    assert captured_options.include_c_inferred_source_comment is True


def test_cli_enables_docstrings_with_explicit_flag(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_options: StubGenerationOptions | None = None

    def fake_write_stubs(*, module_name: str, output: Path, options: StubGenerationOptions) -> None:
        nonlocal captured_options
        _ = module_name
        _ = output
        captured_options = options

    monkeypatch.setattr(main_module, "write_stubs", fake_write_stubs)

    result = RUNNER.invoke(
        main_module.app,
        [
            "math",
            "--output",
            str(tmp_path),
            "--include-docstrings",
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert captured_options is not None
    assert captured_options.include_docstrings is True


def test_cli_creates_timestamped_log_file_for_leaf_module_name(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_write_stubs(*, module_name: str, output: Path, options: StubGenerationOptions) -> None:
        _ = module_name
        _ = output
        _ = options

    monkeypatch.setattr(main_module, "write_stubs", fake_write_stubs)

    result = RUNNER.invoke(
        main_module.app,
        [
            "math",
            "--output",
            str(tmp_path),
        ],
        prog_name="pcstubgen",
    )

    log_files = list(tmp_path.glob("pcstubgen_math_*.log"))

    assert result.exit_code == 0
    assert len(log_files) == 1
    assert re.fullmatch(r"pcstubgen_math_\d{8}_\d{6}\.log", log_files[0].name) is not None
    assert not (tmp_path / "pcstubgen.log").exists()


def test_cli_uses_leaf_module_name_in_log_file_name(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_write_stubs(*, module_name: str, output: Path, options: StubGenerationOptions) -> None:
        _ = module_name
        _ = output
        _ = options

    monkeypatch.setattr(main_module, "write_stubs", fake_write_stubs)

    result = RUNNER.invoke(
        main_module.app,
        [
            "numpy.linalg",
            "--output",
            str(tmp_path),
        ],
        prog_name="pcstubgen",
    )

    log_files = list(tmp_path.glob("pcstubgen_linalg_*.log"))

    assert result.exit_code == 0
    assert len(log_files) == 1
    assert re.fullmatch(r"pcstubgen_linalg_\d{8}_\d{6}\.log", log_files[0].name) is not None
    assert list(tmp_path.glob("pcstubgen_numpy.linalg_*.log")) == []


def test_cli_logs_parsed_arguments_to_log_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_write_stubs(*, module_name: str, output: Path, options: StubGenerationOptions) -> None:
        _ = module_name
        _ = output
        _ = options

    monkeypatch.setattr(main_module, "write_stubs", fake_write_stubs)

    compilation_database = tmp_path / "compile_commands.json"
    result = RUNNER.invoke(
        main_module.app,
        [
            "numpy.linalg",
            "--output",
            str(tmp_path),
            "--compilation-database",
            str(compilation_database),
            "--include-docstrings",
            "--include-module-type-comment",
            "--include-c-inferred-source-comment",
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0

    log_text = _read_single_log_file(tmp_path, "linalg")

    assert "CLI参数:" in log_text
    assert "module_name=numpy.linalg" in log_text
    assert f"output={tmp_path}" in log_text
    assert f"compilation_database={compilation_database}" in log_text
    assert "include_docstrings=True" in log_text
    assert "include_module_type_comment=True" in log_text
    assert "include_c_inferred_source_comment=True" in log_text


def test_cli_logs_default_argument_values_to_log_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_write_stubs(*, module_name: str, output: Path, options: StubGenerationOptions) -> None:
        _ = module_name
        _ = output
        _ = options

    monkeypatch.setattr(main_module, "write_stubs", fake_write_stubs)

    result = RUNNER.invoke(
        main_module.app,
        [
            "math",
            "--output",
            str(tmp_path),
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0

    log_text = _read_single_log_file(tmp_path, "math")

    assert "CLI参数:" in log_text
    assert "module_name=math" in log_text
    assert f"output={tmp_path}" in log_text
    assert "compilation_database=None" in log_text
    assert "include_docstrings=False" in log_text
    assert "include_module_type_comment=False" in log_text
    assert "include_c_inferred_source_comment=False" in log_text


def test_cli_rejects_legacy_source_root_option(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        main_module.app,
        [
            "math",
            "--output",
            str(tmp_path),
            "--source-root",
            str(tmp_path / "src"),
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code != 0
    assert "No such option: --source-root" in result.output


def test_cli_rejects_removed_no_docstrings_option(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        main_module.app,
        [
            "math",
            "--output",
            str(tmp_path),
            "--no-docstrings",
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code != 0
    assert "No such option: --no-docstrings" in result.output


def test_cli_rejects_removed_c_parse_options(tmp_path: Path) -> None:
    for option, value in [
        ("--source", str(tmp_path / "src")),
        ("--include", "Python.h"),
        ("--include-directory", str(tmp_path / "include")),
        ("--c-std", "c11"),
        ("--cpp-std", "c++17"),
    ]:
        result = RUNNER.invoke(
            main_module.app,
            [
                "math",
                "--output",
                str(tmp_path),
                option,
                value,
            ],
            prog_name="pcstubgen",
        )

        assert result.exit_code != 0
        assert f"No such option: {option}" in result.output
