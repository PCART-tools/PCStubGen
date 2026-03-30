from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from pcstubgen import cli
from pcstubgen.stub_generation_options import StubGenerationOptions

RUNNER = CliRunner()


def test_cli_passes_stub_generation_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_module_name: str | None = None
    captured_output_dir: Path | None = None
    captured_options: StubGenerationOptions | None = None

    def fake_write_stubs(*, module_name: str, output_dir: Path, options: StubGenerationOptions) -> None:
        nonlocal captured_module_name, captured_output_dir, captured_options
        captured_module_name = module_name
        captured_output_dir = output_dir
        captured_options = options

    monkeypatch.setattr(cli, "write_stubs", fake_write_stubs)

    result = RUNNER.invoke(
        cli.app,
        [
            "math",
            "--output-dir",
            str(tmp_path),
            "--include",
            "Python.h",
            "--include=numpy/arrayobject.h",
            "--include-directory",
            "C:/IncludeA",
            "--include-directory=C:/IncludeB",
            "--source-root",
            str(tmp_path / "src"),
            "--c-std",
            "c99",
            "--cpp-std",
            "c++20",
            "--include-c-inferred-source-comment",
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert captured_module_name == "math"
    assert captured_output_dir == tmp_path
    assert captured_options is not None
    assert captured_options.include == ["Python.h", "numpy/arrayobject.h"]
    assert captured_options.include_directory == [
        Path("C:/IncludeA"),
        Path("C:/IncludeB"),
    ]
    assert captured_options.source_root == tmp_path / "src"
    assert captured_options.c_std == "c99"
    assert captured_options.cpp_std == "c++20"
    assert captured_options.include_c_inferred_source_comment is True


def test_invalid_include_reports_chinese_validation_error() -> None:
    result = RUNNER.invoke(
        cli.app,
        ["math", "--include=-bad"],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 2
    assert "Invalid value for '--include'" in result.stderr
    assert "'-bad'" in result.stderr


def test_validate_include_preserves_chinese_error_message() -> None:
    with pytest.raises(cli.typer.BadParameter) as ex:
        cli._validate_include(["-bad"])

    message = str(ex.value)
    assert "include" in message
    assert "-bad" in message
