from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import pcstubgen.__main__ as main_module
from pcstubgen.stub_generation_options import StubGenerationOptions

RUNNER = CliRunner()


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
            "--include",
            "Python.h",
            "--include",
            "  spaced/header.h  ",
            "--include=numpy/arrayobject.h",
            "--include",
            "Python.h",
            "--include-directory",
            "C:/IncludeA",
            "--include-directory=C:/IncludeB",
            "--source",
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
    assert captured_output == tmp_path
    assert captured_options is not None
    assert captured_options.include == [
        "Python.h",
        "  spaced/header.h  ",
        "numpy/arrayobject.h",
        "Python.h",
    ]
    assert captured_options.include_directory == [
        Path("C:/IncludeA"),
        Path("C:/IncludeB"),
    ]
    assert captured_options.source == tmp_path / "src"
    assert captured_options.c_std == "c99"
    assert captured_options.cpp_std == "c++20"
    assert captured_options.include_c_inferred_source_comment is True


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


def test_cli_preserves_include_without_validation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_options: StubGenerationOptions | None = None

    def fake_write_stubs(*, module_name: str, output: Path, options: StubGenerationOptions) -> None:
        nonlocal captured_options
        _ = output
        captured_options = options

    monkeypatch.setattr(main_module, "write_stubs", fake_write_stubs)

    result = RUNNER.invoke(
        main_module.app,
        [
            "math",
            "--output",
            str(tmp_path),
            "--include=-bad",
            "--include",
            "   ",
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert captured_options is not None
    assert captured_options.include == ["-bad", "   "]
