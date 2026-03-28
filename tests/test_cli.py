from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core import cli
from core.stub_generation_options import StubGenerationOptions

RUNNER = CliRunner()


def test_removed_enable_c_signature_inference_flag_is_rejected() -> None:
    result = RUNNER.invoke(
        cli.app,
        ["math", "--enable-c-signature-inference"],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 2


def test_removed_ignore_invalid_expressions_flag_is_rejected() -> None:
    result = RUNNER.invoke(
        cli.app,
        ["math", "--ignore-invalid-expressions", ".*"],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 2


def test_removed_print_invalid_expressions_flag_is_rejected() -> None:
    result = RUNNER.invoke(
        cli.app,
        ["math", "--print-invalid-expressions-as-is"],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 2


def test_removed_ignore_all_errors_flag_is_rejected() -> None:
    result = RUNNER.invoke(
        cli.app,
        ["math", "--ignore-all-errors"],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 2


def test_removed_enum_class_locations_flag_is_rejected() -> None:
    result = RUNNER.invoke(
        cli.app,
        ["math", "--enum-class-locations", "MyEnum:pkg.enums"],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 2
def test_cli_passes_repeated_include_directory(
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
            "--include-directory",
            "C:/IncludeA",
            "--include-directory=C:/IncludeB",
            "--c-std",
            "c99",
            "--cpp-std",
            "c++20",
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert captured_module_name == "math"
    assert captured_output_dir == tmp_path
    assert captured_options is not None
    assert captured_options.include_directory == [
        Path("C:/IncludeA"),
        Path("C:/IncludeB"),
    ]
    assert captured_options.c_std == "c99"
    assert captured_options.cpp_std == "c++20"


def test_cli_passes_source_root_as_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_options: StubGenerationOptions | None = None

    def fake_write_stubs(*, module_name: str, output_dir: Path, options: StubGenerationOptions) -> None:
        nonlocal captured_options
        _ = (module_name, output_dir)
        captured_options = options

    monkeypatch.setattr(cli, "write_stubs", fake_write_stubs)

    result = RUNNER.invoke(
        cli.app,
        [
            "math",
            "--output-dir",
            str(tmp_path),
            "--source-root",
            str(tmp_path / "src"),
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert captured_options is not None
    assert captured_options.source_root == tmp_path / "src"
    assert captured_options.include_c_inferred_source_comment is False


def test_cli_passes_include_c_inferred_source_comment_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_options: StubGenerationOptions | None = None

    def fake_write_stubs(*, module_name: str, output_dir: Path, options: StubGenerationOptions) -> None:
        nonlocal captured_options
        _ = (module_name, output_dir)
        captured_options = options

    monkeypatch.setattr(cli, "write_stubs", fake_write_stubs)

    result = RUNNER.invoke(
        cli.app,
        [
            "math",
            "--output-dir",
            str(tmp_path),
            "--include-c-inferred-source-comment",
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert captured_options is not None
    assert captured_options.include_c_inferred_source_comment is True


def test_cli_passes_repeated_include(
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
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert captured_module_name == "math"
    assert captured_output_dir == tmp_path
    assert captured_options is not None
    assert captured_options.include == ["Python.h", "numpy/arrayobject.h"]
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

    assert str(ex.value) == "include 条目必须是 header，不能是类似选项的值: '-bad'"


def test_old_clang_prefixed_options_are_rejected() -> None:
    result = RUNNER.invoke(
        cli.app,
        [
            "math",
            "--clang-include",
            "Python.h",
            "--clang-include-directory",
            "C:/IncludeA",
            "--clang-c-std",
            "c11",
            "--clang-cpp-std",
            "c++17",
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 2
    assert "No such option" in result.stderr


def test_removed_stub_extension_flag_is_rejected() -> None:
    result = RUNNER.invoke(
        cli.app,
        ["math", "--stub-extension", "txt"],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 2
    assert "No such option: --stub-extension" in result.stderr
