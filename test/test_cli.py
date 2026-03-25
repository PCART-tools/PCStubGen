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


def test_build_options_keeps_none_source_root() -> None:
    options = cli._build_options(
        enable_docstring_signature_parser=True,
        source_root=None,
        clang_include=[],
        clang_include_directory=[],
        clang_c_std=None,
        clang_cpp_std=None,
        include_docstrings=True,
        include_module_type_comment=False,
    )

    assert options.source_root is None


def test_build_options_keeps_path_source_root() -> None:
    options = cli._build_options(
        enable_docstring_signature_parser=True,
        source_root=Path("C:/tmp/src"),
        clang_include=[],
        clang_include_directory=[],
        clang_c_std=None,
        clang_cpp_std=None,
        include_docstrings=True,
        include_module_type_comment=False,
    )

    assert options.source_root == Path("C:/tmp/src")


def test_cli_passes_repeated_clang_include_directory(
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
            "--clang-include-directory",
            "C:/IncludeA",
            "--clang-include-directory=C:/IncludeB",
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert captured_module_name == "math"
    assert captured_output_dir == tmp_path
    assert captured_options is not None
    assert captured_options.clang_include_directory == [
        str(Path("C:/IncludeA")),
        str(Path("C:/IncludeB")),
    ]


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


def test_cli_passes_repeated_clang_include(
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
            "--clang-include",
            "Python.h",
            "--clang-include=numpy/arrayobject.h",
        ],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert captured_module_name == "math"
    assert captured_output_dir == tmp_path
    assert captured_options is not None
    assert captured_options.clang_include == ["Python.h", "numpy/arrayobject.h"]


def test_help_contains_chinese_project_text() -> None:
    result = RUNNER.invoke(cli.app, ["--help"], prog_name="pcstubgen")

    assert result.exit_code == 0
    assert "使用 pcstubgen 为模块生成 Python stub。" in result.stdout
    assert "--output-dir" in result.stdout
    assert "输出 stub 的根目录" in result.stdout


def test_invalid_clang_include_reports_chinese_validation_error() -> None:
    result = RUNNER.invoke(
        cli.app,
        ["math", "--clang-include=-bad"],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 2
    assert "Invalid value for '--clang-include'" in result.stderr
    assert "'-bad'" in result.stderr


def test_validate_clang_include_preserves_chinese_error_message() -> None:
    with pytest.raises(cli.typer.BadParameter) as ex:
        cli._validate_clang_include(["-bad"])

    assert str(ex.value) == "clang_include 条目必须是 header，不能是类似选项的值: '-bad'"


def test_removed_stub_extension_flag_is_rejected() -> None:
    result = RUNNER.invoke(
        cli.app,
        ["math", "--stub-extension", "txt"],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 2
    assert "No such option: --stub-extension" in result.stderr
