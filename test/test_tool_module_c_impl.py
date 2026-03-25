from __future__ import annotations

import csv
import importlib
import importlib.machinery
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tool import module_c_impl

RUNNER = CliRunner()


def test_cli_help_contains_chinese_text() -> None:
    result = RUNNER.invoke(module_c_impl.app, ["--help"], prog_name="module_c_impl")

    assert result.exit_code == 0
    assert "检查模块及其子模块是否为 C 实现，并导出 CSV 报告。" in result.stdout
    assert "--output-dir" in result.stdout
    assert "待检查的模块名" in result.stdout


def test_cli_requires_module_name() -> None:
    result = RUNNER.invoke(module_c_impl.app, [], prog_name="module_c_impl")

    assert result.exit_code == 2
    assert "Missing argument 'MODULE_NAME'" in result.stderr


def test_cli_passes_explicit_module_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_module_name: str | None = None
    captured_output_dir: Path | None = None

    def fake_run_single_module(module_name: str, *, output_dir: Path) -> int:
        nonlocal captured_module_name, captured_output_dir
        captured_module_name = module_name
        captured_output_dir = output_dir
        return 0

    monkeypatch.setattr(module_c_impl, "run_single_module", fake_run_single_module)
    monkeypatch.setattr(module_c_impl, "DEFAULT_OUTPUT_DIR", Path("C:/tmp/module_c_impl_output"))

    result = RUNNER.invoke(module_c_impl.app, ["math"], prog_name="module_c_impl")

    assert result.exit_code == 0
    assert captured_module_name == "math"
    assert captured_output_dir == Path("C:/tmp/module_c_impl_output")


def test_cli_passes_explicit_module_name_and_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_module_name: str | None = None
    captured_output_dir: Path | None = None

    def fake_run_single_module(module_name: str, *, output_dir: Path) -> int:
        nonlocal captured_module_name, captured_output_dir
        captured_module_name = module_name
        captured_output_dir = output_dir
        return 0

    monkeypatch.setattr(module_c_impl, "run_single_module", fake_run_single_module)

    result = RUNNER.invoke(
        module_c_impl.app,
        [
            "math",
            "--output-dir",
            str(tmp_path),
        ],
        prog_name="module_c_impl",
    )

    assert result.exit_code == 0
    assert captured_module_name == "math"
    assert captured_output_dir == tmp_path


def test_run_single_module_returns_error_when_collect_module_names_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def raise_collect_error(module_name: str) -> list[str]:
        raise RuntimeError(f"boom: {module_name}")

    monkeypatch.setattr(module_c_impl, "collect_module_names", raise_collect_error)

    exit_code = module_c_impl.run_single_module("bad.module", output_dir=tmp_path)
    captured = capsys.readouterr()

    assert exit_code == module_c_impl.EXIT_ERROR
    assert "检查失败: boom: bad.module" in captured.out


def test_write_report_writes_csv_header_and_c_modules_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    status_map = {
        "pkg.alpha": False,
        "pkg.beta": True,
        "pkg.gamma": True,
    }

    monkeypatch.setattr(module_c_impl, "is_c_implemented", lambda module_name: status_map[module_name])
    report_path = tmp_path / "pkg.csv"

    c_count = module_c_impl.write_report(
        ["pkg.alpha", "pkg.beta", "pkg.gamma"],
        report_path,
    )

    with report_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert c_count == 2
    assert rows[0] == ["module_name", "is_c_implemented"]
    assert rows[1:] == [
        ["pkg.beta", "True"],
        ["pkg.gamma", "True"],
        ["pkg.alpha", "False"],
    ]


def test_is_c_implemented_returns_true_for_builtin_loader() -> None:
    assert module_c_impl.is_c_implemented("sys") is True


def test_is_c_implemented_returns_true_for_extension_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = importlib.machinery.ExtensionFileLoader("demo.module", "C:/demo/module.pyd")
    monkeypatch.setattr(
        module_c_impl.importlib.util,
        "find_spec",
        lambda module_name: SimpleNamespace(loader=loader),
    )

    assert module_c_impl.is_c_implemented("demo.module") is True
