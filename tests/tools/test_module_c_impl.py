from __future__ import annotations

import csv
import importlib.machinery
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import module_c_impl


def test_run_single_module_returns_error_when_collect_module_names_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        module_c_impl,
        "collect_module_names",
        lambda module_name: (_ for _ in ()).throw(RuntimeError(f"boom: {module_name}")),
    )

    exit_code = module_c_impl.run_single_module("bad.module", output=tmp_path)
    captured = capsys.readouterr()

    assert exit_code == module_c_impl.EXIT_ERROR
    assert "boom: bad.module" in captured.out


def test_write_report_writes_csv_header_and_places_c_modules_first(
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


@pytest.mark.parametrize(
    ("module_name", "loader"),
    [
        ("sys", importlib.machinery.BuiltinImporter),
        (
            "demo.module",
            importlib.machinery.ExtensionFileLoader("demo.module", "C:/demo/module.pyd"),
        ),
    ],
)
def test_is_c_implemented_detects_builtin_and_extension_loaders(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    loader: object,
) -> None:
    monkeypatch.setattr(
        module_c_impl.importlib.util,
        "find_spec",
        lambda requested_module_name: SimpleNamespace(loader=loader),
    )

    assert module_c_impl.is_c_implemented(module_name) is True
