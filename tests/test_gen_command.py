from __future__ import annotations

import tomllib

from typer.testing import CliRunner

from pcstubgen.__main__ import app
from pcstubgen.ir_modules import IRArgument, IRFunction, IRModule, QualifiedName
from pcstubgen.types import RawType
from tests._c_extension_test_support import _signature


def test_gen_command_writes_toml_instead_of_stub_when_toml_flag_is_enabled(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.ClangParser",
        lambda compilation_database: object(),
    )
    monkeypatch.setattr(
        "pcstubgen.api.ModuleCollector.run",
        lambda self, module_name: IRModule(
            full_name=QualifiedName.from_str("pkg.mod"),
            functions=[
                IRFunction(
                    name="foo",
                    runtime_handle=object(),
                    signatures=[
                        _signature(
                            args=[IRArgument(name="value", type=RawType("int"))],
                            return_type=RawType("bool"),
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
    assert tomllib.loads((tmp_path / "mod.toml").read_text(encoding="utf-8")) == {
        "entries": [
            {
                "module_name": "pkg.mod",
                "function_name": "foo",
                "signature": "def foo(value: int) -> bool:",
            }
        ]
    }


def test_gen_command_keeps_stub_output_when_toml_flag_is_disabled(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.ClangParser",
        lambda compilation_database: object(),
    )
    monkeypatch.setattr(
        "pcstubgen.api.ModuleCollector.run",
        lambda self, module_name: IRModule(
            full_name=QualifiedName.from_str("pkg.mod"),
            functions=[
                IRFunction(
                    name="foo",
                    runtime_handle=object(),
                    signatures=[
                        _signature(
                            args=[IRArgument(name="value", type=RawType("str"))],
                            return_type=RawType("bool"),
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
    assert (tmp_path / "mod.pyi").read_text(encoding="utf-8") == "def foo(value: str) -> bool:\n    ...\n"


def test_gen_command_rejects_removed_json_flag(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "gen",
            "pkg.mod",
            "--compilation-database",
            str(tmp_path / "compile_commands.json"),
            "--output",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code != 0
    assert "--json" in result.output
