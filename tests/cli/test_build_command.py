from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from pcstubgen.__main__ import app
from pcstubgen._build_command import BuildContext
import pcstubgen._build_command as build_command_module


def test_build_command_rejects_removed_clean_build_env_option(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["build", str(tmp_path), "--clean-build-env"])

    assert result.exit_code == 2
    assert "No such option: --clean-build-env" in result.output


def test_build_command_reports_new_build_env_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    wheel_path = tmp_path / "dist" / "demo.whl"
    compile_commands_path = tmp_path / "build" / "compile_commands.json"

    monkeypatch.setattr(
        build_command_module, "ensure_build_programs_available", lambda: None
    )
    monkeypatch.setattr(
        build_command_module,
        "get_build_context",
        lambda srcdir: BuildContext(
            build_backend="mesonpy",
            runner=build_command_module.default_runner,
            config_settings={},
        ),
    )
    monkeypatch.setattr(build_command_module, "build_wheel", lambda *args: wheel_path)
    monkeypatch.setattr(
        build_command_module,
        "find_compile_commands_path",
        lambda srcdir: compile_commands_path,
    )

    result = CliRunner().invoke(app, ["build", str(tmp_path)])

    assert result.exit_code == 0
    assert f"- 构建环境: {tmp_path / '.pcstubgen-build-env'}" in result.output
    assert "持久构建环境" not in result.output


def test_build_command_fails_when_build_env_path_is_not_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    build_env_path = tmp_path / ".pcstubgen-build-env"
    build_env_path.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(
        build_command_module, "ensure_build_programs_available", lambda: None
    )
    monkeypatch.setattr(
        build_command_module,
        "get_build_context",
        lambda srcdir: BuildContext(
            build_backend="mesonpy",
            runner=build_command_module.default_runner,
            config_settings={},
        ),
    )

    result = CliRunner().invoke(app, ["build", str(tmp_path)])

    assert result.exit_code == 1
    assert f"构建环境路径存在但不是可清理目录: {build_env_path}" in result.output
