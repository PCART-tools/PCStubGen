from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest
from typer.testing import CliRunner

from pcstubgen.__main__ import app
import pcstubgen._wrap_command as wrap_command_module


@pytest.mark.parametrize(
    ("args", "expected_command", "expected_output_path"),
    [
        (
            ["wrap", "--", "python", "-m", "build"],
            ["python", "-m", "build"],
            Path("compile_commands.json"),
        ),
        (
            ["wrap", "--output", "out/compile_commands.json", "--", "uv", "build", "--wheel"],
            ["uv", "build", "--wheel"],
            Path("out/compile_commands.json"),
        ),
    ],
)
def test_wrap_command_passes_cli_arguments_to_runner(
    monkeypatch,
    args: list[str],
    expected_command: list[str],
    expected_output_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_wrap_command(command: list[str], output_path: Path) -> int:
        captured["command"] = command
        captured["output_path"] = output_path
        return 0

    monkeypatch.setattr(
        wrap_command_module, "ensure_wrap_programs_available", lambda: None
    )
    monkeypatch.setattr(wrap_command_module, "run_wrap_command", fake_run_wrap_command)

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0
    assert captured == {
        "command": expected_command,
        "output_path": expected_output_path,
    }


def test_run_wrap_command_invokes_bear_with_clang_debug_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_command: list[str] | None = None
    captured_env: dict[str, str] | None = None
    captured_check: bool | None = None
    output_path = tmp_path / "compile_commands.json"
    monkeypatch.setenv("LIBRARY_PATH", "/existing/lib")

    def fake_check_output(command: list[str], text: bool) -> str:
        assert command == ["llvm-config", "--libdir"]
        assert text is True
        return "/llvm/lib\n"

    def fake_run(
        command: list[str], env: dict[str, str], check: bool
    ) -> subprocess.CompletedProcess[str]:
        nonlocal captured_command, captured_env, captured_check
        captured_command = command
        captured_env = env
        captured_check = check
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        wrap_command_module.subprocess, "check_output", fake_check_output
    )
    monkeypatch.setattr(wrap_command_module.subprocess, "run", fake_run)

    return_code = wrap_command_module.run_wrap_command(
        ["python", "-m", "build"], output_path
    )

    assert return_code == 0
    assert captured_command == [
        "bear",
        "--output",
        str(output_path),
        "--",
        "python",
        "-m",
        "build",
    ]
    assert captured_check is False
    assert captured_env is not None
    assert captured_env["CC"] == "clang"
    assert captured_env["CXX"] == "clang++"
    assert captured_env["DEBUG"] == "1"
    assert captured_env["CMAKE_BUILD_TYPE"] == "Debug"
    assert captured_env["CFLAGS"] == "-O0 -g -UNDEBUG"
    assert captured_env["CXXFLAGS"] == "-O0 -g -UNDEBUG"
    assert captured_env["LIBRARY_PATH"] == os.pathsep.join(
        ["/llvm/lib", "/existing/lib"]
    )


def test_wrap_command_propagates_wrapped_command_exit_code(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        wrap_command_module, "ensure_wrap_programs_available", lambda: None
    )
    monkeypatch.setattr(
        wrap_command_module, "run_wrap_command", lambda command, output_path: 7
    )

    result = CliRunner().invoke(app, ["wrap", "--", "python", "-m", "build"])

    assert result.exit_code == 7
    assert "构建完成" not in result.output


def test_wrap_command_reports_missing_external_programs(monkeypatch) -> None:
    program_paths = {
        "clang": "/usr/bin/clang",
        "clang++": "/usr/bin/clang++",
        "llvm-config": "/usr/bin/llvm-config",
        "bear": None,
    }
    monkeypatch.setattr(
        wrap_command_module.shutil, "which", lambda program: program_paths[program]
    )

    result = CliRunner().invoke(app, ["wrap", "--", "python", "-m", "build"])

    assert result.exit_code == 1
    assert "bear" in result.output
