from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import pytest
from typer.testing import CliRunner

from build import env as build_env

import pcstubgen.__main__ as main_module
import pcstubgen._build as build_module
from pcstubgen._persistent_build_env import PersistentIsolatedEnv


RUNNER = CliRunner()


def test_build_clang_environ_includes_debug_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_library_path = build_module.os.environ.get("LIBRARY_PATH")
    build_module.os.environ.pop("LIBRARY_PATH", None)

    def fake_check_output(cmd: list[str], text: bool = False) -> str:
        assert cmd == ["llvm-config", "--libdir"]
        assert text is True
        return "/opt/llvm/lib\n"

    monkeypatch.setattr(build_module.subprocess, "check_output", fake_check_output)
    env = build_module.build_clang_environ()

    try:
        assert env["CC"] == "clang"
        assert env["CXX"] == "clang++"
        assert env["LIBRARY_PATH"] == "/opt/llvm/lib"
        assert env["DEBUG"] == "1"
        assert env["CMAKE_BUILD_TYPE"] == "Debug"
        assert env["CFLAGS"] == "-O0 -g -UNDEBUG"
        assert env["CXXFLAGS"] == "-O0 -g -UNDEBUG"
    finally:
        if original_library_path is None:
            build_module.os.environ.pop("LIBRARY_PATH", None)
        else:
            build_module.os.environ["LIBRARY_PATH"] = original_library_path


def test_clang_runner_passes_debug_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_default_subprocess_runner(
        cmd: object,
        cwd: object,
        extra_environ: dict[str, str] | None = None,
    ) -> None:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["extra_environ"] = extra_environ

    monkeypatch.setattr(
        build_module.pyproject_hooks,
        "default_subprocess_runner",
        fake_default_subprocess_runner,
    )
    monkeypatch.setattr(
        build_module.subprocess,
        "check_output",
        lambda cmd, text=False: "/opt/llvm/lib\n",
    )

    build_module.clang_runner(["python", "-m", "build"], cwd="/tmp/demo")

    assert captured["cmd"] == ["python", "-m", "build"]
    assert captured["cwd"] == "/tmp/demo"
    assert captured["extra_environ"]["DEBUG"] == "1"
    assert captured["extra_environ"]["CMAKE_BUILD_TYPE"] == "Debug"
    assert captured["extra_environ"]["LIBRARY_PATH"].startswith("/opt/llvm/lib")


def test_bear_runner_passes_debug_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_check_call(
        cmd: object,
        cwd: object = None,
        env: dict[str, str] | None = None,
    ) -> None:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env

    monkeypatch.setattr(build_module.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(
        build_module.subprocess,
        "check_output",
        lambda cmd, text=False: "/opt/llvm/lib\n",
    )

    build_module.bear_runner(["python", "-m", "build"], cwd="/tmp/demo")

    assert captured["cmd"] == ["bear", "--", "python", "-m", "build"]
    assert captured["cwd"] == "/tmp/demo"
    assert captured["env"]["DEBUG"] == "1"
    assert captured["env"]["CMAKE_BUILD_TYPE"] == "Debug"
    assert captured["env"]["LIBRARY_PATH"].startswith("/opt/llvm/lib")


def _fake_build_context(srcdir: Path) -> build_module.BuildContext:
    return build_module.BuildContext(
        builder=None,
        build_backend="test-backend",
        runner=build_module.clang_runner,
        config_settings={},
    )


def _fake_build_wheel(
    observed_verbosity: list[int],
):
    def _impl(
        srcdir: Path,
        runner: object,
        config_settings: object,
    ) -> Path:
        _ = (runner, config_settings)
        observed_verbosity.append(build_env._ctx.verbosity)
        build_env._ctx.log("build verbose message")
        return srcdir / "dist" / "demo.whl"

    return _impl


def test_persistent_isolated_env_build_path_uses_class_dirname(tmp_path: Path) -> None:
    assert PersistentIsolatedEnv.get_build_env_path(tmp_path) == (
        tmp_path / PersistentIsolatedEnv.BUILD_ENV_DIRNAME
    )


@pytest.mark.parametrize(
    ("cli_args", "expected_verbosity", "should_log"),
    [
        ([], 0, False),
        (["-v"], 1, True),
        (["-vv"], 2, True),
    ],
)
def test_build_command_sets_build_verbosity_and_restores_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cli_args: list[str],
    expected_verbosity: int,
    should_log: bool,
) -> None:
    observed_verbosity: list[int] = []
    original_logger = build_env._ctx.LOGGER.get()
    original_verbosity = build_env._ctx.verbosity

    def fake_ensure_clang_compilers_available() -> None:
        return None

    monkeypatch.setattr(build_module, "resolve_build_context", _fake_build_context)
    monkeypatch.setattr(
        build_module,
        "ensure_clang_compilers_available",
        fake_ensure_clang_compilers_available,
    )
    monkeypatch.setattr(build_module, "build_wheel", _fake_build_wheel(observed_verbosity))

    result = RUNNER.invoke(
        main_module.app,
        ["build", *cli_args, str(tmp_path)],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert observed_verbosity == [expected_verbosity]
    assert build_env._ctx.LOGGER.get() is original_logger
    assert build_env._ctx.verbosity == original_verbosity
    assert "构建完成" in result.output
    assert "- build-backend: test-backend" in result.output
    assert f"- 持久构建环境: {PersistentIsolatedEnv.get_build_env_path(tmp_path)}" in result.output
    assert "- wheel 文件:" in result.output
    assert "- compile_commands.json: None" in result.output
    if should_log:
        assert "build verbose message" in result.output
    else:
        assert "build verbose message" not in result.output


@pytest.mark.parametrize(
    ("build_exists", "root_exists", "expected_path"),
    [
        (True, False, "build/compile_commands.json"),
        (False, True, "compile_commands.json"),
        (True, True, "build/compile_commands.json"),
    ],
)
def test_build_command_detects_compile_commands_path_after_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    build_exists: bool,
    root_exists: bool,
    expected_path: str,
) -> None:
    observed_verbosity: list[int] = []

    def fake_ensure_clang_compilers_available() -> None:
        return None

    monkeypatch.setattr(build_module, "resolve_build_context", _fake_build_context)
    monkeypatch.setattr(
        build_module,
        "ensure_clang_compilers_available",
        fake_ensure_clang_compilers_available,
    )
    monkeypatch.setattr(build_module, "build_wheel", _fake_build_wheel(observed_verbosity))

    if build_exists:
        build_compile_commands = tmp_path / "build" / "compile_commands.json"
        build_compile_commands.parent.mkdir(parents=True, exist_ok=True)
        build_compile_commands.write_text("[]", encoding="utf-8")
    if root_exists:
        root_compile_commands = tmp_path / "compile_commands.json"
        root_compile_commands.write_text("[]", encoding="utf-8")

    result = RUNNER.invoke(
        main_module.app,
        ["build", str(tmp_path)],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert observed_verbosity == [0]
    assert f"- compile_commands.json: {tmp_path / expected_path}" in result.output


def test_build_command_reports_missing_compile_commands_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_verbosity: list[int] = []

    def fake_ensure_clang_compilers_available() -> None:
        return None

    monkeypatch.setattr(build_module, "resolve_build_context", _fake_build_context)
    monkeypatch.setattr(
        build_module,
        "ensure_clang_compilers_available",
        fake_ensure_clang_compilers_available,
    )
    monkeypatch.setattr(build_module, "build_wheel", _fake_build_wheel(observed_verbosity))

    result = RUNNER.invoke(
        main_module.app,
        ["build", str(tmp_path)],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert observed_verbosity == [0]
    assert "- compile_commands.json: None" in result.output


def test_build_command_keeps_existing_build_directory_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_verbosity: list[int] = []
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    def fake_ensure_clang_compilers_available() -> None:
        return None

    def fail_rmtree(path: Path) -> None:
        raise AssertionError(f"不应删除 build 目录: {path}")

    monkeypatch.setattr(build_module, "resolve_build_context", _fake_build_context)
    monkeypatch.setattr(
        build_module,
        "ensure_clang_compilers_available",
        fake_ensure_clang_compilers_available,
    )
    monkeypatch.setattr(build_module, "build_wheel", _fake_build_wheel(observed_verbosity))
    monkeypatch.setattr(build_module.shutil, "rmtree", fail_rmtree)

    result = RUNNER.invoke(
        main_module.app,
        ["build", str(tmp_path)],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert observed_verbosity == [0]
    assert build_dir.exists()
    assert "- 已清理目录:" not in result.output


def test_build_command_removes_existing_build_directory_when_clean_build_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_verbosity: list[int] = []
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    removed_paths: list[Path] = []
    original_rmtree = shutil.rmtree

    def fake_ensure_clang_compilers_available() -> None:
        return None

    def fake_rmtree(path: Path) -> None:
        removed_paths.append(path)
        original_rmtree(path)

    monkeypatch.setattr(build_module, "resolve_build_context", _fake_build_context)
    monkeypatch.setattr(
        build_module,
        "ensure_clang_compilers_available",
        fake_ensure_clang_compilers_available,
    )
    monkeypatch.setattr(build_module, "build_wheel", _fake_build_wheel(observed_verbosity))
    monkeypatch.setattr(build_module.shutil, "rmtree", fake_rmtree)

    result = RUNNER.invoke(
        main_module.app,
        ["build", "--clean-build", str(tmp_path)],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 0
    assert observed_verbosity == [0]
    assert removed_paths == [build_dir]
    assert not build_dir.exists()
    assert f"- 已清理目录: {build_dir}" in result.output


def test_build_command_rejects_non_directory_build_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    build_file = tmp_path / "build"
    build_file.write_text("not a directory", encoding="utf-8")

    def fail_ensure_clang_compilers_available() -> None:
        raise AssertionError("不应进入编译器检查")

    def fail_build_wheel(
        srcdir: Path,
        runner: object,
        config_settings: object,
    ) -> Path:
        raise AssertionError(f"不应调用 build_wheel: {srcdir}, {runner}, {config_settings}")

    monkeypatch.setattr(build_module, "resolve_build_context", _fake_build_context)
    monkeypatch.setattr(
        build_module,
        "ensure_clang_compilers_available",
        fail_ensure_clang_compilers_available,
    )
    monkeypatch.setattr(build_module, "build_wheel", fail_build_wheel)

    result = RUNNER.invoke(
        main_module.app,
        ["build", str(tmp_path)],
        prog_name="pcstubgen",
    )

    assert result.exit_code == 1
    assert f"错误: 构建路径存在但不是目录: {build_file}" in result.output
