from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tools import build_wheel

RUNNER = CliRunner()


class FakeStream:
    def __init__(self, *, interactive: bool) -> None:
        self._interactive = interactive

    def isatty(self) -> bool:
        return self._interactive


def set_terminal_interactive(
    monkeypatch: pytest.MonkeyPatch,
    *,
    interactive: bool,
) -> None:
    monkeypatch.setattr(
        build_wheel,
        "sys",
        SimpleNamespace(
            stdin=FakeStream(interactive=interactive),
            stdout=FakeStream(interactive=interactive),
            stderr=FakeStream(interactive=False),
        ),
    )


def write_pyproject(project_dir: Path, backend: str) -> None:
    (project_dir / "pyproject.toml").write_text(
        "[build-system]\n"
        f'build-backend = "{backend}"\n',
        encoding="utf-8",
    )


def set_clang_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        build_wheel.shutil,
        "which",
        lambda executable: f"/usr/bin/{executable}",
    )


def test_cli_reports_mesonpy_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject(project_dir, "mesonpy")
    wheel_path = project_dir / "dist" / "demo.whl"
    captured: dict[str, object] = {}

    def fake_build_wheel(
        srcdir: Path,
        runner: build_wheel.SubprocessRunner,
        config_settings: build_wheel.ConfigSettings,
    ) -> Path:
        captured["srcdir"] = srcdir
        captured["runner"] = runner
        captured["config_settings"] = dict(config_settings)
        return wheel_path

    set_terminal_interactive(monkeypatch, interactive=False)
    set_clang_available(monkeypatch)
    monkeypatch.setattr(build_wheel, "build_wheel", fake_build_wheel)

    result = RUNNER.invoke(build_wheel.app, [str(project_dir)], prog_name="build_wheel")

    assert result.exit_code == 0
    assert captured["srcdir"] == project_dir
    assert captured["runner"] is build_wheel.clang_runner
    assert captured["config_settings"] == {
        "build-dir": "build",
        "setup-args": ["-Dbuildtype=debug", "-Db_ndebug=false"],
    }
    assert "构建方式: mesonpy" in result.output
    assert f"build-backend: mesonpy" in result.output
    assert str(wheel_path) in result.output
    assert str(project_dir / "build" / "compile_commands.json") in result.output


def test_cli_reports_bear_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject(project_dir, "setuptools.build_meta")
    wheel_path = project_dir / "dist" / "demo.whl"
    captured: dict[str, object] = {}

    def fake_build_wheel(
        srcdir: Path,
        runner: build_wheel.SubprocessRunner,
        config_settings: build_wheel.ConfigSettings,
    ) -> Path:
        captured["srcdir"] = srcdir
        captured["runner"] = runner
        captured["config_settings"] = dict(config_settings)
        return wheel_path

    set_terminal_interactive(monkeypatch, interactive=False)
    set_clang_available(monkeypatch)
    monkeypatch.setattr(build_wheel, "build_wheel", fake_build_wheel)

    result = RUNNER.invoke(build_wheel.app, [str(project_dir)], prog_name="build_wheel")

    assert result.exit_code == 0
    assert captured["srcdir"] == project_dir
    assert captured["runner"] is build_wheel.bear_runner
    assert captured["config_settings"] == {}
    assert "构建方式: bear" in result.output
    assert f"build-backend: setuptools.build_meta" in result.output
    assert str(project_dir / "compile_commands.json") in result.output


def test_cli_returns_error_when_build_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject(project_dir, "setuptools.build_meta")

    def fake_build_wheel(
        srcdir: Path,
        runner: build_wheel.SubprocessRunner,
        config_settings: build_wheel.ConfigSettings,
    ) -> Path:
        _ = srcdir
        _ = runner
        _ = config_settings
        raise RuntimeError("boom")

    set_terminal_interactive(monkeypatch, interactive=False)
    set_clang_available(monkeypatch)
    monkeypatch.setattr(build_wheel, "build_wheel", fake_build_wheel)

    result = RUNNER.invoke(build_wheel.app, [str(project_dir)], prog_name="build_wheel")

    assert result.exit_code == 1
    assert "错误: boom" in result.output


def test_cli_requires_srcdir_argument() -> None:
    result = RUNNER.invoke(build_wheel.app, [], prog_name="build_wheel")

    assert result.exit_code != 0
    assert "Missing argument 'SRCDIR'" in result.output


def test_cli_rejects_nonexistent_srcdir() -> None:
    result = RUNNER.invoke(
        build_wheel.app,
        ["/definitely/missing/project"],
        prog_name="build_wheel",
    )

    assert result.exit_code != 0
    assert "SRCDIR" in result.output
    assert "/definitely/missing/project" in result.output


def test_cli_rejects_build_path_that_is_not_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject(project_dir, "mesonpy")
    (project_dir / "build").write_text("not a directory", encoding="utf-8")
    set_terminal_interactive(monkeypatch, interactive=True)
    set_clang_available(monkeypatch)

    result = RUNNER.invoke(build_wheel.app, [str(project_dir)], prog_name="build_wheel")

    assert result.exit_code == 1
    assert "构建路径存在但不是目录" in result.output


def test_cli_rejects_existing_build_dir_without_interactive_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject(project_dir, "mesonpy")
    (project_dir / "build").mkdir()
    set_terminal_interactive(monkeypatch, interactive=False)
    set_clang_available(monkeypatch)

    result = RUNNER.invoke(build_wheel.app, [str(project_dir)], prog_name="build_wheel")

    assert result.exit_code == 1
    assert "无法交互确认删除" in result.output


def test_cli_preserves_build_dir_when_user_declines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject(project_dir, "mesonpy")
    build_dir = project_dir / "build"
    build_dir.mkdir()

    set_terminal_interactive(monkeypatch, interactive=True)
    set_clang_available(monkeypatch)
    monkeypatch.setattr(build_wheel.typer, "confirm", lambda message, default=False: False)

    result = RUNNER.invoke(build_wheel.app, [str(project_dir)], prog_name="build_wheel")

    assert result.exit_code == 1
    assert "用户取消删除 build 目录" in result.output
    assert build_dir.exists()


def test_cli_removes_build_dir_after_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject(project_dir, "setuptools.build_meta")
    build_dir = project_dir / "build"
    build_dir.mkdir()
    wheel_path = project_dir / "dist" / "demo.whl"

    set_terminal_interactive(monkeypatch, interactive=True)
    set_clang_available(monkeypatch)
    monkeypatch.setattr(build_wheel.typer, "confirm", lambda message, default=False: True)
    monkeypatch.setattr(
        build_wheel,
        "build_wheel",
        lambda srcdir, runner, config_settings: wheel_path,
    )

    result = RUNNER.invoke(build_wheel.app, [str(project_dir)], prog_name="build_wheel")

    assert result.exit_code == 0
    assert not build_dir.exists()
    assert f"已清理目录: {build_dir}" in result.output


def test_load_build_backend_reads_mesonpy(tmp_path: Path) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject(project_dir, "mesonpy")

    assert build_wheel.load_build_backend(project_dir) == "mesonpy"


def test_load_build_backend_rejects_invalid_toml(tmp_path: Path) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[build-system\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="解析失败"):
        build_wheel.load_build_backend(project_dir)


def test_load_build_backend_rejects_missing_build_backend(tmp_path: Path) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        "[build-system]\n"
        "requires = ['setuptools']\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="build-backend"):
        build_wheel.load_build_backend(project_dir)


def test_bear_runner_wraps_command_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_check_call(cmd: list[str], cwd: str | None, env: dict[str, str]) -> None:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env

    monkeypatch.setattr(build_wheel.subprocess, "check_call", fake_check_call)

    build_wheel.bear_runner(
        ["python", "-m", "build"],
        cwd="/tmp/demo",
        extra_environ={"PATH": "/venv/bin", "EXTRA": "1"},
    )

    assert captured["cmd"] == ["bear", "--", "python", "-m", "build"]
    assert captured["cwd"] == "/tmp/demo"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["PATH"] == "/venv/bin"
    assert env["EXTRA"] == "1"
    assert env["CC"] == "clang"
    assert env["CXX"] == "clang++"
    assert env["CFLAGS"] == "-O0 -g -UNDEBUG"
    assert env["CXXFLAGS"] == "-O0 -g -UNDEBUG"


def test_bear_runner_overrides_compiler_related_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_check_call(cmd: list[str], cwd: str | None, env: dict[str, str]) -> None:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env

    monkeypatch.setattr(build_wheel.subprocess, "check_call", fake_check_call)

    build_wheel.bear_runner(
        ["python", "-m", "build"],
        extra_environ={
            "CC": "gcc",
            "CXX": "g++",
            "CFLAGS": "-O2",
            "CXXFLAGS": "-O3",
        },
    )

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["CC"] == "clang"
    assert env["CXX"] == "clang++"
    assert env["CFLAGS"] == "-O0 -g -UNDEBUG"
    assert env["CXXFLAGS"] == "-O0 -g -UNDEBUG"


def test_bear_runner_requires_bear(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_check_call(cmd: list[str], cwd: str | None, env: dict[str, str]) -> None:
        _ = cmd
        _ = cwd
        _ = env
        raise FileNotFoundError("bear")

    monkeypatch.setattr(build_wheel.subprocess, "check_call", fake_check_call)

    with pytest.raises(RuntimeError, match="未找到 bear"):
        build_wheel.bear_runner(["python", "-m", "build"])


def test_clang_runner_injects_clang_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_default_subprocess_runner(
        cmd: list[str],
        cwd: str | None = None,
        extra_environ: dict[str, str] | None = None,
    ) -> None:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["extra_environ"] = extra_environ

    monkeypatch.setattr(
        build_wheel.pyproject_hooks,
        "default_subprocess_runner",
        fake_default_subprocess_runner,
    )

    build_wheel.clang_runner(
        ["python", "-m", "build"],
        cwd="/tmp/demo",
        extra_environ={"PATH": "/venv/bin", "CC": "gcc"},
    )

    assert captured["cmd"] == ["python", "-m", "build"]
    assert captured["cwd"] == "/tmp/demo"
    env = captured["extra_environ"]
    assert isinstance(env, dict)
    assert env["PATH"] == "/venv/bin"
    assert env["CC"] == "clang"
    assert env["CXX"] == "clang++"
    assert env["CFLAGS"] == "-O0 -g -UNDEBUG"
    assert env["CXXFLAGS"] == "-O0 -g -UNDEBUG"


def test_cli_reports_missing_clang_before_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject(project_dir, "mesonpy")

    set_terminal_interactive(monkeypatch, interactive=False)

    def fake_which(executable: str) -> str | None:
        if executable == "clang":
            return None
        return f"/usr/bin/{executable}"

    monkeypatch.setattr(build_wheel.shutil, "which", fake_which)

    result = RUNNER.invoke(build_wheel.app, [str(project_dir)], prog_name="build_wheel")

    assert result.exit_code == 1
    assert "未找到 clang 编译器: clang" in result.output
    assert "默认要求 clang 工具链" in result.output


def test_build_wheel_uses_isolated_env_and_build_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    captured: dict[str, object] = {}

    class FakeEnv:
        def __init__(self, *, installer: str) -> None:
            captured["installer"] = installer

        def __enter__(self) -> "FakeEnv":
            captured["entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            captured["exited"] = True

        def install(self, requirements: set[str]) -> None:
            captured.setdefault("installs", []).append(set(requirements))

    class FakeBuilder:
        build_system_requires = {"setuptools", "wheel"}

        def get_requires_for_build(
            self,
            distribution: str,
            config_settings: build_wheel.ConfigSettings,
        ) -> set[str]:
            captured["requires"] = (distribution, dict(config_settings))
            return {"backend-extra"}

        def build(
            self,
            distribution: str,
            output_directory: str,
            config_settings: build_wheel.ConfigSettings,
        ) -> str:
            captured["build"] = (distribution, output_directory, dict(config_settings))
            return str(Path(output_directory) / "demo-0.1.0-py3-none-any.whl")

    def fake_from_isolated_env(
        env: FakeEnv,
        source_dir: str,
        runner: build_wheel.SubprocessRunner,
    ) -> FakeBuilder:
        captured["source_dir"] = source_dir
        captured["runner"] = runner
        captured["env_object"] = env
        return FakeBuilder()

    monkeypatch.setattr(build_wheel, "DefaultIsolatedEnv", FakeEnv)
    monkeypatch.setattr(
        build_wheel,
        "ProjectBuilder",
        SimpleNamespace(from_isolated_env=fake_from_isolated_env),
    )

    runner = lambda cmd, cwd=None, extra_environ=None: None
    wheel_path = build_wheel.build_wheel(
        project_dir,
        runner,
        {"build-dir": "build"},
    )

    assert wheel_path == project_dir / "dist" / "demo-0.1.0-py3-none-any.whl"
    assert captured["installer"] == "pip"
    assert captured["source_dir"] == str(project_dir)
    assert captured["runner"] is runner
    assert captured["installs"] == [
        {"setuptools", "wheel"},
        {"backend-extra"},
    ]
    assert captured["requires"] == ("wheel", {"build-dir": "build"})
    assert captured["build"] == (
        "wheel",
        str(project_dir / "dist"),
        {"build-dir": "build"},
    )
