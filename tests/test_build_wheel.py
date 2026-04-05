from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import pcstubgen.__main__ as main_module
import pcstubgen._build as build_impl

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
        build_impl,
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
        "requires = ['setuptools']\n"
        f'build-backend = "{backend}"\n',
        encoding="utf-8",
    )


def write_pyproject_without_build_system(project_dir: Path) -> None:
    (project_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "demo"\n'
        'version = "0.1.0"\n',
        encoding="utf-8",
    )


def write_pyproject_missing_build_backend(project_dir: Path) -> None:
    (project_dir / "pyproject.toml").write_text(
        "[build-system]\n"
        "requires = ['setuptools']\n",
        encoding="utf-8",
    )


def write_setup_py(project_dir: Path) -> None:
    (project_dir / "setup.py").write_text(
        "from setuptools import setup\n"
        "setup(name='demo', version='0.1.0')\n",
        encoding="utf-8",
    )


def invoke_build(*args: str) -> object:
    return RUNNER.invoke(
        main_module.app,
        ["build", *args],
        prog_name="pcstubgen",
    )


def set_clang_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        build_impl.shutil,
        "which",
        lambda executable: f"/usr/bin/{executable}",
    )


def test_build_help_displays_srcdir_argument() -> None:
    result = invoke_build("--help")

    assert result.exit_code == 0
    assert "SRCDIR" in result.output
    assert "compile_commands.json" in result.output


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
        runner: build_impl.SubprocessRunner,
        config_settings: build_impl.ConfigSettings,
    ) -> Path:
        captured["srcdir"] = srcdir
        captured["runner"] = runner
        captured["config_settings"] = dict(config_settings)
        return wheel_path

    set_terminal_interactive(monkeypatch, interactive=False)
    set_clang_available(monkeypatch)
    monkeypatch.setattr(build_impl, "build_wheel", fake_build_wheel)

    result = invoke_build(str(project_dir))

    assert result.exit_code == 0
    assert captured["srcdir"] == project_dir
    assert captured["runner"] is build_impl.clang_runner
    assert captured["config_settings"] == {
        "build-dir": "build",
        "setup-args": ["-Dbuildtype=debug", "-Db_ndebug=false"],
    }
    assert f"build-backend: mesonpy" in result.output
    assert str(build_impl.get_persistent_build_env_path(project_dir)) in result.output
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
        runner: build_impl.SubprocessRunner,
        config_settings: build_impl.ConfigSettings,
    ) -> Path:
        captured["srcdir"] = srcdir
        captured["runner"] = runner
        captured["config_settings"] = dict(config_settings)
        return wheel_path

    set_terminal_interactive(monkeypatch, interactive=False)
    set_clang_available(monkeypatch)
    monkeypatch.setattr(build_impl, "build_wheel", fake_build_wheel)

    result = invoke_build(str(project_dir))

    assert result.exit_code == 0
    assert captured["srcdir"] == project_dir
    assert captured["runner"] is build_impl.bear_runner
    assert captured["config_settings"] == {}
    assert f"build-backend: setuptools.build_meta" in result.output
    assert str(build_impl.get_persistent_build_env_path(project_dir)) in result.output
    assert str(project_dir / "compile_commands.json") in result.output


def test_cli_reports_legacy_setuptools_mode_for_setup_py_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_setup_py(project_dir)
    wheel_path = project_dir / "dist" / "demo.whl"
    captured: dict[str, object] = {}

    def fake_build_wheel(
        srcdir: Path,
        runner: build_impl.SubprocessRunner,
        config_settings: build_impl.ConfigSettings,
    ) -> Path:
        captured["srcdir"] = srcdir
        captured["runner"] = runner
        captured["config_settings"] = dict(config_settings)
        return wheel_path

    set_terminal_interactive(monkeypatch, interactive=False)
    set_clang_available(monkeypatch)
    monkeypatch.setattr(build_impl, "build_wheel", fake_build_wheel)

    result = invoke_build(str(project_dir))

    assert result.exit_code == 0
    assert captured["srcdir"] == project_dir
    assert captured["runner"] is build_impl.bear_runner
    assert captured["config_settings"] == {}
    assert f"build-backend: {build_impl.LEGACY_SETUPTOOLS_BACKEND}" in result.output
    assert str(project_dir / "compile_commands.json") in result.output


def test_cli_reports_legacy_setuptools_mode_for_pyproject_without_build_system(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject_without_build_system(project_dir)
    wheel_path = project_dir / "dist" / "demo.whl"
    captured: dict[str, object] = {}

    def fake_build_wheel(
        srcdir: Path,
        runner: build_impl.SubprocessRunner,
        config_settings: build_impl.ConfigSettings,
    ) -> Path:
        captured["srcdir"] = srcdir
        captured["runner"] = runner
        captured["config_settings"] = dict(config_settings)
        return wheel_path

    set_terminal_interactive(monkeypatch, interactive=False)
    set_clang_available(monkeypatch)
    monkeypatch.setattr(build_impl, "build_wheel", fake_build_wheel)

    result = invoke_build(str(project_dir))

    assert result.exit_code == 0
    assert captured["srcdir"] == project_dir
    assert captured["runner"] is build_impl.bear_runner
    assert captured["config_settings"] == {}
    assert f"build-backend: {build_impl.LEGACY_SETUPTOOLS_BACKEND}" in result.output
    assert str(project_dir / "compile_commands.json") in result.output


def test_cli_reports_legacy_setuptools_mode_for_missing_build_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject_missing_build_backend(project_dir)
    wheel_path = project_dir / "dist" / "demo.whl"
    captured: dict[str, object] = {}

    def fake_build_wheel(
        srcdir: Path,
        runner: build_impl.SubprocessRunner,
        config_settings: build_impl.ConfigSettings,
    ) -> Path:
        captured["srcdir"] = srcdir
        captured["runner"] = runner
        captured["config_settings"] = dict(config_settings)
        return wheel_path

    set_terminal_interactive(monkeypatch, interactive=False)
    set_clang_available(monkeypatch)
    monkeypatch.setattr(build_impl, "build_wheel", fake_build_wheel)

    result = invoke_build(str(project_dir))

    assert result.exit_code == 0
    assert captured["srcdir"] == project_dir
    assert captured["runner"] is build_impl.bear_runner
    assert captured["config_settings"] == {}
    assert f"build-backend: {build_impl.LEGACY_SETUPTOOLS_BACKEND}" in result.output
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
        runner: build_impl.SubprocessRunner,
        config_settings: build_impl.ConfigSettings,
    ) -> Path:
        _ = srcdir
        _ = runner
        _ = config_settings
        raise RuntimeError("boom")

    set_terminal_interactive(monkeypatch, interactive=False)
    set_clang_available(monkeypatch)
    monkeypatch.setattr(build_impl, "build_wheel", fake_build_wheel)

    result = invoke_build(str(project_dir))

    assert result.exit_code == 1
    assert "错误: boom" in result.output


def test_cli_rejects_invalid_pyproject_even_when_setup_py_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_setup_py(project_dir)
    (project_dir / "pyproject.toml").write_text("[build-system\n", encoding="utf-8")

    set_terminal_interactive(monkeypatch, interactive=False)
    set_clang_available(monkeypatch)

    result = invoke_build(str(project_dir))

    assert result.exit_code == 1
    assert "Failed to parse" in result.output


def test_cli_requires_srcdir_argument() -> None:
    result = invoke_build()

    assert result.exit_code != 0
    assert "Missing argument 'SRCDIR'" in result.output


def test_cli_rejects_nonexistent_srcdir() -> None:
    result = invoke_build("/definitely/missing/project")

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

    result = invoke_build(str(project_dir))

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

    result = invoke_build(str(project_dir))

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
    monkeypatch.setattr(build_impl.typer, "confirm", lambda message, default=False: False)

    result = invoke_build(str(project_dir))

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
    monkeypatch.setattr(build_impl.typer, "confirm", lambda message, default=False: True)
    monkeypatch.setattr(
        build_impl,
        "build_wheel",
        lambda srcdir, runner, config_settings: wheel_path,
    )

    result = invoke_build(str(project_dir))

    assert result.exit_code == 0
    assert not build_dir.exists()
    assert f"已清理目录: {build_dir}" in result.output


def test_resolve_build_context_reads_mesonpy(tmp_path: Path) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject(project_dir, "mesonpy")

    context = build_impl.resolve_build_context(project_dir)

    assert context.build_backend == "mesonpy"
    assert context.runner is build_impl.clang_runner
    assert context.config_settings == {
        "build-dir": "build",
        "setup-args": ["-Dbuildtype=debug", "-Db_ndebug=false"],
    }
    assert context.compile_commands_path == project_dir / "build" / "compile_commands.json"


def test_resolve_build_context_rejects_invalid_toml(tmp_path: Path) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[build-system\n", encoding="utf-8")

    with pytest.raises(Exception, match="Failed to parse"):
        build_impl.resolve_build_context(project_dir)


def test_resolve_build_context_allows_missing_build_backend(tmp_path: Path) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject_missing_build_backend(project_dir)

    context = build_impl.resolve_build_context(project_dir)

    assert context.build_backend == build_impl.LEGACY_SETUPTOOLS_BACKEND
    assert context.runner is build_impl.bear_runner
    assert context.config_settings == {}
    assert context.compile_commands_path == project_dir / "compile_commands.json"


def test_resolve_build_context_allows_pyproject_without_build_system(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject_without_build_system(project_dir)

    context = build_impl.resolve_build_context(project_dir)

    assert context.build_backend == build_impl.LEGACY_SETUPTOOLS_BACKEND
    assert context.runner is build_impl.bear_runner


def test_resolve_build_context_allows_setup_py_only(tmp_path: Path) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_setup_py(project_dir)

    context = build_impl.resolve_build_context(project_dir)

    assert context.build_backend == build_impl.LEGACY_SETUPTOOLS_BACKEND
    assert context.runner is build_impl.bear_runner


def test_resolve_build_context_reports_pyproject_backend(tmp_path: Path) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject(project_dir, "setuptools.build_meta")

    context = build_impl.resolve_build_context(project_dir)

    assert context.build_backend == "setuptools.build_meta"
    assert context.runner is build_impl.bear_runner
    assert context.config_settings == {}
    assert context.compile_commands_path == project_dir / "compile_commands.json"


def test_resolve_build_context_rejects_missing_requires_in_build_system(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        "[build-system]\n"
        'build-backend = "setuptools.build_meta"\n',
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="`requires` is a required property"):
        build_impl.resolve_build_context(project_dir)


def test_bear_runner_wraps_command_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_check_call(cmd: list[str], cwd: str | None, env: dict[str, str]) -> None:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env

    monkeypatch.setattr(build_impl.subprocess, "check_call", fake_check_call)

    build_impl.bear_runner(
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

    monkeypatch.setattr(build_impl.subprocess, "check_call", fake_check_call)

    build_impl.bear_runner(
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

    monkeypatch.setattr(build_impl.subprocess, "check_call", fake_check_call)

    with pytest.raises(RuntimeError, match="未找到 bear"):
        build_impl.bear_runner(["python", "-m", "build"])


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
        build_impl.pyproject_hooks,
        "default_subprocess_runner",
        fake_default_subprocess_runner,
    )

    build_impl.clang_runner(
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

    monkeypatch.setattr(build_impl.shutil, "which", fake_which)

    result = invoke_build(str(project_dir))

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
        def __init__(self, srcdir: Path, *, installer: str = "pip") -> None:
            captured["srcdir_for_env"] = srcdir
            captured["installer"] = installer
            self.path = str(srcdir / build_impl.PERSISTENT_BUILD_ENV_DIRNAME)
            self._python_executable = str(Path(self.path) / "bin" / "python")

        def __enter__(self) -> "FakeEnv":
            captured["entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            captured["exited"] = True

        def install(self, requirements: set[str]) -> None:
            captured.setdefault("installs", []).append(set(requirements))

        @property
        def python_executable(self) -> str:
            return self._python_executable

        def make_extra_environ(self) -> dict[str, str]:
            return {"PATH": str(Path(self.path) / "bin")}

    class FakeBuilder:
        build_system_requires = {"setuptools", "wheel"}

        def get_requires_for_build(
            self,
            distribution: str,
            config_settings: build_impl.ConfigSettings,
        ) -> set[str]:
            captured["requires"] = (distribution, dict(config_settings))
            return {"backend-extra"}

        def build(
            self,
            distribution: str,
            output_directory: str,
            config_settings: build_impl.ConfigSettings,
        ) -> str:
            captured["build"] = (distribution, output_directory, dict(config_settings))
            return str(Path(output_directory) / "demo-0.1.0-py3-none-any.whl")

    def fake_from_isolated_env(
        env: FakeEnv,
        source_dir: str,
        runner: build_impl.SubprocessRunner,
    ) -> FakeBuilder:
        captured["source_dir"] = source_dir
        captured["runner"] = runner
        captured["env_object"] = env
        return FakeBuilder()

    monkeypatch.setattr(build_impl, "PersistentIsolatedEnv", FakeEnv)
    monkeypatch.setattr(
        build_impl,
        "ProjectBuilder",
        SimpleNamespace(from_isolated_env=fake_from_isolated_env),
    )

    runner = lambda cmd, cwd=None, extra_environ=None: None
    wheel_path = build_impl.build_wheel(
        project_dir,
        runner,
        {"build-dir": "build"},
    )

    assert wheel_path == project_dir / "dist" / "demo-0.1.0-py3-none-any.whl"
    assert captured["srcdir_for_env"] == project_dir
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


def test_persistent_isolated_env_reuses_existing_environment_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    env_path = build_impl.get_persistent_build_env_path(project_dir)
    env_path.mkdir()
    captured: dict[str, object] = {}

    def fake_find_executable_and_scripts(path: str) -> tuple[str, str, str]:
        captured["find_path"] = path
        return (f"{path}/bin/python", f"{path}/bin", f"{path}/lib/python3.12/site-packages")

    monkeypatch.setattr(build_impl.build_env, "_find_executable_and_scripts", fake_find_executable_and_scripts)

    env = build_impl.PersistentIsolatedEnv(project_dir)
    with env:
        assert env.python_executable == f"{env.path}/bin/python"
        assert env.make_extra_environ() == {
            "PATH": f"{env.path}/bin:{build_impl.os.environ.get('PATH')}"
        }

    assert captured["find_path"] == str(env_path.resolve())
    assert env_path.exists()


def test_persistent_isolated_env_install_uses_backend_install_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    captured: dict[str, object] = {}

    class FakeBackend:
        python_executable = "/tmp/python"
        scripts_dir = "/tmp/bin"

        def install_dependencies(self, requirements: set[str], constraints: list[str]) -> None:
            captured["requirements"] = set(requirements)
            captured["constraints"] = list(constraints)

    env = build_impl.PersistentIsolatedEnv(project_dir)
    env._env_backend = FakeBackend()
    env.install(["b>=1", "a>=2"])

    assert captured["requirements"] == {"a>=2", "b>=1"}
    assert captured["constraints"] == []


def test_persistent_isolated_env_creates_environment_with_build_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    captured: dict[str, object] = {}

    class FakeBackend:
        display_name = "fake-backend"

        def __init__(self) -> None:
            self.python_executable = ""
            self.scripts_dir = ""

        def create(self, path: str) -> None:
            captured["create_path"] = path
            self.python_executable = f"{path}/bin/python"
            self.scripts_dir = f"{path}/bin"

        def install_dependencies(self, requirements: set[str], constraints: list[str]) -> None:
            _ = requirements
            _ = constraints

    monkeypatch.setattr(build_impl.build_env, "_PipBackend", FakeBackend)

    env = build_impl.PersistentIsolatedEnv(project_dir)
    with env:
        assert env.python_executable == f"{env.path}/bin/python"

    assert captured["create_path"] == str(build_impl.get_persistent_build_env_path(project_dir).resolve())


def test_persistent_isolated_env_rejects_invalid_existing_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    env_path = build_impl.get_persistent_build_env_path(project_dir)
    env_path.mkdir()

    def fake_find_executable_and_scripts(path: str) -> tuple[str, str, str]:
        _ = path
        raise RuntimeError("broken env")

    monkeypatch.setattr(build_impl.build_env, "_find_executable_and_scripts", fake_find_executable_and_scripts)

    env = build_impl.PersistentIsolatedEnv(project_dir)
    with pytest.raises(RuntimeError, match="无效持久构建环境"):
        with env:
            pass


def test_cli_reports_persistent_build_env_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    write_pyproject(project_dir, "mesonpy")
    wheel_path = project_dir / "dist" / "demo.whl"

    set_terminal_interactive(monkeypatch, interactive=False)
    set_clang_available(monkeypatch)
    monkeypatch.setattr(
        build_impl,
        "build_wheel",
        lambda srcdir, runner, config_settings: wheel_path,
    )

    result = invoke_build(str(project_dir))

    assert result.exit_code == 0
    assert f"持久构建环境: {build_impl.get_persistent_build_env_path(project_dir)}" in result.output


def test_build_wheel_failure_keeps_persistent_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    env_path = build_impl.get_persistent_build_env_path(project_dir)
    env_path.mkdir()

    class FakeBuilder:
        build_system_requires = {"setuptools"}

        def get_requires_for_build(
            self,
            distribution: str,
            config_settings: build_impl.ConfigSettings,
        ) -> set[str]:
            _ = distribution
            _ = config_settings
            return set()

        def build(
            self,
            distribution: str,
            output_directory: str,
            config_settings: build_impl.ConfigSettings,
        ) -> str:
            _ = distribution
            _ = output_directory
            _ = config_settings
            raise RuntimeError("backend boom")

    monkeypatch.setattr(
        build_impl,
        "ProjectBuilder",
        SimpleNamespace(from_isolated_env=lambda env, source_dir, runner: FakeBuilder()),
    )
    monkeypatch.setattr(build_impl.PersistentIsolatedEnv, "install", lambda self, requirements: None)
    monkeypatch.setattr(
        build_impl.build_env,
        "_find_executable_and_scripts",
        lambda path: (f"{path}/bin/python", f"{path}/bin", f"{path}/lib/python3.12/site-packages"),
    )

    with pytest.raises(RuntimeError, match="持久构建环境失败"):
        build_impl.build_wheel(project_dir, lambda *args, **kwargs: None, {})

    assert env_path.exists()
