from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

from build import env as build_env
from build import ProjectBuilder
from build._types import ConfigSettings, SubprocessRunner
from build.env import DefaultIsolatedEnv
import pyproject_hooks
import typer

app = typer.Typer(add_completion=False)

DEBUG_COMPILE_FLAGS = "-O0 -g -UNDEBUG"
CLANG_CC = "clang"
CLANG_CXX = "clang++"
PERSISTENT_BUILD_ENV_DIRNAME = ".pcstubgen-build-venv"


def load_build_backend(srcdir: Path) -> str:
    pyproject_path = srcdir / "pyproject.toml"
    if not pyproject_path.is_file():
        raise RuntimeError(f"未找到 pyproject.toml: {pyproject_path}")

    try:
        with pyproject_path.open("rb") as file:
            pyproject_data = tomllib.load(file)
    except tomllib.TOMLDecodeError as ex:
        raise RuntimeError(f"pyproject.toml 解析失败: {ex}") from ex
    except OSError as ex:
        raise RuntimeError(f"读取 pyproject.toml 失败: {ex}") from ex

    build_system = pyproject_data.get("build-system")
    if not isinstance(build_system, dict):
        raise RuntimeError("pyproject.toml 缺少 [build-system] 表。")

    build_backend = build_system.get("build-backend")
    if not isinstance(build_backend, str) or not build_backend.strip():
        raise RuntimeError("pyproject.toml 缺少 build-system.build-backend。")

    return build_backend


def build_clang_environ(
    extra_environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if extra_environ is not None:
        env.update(extra_environ)
    env["CC"] = CLANG_CC
    env["CXX"] = CLANG_CXX
    env["CFLAGS"] = DEBUG_COMPILE_FLAGS
    env["CXXFLAGS"] = DEBUG_COMPILE_FLAGS
    return env


def clang_runner(
    cmd: Sequence[str],
    cwd: str | None = None,
    extra_environ: Mapping[str, str] | None = None,
) -> None:
    pyproject_hooks.default_subprocess_runner(
        cmd,
        cwd=cwd,
        extra_environ=build_clang_environ(extra_environ),
    )


def bear_runner(
    cmd: Sequence[str],
    cwd: str | None = None,
    extra_environ: Mapping[str, str] | None = None,
) -> None:
    try:
        subprocess.check_call(
            ["bear", "--", *cmd],
            cwd=cwd,
            env=build_clang_environ(extra_environ),
        )
    except FileNotFoundError as ex:
        raise RuntimeError("未找到 bear 命令，无法为非 mesonpy 项目生成 compile_commands.json。") from ex


def ensure_clang_compilers_available() -> None:
    missing_compilers = [
        compiler for compiler in (CLANG_CC, CLANG_CXX) if shutil.which(compiler) is None
    ]
    if missing_compilers:
        missing_display = ", ".join(missing_compilers)
        raise RuntimeError(
            f"未找到 clang 编译器: {missing_display}。build_wheel 默认要求 clang 工具链。"
        )


def get_persistent_build_env_path(srcdir: Path) -> Path:
    return srcdir / PERSISTENT_BUILD_ENV_DIRNAME


class PersistentIsolatedEnv(DefaultIsolatedEnv):
    """在项目目录下维护可复用的持久隔离构建环境。"""

    def __init__(
        self,
        srcdir: Path,
        *,
        installer: build_env.Installer = "pip",
    ) -> None:
        super().__init__(installer=installer)
        self._persistent_srcdir = srcdir

    def __enter__(self) -> PersistentIsolatedEnv:
        try:
            path = get_persistent_build_env_path(self._persistent_srcdir).resolve()
            # 与 DefaultIsolatedEnv 保持一致，统一真实路径表示。
            self._path = os.path.realpath(path)

            self._env_backend: build_env._EnvBackend

            if self.installer == "uv":
                self._env_backend = build_env._UvBackend()
            else:
                self._env_backend = build_env._PipBackend()

            if os.path.exists(self._path):
                try:
                    python_executable, scripts_dir, _ = build_env._find_executable_and_scripts(
                        self._path
                    )
                except Exception as ex:
                    raise RuntimeError(f"无效持久构建环境: {self._path}") from ex
                self._env_backend.python_executable = python_executable
                self._env_backend.scripts_dir = scripts_dir
            else:
                build_env._ctx.log(
                    f"Creating isolated environment: {self._env_backend.display_name}..."
                )
                self._env_backend.create(self._path)
        except Exception:
            self.__exit__(*sys.exc_info())
            raise

        return self

    def __exit__(self, *args: object) -> None:
        _ = args


def build_wheel(
    srcdir: Path,
    runner: SubprocessRunner,
    config_settings: ConfigSettings,
) -> Path:
    """
    在源目录中通过 build API 构建 wheel，并返回产物绝对路径。
    """
    output_directory = srcdir / "dist"
    output_directory.mkdir(parents=True, exist_ok=True)
    env = PersistentIsolatedEnv(srcdir)

    with contextlib.chdir(srcdir):
        try:
            with env:
                builder = ProjectBuilder.from_isolated_env(env, str(srcdir), runner)
                env.install(builder.build_system_requires)
                env.install(builder.get_requires_for_build("wheel", config_settings))
                wheel_path = Path(builder.build("wheel", str(output_directory), config_settings))
        except Exception as ex:
            raise RuntimeError(f"持久构建环境失败 [{env.path}]: {ex}") from ex

    return wheel_path.resolve()


@app.command(help="根据 pyproject.toml 构建 wheel，并在非 mesonpy 项目中生成 compile_commands.json。")
def command(
    srcdir: Path = typer.Argument(
        ...,
        metavar="SRCDIR",
        help="待构建的项目根目录。",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    build_dir_name = "build"
    build_dir = srcdir / build_dir_name
    try:
        build_backend = load_build_backend(srcdir)

        if build_dir.exists():
            if not build_dir.is_dir():
                raise RuntimeError(f"构建路径存在但不是目录: {build_dir}")
            if not sys.stdin.isatty() or not sys.stdout.isatty():
                raise RuntimeError("检测到 build 目录存在，但当前环境无法交互确认删除。")
            if not typer.confirm(f"将删除构建目录 {build_dir}，是否继续？", default=False):
                raise RuntimeError("用户取消删除 build 目录，构建已终止。")
            shutil.rmtree(build_dir)
            print(f"- 已清理目录: {build_dir}")

        if build_backend == "mesonpy":
            build_mode_label = "mesonpy"
            runner = clang_runner
            config_settings: ConfigSettings = {
                "build-dir": build_dir_name,
                "setup-args": ["-Dbuildtype=debug", "-Db_ndebug=false"],
            }
            compile_commands_path = srcdir / build_dir_name / "compile_commands.json"
        else:
            build_mode_label = "bear"
            runner = bear_runner
            config_settings = {}
            compile_commands_path = srcdir / "compile_commands.json"

        ensure_clang_compilers_available()
        wheel_path = build_wheel(srcdir, runner, config_settings)
    except Exception as ex:
        print(f"错误: {ex}")
        raise typer.Exit(1) from ex

    print("构建完成")
    print(f"- 构建方式: {build_mode_label}")
    print(f"- build-backend: {build_backend}")
    print(f"- 持久构建环境: {get_persistent_build_env_path(srcdir)}")
    print(f"- wheel 文件: {wheel_path}")
    print(f"- compile_commands.json: {compile_commands_path}")


if __name__ == "__main__":
    app()
