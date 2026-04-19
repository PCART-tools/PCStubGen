from __future__ import annotations

import contextlib
from dataclasses import dataclass
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
import subprocess
from build import ProjectBuilder
from build._types import ConfigSettings, SubprocessRunner
import typer

from ._persistent_build_env import PersistentIsolatedEnv

BUILD_COMMAND_HELP = (
    "构建 Python 项目，并为 stub 工作流提供可用的 compile_commands.json。"
)
LEGACY_SETUPTOOLS_BACKEND = "setuptools.build_meta:__legacy__"


@dataclass(frozen=True)
class BuildContext:
    """封装 build 解析后的构建模式信息。"""

    build_backend: str
    runner: SubprocessRunner
    config_settings: ConfigSettings


def find_compile_commands_path(srcdir: Path) -> Path | None:
    """按优先级探测可用的 compile_commands.json 路径。"""
    for path in (
        srcdir / "build" / "compile_commands.json",
        srcdir / "compile_commands.json",
    ):
        if path.exists():
            return path
    return None


def get_build_context(srcdir: Path) -> BuildContext:
    """借用 ProjectBuilder 解析后的 backend，确定构建模式。"""
    builder = ProjectBuilder(str(srcdir))
    build_backend = builder._backend

    if build_backend == "mesonpy":
        return BuildContext(
            build_backend=build_backend,
            runner=default_runner,
            config_settings={
                "build-dir": "build",
                "setup-args": ["-Dbuildtype=debug", "-Db_ndebug=false"],
            },
        )

    return BuildContext(
        build_backend=build_backend,
        runner=bear_runner,
        config_settings={},
    )


def add_clang_environ(env: dict[str, str]) -> None:
    # C/C++ 编译器前端。
    env["CC"] = "libclang"
    env["CXX"] = "libclang++"

    # 让 LLVM lib 目录进入隐式库搜索路径，便于上游 CMake 的
    # find_library(NAMES omp gomp iomp5 ...) 优先命中 libomp 而不是 libgomp。
    llvm_libdir = subprocess.check_output(
        ["llvm-config", "--libdir"],
        text=True,
    ).strip()
    existing_library_path = env.get("LIBRARY_PATH")
    if existing_library_path:
        env["LIBRARY_PATH"] = os.pathsep.join([llvm_libdir, existing_library_path])
    else:
        env["LIBRARY_PATH"] = llvm_libdir

    # 显式打开 debug 构建，避免上游构建后端按 release 路径生成产物。
    env["DEBUG"] = "1"
    env["CMAKE_BUILD_TYPE"] = "Debug"
    env["CFLAGS"] = "-O0 -g -UNDEBUG"
    env["CXXFLAGS"] = "-O0 -g -UNDEBUG"


def default_runner(
    cmd: Sequence[str],
    cwd: str | None = None,
    extra_environ: Mapping[str, str] | None = None,
) -> None:
    env = os.environ.copy()
    if extra_environ is not None:
        env.update(extra_environ)
    add_clang_environ(env)
    subprocess.check_call(
        cmd,
        cwd=cwd,
        env=env,
    )


def bear_runner(
    cmd: Sequence[str],
    cwd: str | None = None,
    extra_environ: Mapping[str, str] | None = None,
) -> None:
    env = os.environ.copy()
    if extra_environ is not None:
        env.update(extra_environ)
    add_clang_environ(env)
    subprocess.check_call(
        ["bear", "--", *cmd],
        cwd=cwd,
        env=env,
    )


def ensure_build_programs_available() -> None:
    missing_programs = [
        program
        for program in ("libclang", "libclang++", "llvm-config", "bear")
        if shutil.which(program) is None
    ]
    if missing_programs:
        missing_display = ", ".join(missing_programs)
        raise RuntimeError(
            "build 命令缺少外部程序依赖: "
            f"{missing_display}。请先安装这些程序并确保它们在 PATH 中。"
        )


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
                wheel_path = Path(
                    builder.build("wheel", str(output_directory), config_settings)
                )
        except Exception as ex:
            raise RuntimeError(f"持久构建环境失败 [{env.path}]: {ex}") from ex

    return wheel_path.resolve()


def _build_command(
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
    clean_env: bool = typer.Option(
        False,
        "--clean-env",
        help="构建前删除已有的持久构建环境。",
    ),
    clean_build: bool = typer.Option(
        False,
        "--clean-build",
        help="构建前删除已有的 build 目录。",
    ),
) -> None:
    build_dir = srcdir / "build"
    env_dir = PersistentIsolatedEnv.get_build_env_path(srcdir)
    try:
        ensure_build_programs_available()

        if env_dir.exists():
            if env_dir.is_symlink() or not env_dir.is_dir():
                raise RuntimeError(f"持久构建环境路径存在但不是可清理目录: {env_dir}")
            if clean_env:
                shutil.rmtree(env_dir)
                print(f"- 已清理持久构建环境: {env_dir}")
        if build_dir.exists():
            if not build_dir.is_dir():
                raise RuntimeError(f"构建路径存在但不是目录: {build_dir}")
            if clean_build:
                shutil.rmtree(build_dir)
                print(f"- 已清理目录: {build_dir}")

        build_context = get_build_context(srcdir)

        wheel_path = build_wheel(
            srcdir,
            build_context.runner,
            build_context.config_settings,
        )

    except Exception as ex:
        print(f"错误: {ex}")
        raise typer.Exit(1) from ex

    print("构建完成")
    print(f"- build-backend: {build_context.build_backend}")
    print(f"- 持久构建环境: {env_dir}")
    print(f"- wheel 文件: {wheel_path}")
    print(f"- compile_commands.json: {find_compile_commands_path(srcdir)}")
