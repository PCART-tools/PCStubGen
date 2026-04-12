from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
import shutil
import subprocess
import typer

def ensure_build_programs_available() -> None:
    """检查 build 包装命令依赖的外部程序。"""
    missing_programs = [
        program
        for program in ("clang", "clang++", "llvm-config", "bear")
        if shutil.which(program) is None
    ]
    if missing_programs:
        missing_display = ", ".join(missing_programs)
        raise RuntimeError(
            "build 命令缺少外部程序依赖: "
            f"{missing_display}。请先安装这些程序并确保它们在 PATH 中。"
        )


def add_clang_environ(env: dict[str, str]) -> None:
    """向构建进程注入 clang 与 debug 构建环境变量。"""
    env["CC"] = "clang"
    env["CXX"] = "clang++"

    try:
        llvm_libdir = subprocess.check_output(
            ["llvm-config", "--libdir"],
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError) as ex:
        raise RuntimeError(f"获取 llvm-config libdir 失败: {ex}") from ex

    existing_library_path = env.get("LIBRARY_PATH")
    if existing_library_path:
        env["LIBRARY_PATH"] = os.pathsep.join([llvm_libdir, existing_library_path])
    else:
        env["LIBRARY_PATH"] = llvm_libdir

    env["DEBUG"] = "1"
    env["CMAKE_BUILD_TYPE"] = "Debug"
    env["CFLAGS"] = "-O0 -g -UNDEBUG"
    env["CXXFLAGS"] = "-O0 -g -UNDEBUG"


def build_bear_command(output: Path, command: Sequence[str]) -> list[str]:
    """组装 bear 前缀包装命令。"""
    return ["bear", "--output", str(output), "--", *command]


def run_build_command(command: Sequence[str], output: Path) -> int:
    """执行包装后的构建命令并返回退出码。"""
    env = os.environ.copy()
    add_clang_environ(env)
    try:
        completed_process = subprocess.run(
            build_bear_command(output, command),
            env=env,
            check=False,
        )
    except OSError as ex:
        raise RuntimeError(f"执行 build 命令失败: {ex}") from ex

    return completed_process.returncode


def _build_command(
    command: list[str] = typer.Argument(
        ...,
        metavar="-- ...",
        help="构建命令",
    ),
    output: Path = typer.Option(
        Path("compile_commands.json"),
        "--output",
        help="结果文件路径",
    ),
) -> None:
    """
    使用clang编译器，开启调试FLAG，使用bear包装原始构建命令产生compile_commands.json
    """
    try:
        ensure_build_programs_available()
        return_code = run_build_command(command, output)
    except RuntimeError as ex:
        print(f"错误: {ex}")
        raise typer.Exit(1) from ex

    if return_code != 0:
        raise typer.Exit(return_code)

    print("构建完成")
    print(f"bear compile_commands.json: {output}")
