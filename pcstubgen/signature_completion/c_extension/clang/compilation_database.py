from __future__ import annotations

import dataclasses
from pathlib import Path

from clang.cindex import CompilationDatabase

_ARG_FLAGS_WITH_VALUE = {
    "-o",
    "-MF",
    "-MT",
    "-MQ",
    "-MJ",
}
_ARG_FLAGS_WITHOUT_VALUE = {
    "-c",
    "-MD",
    "-MMD",
    "-MG",
    "-MP",
}


@dataclasses.dataclass(frozen=True)
class CompilationCommand:
    file_path: Path
    working_directory: Path
    parse_args: list[str]


def validate_compilation_database_path(compilation_database: Path) -> Path:
    """校验 compile_commands.json 路径。"""
    if not compilation_database.exists():
        raise RuntimeError(f"编译数据库不存在: {compilation_database}")
    if not compilation_database.is_file():
        raise RuntimeError(f"编译数据库不是文件: {compilation_database}")
    if compilation_database.name != "compile_commands.json":
        raise RuntimeError(
            f"编译数据库文件名必须为 compile_commands.json: {compilation_database}"
        )
    return compilation_database.resolve()


def load_compilation_database(compilation_database: Path) -> CompilationDatabase:
    """从 compile_commands.json 所在目录加载编译数据库。"""
    validated_path = validate_compilation_database_path(compilation_database)
    try:
        return CompilationDatabase.fromDirectory(str(validated_path.parent))
    except Exception as ex:
        raise RuntimeError(f"编译数据库加载失败: {validated_path}") from ex


def resolve_compile_command_file_path(command: object) -> Path:
    """将 compile command 的 file 字段解析为绝对路径。"""
    file_path = Path(str(command.filename))
    if file_path.is_absolute():
        return file_path.resolve()
    return (Path(str(command.directory)) / file_path).resolve()


def _is_source_file_operand(
    arg: str,
    *,
    file_path: Path,
    working_directory: Path,
) -> bool:
    if not arg or arg.startswith("-"):
        return False
    candidate = Path(arg)
    if not candidate.is_absolute():
        candidate = working_directory / candidate
    return candidate.resolve() == file_path


def sanitize_compile_command_arguments(command: object) -> list[str]:
    """清理 compile command 中不应传递给 libclang parse 的参数。"""
    arguments = [str(arg) for arg in command.arguments]
    if not arguments:
        return []

    working_directory = Path(str(command.directory)).resolve()
    file_path = resolve_compile_command_file_path(command)
    parse_args: list[str] = []
    skip_next = False
    for arg in arguments[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in _ARG_FLAGS_WITH_VALUE:
            skip_next = True
            continue
        if arg in _ARG_FLAGS_WITHOUT_VALUE:
            continue
        if _is_source_file_operand(
            arg,
            file_path=file_path,
            working_directory=working_directory,
        ):
            continue
        parse_args.append(arg)
    return parse_args


def resolve_compilation_command(
    database: CompilationDatabase,
    source_path: Path,
) -> CompilationCommand:
    """按源码绝对路径查询首条编译命令。"""
    resolved_source_path = source_path.resolve()
    compile_commands = database.getCompileCommands(str(resolved_source_path))
    if compile_commands is None:
        raise RuntimeError(f"未在编译数据库中定位到编译单元: {resolved_source_path}")

    commands = list(compile_commands)
    if not commands:
        raise RuntimeError(f"未在编译数据库中定位到编译单元: {resolved_source_path}")

    command = commands[0]
    return CompilationCommand(
        file_path=resolve_compile_command_file_path(command),
        working_directory=Path(str(command.directory)).resolve(),
        parse_args=sanitize_compile_command_arguments(command),
    )
