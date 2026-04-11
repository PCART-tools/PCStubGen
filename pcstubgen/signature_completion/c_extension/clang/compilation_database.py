from __future__ import annotations

from pathlib import Path

from clang.cindex import CompilationDatabase, CompileCommand


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


def get_compile_command(
    database: CompilationDatabase,
    source_path: Path,
) -> CompileCommand:
    """按源码绝对路径查询首条编译命令。"""
    absolute_source_path = source_path.resolve()
    compile_commands = database.getCompileCommands(str(absolute_source_path))
    if compile_commands is None:
        raise RuntimeError(f"未在编译数据库中定位到编译单元: {absolute_source_path}")

    commands = list(compile_commands)
    if not commands:
        raise RuntimeError(f"未在编译数据库中定位到编译单元: {absolute_source_path}")

    return commands[0]
