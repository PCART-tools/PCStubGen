from __future__ import annotations

import contextlib
import dataclasses
import functools
from pathlib import Path
import subprocess

import clang
from clang.cindex import CompilationDatabase, Diagnostic, Index, TranslationUnit

from .source_suffixes import NATIVE_SOURCE_SUFFIXES

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


@functools.cache
def detect_clang_resource_dir() -> str | None:
    """调用系统 clang 自动探测 resource dir。"""
    completed_process = subprocess.run(
        ["clang", "-print-resource-dir"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed_process.returncode != 0:
        return None

    resource_dir = completed_process.stdout.strip()
    if not resource_dir:
        return None
    return resource_dir


def diagnostic_severity_to_str(severity: int) -> str:
    """把 libclang severity 严重程度数值转换成可读名称。"""
    match severity:
        case clang.cindex.Diagnostic.Ignored:
            return "IGNORED"
        case clang.cindex.Diagnostic.Note:
            return "NOTE"
        case clang.cindex.Diagnostic.Warning:
            return "WARNING"
        case clang.cindex.Diagnostic.Error:
            return "ERROR"
        case clang.cindex.Diagnostic.Fatal:
            return "FATAL"
        case _:
            return f"SEVERITY_{severity}"


def diagnostic_to_str(diagnostic: Diagnostic) -> str:
    """将单条 clang diagnostic 格式化为稳定的一行文本。"""
    severity = diagnostic_severity_to_str(diagnostic.severity)
    location = diagnostic.location
    message = diagnostic.spelling
    return f"[{severity}] {location}: {message}"


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


def list_compilation_commands(compilation_database: Path) -> list[CompilationCommand]:
    """按编译数据库顺序列出首条唯一源码编译命令。"""
    database = load_compilation_database(compilation_database)
    result: list[CompilationCommand] = []
    seen_file_paths: set[Path] = set()
    all_compile_commands = database.getAllCompileCommands()
    if all_compile_commands is None:
        return result
    for command in all_compile_commands:
        file_path = resolve_compile_command_file_path(command)
        if file_path in seen_file_paths:
            continue
        seen_file_paths.add(file_path)
        result.append(
            CompilationCommand(
                file_path=file_path,
                working_directory=Path(str(command.directory)).resolve(),
                parse_args=sanitize_compile_command_arguments(command),
            )
        )
    return result


def has_error_diagnostics(diagnostics: list[Diagnostic]) -> bool:
    """判断 diagnostics 中是否包含 Error/Fatal 级别。"""
    for diagnostic in diagnostics:
        if diagnostic.severity >= clang.cindex.Diagnostic.Error:
            return True
    return False


def build_effective_parse_args(compilation_command: CompilationCommand) -> list[str]:
    """构造实际传给 libclang parse 的参数列表。"""
    effective_parse_args = list(compilation_command.parse_args)
    resource_dir = detect_clang_resource_dir()
    if resource_dir is not None:
        effective_parse_args.extend(["-resource-dir", resource_dir])
    return effective_parse_args


def parse(
    index: Index,
    compilation_command: CompilationCommand,
    *,
    effective_parse_args: list[str] | None = None,
) -> TranslationUnit:
    """解析单个编译数据库条目为 clang translation unit。"""
    if effective_parse_args is None:
        effective_parse_args = build_effective_parse_args(compilation_command)

    with contextlib.chdir(compilation_command.working_directory):
        translation_unit = index.parse(
            str(compilation_command.file_path),
            args=effective_parse_args,
        )
    return translation_unit
