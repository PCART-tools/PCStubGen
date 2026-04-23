from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
import functools
from pathlib import Path
import subprocess

import clang
from clang.cindex import (
    CompilationDatabase,
    CompileCommand,
    Cursor,
    CursorKind,
    Diagnostic,
    Index,
    TranslationUnit,
)
from loguru import logger

from .libclang_wrap import parse_translation_unit_full_argv

FUNCTION_DECL_CONTEXT_KINDS = {
    CursorKind.TRANSLATION_UNIT,
    CursorKind.NAMESPACE,
    CursorKind.LINKAGE_SPEC,
}


@dataclass(slots=True)
class _FunctionCursorIndex:
    """保存单个源码文件中可定位函数定义的索引。"""

    cursor_by_linkage_name: dict[str, Cursor]
    cursor_by_function_name: dict[str, Cursor]


@dataclass(slots=True)
class _ParsedSource:
    """保存 translation unit 及其派生索引。"""

    translation_unit: TranslationUnit
    function_index: _FunctionCursorIndex


class ClangFunctionLocator:
    def __init__(
        self,
        compilation_database: Path,
    ) -> None:
        """创建持有编译数据库、Index 与函数定位缓存的 libclang 定位器。"""
        self._compilation_database = _load_compilation_database(compilation_database)
        self._index = Index.create()
        self._resource_dir = try_get_clang_resource_dir()
        self._get_parsed_source = functools.lru_cache(maxsize=8)(self._build_parsed_source)

    def get_function_cursor(
        self,
        path: Path,
        function_name: str,
        linkage_name: str | None,
    ) -> Cursor:
        """按源码路径与函数身份定位函数定义 cursor。"""
        source_path = path.resolve()
        parsed_source = self._get_parsed_source(source_path)
        function_index = parsed_source.function_index

        if linkage_name is not None:
            matched = function_index.cursor_by_linkage_name.get(linkage_name)
            if matched is not None:
                return matched
        else:
            matched = function_index.cursor_by_function_name.get(function_name)
            if matched is not None:
                return matched

        translation_unit = parsed_source.translation_unit
        raise RuntimeError(
            "未在 translation unit 中定位到函数定义, "
            f"translation_unit: {translation_unit.cursor.location}, "
            f"function_name: {function_name}, "
            f"linkage_name: {linkage_name}"
        )

    def _build_parsed_source(self, source_path: Path) -> _ParsedSource:
        """按源码路径解析 translation unit 并构建函数索引。"""
        compile_command = self._get_compile_command(source_path)
        compile_arguments = list(compile_command.arguments)

        try:
            translation_unit = self._parse_translation_unit(compile_command)
        except clang.cindex.TranslationUnitLoadError as ex:
            raise RuntimeError(
                "Parse失败, "
                f"文件路径: {source_path}, "
                f"解析参数: {' '.join(str(argument) for argument in compile_arguments)}"
            ) from ex

        diagnostics = list(translation_unit.diagnostics)
        if has_error_diagnostics(diagnostics):
            logger.warning(
                "Parse诊断, 文件路径: {}, 诊断: {}",
                source_path,
                "\n".join(diagnostic_to_str(diagnostic) for diagnostic in diagnostics),
            )

        function_index = _build_function_cursor_index(translation_unit)
        return _ParsedSource(
            translation_unit=translation_unit,
            function_index=function_index,
        )

    def _get_compile_command(self, source_path: Path) -> CompileCommand:
        """按源码绝对路径查询首条编译命令。"""
        compile_commands = self._compilation_database.getCompileCommands(str(source_path))
        if compile_commands is None:
            raise RuntimeError(f"未在编译数据库中定位到编译单元: {source_path}")

        commands = list(compile_commands)
        if not commands:
            raise RuntimeError(f"未在编译数据库中定位到编译单元: {source_path}")

        return commands[0]

    def _parse_translation_unit(self, compile_command: CompileCommand) -> TranslationUnit:
        """在编译命令工作目录下用完整 argv 解析 translation unit。"""
        arguments = list(compile_command.arguments)
        if self._resource_dir is not None:
            arguments.extend(["-resource-dir", str(self._resource_dir)])

        with contextlib.chdir(Path(str(compile_command.directory)).resolve()):
            return parse_translation_unit_full_argv(self._index, arguments)


def _build_function_cursor_index(translation_unit: TranslationUnit) -> _FunctionCursorIndex:
    """从 translation unit 中收集函数定义索引。"""
    cursor_by_linkage_name: dict[str, Cursor] = {}
    cursor_by_function_name: dict[str, Cursor] = {}

    for cursor in _iter_function_definition_candidates(translation_unit.cursor):
        if cursor.mangled_name:
            cursor_by_linkage_name[cursor.mangled_name] = cursor
        cursor_by_function_name.setdefault(cursor.spelling, cursor)

    return _FunctionCursorIndex(
        cursor_by_linkage_name=cursor_by_linkage_name,
        cursor_by_function_name=cursor_by_function_name,
    )


def _iter_function_definition_candidates(node: Cursor) -> Iterator[Cursor]:
    """仅在函数声明上下文中递归收集函数定义节点。"""
    for child in node.get_children():
        if child.kind == CursorKind.FUNCTION_DECL:
            if child.is_definition():
                yield child
            continue
        if child.kind in FUNCTION_DECL_CONTEXT_KINDS:
            yield from _iter_function_definition_candidates(child)


def try_get_clang_resource_dir() -> Path | None:
    """尝试解析 clang resource dir，失败时返回 None。"""
    try:
        resource_dir_text = subprocess.check_output(
            ["clang", "-print-resource-dir"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as ex:
        logger.warning("clang resource dir 获取失败: {!r}", ex)
        return None

    if not resource_dir_text:
        return None

    return Path(resource_dir_text)


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


def _load_compilation_database(compilation_database: Path) -> CompilationDatabase:
    """从 compile_commands.json 所在目录加载编译数据库。"""
    validated_path = validate_compilation_database_path(compilation_database)
    try:
        return CompilationDatabase.fromDirectory(str(validated_path.parent))
    except Exception as ex:
        raise RuntimeError(f"编译数据库加载失败: {validated_path}") from ex


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
    """将单条 libclang diagnostic 格式化为稳定的一行文本。"""
    severity = diagnostic_severity_to_str(diagnostic.severity)
    location = diagnostic.location
    message = diagnostic.spelling
    return f"[{severity}] {location}: {message}"


def has_error_diagnostics(diagnostics: list[Diagnostic]) -> bool:
    """判断 diagnostics 中是否包含 Error/Fatal 级别。"""
    for diagnostic in diagnostics:
        if diagnostic.severity >= clang.cindex.Diagnostic.Error:
            return True
    return False
