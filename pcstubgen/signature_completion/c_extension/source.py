from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

from clang.cindex import Cursor, CursorKind, Index, TranslationUnit, TranslationUnitLoadError
from loguru import logger

from ...ir_modules import IRFunction, IRModule, IRModuleType, IRSignature
from .address_resolver import SymbolizedAddressLocation, resolve_symbolized_address
from .clang import parser as clang_parser
from .clang.cursor_utils import source_range_get_text
from .runtime import resolve_runtime_pymethoddef
from .signatures import inference

ResolvedCExtensionFunction: TypeAlias = tuple[list[IRSignature], str | None]

_FUNCTION_DECL_CONTEXT_KINDS = {
    CursorKind.TRANSLATION_UNIT,
    CursorKind.NAMESPACE,
    CursorKind.LINKAGE_SPEC,
}


class CExtensionSource:
    def __init__(
        self,
        *,
        compilation_database: Path | None = None,
    ) -> None:
        self._compilation_database = compilation_database
        self._compilation_commands: list[clang_parser.CompilationCommand] | None = None
        self._translation_units: dict[Path, TranslationUnit] = {}
        self._index: Index | None = None

    def resolve_function(
        self,
        irmodule: IRModule,
        irfunction: IRFunction,
    ) -> ResolvedCExtensionFunction:
        """按函数懒解析 C 扩展签名。"""
        if irmodule.module_type is not IRModuleType.EXTENSION:
            raise RuntimeError(f"模块 {irmodule.full_name} 不是扩展模块。")

        runtime_method = resolve_runtime_pymethoddef(irfunction.runtime_handle)
        location = resolve_symbolized_address(runtime_method.method_address)
        function_cursor = self._resolve_function_cursor(location=location)
        source_comment = self._get_source_comment(function_cursor)

        if function_cursor is not None:
            try:
                signatures = inference.infer_signature(
                    function_cursor,
                    ml_flags=runtime_method.flags,
                )
            except RuntimeError as ex:
                logger.info(
                    "AST推断失败，回退到最小签名, module: {}, func: {}, reason: {}",
                    irmodule.full_name,
                    irfunction.name,
                    ex,
                )
                signatures = inference.infer_minimal_signatures(runtime_method.flags)
        else:
            signatures = inference.infer_minimal_signatures(runtime_method.flags)

        if not signatures:
            raise RuntimeError(f"C函数 {irmodule.full_name}.{irfunction.name} 没有可用签名")

        return signatures, source_comment

    def _resolve_function_cursor(
        self,
        *,
        location: SymbolizedAddressLocation,
    ) -> Cursor | None:
        """按需 parse 已定位到的源码文件，并找到对应的函数 cursor。"""
        if self._compilation_database is None:
            return None

        compilation_command = self._match_compilation_command(location.compilation_unit_path)
        if compilation_command is None:
            logger.info(
                "未在编译数据库中定位到编译单元, source_path: {}, symbol_name: {}",
                location.compilation_unit_path,
                location.function_name,
            )
            return None

        translation_unit = self._load_translation_unit(compilation_command)
        if translation_unit is None:
            return None

        matched = self._find_function_cursor(
            translation_unit=translation_unit,
            symbol_name=location.function_name,
            linkage_name=location.linkage_name,
        )
        if matched is not None:
            return matched

        logger.info(
            "未在 translation unit 中定位到函数定义, source_path: {}, symbol_name: {}, linkage_name: {}",
            compilation_command.file_path,
            location.function_name,
            location.linkage_name,
        )
        return None

    def _match_compilation_command(
        self,
        source_path: Path,
    ) -> clang_parser.CompilationCommand | None:
        commands = self._get_compilation_commands()
        resolved_source_path = source_path.resolve()
        for command in commands:
            if command.file_path == resolved_source_path:
                return command
        return None

    def _get_compilation_commands(self) -> list[clang_parser.CompilationCommand]:
        if self._compilation_commands is None:
            compilation_database = self._compilation_database
            if compilation_database is None:
                raise RuntimeError("缺少 compile_commands.json，无法加载编译数据库。")
            self._compilation_commands = clang_parser.list_compilation_commands(
                compilation_database
            )
        compilation_commands = self._compilation_commands
        if compilation_commands is None:
            raise RuntimeError("编译数据库加载失败。")
        return compilation_commands

    def _load_translation_unit(
        self,
        compilation_command: clang_parser.CompilationCommand,
    ) -> TranslationUnit | None:
        cached = self._translation_units.get(compilation_command.file_path)
        if cached is not None:
            return cached

        if self._index is None:
            self._index = Index.create()
        index = self._index
        if index is None:
            raise RuntimeError("libclang Index 初始化失败。")

        effective_parse_args = clang_parser.build_effective_parse_args(compilation_command)
        try:
            translation_unit = clang_parser.parse(
                index,
                compilation_command,
                effective_parse_args=effective_parse_args,
            )
        except TranslationUnitLoadError:
            logger.warning(
                "按需Parse失败, 文件路径: {}, 工作目录: {}, 解析参数: {}",
                compilation_command.file_path,
                compilation_command.working_directory,
                " ".join(effective_parse_args),
            )
            return None

        diagnostics = translation_unit.diagnostics
        if clang_parser.has_error_diagnostics(diagnostics):
            logger.warning(
                "按需Parse诊断, 文件路径: {}, 诊断: {}",
                compilation_command.file_path,
                "\n".join(
                    clang_parser.diagnostic_to_str(diagnostic)
                    for diagnostic in diagnostics
                ),
            )

        self._translation_units[compilation_command.file_path] = translation_unit
        return translation_unit

    @staticmethod
    def _find_function_cursor(
        *,
        translation_unit: TranslationUnit,
        symbol_name: str,
        linkage_name: str | None = None,
    ) -> Cursor | None:
        for cursor in _iter_function_definition_candidates(translation_unit.cursor):
            if linkage_name is not None:
                if cursor.mangled_name == linkage_name:
                    return cursor
                continue

            if cursor.spelling == symbol_name:
                return cursor

        return None

    @staticmethod
    def _get_source_comment(function_cursor: Cursor | None) -> str | None:
        if function_cursor is None or function_cursor.extent is None:
            return None
        source_text = source_range_get_text(function_cursor.extent)
        if not source_text:
            return None
        return source_text


def _iter_function_definition_candidates(node: Cursor) -> Iterator[Cursor]:
    """仅在声明上下文中递归收集函数定义节点。"""
    for child in node.get_children():
        if child.kind == CursorKind.FUNCTION_DECL:
            if child.is_definition():
                yield child
            continue
        if child.kind in _FUNCTION_DECL_CONTEXT_KINDS:
            yield from _iter_function_definition_candidates(child)
