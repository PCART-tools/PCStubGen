from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

from clang.cindex import Cursor, CursorKind, Index, TranslationUnit, TranslationUnitLoadError
from loguru import logger

from ...ir_modules import IRFunction, IRModule, IRModuleType, IRSignature
from .address_resolver import SymbolizedAddressLocation, resolve_symbolized_address
from .clang import compilation_database as compilation_database_loader
from .clang import translation_unit as translation_unit_loader
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
        compilation_database: Path,
    ) -> None:
        self._compilation_database = compilation_database_loader.load_compilation_database(
            compilation_database
        )
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
        signatures = inference.infer_signature(
            function_cursor,
            ml_flags=runtime_method.flags,
        )

        if not signatures:
            raise RuntimeError(f"C函数 {irmodule.full_name}.{irfunction.name} 没有可用签名")

        return signatures, source_comment

    def _resolve_function_cursor(
        self,
        *,
        location: SymbolizedAddressLocation,
    ) -> Cursor:
        """按需 parse 已定位到的源码文件，并找到对应的函数 cursor。"""
        compile_command = compilation_database_loader.resolve_compile_command(
            self._compilation_database,
            location.compilation_unit_path,
        )

        translation_unit = self._load_translation_unit(compile_command)

        matched = self._find_function_cursor(
            translation_unit=translation_unit,
            symbol_name=location.function_name,
            linkage_name=location.linkage_name,
        )
        if matched is not None:
            return matched

        raise RuntimeError(
            "未在 translation unit 中定位到函数定义, "
            f"source_path: {compile_command.filename}, "
            f"symbol_name: {location.function_name}, "
            f"linkage_name: {location.linkage_name}"
        )

    def _load_translation_unit(
        self,
        compile_command: compilation_database_loader.MyCompileCommand,
    ) -> TranslationUnit:
        file_path = compile_command.filename
        cached = self._translation_units.get(file_path)
        if cached is not None:
            return cached

        if self._index is None:
            self._index = Index.create()
        index = self._index
        if index is None:
            raise RuntimeError("libclang Index 初始化失败。")

        try:
            translation_unit = translation_unit_loader.parse(
                index,
                compile_command,
            )
        except TranslationUnitLoadError as ex:
            raise RuntimeError(
                "按需Parse失败, "
                f"文件路径: {file_path}, "
                f"工作目录: {compile_command.directory}, "
                f"解析参数: {' '.join(compile_command.arguments)}"
            ) from ex

        diagnostics = translation_unit.diagnostics
        if translation_unit_loader.has_error_diagnostics(diagnostics):
            logger.warning(
                "按需Parse诊断, 文件路径: {}, 诊断: {}",
                file_path,
                "\n".join(
                    translation_unit_loader.diagnostic_to_str(diagnostic)
                    for diagnostic in diagnostics
                ),
            )

        self._translation_units[file_path] = translation_unit
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
