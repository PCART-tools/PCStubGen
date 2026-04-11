from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from clang.cindex import Cursor, CursorKind, Index, TranslationUnit, TranslationUnitLoadError
from loguru import logger

from ...ir_modules import IRFunction, IRModule, IRSignature
from .address_resolver import (
    get_func_file_location,
)
from .clang import compilation_database as compilation_database_loader
from .clang import translation_unit as translation_unit_loader
from .clang.cursor_utils import source_range_get_text
from .runtime import read_builtin_function_runtime_info
from .signatures import inference

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
        self._index = Index.create()

    def infer_function_signatures(
        self,
        irmodule: IRModule,
        irfunction: IRFunction,
    ) -> tuple[list[IRSignature], str | None]:
        """按函数懒解析 builtin function 的 C 扩展签名。"""
        runtime_info = read_builtin_function_runtime_info(irfunction.runtime_handle)
        location = get_func_file_location(runtime_info.address)
        tu = self.get_translation_unit(location.compilation_unit_path)
        func_cursor = get_function_cursor(
            translation_unit=tu,
            function_name=location.function_name,
            linkage_name=location.linkage_name,
        )
        source_comment = source_range_get_text(func_cursor.extent)
        signatures = inference.infer_signature(
            func_cursor,
            ml_flags=runtime_info.flags,
        )

        if not signatures:
            raise RuntimeError(f"C函数 {irmodule.full_name}.{irfunction.name} 没有可用签名")

        return signatures, source_comment

    def get_translation_unit(self, path: Path) -> TranslationUnit:
        cached = self._translation_units.get(path)
        if cached is not None:
            return cached

        compile_command = compilation_database_loader.get_compile_command(
            self._compilation_database,
            path,
        )
        compile_arguments = list(compile_command.arguments)

        try:
            translation_unit = translation_unit_loader.parse(
                self._index,
                compile_command,
            )
        except TranslationUnitLoadError as ex:
            raise RuntimeError(
                "Parse失败, "
                f"文件路径: {path}, "
                f"解析参数: {' '.join(str(argument) for argument in compile_arguments)}"
            ) from ex

        diagnostics = translation_unit.diagnostics
        if translation_unit_loader.has_error_diagnostics(diagnostics):
            logger.warning(
                "Parse诊断, 文件路径: {}, 诊断: {}",
                path,
                "\n".join(
                    translation_unit_loader.diagnostic_to_str(diagnostic)
                    for diagnostic in diagnostics
                ),
            )

        self._translation_units[path] = translation_unit
        return translation_unit


def get_function_cursor(
        *,
        translation_unit: TranslationUnit,
        function_name: str,
        linkage_name: str | None = None,
) -> Cursor:
    for cursor in _iter_function_definition_candidates(translation_unit.cursor):
        if cursor.spelling == function_name and cursor.mangled_name == function_name:
            return cursor

    raise RuntimeError(
        "未在 translation unit 中定位到函数定义, "
        f"translation_unit: {translation_unit.cursor.location}, "
        f"function_name: {function_name}, "
        f"linkage_name: {linkage_name}"
    )

def _iter_function_definition_candidates(node: Cursor) -> Iterator[Cursor]:
    """仅在声明上下文中递归收集函数定义节点。"""
    for child in node.get_children():
        if child.kind == CursorKind.FUNCTION_DECL:
            if child.is_definition():
                yield child
            continue
        if child.kind in _FUNCTION_DECL_CONTEXT_KINDS:
            yield from _iter_function_definition_candidates(child)
