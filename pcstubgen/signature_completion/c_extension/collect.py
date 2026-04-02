from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clang.cindex import Cursor, Index, TranslationUnit
from loguru import logger

from ...checks import check
from .clang.cursor_utils import walk_cursor
from .models import CModule
from .clang import parser as clang_parser
from .modules import collect_modules as module_collection
from .signatures import inference


@dataclass(frozen=True)
class _IndexedDefinition:
    cursor: Cursor
    file_name: str | None
    line: int
    column: int


def _get_stable_usr(cursor: Cursor) -> str | None:
    canonical = getattr(cursor, "canonical", None)
    if canonical is not None:
        getter = getattr(canonical, "get_usr", None)
        if callable(getter):
            stable_usr = getter()
            if stable_usr:
                return stable_usr

    getter = getattr(cursor, "get_usr", None)
    if not callable(getter):
        return None
    stable_usr = getter()
    if stable_usr:
        return stable_usr
    return None


def _is_declaration_definition(cursor: Cursor) -> bool:
    kind = getattr(cursor, "kind", None)
    if kind is None or not kind.is_declaration():
        return False
    return cursor.is_definition()


def _build_definition_resolver(
    translation_units: list[TranslationUnit],
) -> module_collection.DefinitionResolver:
    indexed_definitions: dict[str, _IndexedDefinition] = {}

    for translation_unit in translation_units:
        for cursor in walk_cursor(translation_unit.cursor):
            if not _is_declaration_definition(cursor):
                continue

            stable_usr = _get_stable_usr(cursor)
            if stable_usr is None:
                continue

            location = getattr(cursor, "location", None)
            file = getattr(location, "file", None)
            record = _IndexedDefinition(
                cursor=cursor,
                file_name=None if file is None else str(file.name),
                line=0 if location is None else int(getattr(location, "line", 0)),
                column=0 if location is None else int(getattr(location, "column", 0)),
            )

            existing = indexed_definitions.get(stable_usr)
            if existing is None:
                indexed_definitions[stable_usr] = record
                continue
            if (
                existing.file_name == record.file_name
                and existing.line == record.line
                and existing.column == record.column
            ):
                continue

            logger.warning(
                "USR 定义冲突, 保留首个定义, usr: {}, first: {}:{}:{}, second: {}:{}:{}",
                stable_usr,
                existing.file_name,
                existing.line,
                existing.column,
                record.file_name,
                record.line,
                record.column,
            )

    return module_collection.DefinitionResolver(
        {
            stable_usr: indexed.cursor
            for stable_usr, indexed in indexed_definitions.items()
        }
    )


def collect_modules(
    source_root: Path,
    *,
    include: list[str] = (),
    include_directory: list[Path] = (),
    c_std: str = "c11",
    cpp_std: str = "c++17",
) -> dict[str, CModule]:
    """
    基于 libclang 提取模块级 C 签名。

    该流程从 `PyModuleDef` 变量定义出发，读取 `m_name` / `m_methods`
    还原模块级 `PyMethodDef`，再结合 `PyArg_*` 调用和格式串规则推断
    Python 侧参数信息。
    """
    check(source_root.exists())

    normalized_include_dirs = clang_parser.inject_python_include_directories(include_directory)

    source_files = clang_parser.list_files(source_root)

    index = Index.create()
    translation_units = []
    for file_path in source_files:
        tu = clang_parser.parse(
            index,
            file_path,
            source_root=source_root,
            include=include,
            include_directory=normalized_include_dirs,
            c_std=c_std,
            cpp_std=cpp_std,
        )
        translation_units.append(tu)

    definition_resolver = _build_definition_resolver(translation_units)

    result: dict[str, CModule] = {}
    for tu in translation_units:
        try:
            modules = module_collection.collect_modules_from_translation_unit(
                tu.cursor,
                definition_resolver=definition_resolver,
            )
        except AssertionError:
            logger.exception("处理 translation unit 时触发 AssertionError")
            continue
        for module in modules:
            existing = result.get(module.name)
            if existing is not None:
                logger.warning("模块重复, 丢弃新模块, module: {}", existing.name)
                continue
            result[module.name] = module

    for module in result.values():
        for function in module.functions.values():
            try:
                function.signatures = inference.infer_signature(
                    function.function_cursor
                )
            except Exception:
                logger.exception(
                    "推断 C 函数签名失败, module_name: {}, func_name: {}",
                    module.name,
                    function.ml_name,
                )

    return result
