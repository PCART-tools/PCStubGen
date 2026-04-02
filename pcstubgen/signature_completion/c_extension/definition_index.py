from __future__ import annotations

from collections.abc import Iterable

from clang.cindex import Cursor, CursorKind, LinkageKind, TranslationUnit
from loguru import logger

_INDEXED_DEFINITION_KINDS = {
    CursorKind.FUNCTION_DECL,
    CursorKind.VAR_DECL,
}

_EXPORTABLE_LINKAGES = {
    LinkageKind.EXTERNAL,
    LinkageKind.UNIQUE_EXTERNAL,
}

_DECL_CONTEXT_KINDS = {
    CursorKind.TRANSLATION_UNIT,
    CursorKind.NAMESPACE,
    CursorKind.LINKAGE_SPEC,
}


class DefinitionIndex:
    """为跨 translation unit 的定义查询建立索引。"""

    def __init__(self, translation_units: Iterable[TranslationUnit]) -> None:
        self._usr_to_cursor: dict[str, Cursor] = {}
        for translation_unit in translation_units:
            for cursor in _iter_exportable_definition_candidates(translation_unit.cursor):
                if (
                    not cursor.is_definition()
                    or cursor.linkage not in _EXPORTABLE_LINKAGES
                ):
                    continue

                stable_usr = _get_usr_from_definition_cursor(cursor)
                if stable_usr is None:
                    continue

                existing = self._usr_to_cursor.get(stable_usr)
                if existing is None:
                    self._usr_to_cursor[stable_usr] = cursor
                    continue
                existing_file_name, existing_line, existing_column = (
                    _get_cursor_location(existing)
                )
                file_name, line, column = _get_cursor_location(cursor)
                if (
                    existing_file_name == file_name
                    and existing_line == line
                    and existing_column == column
                ):
                    continue

                logger.warning(
                    "USR 定义冲突, 保留首个定义, usr: {}, first: {}:{}:{}, second: {}:{}:{}",
                    stable_usr,
                    existing_file_name,
                    existing_line,
                    existing_column,
                    file_name,
                    line,
                    column,
                )

    def get_definition(self, cursor: Cursor) -> Cursor | None:
        local_definition = cursor.get_definition()
        if local_definition is not None:
            return local_definition

        stable_usr = _get_usr_from_lookup_cursor(cursor)
        if stable_usr is None:
            return None
        return self._usr_to_cursor.get(stable_usr)


def _iter_exportable_definition_candidates(node: Cursor) -> Iterable[Cursor]:
    for child in node.get_children():
        if child.kind in _INDEXED_DEFINITION_KINDS:
            yield child
            continue
        if child.kind in _DECL_CONTEXT_KINDS:
            yield from _iter_exportable_definition_candidates(child)


def _get_cursor_location(cursor: Cursor) -> tuple[str | None, int, int]:
    location = cursor.location
    file = location.file
    return (
        None if file is None else str(file.name),
        int(location.line),
        int(location.column),
    )


def _get_usr_from_definition_cursor(cursor: Cursor) -> str | None:
    stable_usr = cursor.canonical.get_usr()
    if stable_usr:
        return stable_usr

    stable_usr = cursor.get_usr()
    if stable_usr:
        return stable_usr
    return None


def _get_usr_from_lookup_cursor(cursor: Cursor) -> str | None:
    seen: set[int] = set()
    candidates: list[Cursor] = []

    referenced = cursor.referenced
    if referenced is not None:
        candidates.append(referenced)
        candidates.append(referenced.canonical)

    candidates.append(cursor.canonical)
    candidates.append(cursor)

    for candidate in candidates:
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)

        stable_usr = candidate.get_usr()
        if stable_usr:
            return stable_usr
    return None
