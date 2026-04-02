from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from clang.cindex import Cursor, TranslationUnit
from loguru import logger

from .clang.cursor_utils import walk_cursor


@dataclass(frozen=True)
class _IndexedDefinition:
    cursor: Cursor
    file_name: str | None
    line: int
    column: int


class DefinitionIndex:
    """为跨 translation unit 的定义查询建立索引。"""

    def __init__(self, translation_units: Iterable[TranslationUnit]) -> None:
        self._definitions_by_usr: dict[str, Cursor] = {}

        indexed_definitions: dict[str, _IndexedDefinition] = {}
        for translation_unit in translation_units:
            for cursor in walk_cursor(translation_unit.cursor):
                if not cursor.is_definition():
                    continue

                stable_usr = _get_usr_from_definition_cursor(cursor)
                if stable_usr is None:
                    continue

                location = cursor.location
                file = location.file
                record = _IndexedDefinition(
                    cursor=cursor,
                    file_name=None if file is None else str(file.name),
                    line=int(location.line),
                    column=int(location.column),
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

        self._definitions_by_usr = {
            stable_usr: indexed.cursor
            for stable_usr, indexed in indexed_definitions.items()
        }

    def get_definition(self, cursor: Cursor) -> Cursor | None:
        local_definition = cursor.get_definition()
        if local_definition is not None:
            return local_definition

        stable_usr = _get_usr_from_lookup_cursor(cursor)
        if stable_usr is None:
            return None
        return self._definitions_by_usr.get(stable_usr)


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
