from __future__ import annotations

import clang.cindex
import pytest

from pcstubgen.signature_completion.c_extension import (
    definition_index as definition_index_module,
)
from pcstubgen.signature_completion.c_extension.definition_index import DefinitionIndex
from tests._c_extension_test_support import *


def test_definition_index_returns_local_definition_before_usr_lookup() -> None:
    definition = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        usr="usr:local",
        is_definition=True,
    )
    cursor = _token_identifier_node(
        "local_func",
        referenced=definition,
        canonical=definition,
    )

    definition_index = _definition_index()

    assert definition_index.get_definition(cursor) is definition


def test_definition_index_falls_back_to_referenced_canonical_usr() -> None:
    indexed_definition = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        location=_FakeCursorLocation("module.c", line=12, column=8),
    )
    referenced_canonical = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        usr="usr:canonical",
    )
    referenced = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        usr="",
        canonical=referenced_canonical,
    )
    cursor = _token_identifier_node(
        "target_func",
        referenced=referenced,
        usr="",
    )

    definition_index = _definition_index({"usr:canonical": indexed_definition})

    assert definition_index.get_definition(cursor) is indexed_definition


def test_definition_index_keeps_first_duplicate_definition_and_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged_messages: list[str] = []

    def fake_warning(message: str, *args: object) -> None:
        logged_messages.append(message.format(*args))

    monkeypatch.setattr(definition_index_module.logger, "warning", fake_warning)

    first_definition = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        usr="usr:dup",
        is_definition=True,
        location=_FakeCursorLocation("first.c", line=3, column=4),
    )
    second_definition = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        usr="usr:dup",
        is_definition=True,
        location=_FakeCursorLocation("second.c", line=7, column=9),
    )
    translation_unit = _FakeTranslationUnit(
        diagnostics=[],
        cursor=_FakeNode(
            kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
            children=[first_definition, second_definition],
        ),
    )
    cursor = _token_identifier_node("dup_func", usr="usr:dup")

    definition_index = DefinitionIndex([translation_unit])

    assert definition_index.get_definition(cursor) is first_definition
    assert logged_messages == [
        "USR 定义冲突, 保留首个定义, usr: usr:dup, first: first.c:3:4, second: second.c:7:9"
    ]


def test_definition_index_ignores_empty_usr_for_index_and_lookup() -> None:
    translation_unit = _FakeTranslationUnit(
        diagnostics=[],
        cursor=_FakeNode(
            kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
            children=[
                _FakeNode(
                    kind=clang.cindex.CursorKind.FUNCTION_DECL,
                    usr="",
                    is_definition=True,
                )
            ],
        ),
    )
    cursor = _token_identifier_node("missing_usr", usr="")

    definition_index = DefinitionIndex([translation_unit])

    assert definition_index.get_definition(cursor) is None
