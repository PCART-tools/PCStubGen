from __future__ import annotations

import clang.cindex
import pytest
from clang.cindex import LinkageKind

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


def test_definition_index_indexes_external_function_definition() -> None:
    definition = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        usr="usr:external_func",
        is_definition=True,
        linkage=LinkageKind.EXTERNAL,
        location=_FakeCursorLocation("module.c", line=10, column=2),
    )
    cursor = _token_identifier_node("external_func", usr="usr:external_func")

    definition_index = DefinitionIndex([
        _FakeTranslationUnit(
            diagnostics=[],
            cursor=_FakeNode(
                kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
                children=[definition],
            ),
        )
    ])

    assert definition_index.get_definition(cursor) is definition


def test_definition_index_indexes_external_variable_definition() -> None:
    definition = _FakeNode(
        kind=clang.cindex.CursorKind.VAR_DECL,
        usr="usr:methods",
        is_definition=True,
        linkage=LinkageKind.EXTERNAL,
        location=_FakeCursorLocation("module.c", line=12, column=4),
    )
    cursor = _token_identifier_node("Methods", usr="usr:methods")

    definition_index = DefinitionIndex([
        _FakeTranslationUnit(
            diagnostics=[],
            cursor=_FakeNode(
                kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
                children=[definition],
            ),
        )
    ])

    assert definition_index.get_definition(cursor) is definition


def test_definition_index_ignores_internal_function_definition() -> None:
    definition = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        usr="usr:internal_func",
        is_definition=True,
        linkage=LinkageKind.INTERNAL,
    )
    cursor = _token_identifier_node("internal_func", usr="usr:internal_func")

    definition_index = DefinitionIndex([
        _FakeTranslationUnit(
            diagnostics=[],
            cursor=_FakeNode(
                kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
                children=[definition],
            ),
        )
    ])

    assert definition_index.get_definition(cursor) is None


def test_definition_index_ignores_internal_variable_definition() -> None:
    definition = _FakeNode(
        kind=clang.cindex.CursorKind.VAR_DECL,
        usr="usr:internal_methods",
        is_definition=True,
        linkage=LinkageKind.INTERNAL,
    )
    cursor = _token_identifier_node("Methods", usr="usr:internal_methods")

    definition_index = DefinitionIndex([
        _FakeTranslationUnit(
            diagnostics=[],
            cursor=_FakeNode(
                kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
                children=[definition],
            ),
        )
    ])

    assert definition_index.get_definition(cursor) is None


def test_definition_index_ignores_local_variable_definition_in_function_body() -> None:
    local_definition = _FakeNode(
        kind=clang.cindex.CursorKind.VAR_DECL,
        usr="usr:local_methods",
        is_definition=True,
        linkage=LinkageKind.NO_LINKAGE,
    )
    cursor = _token_identifier_node("Methods", usr="usr:local_methods")

    definition_index = DefinitionIndex([
        _FakeTranslationUnit(
            diagnostics=[],
            cursor=_FakeNode(
                kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
                children=[
                    _FakeNode(
                        kind=clang.cindex.CursorKind.FUNCTION_DECL,
                        spelling="PyInit_mod",
                        is_definition=True,
                        linkage=LinkageKind.EXTERNAL,
                        children=[local_definition],
                    )
                ],
            ),
        )
    ])

    assert definition_index.get_definition(cursor) is None


def test_definition_index_does_not_recurse_into_namespace() -> None:
    definition = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        usr="usr:namespace_func",
        is_definition=True,
        linkage=LinkageKind.EXTERNAL,
        location=_FakeCursorLocation("module.cpp", line=6, column=3),
    )
    cursor = _token_identifier_node("namespace_func", usr="usr:namespace_func")

    definition_index = DefinitionIndex([
        _FakeTranslationUnit(
            diagnostics=[],
            cursor=_FakeNode(
                kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
                children=[
                    _FakeNode(
                        kind=clang.cindex.CursorKind.NAMESPACE,
                        spelling="ns",
                        children=[definition],
                    )
                ],
            ),
        )
    ])

    assert definition_index.get_definition(cursor) is None


def test_definition_index_does_not_recurse_into_linkage_spec() -> None:
    definition = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        usr="usr:extern_c_func",
        is_definition=True,
        linkage=LinkageKind.EXTERNAL,
        location=_FakeCursorLocation("module.cpp", line=8, column=5),
    )
    cursor = _token_identifier_node("extern_c_func", usr="usr:extern_c_func")

    definition_index = DefinitionIndex([
        _FakeTranslationUnit(
            diagnostics=[],
            cursor=_FakeNode(
                kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
                children=[
                    _FakeNode(
                        kind=clang.cindex.CursorKind.LINKAGE_SPEC,
                        children=[definition],
                    )
                ],
            ),
        )
    ])

    assert definition_index.get_definition(cursor) is None
