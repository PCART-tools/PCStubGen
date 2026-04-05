from __future__ import annotations

import clang.cindex
import pytest
from clang.cindex import LinkageKind

from pcstubgen.signature_completion.c_extension import (
    definition_index as definition_index_module,
)
from pcstubgen.signature_completion.c_extension.definition_index import DefinitionIndex
from tests._c_extension_test_support import (
    AnyType,
    CArgument,
    CExtensionSource,
    CFunction,
    CModule,
    CPP_SOURCE_SUFFIXES,
    CSignature,
    CSignatureExtractor,
    CSignatureResolver,
    DefinitionIndex,
    ExtractedArgument,
    ExtractedFunction,
    ExtractedModule,
    ExtractedSignature,
    IRArgument,
    IRArgumentKind,
    IRClass,
    IRFunction,
    IRMethod,
    IRModule,
    IRModuleType,
    IRSignature,
    Iterable,
    LinkageKind,
    ListType,
    METH_KEYWORDS,
    METH_NOARGS,
    METH_O,
    METH_VARARGS,
    NATIVE_SOURCE_SUFFIXES,
    Path,
    QualifiedName,
    RawType,
    SignatureCompleter,
    SignatureCompletionResult,
    SimpleNamespace,
    StubGenerationOptions,
    TupleType,
    Type,
    UnionType,
    _FakeClangWithDiagnostics,
    _FakeCursorFile,
    _FakeCursorLocation,
    _FakeDiagnostic,
    _FakeDiagnosticFile,
    _FakeDiagnosticLocation,
    _FakeDiagnosticType,
    _FakeExtractor,
    _FakeIndex,
    _FakeNode,
    _FakeSourceRange,
    _FakeToken,
    _FakeTranslationUnit,
    _SequentialIndex,
    _address_of,
    _arg,
    _build_definition_translation_unit,
    _call_expr,
    _collect_PyMethodDef_INIT_LIST_EXPR,
    _collect_PyMethodDef_INIT_LIST_EXPR_impl,
    _collect_method_table,
    _collect_method_table_impl,
    _conditional_expr,
    _definition_index,
    _designated_initializer,
    _extent_for_source_snippet,
    _extract_PyMethodDef_INIT_LIST_EXPR,
    _extract_method_table,
    _fake_function_cursor,
    _fake_function_cursor_with_children,
    _float_literal,
    _get_packaged_libclang_path,
    _gnu_null_literal,
    _has_include_directory_arg,
    _has_std_arg,
    _identifier_node,
    _init_list,
    _int_literal,
    _kwlist_decl,
    _macro_expr,
    _ml_flags_identifier_field,
    _ml_meth_cast_field,
    _ml_meth_field,
    _ml_name_field,
    _module_fixture,
    _null_ptr_literal,
    _patch_c_signature_extractor,
    _patch_fake_eval_int,
    _patch_raising_c_signature_extractor,
    _resolve_INIT_LIST_EXPR,
    _return_stmt,
    _signature,
    _signed_numeric_literal,
    _string_literal,
    _token_identifier_node,
    _type_object_decl,
    _unknown_function,
    _var_decl,
    _wrap,
    _write_compilation_database,
    annotations,
    c_extension_collect_module,
    c_extension_source_module,
    c_signature_extraction_module,
    cast,
    clang,
    collect_modules,
    cursor_utils_module,
    extract_c_signature_modules,
    json,
    module_collection_module,
    module_table_module,
    pytest,
    resolve_docstring_signatures,
    signature_rules_module,
    translation_unit_module,
)


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


def test_definition_index_indexes_definition_in_namespace() -> None:
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

    assert definition_index.get_definition(cursor) is definition


def test_definition_index_indexes_definition_in_linkage_spec() -> None:
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

    assert definition_index.get_definition(cursor) is definition
