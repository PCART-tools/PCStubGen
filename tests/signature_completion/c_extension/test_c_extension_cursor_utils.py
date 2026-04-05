from __future__ import annotations

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


def test_extract_cursor_source_text_reads_text_from_extent(tmp_path: Path) -> None:
    source = tmp_path / "extent_text.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O!\", (&PyUnicode_Type), &value);",
            ]
        ),
        encoding="utf-8",
    )

    extracted = cursor_utils_module.source_range_get_text(
        _extent_for_source_snippet(source, "(&PyUnicode_Type)")
    )

    assert extracted == "(&PyUnicode_Type)"


def test_extract_cursor_source_text_returns_none_when_extent_start_file_is_missing() -> None:
    extracted = cursor_utils_module.source_range_get_text(
        _FakeSourceRange(
            _FakeCursorLocation(None, 0),
            _FakeCursorLocation("extent_text.c", 1),
        )
    )

    assert extracted is None


def test_extract_cursor_source_text_returns_none_for_cross_file_extent(tmp_path: Path) -> None:
    first = tmp_path / "first.c"
    second = tmp_path / "second.c"
    first.write_text("abc", encoding="utf-8")
    second.write_text("def", encoding="utf-8")

    extracted = cursor_utils_module.source_range_get_text(
        _FakeSourceRange(
            _FakeCursorLocation(str(first), 0),
            _FakeCursorLocation(str(second), 1),
        )
    )

    assert extracted is None


def test_extract_cursor_source_text_returns_none_when_file_read_fails(tmp_path: Path) -> None:
    missing = tmp_path / "missing_extent_text.c"

    extracted = cursor_utils_module.source_range_get_text(
        _FakeSourceRange(
            _FakeCursorLocation(str(missing), 0),
            _FakeCursorLocation(str(missing), 1),
        )
    )

    assert extracted is None

