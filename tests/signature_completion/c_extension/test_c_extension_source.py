from __future__ import annotations

from pathlib import Path

import clang.cindex
import pytest

from pcstubgen.ir_modules import IRFunction, IRModule, IRSignature, QualifiedName
from pcstubgen.signature_completion.c_extension.address_resolver import SymbolizedAddressLocation
from pcstubgen.signature_completion.c_extension.method_flags import (
    METH_FASTCALL,
    METH_KEYWORDS,
    METH_O,
    METH_VARARGS,
)
from pcstubgen.signature_completion.c_extension.runtime import RuntimePyMethodDef
from pcstubgen.signature_completion.c_extension.source import CExtensionSource
from pcstubgen.types import RawType
from tests._c_extension_test_support import _FakeNode, _arg, _extent_for_source_snippet


def _make_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> CExtensionSource:
    compilation_database = tmp_path / "compile_commands.json"
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.compilation_database_loader.load_compilation_database",
        lambda path: object(),
    )
    return CExtensionSource(compilation_database=compilation_database)


def _symbolized_location(
    *,
    compilation_unit_path: Path,
    function_name: str = "foo_impl",
    linkage_name: str | None = None,
) -> SymbolizedAddressLocation:
    return SymbolizedAddressLocation(
        compilation_unit_path=compilation_unit_path,
        function_name=function_name,
        linkage_name=linkage_name,
    )


def test_c_extension_source_prefers_ast_inference_and_preserves_source_comment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "foo_impl.c"
    snippet = "\n".join(
        [
            "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
            "    return (PyObject*)0;",
            "}",
        ]
    )
    source_path.write_text(snippet, encoding="utf-8", newline="\n")
    function_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo_impl",
        extent=_extent_for_source_snippet(source_path, snippet),
    )

    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_runtime_pymethoddef",
        lambda handle: RuntimePyMethodDef(method_address=0x1234, flags=METH_VARARGS),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_symbolized_address",
        lambda address: _symbolized_location(
            compilation_unit_path=source_path,
            function_name="foo_impl",
        ),
    )
    monkeypatch.setattr(
        CExtensionSource,
        "_resolve_function_cursor",
        lambda self, **kwargs: function_cursor,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.inference.infer_signature",
        lambda function_cursor, *, ml_flags=0: [
            IRSignature(
                args=[_arg("value", "int")],
                return_type=RawType("bool"),
            )
        ],
    )

    source = _make_source(monkeypatch, tmp_path)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[IRFunction(name="foo", runtime_handle=object())],
    )

    signatures, source_comment = source.resolve_function(module, module.functions[0])

    assert signatures[0].args[0].name == "value"
    assert signatures[0].args[0].type is not None
    assert signatures[0].args[0].type.render() == "int"
    assert signatures[0].return_type is not None
    assert signatures[0].return_type.render() == "bool"
    assert source_comment == snippet


@pytest.mark.parametrize(
    "flags",
    [
        METH_O,
        METH_VARARGS | METH_KEYWORDS,
        METH_FASTCALL | METH_KEYWORDS,
    ],
)
def test_c_extension_source_propagates_cursor_lookup_failures_for_runtime_functions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    flags: int,
) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_runtime_pymethoddef",
        lambda handle: RuntimePyMethodDef(method_address=0x1234, flags=flags),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_symbolized_address",
        lambda address: _symbolized_location(
            compilation_unit_path=tmp_path / "foo_impl.c",
            function_name="foo_impl",
        ),
    )
    monkeypatch.setattr(
        CExtensionSource,
        "_resolve_function_cursor",
        lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("cursor missing")),
    )

    source = _make_source(monkeypatch, tmp_path)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[IRFunction(name="foo", runtime_handle=object())],
    )

    with pytest.raises(RuntimeError, match="cursor missing"):
        source.resolve_function(module, module.functions[0])


def test_c_extension_source_propagates_cursor_lookup_failures_for_method_descriptors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_symbolized_address",
        lambda address: _symbolized_location(
            compilation_unit_path=tmp_path / "append.c",
            function_name="append",
        ),
    )
    monkeypatch.setattr(
        CExtensionSource,
        "_resolve_function_cursor",
        lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("cursor missing")),
    )

    source = _make_source(monkeypatch, tmp_path)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[IRFunction(name="append", runtime_handle=list.__dict__["append"])],
    )

    with pytest.raises(RuntimeError, match="cursor missing"):
        source.resolve_function(module, module.functions[0])


def test_c_extension_source_raises_when_ast_inference_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "foo_impl.c"
    snippet = "\n".join(
        [
            "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
            "    return (PyObject*)0;",
            "}",
        ]
    )
    source_path.write_text(snippet, encoding="utf-8", newline="\n")
    function_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo_impl",
        extent=_extent_for_source_snippet(source_path, snippet),
    )

    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_runtime_pymethoddef",
        lambda handle: RuntimePyMethodDef(method_address=0x1234, flags=METH_O),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_symbolized_address",
        lambda address: _symbolized_location(
            compilation_unit_path=source_path,
            function_name="foo_impl",
        ),
    )
    monkeypatch.setattr(
        CExtensionSource,
        "_resolve_function_cursor",
        lambda self, **kwargs: function_cursor,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.inference.infer_signature",
        lambda function_cursor, *, ml_flags=0: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    source = _make_source(monkeypatch, tmp_path)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[IRFunction(name="foo", runtime_handle=object())],
    )

    with pytest.raises(RuntimeError, match="boom"):
        source.resolve_function(module, module.functions[0])


def test_find_function_cursor_matches_linkage_name_before_symbol_name(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "foo_impl.cpp"
    source_path.write_text(
        "\n".join(
            [
                "int foo(int value) {",
                "    return value;",
                "}",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    first_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo",
        is_definition=True,
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
        extent=_extent_for_source_snippet(source_path, "int foo(int value) {\n    return value;\n}"),
    )
    first_cursor.mangled_name = "_Z3fooi"

    second_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo",
        is_definition=True,
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
        extent=_extent_for_source_snippet(source_path, "int foo(int value) {\n    return value;\n}"),
    )
    second_cursor.mangled_name = "_Z3food"

    translation_unit = type(
        "FakeTranslationUnit",
        (),
        {
            "cursor": _FakeNode(
                kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
                children=[first_cursor, second_cursor],
            )
        },
    )()

    matched = CExtensionSource._find_function_cursor(
        translation_unit=translation_unit,
        symbol_name="foo",
        linkage_name="_Z3fooi",
    )

    assert matched is first_cursor


def test_find_function_cursor_matches_spelling_without_linkage_name(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "foo_impl.cpp"
    source_path.write_text(
        "\n".join(
            [
                "int ignored() { return 0; }",
                "int foo_impl(int value) {",
                "    return value;",
                "}",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    function_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo_impl",
        is_definition=True,
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
        extent=_extent_for_source_snippet(source_path, "int foo_impl(int value) {\n    return value;\n}"),
    )
    translation_unit = type(
        "FakeTranslationUnit",
        (),
        {
            "cursor": _FakeNode(
                kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
                children=[function_cursor],
            )
        },
    )()

    matched = CExtensionSource._find_function_cursor(
        translation_unit=translation_unit,
        symbol_name="foo_impl",
    )

    assert matched is function_cursor


def test_find_function_cursor_matches_symbol_across_different_location_file(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "foo_impl.cpp"
    other_path = tmp_path / "other.cpp"
    source_path.write_text(
        "\n".join(
            [
                "int foo_impl(int value) {",
                "    return value;",
                "}",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    function_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo_impl",
        is_definition=True,
        location=type("Loc", (), {"file": type("File", (), {"name": str(other_path)})()})(),
        extent=_extent_for_source_snippet(source_path, "int foo_impl(int value) {\n    return value;\n}"),
    )
    translation_unit = type(
        "FakeTranslationUnit",
        (),
        {
            "cursor": _FakeNode(
                kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
                children=[function_cursor],
            )
        },
    )()

    matched = CExtensionSource._find_function_cursor(
        translation_unit=translation_unit,
        symbol_name="foo_impl",
    )

    assert matched is function_cursor


def test_find_function_cursor_returns_first_name_match_without_linkage_name(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "foo_impl.c"
    source_path.write_text(
        "\n".join(
            [
                "int foo_impl(void) {",
                "    return 1;",
                "}",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    shared_extent = _extent_for_source_snippet(source_path, "int foo_impl(void) {\n    return 1;\n}")
    first_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo_impl",
        is_definition=True,
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
        extent=shared_extent,
    )
    second_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo_impl",
        is_definition=True,
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
        extent=shared_extent,
    )
    translation_unit = type(
        "FakeTranslationUnit",
        (),
        {
            "cursor": _FakeNode(
                kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
                children=[first_cursor, second_cursor],
            )
        },
    )()

    matched = CExtensionSource._find_function_cursor(
        translation_unit=translation_unit,
        symbol_name="foo_impl",
    )

    assert matched is first_cursor


def test_find_function_cursor_prefers_definition_over_matching_declaration(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "foo_impl.c"
    source_path.write_text(
        "\n".join(
            [
                "int foo_impl(int value);",
                "int foo_impl(int value) {",
                "    return value;",
                "}",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    declaration_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo_impl",
        is_definition=False,
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
    )

    definition_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo_impl",
        is_definition=True,
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
        extent=_extent_for_source_snippet(source_path, "int foo_impl(int value) {\n    return value;\n}"),
    )

    translation_unit = type(
        "FakeTranslationUnit",
        (),
        {
            "cursor": _FakeNode(
                kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
                children=[declaration_cursor, definition_cursor],
            )
        },
    )()

    matched = CExtensionSource._find_function_cursor(
        translation_unit=translation_unit,
        symbol_name="foo_impl",
    )

    assert matched is definition_cursor


def test_find_function_cursor_matches_definition_in_nested_decl_contexts(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "foo_impl.cpp"
    source_path.write_text(
        "\n".join(
            [
                "namespace outer {",
                'extern "C" {',
                "int foo_impl(int value) {",
                "    return value;",
                "}",
                "}",
                "}",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    function_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo_impl",
        is_definition=True,
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
        extent=_extent_for_source_snippet(source_path, "int foo_impl(int value) {\n    return value;\n}"),
    )

    translation_unit = type(
        "FakeTranslationUnit",
        (),
        {
            "cursor": _FakeNode(
                kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
                children=[
                    _FakeNode(
                        kind=clang.cindex.CursorKind.NAMESPACE,
                        children=[
                            _FakeNode(
                                kind=clang.cindex.CursorKind.LINKAGE_SPEC,
                                children=[function_cursor],
                            )
                        ],
                    )
                ],
            )
        },
    )()

    matched = CExtensionSource._find_function_cursor(
        translation_unit=translation_unit,
        symbol_name="foo_impl",
    )

    assert matched is function_cursor


def test_find_function_cursor_matches_definition_from_header_candidate(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "foo_impl.c"
    header_path = tmp_path / "foo_impl.h"
    source_path.write_text('#include "foo_impl.h"\n', encoding="utf-8", newline="\n")
    header_path.write_text(
        "\n".join(
            [
                "int foo_impl(int value) {",
                "    return value;",
                "}",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    function_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo_impl",
        is_definition=True,
        location=type("Loc", (), {"file": type("File", (), {"name": str(header_path)})()})(),
        extent=_extent_for_source_snippet(header_path, "int foo_impl(int value) {\n    return value;\n}"),
    )

    translation_unit = type(
        "FakeTranslationUnit",
        (),
        {
            "cursor": _FakeNode(
                kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
                children=[function_cursor],
            )
        },
    )()

    matched = CExtensionSource._find_function_cursor(
        translation_unit=translation_unit,
        symbol_name="foo_impl",
    )

    assert matched is function_cursor
