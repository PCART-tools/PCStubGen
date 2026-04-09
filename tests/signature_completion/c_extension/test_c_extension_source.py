from __future__ import annotations

from pathlib import Path

import clang.cindex
import pytest

from pcstubgen.ir_modules import IRArgumentKind, IRFunction, IRModule, IRModuleType, IRSignature, QualifiedName
from pcstubgen.signature_completion.c_extension.address_resolver import SymbolizedAddressLocation
from pcstubgen.signature_completion.c_extension.modules.method_flags import (
    METH_FASTCALL,
    METH_KEYWORDS,
    METH_NOARGS,
    METH_O,
    METH_VARARGS,
)
from pcstubgen.signature_completion.c_extension.runtime import RuntimePyMethodDef
from pcstubgen.signature_completion.c_extension.source import CExtensionSource
from pcstubgen.types import AnyType, RawType
from tests._c_extension_test_support import _FakeNode, _arg, _extent_for_source_snippet


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

    source = CExtensionSource(compilation_database=tmp_path / "compile_commands.json")
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", runtime_handle=object())],
    )

    signatures, source_comment = source.resolve_function(module, module.functions[0])

    assert signatures[0].args[0].name == "value"
    assert signatures[0].args[0].type is not None
    assert signatures[0].args[0].type.render() == "int"
    assert signatures[0].return_type is not None
    assert signatures[0].return_type.render() == "bool"
    assert source_comment == snippet


def test_c_extension_source_passes_resolved_scope_to_cursor_resolver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    compilation_unit_path = tmp_path / "foo_impl.c"

    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_runtime_pymethoddef",
        lambda handle: RuntimePyMethodDef(method_address=0x1234, flags=METH_NOARGS),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_symbolized_address",
        lambda address: _symbolized_location(
            compilation_unit_path=compilation_unit_path,
            function_name="foo_impl",
        ),
    )

    def fake_resolve_function_cursor(
        self: CExtensionSource,
        *,
        location: SymbolizedAddressLocation,
    ):
        _ = self
        captured["location"] = location
        return None

    monkeypatch.setattr(CExtensionSource, "_resolve_function_cursor", fake_resolve_function_cursor)

    source = CExtensionSource()
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", runtime_handle=object())],
    )

    signatures, _ = source.resolve_function(module, module.functions[0])

    assert signatures[0].args == []
    assert captured["location"] == _symbolized_location(
        compilation_unit_path=compilation_unit_path,
        function_name="foo_impl",
    )


def test_c_extension_source_uses_minimal_inference_without_ast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_runtime_pymethoddef",
        lambda handle: RuntimePyMethodDef(method_address=0x1234, flags=METH_O),
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
        lambda self, **kwargs: None,
    )

    source = CExtensionSource()
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", runtime_handle=object())],
    )

    signatures, _ = source.resolve_function(module, module.functions[0])

    assert signatures[0].args[0].name == "arg"
    assert signatures[0].args[0].kind is IRArgumentKind.POSITIONAL_ONLY
    assert signatures[0].args[0].type == RawType("object")
    assert signatures[0].return_type == AnyType()


def test_c_extension_source_uses_minimal_varargs_keywords_shape_without_ast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_runtime_pymethoddef",
        lambda handle: RuntimePyMethodDef(
            method_address=0x1234,
            flags=METH_VARARGS | METH_KEYWORDS,
        ),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_symbolized_address",
        lambda address: _symbolized_location(
            compilation_unit_path=tmp_path / "foo_impl.c",
            function_name="foo_impl",
        ),
    )

    source = CExtensionSource()
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", runtime_handle=object())],
    )

    signatures, _ = source.resolve_function(module, module.functions[0])

    assert [arg.name for arg in signatures[0].args] == ["args", "kwargs"]
    assert signatures[0].args[0].kind is IRArgumentKind.VAR_POSITIONAL
    assert signatures[0].args[1].kind is IRArgumentKind.VAR_KEYWORD
    assert signatures[0].args[0].type == RawType("object")
    assert signatures[0].args[1].type == RawType("object")
    assert signatures[0].return_type == AnyType()


def test_c_extension_source_supports_method_descriptors(
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

    source = CExtensionSource()
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="append", runtime_handle=list.__dict__["append"])],
    )

    signatures, _ = source.resolve_function(module, module.functions[0])

    assert signatures[0].args[0].name == "arg"
    assert signatures[0].args[0].kind is IRArgumentKind.POSITIONAL_ONLY
    assert signatures[0].args[0].type == RawType("object")


def test_c_extension_source_uses_fastcall_minimal_shape_without_ast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_runtime_pymethoddef",
        lambda handle: RuntimePyMethodDef(
            method_address=0x1234,
            flags=METH_FASTCALL | METH_KEYWORDS,
        ),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_symbolized_address",
        lambda address: _symbolized_location(
            compilation_unit_path=tmp_path / "foo_impl.c",
            function_name="foo_impl",
        ),
    )

    source = CExtensionSource()
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", runtime_handle=object())],
    )

    signatures, _ = source.resolve_function(module, module.functions[0])

    assert [arg.name for arg in signatures[0].args] == ["args", "kwargs"]
    assert signatures[0].args[0].kind is IRArgumentKind.VAR_POSITIONAL
    assert signatures[0].args[1].kind is IRArgumentKind.VAR_KEYWORD


def test_c_extension_source_falls_back_to_minimal_signatures_when_ast_inference_raises(
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

    source = CExtensionSource(compilation_database=tmp_path / "compile_commands.json")
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", runtime_handle=object())],
    )

    signatures, source_comment = source.resolve_function(module, module.functions[0])

    assert signatures[0].args[0].name == "arg"
    assert signatures[0].args[0].type == RawType("object")
    assert signatures[0].return_type == AnyType()
    assert source_comment == snippet


def test_c_extension_source_rejects_non_extension_modules() -> None:
    source = CExtensionSource()
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[IRFunction(name="foo", runtime_handle=object())],
    )

    with pytest.raises(RuntimeError, match="不是扩展模块"):
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
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
        extent=_extent_for_source_snippet(source_path, "int foo(int value) {\n    return value;\n}"),
    )
    first_cursor.displayname = "foo(int)"
    first_cursor.mangled_name = "_Z3fooi"

    second_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo",
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
        extent=_extent_for_source_snippet(source_path, "int foo(int value) {\n    return value;\n}"),
    )
    second_cursor.displayname = "foo(double)"
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
        source_path=source_path,
        symbol_name="foo",
        linkage_name="_Z3fooi",
    )

    assert matched is first_cursor


def test_find_function_cursor_matches_demangled_name_without_linkage_name(
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
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
        extent=_extent_for_source_snippet(source_path, "int foo_impl(int value) {\n    return value;\n}"),
    )
    function_cursor.displayname = "foo_impl(int)"
    function_cursor.semantic_parent = _FakeNode(
        kind=clang.cindex.CursorKind.NAMESPACE,
        spelling="ns",
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
        source_path=source_path,
        symbol_name="ns::foo_impl(int)",
    )

    assert matched is function_cursor


def test_find_function_cursor_rejects_multiple_name_matches_without_linkage_name(
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
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
        extent=shared_extent,
    )
    second_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo_impl",
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
        source_path=source_path,
        symbol_name="foo_impl",
    )

    assert matched is None
