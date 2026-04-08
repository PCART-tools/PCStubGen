from __future__ import annotations

from pathlib import Path

import clang.cindex
import pytest

from pcstubgen.ir_modules import IRArgumentKind, IRFunction, IRModule, IRModuleType, IRSignature, QualifiedName
from pcstubgen.signature_completion.c_extension.modules.method_flags import (
    METH_FASTCALL,
    METH_KEYWORDS,
    METH_NOARGS,
    METH_O,
    METH_VARARGS,
)
from pcstubgen.signature_completion.c_extension.runtime import RuntimePyMethodDef
from pcstubgen.signature_completion.c_extension.source import CExtensionSource
from pcstubgen.signature_completion.c_extension.address_resolver import SymbolizedAddressLocation
from pcstubgen.types import AnyType, RawType
from tests._c_extension_test_support import _FakeNode, _arg, _extent_for_source_snippet


def _symbolized_location(
    *,
    binary_path: Path,
    relative_address: int,
    function_name: str = "foo_impl",
    resolved_path: Path,
    resolved_line: int,
    function_start_path: Path,
    function_start_line: int,
) -> SymbolizedAddressLocation:
    return SymbolizedAddressLocation(
        binary_path=binary_path,
        relative_address=relative_address,
        function_name=function_name,
        resolved_path=resolved_path,
        resolved_line=resolved_line,
        function_start_path=function_start_path,
        function_start_line=function_start_line,
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
            binary_path=tmp_path / "sample.so",
            relative_address=address,
            function_start_path=source_path,
            function_start_line=1,
            resolved_path=source_path,
            resolved_line=2,
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


def test_c_extension_source_prefers_function_start_location(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    function_start_path = tmp_path / "foo_start.c"
    resolved_path = tmp_path / "foo_body.c"

    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_runtime_pymethoddef",
        lambda handle: RuntimePyMethodDef(method_address=0x1234, flags=METH_NOARGS),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_symbolized_address",
        lambda address: _symbolized_location(
            binary_path=tmp_path / "sample.so",
            relative_address=address,
            function_start_path=function_start_path,
            function_start_line=11,
            resolved_path=resolved_path,
            resolved_line=23,
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
        binary_path=tmp_path / "sample.so",
        relative_address=0x1234,
        function_start_path=function_start_path,
        function_start_line=11,
        resolved_path=resolved_path,
        resolved_line=23,
    )


def test_c_extension_source_uses_function_start_location_for_minimal_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    function_start_path = tmp_path / "foo_start.c"
    resolved_path = tmp_path / "foo_impl.c"

    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_runtime_pymethoddef",
        lambda handle: RuntimePyMethodDef(method_address=0x1234, flags=METH_O),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_symbolized_address",
        lambda address: _symbolized_location(
            binary_path=tmp_path / "sample.so",
            relative_address=address,
            function_start_path=function_start_path,
            function_start_line=7,
            resolved_path=resolved_path,
            resolved_line=19,
        ),
    )

    def fake_resolve_function_cursor(
        self: CExtensionSource,
        *,
        location: SymbolizedAddressLocation,
    ):
        _ = (self, location)
        return None

    monkeypatch.setattr(CExtensionSource, "_resolve_function_cursor", fake_resolve_function_cursor)

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
            binary_path=tmp_path / "sample.so",
            relative_address=address,
            function_start_path=tmp_path / "foo_impl.c",
            function_start_line=1,
            resolved_path=tmp_path / "foo_impl.c",
            resolved_line=1,
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
            binary_path=tmp_path / "sample.so",
            relative_address=address,
            function_start_path=tmp_path / "append.c",
            function_start_line=1,
            resolved_path=tmp_path / "append.c",
            resolved_line=1,
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
            binary_path=tmp_path / "sample.so",
            relative_address=address,
            function_start_path=tmp_path / "foo_impl.c",
            function_start_line=1,
            resolved_path=tmp_path / "foo_impl.c",
            resolved_line=1,
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
            binary_path=tmp_path / "sample.so",
            relative_address=address,
            function_start_path=source_path,
            function_start_line=1,
            resolved_path=source_path,
            resolved_line=2,
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


def test_find_function_cursor_matches_demangled_name_and_line(
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
    function_cursor.extent.start.line = 2
    function_cursor.extent.end.line = 4
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
        line=2,
        symbol_name="ns::foo_impl(int)",
    )

    assert matched is function_cursor


def test_find_function_cursor_rejects_name_match_when_line_is_out_of_range(
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
    function_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo_impl",
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
        extent=_extent_for_source_snippet(source_path, "int foo_impl(void) {\n    return 1;\n}"),
    )
    function_cursor.extent.start.line = 1
    function_cursor.extent.end.line = 3
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
        line=10,
        symbol_name="foo_impl",
    )

    assert matched is None


def test_find_function_cursor_rejects_multiple_name_and_line_matches(
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
    first_cursor.extent.start.line = 1
    first_cursor.extent.end.line = 3
    second_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo_impl",
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
        extent=shared_extent,
    )
    second_cursor.extent.start.line = 1
    second_cursor.extent.end.line = 3
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
        line=1,
        symbol_name="foo_impl",
    )

    assert matched is None
