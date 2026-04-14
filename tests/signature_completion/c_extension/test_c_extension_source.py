from __future__ import annotations

import re
from pathlib import Path

import clang.cindex
import pytest

from pcstubgen.models import Function, Module, Signature, QualifiedName
from pcstubgen.signature_completion.c_extension.address_resolver import FuncFileLocation
from pcstubgen.signature_completion.c_extension.clang import ast_utils as ast_utils_module
from pcstubgen.signature_completion.c_extension.method_flags import (
    METH_O,
    METH_VARARGS,
)
from pcstubgen.signature_completion.c_extension.runtime import BuiltinFunctionRuntimeInfo
from pcstubgen.signature_completion.c_extension.source import (
    CExtensionSource,
    CInferenceResult,
)
from pcstubgen.type_models import RawType
from tests._c_extension_test_support import _FakeNode, _arg, _extent_for_source_snippet


def _location_text(text: str) -> object:
    class _Location:
        def __str__(self) -> str:
            return text

    return _Location()


def _make_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    translation_unit: object | None = None,
) -> CExtensionSource:
    compilation_database = tmp_path / "compile_commands.json"
    fake_translation_unit = translation_unit if translation_unit is not None else object()
    fake_parser = type(
        "FakeClangParser",
        (),
        {
            "get_translation_unit": lambda self, path: fake_translation_unit,
        },
    )()
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.ClangParser",
        lambda compilation_database: fake_parser,
    )
    return CExtensionSource(compilation_database)


def _symbolized_location(
    *,
    compilation_unit_path: Path,
    function_name: str = "foo_impl",
    linkage_name: str | None = None,
) -> FuncFileLocation:
    return FuncFileLocation(
        compilation_unit_path=compilation_unit_path,
        function_name=function_name,
        linkage_name=linkage_name,
    )


def test_c_extension_source_prefers_ast_inference_and_preserves_comment(
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
        location=_location_text(f"{source_path}:1:1"),
        extent=_extent_for_source_snippet(source_path, snippet),
    )

    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.read_builtin_function_runtime_info",
        lambda handle: BuiltinFunctionRuntimeInfo(address=0x1234, flags=METH_VARARGS),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.get_func_file_location",
        lambda address: _symbolized_location(
            compilation_unit_path=source_path,
            function_name="foo_impl",
        ),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.get_func_cursor",
        lambda *args: function_cursor,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.inference.infer_signature",
        lambda function_cursor, *, flags=0: [
            Signature(
                args=[_arg("value", "int")],
                return_type=RawType("bool"),
            )
        ],
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.get_cursor_text",
        lambda cursor: snippet,
    )

    source = _make_source(monkeypatch, tmp_path, translation_unit=object())
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[Function(name="foo", runtime_handle=object())],
    )

    result = source.infer_function_signatures(module, module.functions[0])

    assert isinstance(result, CInferenceResult)
    assert result.signatures[0].args[0].name == "value"
    assert result.signatures[0].args[0].type is not None
    assert result.signatures[0].args[0].type.render() == "int"
    assert result.signatures[0].return_type is not None
    assert result.signatures[0].return_type.render() == "bool"
    assert result.comment == f"{source_path}:1:1\n{snippet}"

def test_c_extension_source_raises_when_ast_inference_returns_no_signatures(
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
        location=_location_text(f"{source_path}:1:1"),
        extent=_extent_for_source_snippet(source_path, snippet),
    )

    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.read_builtin_function_runtime_info",
        lambda handle: BuiltinFunctionRuntimeInfo(address=0x1234, flags=METH_O),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.get_func_file_location",
        lambda address: _symbolized_location(
            compilation_unit_path=source_path,
            function_name="foo_impl",
        ),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.get_func_cursor",
        lambda *args: function_cursor,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.inference.infer_signature",
        lambda function_cursor, *, flags=0: [],
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.get_cursor_text",
        lambda cursor: snippet,
    )

    source = _make_source(monkeypatch, tmp_path, translation_unit=object())
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[Function(name="foo", runtime_handle=object())],
    )

    with pytest.raises(
        RuntimeError,
        match=rf"没有可用签名.*{re.escape(f'{source_path}:1:1')}",
    ):
        source.infer_function_signatures(module, module.functions[0])


def test_get_func_cursor_matches_linkage_name(
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

    matched = ast_utils_module.get_func_cursor(translation_unit, "foo", "_Z3fooi")

    assert matched is first_cursor


def test_get_func_cursor_matches_nested_definition_with_linkage_name(
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
    function_cursor.mangled_name = "_Z8foo_impli"

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

    matched = ast_utils_module.get_func_cursor(translation_unit, "foo_impl", "_Z8foo_impli")

    assert matched is function_cursor


def test_get_func_cursor_matches_spelling_when_linkage_name_is_missing(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "foo_impl.c"
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
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
        extent=_extent_for_source_snippet(source_path, "int foo_impl(int value) {\n    return value;\n}"),
    )
    function_cursor.mangled_name = "foo_impl"

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

    matched = ast_utils_module.get_func_cursor(translation_unit, "foo_impl", None)

    assert matched is function_cursor


def test_get_func_cursor_raises_when_linkage_name_does_not_match(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "foo_impl.cpp"
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
        location=type("Loc", (), {"file": type("File", (), {"name": str(source_path)})()})(),
        extent=_extent_for_source_snippet(source_path, "int foo_impl(int value) {\n    return value;\n}"),
    )
    function_cursor.mangled_name = "_Z8foo_impli"

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

    with pytest.raises(RuntimeError, match="未在 translation unit 中定位到函数定义"):
        ast_utils_module.get_func_cursor(translation_unit, "foo_impl", "_Z8foo_impld")
