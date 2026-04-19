from __future__ import annotations

import re
from pathlib import Path

import clang.cindex
import pytest

from pcstubgen.models import Function, Module, Signature, QualifiedName
from pcstubgen.signature_completion.c_extension.dladdr import FuncFileLocation
from pcstubgen.signature_completion.c_extension.method_flags import (
    METH_O,
    METH_VARARGS,
)
from pcstubgen.runtime import BuiltinFunctionRuntimeInfo
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
        "pcstubgen.signature_completion.c_extension.source.read_cpython_function_runtime_info",
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
        functions=[Function(name="foo", handle=object())],
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
        "pcstubgen.signature_completion.c_extension.source.read_cpython_function_runtime_info",
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
        functions=[Function(name="foo", handle=object())],
    )

    with pytest.raises(
        RuntimeError,
        match=rf"没有可用签名.*{re.escape(f'{source_path}:1:1')}",
    ):
        source.infer_function_signatures(module, module.functions[0])
