from __future__ import annotations

from pathlib import Path
from typing import cast

import clang.cindex
import pytest

from pcstubgen.models import Function, Module, QualifiedName
from pcstubgen.signature_completion import SignatureCompleter
from pcstubgen.signature_completion.c_extension.source import CInferenceResult
from pcstubgen.type_models import RawType
from tests._c_extension_test_support import (
    _FakeNode,
    _arg,
    _extent_for_source_snippet,
    _patch_c_signature_extractor,
    _signature,
    ResolvedFunctionFixture,
    _unknown_function,
)


def _location_text(text: str) -> object:
    class _Location:
        def __str__(self) -> str:
            return text

    return _Location()


def _patch_compilation_database_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.ClangParser",
        lambda compilation_database: object(),
    )


def _patch_c_runtime_support(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.completion.is_cpython_builtin",
        lambda handle: True,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.completion.is_pybind11_builtin",
        lambda handle: True,
    )


def _patch_pybind11_runtime_support(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.completion.is_cpython_builtin",
        lambda handle: False,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.completion.is_pybind11_builtin",
        lambda handle: True,
    )


def test_completer_prefers_c_branch_over_docstring_and_writes_comment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    _patch_c_runtime_support(monkeypatch)
    source = tmp_path / "foo_impl.c"
    snippet = "\n".join(
        [
            "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
            "    return (PyObject*)0;",
            "}",
        ]
    )
    source.write_text(snippet, encoding="utf-8", newline="\n")
    func_cursor = cast(
        clang.cindex.Cursor,
        _FakeNode(
            kind=clang.cindex.CursorKind.FUNCTION_DECL,
            spelling="foo_impl",
            location=_location_text(f"{source}:1:1"),
            extent=_extent_for_source_snippet(source, snippet),
        ),
    )

    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            _unknown_function(
                "foo",
                doc="foo(value: str) -> str\n\nparsed from docstring",
            )
        ],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        functions={
            "foo": ResolvedFunctionFixture(
                function_cursor=func_cursor,
                signatures=[
                    _signature(
                        args=[_arg("value", "int")],
                        return_type=RawType("bool"),
                    )
                ],
            )
        },
    )
    monkeypatch.setattr(
        "tests._c_extension_test_support.ast_utils_module.get_cursor_text",
        lambda cursor: snippet,
    )

    summary = SignatureCompleter(tmp_path / "compile_commands.json").run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.signatures[0].args] == ["value"]
    assert parsed.signatures[0].args[0].type is not None
    assert parsed.signatures[0].args[0].type.render() == "int"
    assert parsed.signatures[0].return_type is not None
    assert parsed.signatures[0].return_type.render() == "bool"
    assert parsed.doc == "foo(value: str) -> str\n\nparsed from docstring"
    assert parsed.comment == f"{source}:1:1\n{snippet}"
    assert summary.c_completed == 1
    assert summary.docstring_completed == 0


def test_completer_reads_signatures_and_comment_from_c_inference_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    _patch_c_runtime_support(monkeypatch)
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[_unknown_function("foo")],
    )

    monkeypatch.setattr(
        "pcstubgen.signature_completion.completion.CExtensionSource.infer_function_signatures",
        lambda self, module_node, function_node: CInferenceResult(
            signatures=[
                _signature(
                    args=[_arg("value", "int")],
                    return_type=RawType("bool"),
                )
            ],
            comment="mock:pkg.mod.foo\nmocked source",
        ),
    )

    summary = SignatureCompleter(tmp_path / "compile_commands.json").run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.signatures[0].args] == ["value"]
    assert parsed.signatures[0].return_type is not None
    assert parsed.signatures[0].return_type.render() == "bool"
    assert parsed.comment == "mock:pkg.mod.foo\nmocked source"
    assert summary.c_completed == 1


def test_completer_uses_docstring_for_pybind11_builtin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    _patch_pybind11_runtime_support(monkeypatch)
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            Function(
                name="fallback",
                runtime_handle=object(),
                doc="fallback(value: str) -> bool\n\nparsed from docstring",
            )
        ],
    )
    summary = SignatureCompleter(tmp_path / "compile_commands.json").run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.signatures[0].args] == ["value"]
    assert parsed.signatures[0].args[0].type is not None
    assert parsed.signatures[0].args[0].type.render() == "str"
    assert parsed.signatures[0].return_type is not None
    assert parsed.signatures[0].return_type.render() == "bool"
    assert summary.docstring_completed == 1
    assert summary.uncompleted == 0


def test_completer_continues_after_pybind11_docstring_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    _patch_pybind11_runtime_support(monkeypatch)
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            Function(
                name="broken",
                runtime_handle=object(),
                doc="broken(value: str) -> bool",
            ),
            Function(
                name="working",
                runtime_handle=object(),
                doc="working(value: str) -> bool",
            ),
        ],
    )

    def _parse_or_raise(_: Module, func: Function):
        if func.name == "broken":
            raise RuntimeError("boom")
        return [_signature(args=[_arg("value", "str")], return_type=RawType("bool"))]

    monkeypatch.setattr(
        "pcstubgen.signature_completion.completion.parse_docstring_signatures",
        _parse_or_raise,
    )

    summary = SignatureCompleter(tmp_path / "compile_commands.json").run(module)

    assert module.functions[0].signatures == []
    assert [arg.name for arg in module.functions[1].signatures[0].args] == ["value"]
    assert module.functions[1].signatures[0].return_type is not None
    assert module.functions[1].signatures[0].return_type.render() == "bool"
    assert summary.docstring_completed == 1
    assert summary.uncompleted == 1


def test_completer_does_not_swallow_pybind11_docstring_base_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    _patch_pybind11_runtime_support(monkeypatch)
    module = Module(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            Function(
                name="broken",
                runtime_handle=object(),
                doc="broken(value: str) -> bool",
            ),
        ],
    )

    def _parse_or_raise(_: Module, __: Function):
        raise BaseException("boom")

    monkeypatch.setattr(
        "pcstubgen.signature_completion.completion.parse_docstring_signatures",
        _parse_or_raise,
    )

    with pytest.raises(BaseException, match="boom"):
        SignatureCompleter(tmp_path / "compile_commands.json").run(module)
