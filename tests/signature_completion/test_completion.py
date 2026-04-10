from __future__ import annotations

from pathlib import Path
from typing import cast

import clang.cindex
import pytest

from pcstubgen.ir_modules import IRArgument, IRFunction, IRModule, IRModuleType, QualifiedName
from pcstubgen.signature_completion import SignatureCompleter
from pcstubgen.types import RawType
from tests._c_extension_test_support import (
    _FakeNode,
    _arg,
    _extent_for_source_snippet,
    _patch_c_signature_extractor,
    _patch_raising_c_signature_extractor,
    _signature,
    ResolvedFunctionFixture,
    _unknown_function,
)


def _patch_compilation_database_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.compilation_database_loader.load_compilation_database",
        lambda path: object(),
    )


def test_completer_prefers_c_over_docstring_and_writes_source_comment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
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
            extent=_extent_for_source_snippet(source, snippet),
        ),
    )

    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
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

    summary = SignatureCompleter(tmp_path / "compile_commands.json").run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.signatures[0].args] == ["value"]
    assert parsed.signatures[0].args[0].type is not None
    assert parsed.signatures[0].args[0].type.render() == "int"
    assert parsed.signatures[0].return_type is not None
    assert parsed.signatures[0].return_type.render() == "bool"
    assert parsed.doc == "foo(value: str) -> str\n\nparsed from docstring"
    assert parsed.c_inferred_source_comment == snippet
    assert summary.c_completed == 1
    assert summary.docstring_completed == 0


def test_completer_raises_when_c_has_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[
            _unknown_function(
                "cdist_minkowski",
                doc=(
                    "cdist_minkowski(x: object, y: object, w: object = None, "
                    "out: object = None, p: typing.SupportsFloat = 2.0) -> numpy.ndarray"
                ),
            )
        ],
    )
    _patch_c_signature_extractor(monkeypatch, functions={})

    with pytest.raises(RuntimeError, match="未找到函数|没有可用签名"):
        SignatureCompleter(tmp_path / "compile_commands.json").run(module)


def test_completer_raises_when_c_source_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo", doc="foo(value: str) -> bool")],
    )
    _patch_raising_c_signature_extractor(monkeypatch, RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        SignatureCompleter(tmp_path / "compile_commands.json").run(module)


def test_completer_uses_docstring_when_available_for_python_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[
            _unknown_function(
                "fallback",
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


def test_completer_keeps_known_signatures_and_counts_unresolved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[
            IRFunction(
                name="known",
                runtime_handle=object(),
                signatures=[_signature(args=[IRArgument(name="value")])],
            ),
            _unknown_function("missing"),
            _unknown_function("fallback"),
        ],
    )
    summary = SignatureCompleter(tmp_path / "compile_commands.json").run(module)

    assert summary.total_functions == 3
    assert summary.uncompleted == 3
    assert module.functions[0].signatures[0].args[0].name == "value"
    assert module.functions[1].signatures == []
    assert module.functions[2].signatures == []


def test_completer_run_recreates_summary_for_each_invocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    completer = SignatureCompleter(tmp_path / "compile_commands.json")

    first_module = IRModule(
        full_name=QualifiedName.from_str("pkg.first"),
        module_type=IRModuleType.PYTHON,
        functions=[_unknown_function("missing")],
    )
    second_module = IRModule(
        full_name=QualifiedName.from_str("pkg.second"),
        module_type=IRModuleType.PYTHON,
        functions=[],
    )

    first_summary = completer.run(first_module)
    second_summary = completer.run(second_module)

    assert first_summary.total_functions == 1
    assert first_summary.uncompleted == 1
    assert second_summary.total_functions == 0
    assert second_summary.uncompleted == 0
    assert second_summary is not first_summary
