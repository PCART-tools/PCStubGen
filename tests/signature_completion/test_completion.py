from __future__ import annotations

from pathlib import Path
from typing import cast

import clang.cindex
import pytest

from pcstubgen.ir_modules import IRArgument, IRFunction, IRModule, QualifiedName
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


def _patch_c_runtime_support(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.completion.supports_builtin_function_inference",
        lambda handle: True,
    )


def _make_pybind11_builtin_handle() -> object:
    """构造满足 pybind11 判定的 builtin-like 句柄。"""
    class _PybindBoundSelf:
        __module__ = "pybind11_builtins.fake_module"

    builtin_function_like = type(
        "builtin_function_or_method",
        (),
        {
            "__module__": "builtins",
            "__self__": _PybindBoundSelf(),
        },
    )
    return builtin_function_like()


def test_completer_prefers_c_branch_over_docstring_and_writes_source_comment(
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
            extent=_extent_for_source_snippet(source, snippet),
        ),
    )

    module = IRModule(
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


def test_completer_marks_uncompleted_when_c_has_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    _patch_c_runtime_support(monkeypatch)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
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

    summary = SignatureCompleter(tmp_path / "compile_commands.json").run(module)

    assert module.functions[0].signatures == []
    assert summary.c_completed == 0
    assert summary.docstring_completed == 0
    assert summary.uncompleted == 1


def test_completer_marks_uncompleted_when_c_source_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    _patch_c_runtime_support(monkeypatch)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[_unknown_function("foo", doc="foo(value: str) -> bool")],
    )
    _patch_raising_c_signature_extractor(monkeypatch, RuntimeError("boom"))

    summary = SignatureCompleter(tmp_path / "compile_commands.json").run(module)

    assert module.functions[0].signatures == []
    assert summary.c_completed == 0
    assert summary.docstring_completed == 0
    assert summary.uncompleted == 1


def test_completer_uses_docstring_for_pybind11_builtin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            IRFunction(
                name="fallback",
                runtime_handle=_make_pybind11_builtin_handle(),
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


def test_completer_does_not_use_docstring_for_unsupported_function(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            _unknown_function(
                "fallback",
                doc="fallback(value: str) -> bool\n\nparsed from docstring",
            )
        ],
    )

    summary = SignatureCompleter(tmp_path / "compile_commands.json").run(module)

    assert module.functions[0].signatures == []
    assert summary.c_completed == 0
    assert summary.docstring_completed == 0
    assert summary.uncompleted == 1


def test_completer_marks_uncompleted_when_pybind11_docstring_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            IRFunction(
                name="fallback",
                runtime_handle=_make_pybind11_builtin_handle(),
                doc="not a signature",
            )
        ],
    )

    summary = SignatureCompleter(tmp_path / "compile_commands.json").run(module)

    assert module.functions[0].signatures == []
    assert summary.c_completed == 0
    assert summary.docstring_completed == 0
    assert summary.uncompleted == 1


def test_completer_marks_uncompleted_when_pybind11_docstring_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            IRFunction(
                name="fallback",
                runtime_handle=_make_pybind11_builtin_handle(),
                doc=None,
            )
        ],
    )

    summary = SignatureCompleter(tmp_path / "compile_commands.json").run(module)

    assert module.functions[0].signatures == []
    assert summary.c_completed == 0
    assert summary.docstring_completed == 0
    assert summary.uncompleted == 1


def test_completer_continues_after_pybind11_docstring_base_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        functions=[
            IRFunction(
                name="broken",
                runtime_handle=_make_pybind11_builtin_handle(),
                doc="broken(value: str) -> bool",
            ),
            IRFunction(
                name="working",
                runtime_handle=_make_pybind11_builtin_handle(),
                doc="working(value: str) -> bool",
            ),
        ],
    )

    def _resolve_or_raise(_: IRModule, func: IRFunction):
        if func.name == "broken":
            raise BaseException("boom")
        return [_signature(args=[_arg("value", "str")], return_type=RawType("bool"))]

    monkeypatch.setattr(
        "pcstubgen.signature_completion.completion.resolve_docstring_signatures",
        _resolve_or_raise,
    )

    summary = SignatureCompleter(tmp_path / "compile_commands.json").run(module)

    assert module.functions[0].signatures == []
    assert [arg.name for arg in module.functions[1].signatures[0].args] == ["value"]
    assert module.functions[1].signatures[0].return_type is not None
    assert module.functions[1].signatures[0].return_type.render() == "bool"
    assert summary.docstring_completed == 1
    assert summary.uncompleted == 1


def test_completer_keeps_known_signatures_and_counts_unresolved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_compilation_database_loader(monkeypatch)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
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
        functions=[_unknown_function("missing")],
    )
    second_module = IRModule(
        full_name=QualifiedName.from_str("pkg.second"),
        functions=[],
    )

    first_summary = completer.run(first_module)
    second_summary = completer.run(second_module)

    assert first_summary.total_functions == 1
    assert first_summary.uncompleted == 1
    assert second_summary.total_functions == 0
    assert second_summary.uncompleted == 0
    assert second_summary is not first_summary
