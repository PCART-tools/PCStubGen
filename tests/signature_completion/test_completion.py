from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import cast

import clang.cindex
import pytest
from loguru import logger

from pcstubgen.ir_modules import IRArgument, IRClass, IRFunction, IRMethod, IRModule, IRModuleType, QualifiedName
from pcstubgen.signature_completion import SignatureCompleter
from pcstubgen.stub_generation_options import StubGenerationOptions
from pcstubgen.types import RawType
from tests._c_extension_test_support import (
    ExtractedFunction,
    ExtractedSignature,
    METH_VARARGS,
    _FakeNode,
    _arg,
    _extent_for_source_snippet,
    _fake_function_cursor,
    _module_fixture,
    _patch_c_signature_extractor,
    _patch_raising_c_signature_extractor,
    _signature,
    _unknown_function,
)


def test_completer_prefers_c_over_docstring_and_writes_source_comment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=func_cursor,
                    ml_flags=METH_VARARGS,
                    signatures=[
                        ExtractedSignature(
                            arguments=[_arg("value", "int")],
                            return_type=RawType("bool"),
                        )
                    ],
                )
            }
        ),
    )

    summary = SignatureCompleter(
        StubGenerationOptions(
            compilation_database=tmp_path / "compile_commands.json",
            include_c_inferred_source_comment=True,
        )
    ).run(module)

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


def test_completer_falls_back_to_docstring_when_c_has_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
    _patch_c_signature_extractor(monkeypatch, modules={})

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        summary = SignatureCompleter(
            StubGenerationOptions(
                compilation_database=tmp_path / "compile_commands.json",
            )
        ).run(module)
    finally:
        logger.remove(sink_id)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.signatures[0].args] == ["x", "y", "w", "out", "p"]
    assert parsed.signatures[0].return_type is not None
    assert parsed.signatures[0].return_type.render() == "numpy.ndarray"
    assert summary.docstring_completed == 1
    assert "docstring" in log_output.getvalue()


def test_completer_falls_back_to_docstring_when_c_source_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo", doc="foo(value: str) -> bool")],
    )
    _patch_raising_c_signature_extractor(monkeypatch, RuntimeError("boom"))

    summary = SignatureCompleter(
        StubGenerationOptions(
            compilation_database=tmp_path / "compile_commands.json",
        )
    ).run(module)

    parsed = module.functions[0]
    assert parsed.signatures[0].args[0].name == "value"
    assert parsed.signatures[0].return_type is not None
    assert parsed.signatures[0].return_type.render() == "bool"
    assert summary.docstring_completed == 1


def test_completer_skips_source_comment_when_option_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=func_cursor,
                    ml_flags=METH_VARARGS,
                    signatures=[
                        ExtractedSignature(
                            arguments=[_arg("value", "int")],
                            return_type=RawType("bool"),
                        )
                    ],
                )
            }
        ),
    )
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )

    SignatureCompleter(
        StubGenerationOptions(
            compilation_database=tmp_path / "compile_commands.json",
            include_c_inferred_source_comment=False,
        )
    ).run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.signatures[0].args] == ["value"]
    assert parsed.c_inferred_source_comment is None


def test_completer_completes_extension_methods_via_c_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "build": ExtractedFunction(
                    ml_name="build",
                    function_cursor=_fake_function_cursor("build"),
                    ml_flags=METH_VARARGS,
                    signatures=[
                        ExtractedSignature(
                            arguments=[_arg("from_c", "bool")],
                            return_type=RawType("bool"),
                        )
                    ],
                )
            }
        ),
    )
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        classes=[
            IRClass(
                name="Builder",
                methods=[IRMethod(function=_unknown_function("build"), decorator=None)],
            )
        ],
    )

    summary = SignatureCompleter(
        StubGenerationOptions(
            compilation_database=tmp_path / "compile_commands.json",
        )
    ).run(module)

    parsed = module.classes[0].methods[0].function
    assert parsed.signatures[0].args[0].name == "from_c"
    assert parsed.signatures[0].return_type is not None
    assert parsed.signatures[0].return_type.render() == "bool"
    assert summary.c_completed == 1
    assert summary.uncompleted == 0


def test_completer_uses_docstring_when_available_for_python_module() -> None:
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

    summary = SignatureCompleter(StubGenerationOptions()).run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.signatures[0].args] == ["value"]
    assert parsed.signatures[0].args[0].type is not None
    assert parsed.signatures[0].args[0].type.render() == "str"
    assert parsed.signatures[0].return_type is not None
    assert parsed.signatures[0].return_type.render() == "bool"
    assert summary.docstring_completed == 1
    assert summary.uncompleted == 0


def test_completer_logs_failure_reasons_when_both_paths_return_no_signature() -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[_unknown_function("fallback", doc="plain fallback docs")],
    )

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        summary = SignatureCompleter(StubGenerationOptions()).run(module)
    finally:
        logger.remove(sink_id)

    assert summary.uncompleted == 1
    assert "c_reason:" in log_output.getvalue()
    assert "docstring_reason:" in log_output.getvalue()


def test_completer_keeps_known_signatures_and_counts_unresolved() -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[
            IRFunction(
                name="known",
                signatures=[_signature(args=[IRArgument(name="value")])],
            ),
            _unknown_function("missing"),
            _unknown_function("fallback"),
        ],
    )

    summary = SignatureCompleter(StubGenerationOptions()).run(module)

    assert summary.total_functions == 3
    assert summary.uncompleted == 3
    assert module.functions[0].signatures[0].args[0].name == "value"
    assert module.functions[1].signatures == []
    assert module.functions[2].signatures == []


def test_completer_run_recreates_summary_for_each_invocation() -> None:
    completer = SignatureCompleter(StubGenerationOptions())

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
