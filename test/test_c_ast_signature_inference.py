from __future__ import annotations

import importlib
import logging
import sysconfig
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import clang.cindex
import pytest

from core.node_visitors.c_signature_extraction.core import extract_c_signature_modules
from core.node_visitors.c_signature_extraction.core.constants import (
    METH_KEYWORDS,
    METH_VARARGS,
)
from core.node_visitors.c_signature_extraction.core import (
    c_signature_extraction as c_signature_extraction_module,
)
from core.node_visitors.c_signature_extraction.core import cursor_utils as cursor_utils_module
from core.node_visitors.c_signature_extraction.core import inference_signature as signature_rules_module
from core.node_visitors.c_signature_extraction.core import module_table as module_table_module
from core.node_visitors.c_signature_extraction.core import translation_unit as translation_unit_module
from core.node_visitors.c_signature_extraction.core.module_table import (
    extract_method_table as _extract_method_table,
    extract_pymethoddef_init_list_expr as _extract_PyMethodDef_INIT_LIST_EXPR,
    resolve_init_list_expr as _resolve_INIT_LIST_EXPR,
)
from core.node_visitors.c_signature_extraction.core.models import (
    ExtractedArgument,
    ExtractedFunction,
    ExtractedModule,
    ExtractedSignature,
)
from core.ir import (
    IRArgument,
    IRArgumentKind,
    IRClass,
    IRFunction,
    IRMethod,
    IRModule,
    IRModuleType,
    QualifiedName,
)
from core.node_visitors.c_signature_extraction.c_signature_extraction_visitor import (
    CSignatureExtractionVisitor,
)
from core.node_visitors.DocStringSignatureParserVisitor import (
    DocStringSignatureParserVisitor,
)
from core.pipeline import Pipeline
from core.stub_generation_options import StubGenerationOptions


def _generic_signature() -> list[IRArgument]:
    return [
        IRArgument(name="args", kind=IRArgumentKind.VAR_POSITIONAL),
        IRArgument(name="kwargs", kind=IRArgumentKind.VAR_KEYWORD),
    ]


def _module_fixture(
    *,
    name: str = "pkg.mod",
    functions: dict[str, ExtractedFunction] | None = None,
    lookup_names: set[str] | None = None,
) -> dict[str, ExtractedModule]:
    module_lookup_names = set(lookup_names or ())
    module_lookup_names.add(name)
    module_lookup_names.add(name.rsplit(".", 1)[-1])
    return {
        name: ExtractedModule(
            name=name,
            lookup_names=module_lookup_names,
            functions=functions or {},
        )
    }


def _make_extraction_config(
    *,
    source_root: Path,
    clang_include: list[str] = (),
    clang_include_directory: list[str] = (),
    clang_c_std: str = "c11",
    clang_cpp_std: str = "c++17",
) -> dict[str, object]:
    return {
        "source_root": source_root,
        "clang_include": list(clang_include),
        "clang_include_directory": translation_unit_module.inject_python_include_directories(
            list(clang_include_directory)
        ),
        "clang_c_std": clang_c_std,
        "clang_cpp_std": clang_cpp_std,
    }


class CSignatureExtractor:
    def __init__(
        self,
        source_root: Path,
        *,
        clang_include: list[str] = (),
        clang_include_directory: list[str] = (),
        clang_c_std: str = "c11",
        clang_cpp_std: str = "c++17",
    ) -> None:
        self._source_root = source_root
        self._clang_include = list(clang_include)
        self._clang_include_directory = translation_unit_module.inject_python_include_directories(
            list(clang_include_directory)
        )
        self._clang_c_std = clang_c_std
        self._clang_cpp_std = clang_cpp_std

    def extract_modules(self) -> dict[str, ExtractedModule]:
        return extract_c_signature_modules(
            self._source_root,
            clang_include=self._clang_include,
            clang_include_directory=self._clang_include_directory,
            clang_c_std=self._clang_c_std,
            clang_cpp_std=self._clang_cpp_std,
        )


class _FakeExtractor:
    def __init__(
        self,
        modules: dict[str, ExtractedModule] | None = None,
    ) -> None:
        self.modules = modules or {}
        self.called = 0

    def extract_modules(self) -> dict[str, ExtractedModule]:
        self.called += 1
        return self.modules


def _patch_c_signature_extractor(
    monkeypatch: pytest.MonkeyPatch,
    modules: dict[str, ExtractedModule] | None = None,
) -> _FakeExtractor:
    extractor = _FakeExtractor(modules=modules)

    def _patched_extract_c_signature_modules(
        source_root: Path,
        *,
        clang_include: list[str] = (),
        clang_include_directory: list[str] = (),
        clang_c_std: str = "c11",
        clang_cpp_std: str = "c++17",
    ) -> dict[str, ExtractedModule]:
        _ = (source_root, clang_include, clang_include_directory, clang_c_std, clang_cpp_std)
        return extractor.extract_modules()

    import core.node_visitors.c_signature_extraction.c_signature_extraction_visitor as visitor_module

    monkeypatch.setattr(visitor_module, "extract_c_signature_modules", _patched_extract_c_signature_modules)
    return extractor


def _patch_raising_c_signature_extractor(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def _patched_extract_c_signature_modules(
        source_root: Path,
        *,
        clang_include: list[str] = (),
        clang_include_directory: list[str] = (),
        clang_c_std: str = "c11",
        clang_cpp_std: str = "c++17",
    ) -> dict[str, ExtractedModule]:
        _ = (source_root, clang_include, clang_include_directory, clang_c_std, clang_cpp_std)
        raise error

    import core.node_visitors.c_signature_extraction.c_signature_extraction_visitor as visitor_module

    monkeypatch.setattr(visitor_module, "extract_c_signature_modules", _patched_extract_c_signature_modules)


def _get_packaged_libclang_path() -> str | None:
    import clang

    native_dir = Path(clang.__file__).resolve().parent / "native"
    for filename in ("libclang.dll", "libclang.so", "libclang.dylib"):
        candidate = native_dir / filename
        if candidate.exists():
            return str(candidate)
    return None


class _FakeDiagnosticType:
    Ignored = 0
    Note = 1
    Warning = 2
    Error = 3
    Fatal = 4


class _FakeClangWithDiagnostics:
    Diagnostic = _FakeDiagnosticType


class _FakeDiagnosticFile:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeDiagnosticLocation:
    def __init__(self, *, file_name: str | None, line: int, column: int) -> None:
        self.file = _FakeDiagnosticFile(file_name) if file_name is not None else None
        self.line = line
        self.column = column


class _FakeDiagnostic:
    def __init__(
        self,
        *,
        severity: int,
        message: str,
        file_name: str | None,
        line: int,
        column: int,
    ) -> None:
        self.severity = severity
        self.spelling = message
        self.location = _FakeDiagnosticLocation(file_name=file_name, line=line, column=column)


class _FakeTranslationUnit:
    def __init__(self, diagnostics: list[_FakeDiagnostic]) -> None:
        self.diagnostics = diagnostics


class _FakeIndex:
    def __init__(self, translation_unit: _FakeTranslationUnit) -> None:
        self.translation_unit = translation_unit

    def parse(self, filename: str, args: list[str]) -> _FakeTranslationUnit:
        return self.translation_unit


class _SequentialIndex:
    def __init__(self, translation_units: list[_FakeTranslationUnit]) -> None:
        self._translation_units = translation_units
        self._index = 0
        self.calls: list[tuple[str, list[str]]] = []

    def parse(self, filename: str, args: list[str]) -> _FakeTranslationUnit:
        self.calls.append((filename, list(args)))
        if not self._translation_units:
            raise AssertionError("translation_units must not be empty")
        if self._index < len(self._translation_units):
            current = self._translation_units[self._index]
            self._index += 1
            return current
        return self._translation_units[-1]


def _has_include_directory_arg(args: list[str], include_dir: str | Path) -> bool:
    include_dir_str = str(include_dir)
    for index, token in enumerate(args):
        if token != "--include-directory":
            continue
        if index + 1 >= len(args):
            continue
        if args[index + 1] == include_dir_str:
            return True
    return False


def _has_std_arg(args: list[str], std_value: str) -> bool:
    for index, token in enumerate(args):
        if token != "--std":
            continue
        if index + 1 >= len(args):
            continue
        if args[index + 1] == std_value:
            return True
    return False


def test_c_ast_visitor_rewrites_module_function_and_drops_self(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    func = IRFunction(name="foo", args=_generic_signature())
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[func],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    ml_flags=METH_VARARGS,
                    signatures=[
                        ExtractedSignature(
                            arguments=[
                                ExtractedArgument(name="self", type_name="object"),
                                ExtractedArgument(name="x", type_name="int"),
                                ExtractedArgument(name="flag", type_name="bool", default_value="False"),
                            ],
                            return_type_name="int",
                        )
                    ],
                )
            }
        ),
    )

    visitor = CSignatureExtractionVisitor(
        source_root=tmp_path,
    )
    visitor.visit_module(module)

    rewritten = module.functions[0]
    assert [arg.name for arg in rewritten.args] == ["x", "flag"]
    assert rewritten.args[0].type_name == "int"
    assert rewritten.args[1].type_name == "bool"
    assert rewritten.args[1].default_value is not None
    assert rewritten.args[1].default_value == "False"
    assert rewritten.return_type_name is not None
    assert rewritten.return_type_name == "int"


def test_c_ast_visitor_does_not_log_for_non_generic_function(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    visitor = CSignatureExtractionVisitor(
        source_root=tmp_path,
    )
    func = IRFunction(name="foo", args=[IRArgument(name="x", kind=IRArgumentKind.POSITIONAL_OR_KEYWORD)])
    signatures = {
        "foo": ExtractedFunction(
            ml_name="foo",
            function_cursor=_fake_function_cursor("foo"),
            ml_flags=METH_VARARGS,
            signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")])],
        )
    }

    with caplog.at_level(logging.INFO, logger="core"):
        rewritten = visitor._rewrite_function(func=func, signatures=signatures, is_method=False)

    assert rewritten == [func]
    assert caplog.records == []


def test_c_ast_visitor_log_summary_resets_after_logging(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    _patch_c_signature_extractor(
        monkeypatch,
        modules={
            **_module_fixture(
                name="pkg.first",
                functions={
                    "foo": ExtractedFunction(
                        ml_name="foo",
                        function_cursor=_fake_function_cursor("foo"),
                        ml_flags=METH_VARARGS,
                        signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")])],
                    )
                },
            ),
            **_module_fixture(
                name="pkg.second",
                functions={
                    "bar": ExtractedFunction(
                        ml_name="bar",
                        function_cursor=_fake_function_cursor("bar"),
                        ml_flags=METH_VARARGS,
                        signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="y", type_name="int")])],
                    )
                },
            ),
        },
    )
    visitor = CSignatureExtractionVisitor(
        source_root=tmp_path,
    )
    first_module = IRModule(
        full_name=QualifiedName.from_str("pkg.first"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", args=_generic_signature())],
    )
    second_module = IRModule(
        full_name=QualifiedName.from_str("pkg.second"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="bar", args=_generic_signature())],
    )

    with caplog.at_level(logging.INFO, logger="core"):
        visitor.visit_module(first_module)
        visitor.visit_module(second_module)
        visitor.log_summary("pkg")
        after_first_summary = len(caplog.records)
        visitor.log_summary("pkg")

    assert len(caplog.records) == after_first_summary
    summary_records = [
        record
        for record in caplog.records
        if record.message.startswith("C AST signature inference summary for pkg:")
    ]
    assert len(summary_records) == 1
    assert summary_records[0].message == (
        "C AST signature inference summary for pkg: "
        "total_generic=2, success=2, failed=0, no_candidates=0, "
        "empty_selected_signatures=0, empty_extract=0"
    )


def test_c_signature_engine_logs_all_diagnostics_when_error_present(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    config = _make_extraction_config(source_root=tmp_path, clang_c_std="c11")
    source = tmp_path / "module.c"
    translation_unit = _FakeTranslationUnit(
        diagnostics=[
            _FakeDiagnostic(
                severity=_FakeDiagnosticType.Warning,
                message="warning detail",
                file_name=str(source),
                line=3,
                column=1,
            ),
            _FakeDiagnostic(
                severity=_FakeDiagnosticType.Error,
                message="error detail",
                file_name=str(source),
                line=7,
                column=9,
            ),
            _FakeDiagnostic(
                severity=_FakeDiagnosticType.Fatal,
                message="fatal detail",
                file_name=str(source),
                line=11,
                column=4,
            ),
        ]
    )

    with caplog.at_level(logging.WARNING, logger="core"):
        result = translation_unit_module.parse_translation_unit(
            index=_FakeIndex(translation_unit),
            file_path=source,
            source_root=config["source_root"],
            clang_include=config["clang_include"],
            clang_include_directory=config["clang_include_directory"],
            clang_c_std=config["clang_c_std"],
            clang_cpp_std=config["clang_cpp_std"],
        )

    assert result is translation_unit
    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert str(source) in message
    assert "suffix: .c" in message
    expected_parse_args = translation_unit_module.build_clang_parse_args(
        source,
        clang_include=config["clang_include"],
        clang_include_directory=config["clang_include_directory"],
        clang_c_std=config["clang_c_std"],
        clang_cpp_std=config["clang_cpp_std"],
    )
    assert f"parse_args: {expected_parse_args!r}" in message
    assert f"[WARNING] {source}:3:1: warning detail" in message
    assert f"[ERROR] {source}:7:9: error detail" in message
    assert f"[FATAL] {source}:11:4: fatal detail" in message


def test_c_signature_engine_skips_logging_for_non_error_diagnostics(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    config = _make_extraction_config(source_root=tmp_path, clang_c_std="c11")
    source = tmp_path / "module.c"
    translation_unit = _FakeTranslationUnit(
        diagnostics=[
            _FakeDiagnostic(
                severity=_FakeDiagnosticType.Note,
                message="note detail",
                file_name=str(source),
                line=2,
                column=5,
            ),
            _FakeDiagnostic(
                severity=_FakeDiagnosticType.Warning,
                message="warning detail",
                file_name=str(source),
                line=4,
                column=6,
            ),
        ]
    )

    with caplog.at_level(logging.WARNING, logger="core"):
        result = translation_unit_module.parse_translation_unit(
            index=_FakeIndex(translation_unit),
            file_path=source,
            source_root=config["source_root"],
            clang_include=config["clang_include"],
            clang_include_directory=config["clang_include_directory"],
            clang_c_std=config["clang_c_std"],
            clang_cpp_std=config["clang_cpp_std"],
        )

    assert result is translation_unit
    assert caplog.records == []


def test_c_signature_engine_auto_adds_include_dir_for_nested_header_literal(tmp_path: Path) -> None:
    config = _make_extraction_config(source_root=tmp_path, clang_c_std="c11")
    source = tmp_path / "src" / "module.c"
    header_path = tmp_path / "numpy_core" / "include" / "numpy" / "npy_common.h"
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text("/* header */", encoding="utf-8")

    first = _FakeTranslationUnit(
        diagnostics=[
            _FakeDiagnostic(
                severity=clang.cindex.Diagnostic.Fatal,
                message="'numpy/npy_common.h' file not found",
                file_name=str(source),
                line=1,
                column=1,
            )
        ]
    )
    second = _FakeTranslationUnit(diagnostics=[])
    index = _SequentialIndex([first, second])

    result = translation_unit_module.parse_translation_unit(
        index=index,
        file_path=source,
        source_root=config["source_root"],
        clang_include=config["clang_include"],
        clang_include_directory=config["clang_include_directory"],
        clang_c_std=config["clang_c_std"],
        clang_cpp_std=config["clang_cpp_std"],
    )

    assert result is second
    expected_include_root = header_path.parents[1]
    assert str(expected_include_root) in config["clang_include_directory"]
    assert str(header_path.parent) not in config["clang_include_directory"]
    assert len(index.calls) == 2
    assert _has_std_arg(index.calls[0][1], "c11")
    assert _has_std_arg(index.calls[1][1], "c11")


def test_c_signature_engine_logs_info_when_auto_include_is_added(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    config = _make_extraction_config(source_root=tmp_path, clang_c_std="c11")
    source = tmp_path / "src" / "module.c"
    header_path = tmp_path / "numpy_core" / "include" / "numpy" / "npy_common.h"
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text("/* header */", encoding="utf-8")

    first = _FakeTranslationUnit(
        diagnostics=[
            _FakeDiagnostic(
                severity=clang.cindex.Diagnostic.Fatal,
                message="'numpy/npy_common.h' file not found",
                file_name=str(source),
                line=1,
                column=1,
            )
        ]
    )
    second = _FakeTranslationUnit(diagnostics=[])
    index = _SequentialIndex([first, second])

    with caplog.at_level(logging.INFO, logger="core"):
        _ = translation_unit_module.parse_translation_unit(
            index=index,
            file_path=source,
            source_root=config["source_root"],
            clang_include=config["clang_include"],
            clang_include_directory=config["clang_include_directory"],
            clang_c_std=config["clang_c_std"],
            clang_cpp_std=config["clang_cpp_std"],
        )

    messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
    assert any("Auto-added clang include path for missing header numpy/npy_common.h" in message for message in messages)
    assert any(str(header_path.parents[1]) in message for message in messages)


def test_c_signature_engine_retries_until_missing_includes_converge(tmp_path: Path) -> None:
    config = _make_extraction_config(source_root=tmp_path, clang_c_std="c11")
    source = tmp_path / "pkg" / "src" / "module.c"

    include_one = tmp_path / "vendor1" / "include"
    include_two = tmp_path / "vendor2" / "include"
    (include_one / "numpy").mkdir(parents=True, exist_ok=True)
    (include_two / "pkg").mkdir(parents=True, exist_ok=True)
    (include_one / "numpy" / "npy_common.h").write_text("/* one */", encoding="utf-8")
    (include_two / "pkg" / "extra.h").write_text("/* two */", encoding="utf-8")

    first = _FakeTranslationUnit(
        diagnostics=[
            _FakeDiagnostic(
                severity=clang.cindex.Diagnostic.Fatal,
                message="'numpy/npy_common.h' file not found",
                file_name=str(source),
                line=2,
                column=7,
            )
        ]
    )
    second = _FakeTranslationUnit(
        diagnostics=[
            _FakeDiagnostic(
                severity=clang.cindex.Diagnostic.Fatal,
                message="'pkg/extra.h' file not found",
                file_name=str(source),
                line=3,
                column=5,
            )
        ]
    )
    third = _FakeTranslationUnit(diagnostics=[])
    index = _SequentialIndex([first, second, third])

    result = translation_unit_module.parse_translation_unit(
        index=index,
        file_path=source,
        source_root=config["source_root"],
        clang_include=config["clang_include"],
        clang_include_directory=config["clang_include_directory"],
        clang_c_std=config["clang_c_std"],
        clang_cpp_std=config["clang_cpp_std"],
    )

    include_arg_one = str(include_one)
    include_arg_two = str(include_two)
    assert result is third
    assert include_arg_one in config["clang_include_directory"]
    assert include_arg_two in config["clang_include_directory"]
    assert len(index.calls) == 3
    assert _has_std_arg(index.calls[0][1], "c11")
    assert _has_std_arg(index.calls[1][1], "c11")
    assert _has_std_arg(index.calls[2][1], "c11")
    assert not _has_include_directory_arg(index.calls[0][1], include_arg_one)
    assert _has_include_directory_arg(index.calls[1][1], include_arg_one)
    assert not _has_include_directory_arg(index.calls[1][1], include_arg_two)
    assert _has_include_directory_arg(index.calls[2][1], include_arg_one)
    assert _has_include_directory_arg(index.calls[2][1], include_arg_two)


def test_c_signature_engine_does_not_retry_when_missing_header_is_unresolved(tmp_path: Path) -> None:
    config = _make_extraction_config(source_root=tmp_path, clang_c_std="c11")
    source = tmp_path / "src" / "module.c"
    initial_include_dirs = list(config["clang_include_directory"])

    unrelated_header = tmp_path / "include" / "numpy" / "arrayobject.h"
    unrelated_header.parent.mkdir(parents=True, exist_ok=True)
    unrelated_header.write_text("/* unrelated */", encoding="utf-8")

    unresolved = _FakeTranslationUnit(
        diagnostics=[
            _FakeDiagnostic(
                severity=clang.cindex.Diagnostic.Fatal,
                message="'numpy/npy_common.h' file not found",
                file_name=str(source),
                line=6,
                column=3,
            )
        ]
    )
    index = _SequentialIndex([unresolved])

    result = translation_unit_module.parse_translation_unit(
        index=index,
        file_path=source,
        source_root=config["source_root"],
        clang_include=config["clang_include"],
        clang_include_directory=config["clang_include_directory"],
        clang_c_std=config["clang_c_std"],
        clang_cpp_std=config["clang_cpp_std"],
    )

    assert result is unresolved
    assert config["clang_include_directory"] == initial_include_dirs
    assert len(index.calls) == 1


def test_c_ast_visitor_matches_candidates_by_module_before_function_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_c_signature_extractor(
        monkeypatch,
        modules={
            "pkg.first": ExtractedModule(
                name="pkg.first",
                lookup_names={"pkg.first", "first"},
                functions={
                    "foo": ExtractedFunction(
                        ml_name="foo",
                        function_cursor=_fake_function_cursor("foo"),
                        ml_flags=METH_VARARGS,
                        signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")])],
                    )
                },
            ),
            "pkg.second": ExtractedModule(
                name="pkg.second",
                lookup_names={"pkg.second", "second"},
                functions={
                    "foo": ExtractedFunction(
                        ml_name="foo",
                        function_cursor=_fake_function_cursor("foo"),
                        ml_flags=METH_VARARGS,
                        signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="value", type_name="float")])],
                    )
                },
            ),
        },
    )
    visitor = CSignatureExtractionVisitor(
        source_root=tmp_path,
    )
    first_module = IRModule(
        full_name=QualifiedName.from_str("pkg.first"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", args=_generic_signature())],
    )
    second_module = IRModule(
        full_name=QualifiedName.from_str("pkg.second"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", args=_generic_signature())],
    )

    visitor.visit_module(first_module)
    visitor.visit_module(second_module)

    assert [arg.name for arg in first_module.functions[0].args] == ["x"]
    assert first_module.functions[0].args[0].type_name == "int"
    assert [arg.name for arg in second_module.functions[0].args] == ["value"]
    assert second_module.functions[0].args[0].type_name == "float"


def test_c_ast_visitor_rejects_ambiguous_leaf_module_match_without_global_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    _patch_c_signature_extractor(
        monkeypatch,
        modules={
            "one": ExtractedModule(
                name="one",
                lookup_names={"mod"},
                functions={
                    "foo": ExtractedFunction(
                        ml_name="foo",
                        function_cursor=_fake_function_cursor("foo"),
                        ml_flags=METH_VARARGS,
                        signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")])],
                    )
                },
            ),
            "two": ExtractedModule(
                name="two",
                lookup_names={"mod"},
                functions={
                    "foo": ExtractedFunction(
                        ml_name="foo",
                        function_cursor=_fake_function_cursor("foo"),
                        ml_flags=METH_VARARGS,
                        signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="y", type_name="float")])],
                    )
                },
            ),
        },
    )
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", args=_generic_signature())],
    )
    visitor = CSignatureExtractionVisitor(
        source_root=tmp_path,
    )

    with caplog.at_level(logging.WARNING, logger="core"):
        visitor.visit_module(module)

    assert module.functions[0].is_generic_signature()
    assert (
        "Failed to rewrite generic signature for foo (is_method=False): no C signature candidates found"
        in caplog.text
    )


def test_c_ast_visitor_overwrites_existing_return_with_raw_inferred_return(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    func = IRFunction(
        name="foo",
        args=_generic_signature(),
        return_type_name="bytes",
    )
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[func],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    ml_flags=METH_VARARGS,
                    signatures=[
                        ExtractedSignature(
                            arguments=[ExtractedArgument(name="x", type_name="int")],
                            return_type_name="typing.Optional[int]",
                        )
                    ],
                )
            }
        ),
    )

    visitor = CSignatureExtractionVisitor(
        source_root=tmp_path,
    )
    visitor.visit_module(module)

    rewritten = module.functions[0]
    assert rewritten.return_type_name is not None
    assert rewritten.return_type_name == "typing.Optional[int]"


def test_c_ast_visitor_skips_python_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[IRFunction(name="foo", args=_generic_signature())],
    )
    extractor = _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")])],
                )
            }
        ),
    )
    visitor = CSignatureExtractionVisitor(
        source_root=tmp_path,
    )
    visitor.visit_module(module)

    assert module.functions[0].is_generic_signature()
    assert extractor.called == 0


def test_c_ast_visitor_propagates_signature_extraction_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", args=_generic_signature())],
    )
    _patch_raising_c_signature_extractor(monkeypatch, RuntimeError("boom"))

    visitor = CSignatureExtractionVisitor(
        source_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="boom"):
        visitor.visit_module(module)


def test_write_stubs_skips_c_ast_visitor_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core as stubgen_module
    from core.stub_generation_options import StubGenerationOptions

    captured_output_dirs: list[Path] = []

    def _unexpected_constructor(*args: object, **kwargs: object) -> object:
        raise AssertionError("CSignatureExtractionVisitor should not be instantiated when disabled")

    def _record_writer_output_dir(self: object, module: IRModule, printer: object, to: Path) -> None:
        _ = (self, module, printer)
        captured_output_dirs.append(to)
        (to / "math.pyi").write_text("", encoding="utf-8")

    monkeypatch.setattr(stubgen_module, "CSignatureExtractionVisitor", _unexpected_constructor)
    monkeypatch.setattr(stubgen_module.Writer, "write", _record_writer_output_dir)

    options = StubGenerationOptions(
        enable_docstring_signature_parser=False,
    )
    stubgen_module.write_stubs("math", tmp_path, options=options)

    assert captured_output_dirs == [tmp_path]
    assert captured_output_dirs[0] is tmp_path
    assert list(tmp_path.rglob("*.pyi"))


def test_write_stubs_skips_c_inference_when_source_root_not_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core as stubgen_module
    from core.stub_generation_options import StubGenerationOptions

    def _unexpected_constructor(*args: object, **kwargs: object) -> object:
        raise AssertionError("CSignatureExtractionVisitor should not be instantiated when source_root is not set")

    monkeypatch.setattr(stubgen_module, "CSignatureExtractionVisitor", _unexpected_constructor)

    options = StubGenerationOptions(
        enable_docstring_signature_parser=False,
        source_root=None,
    )
    stubgen_module.write_stubs("math", tmp_path, options=options)


def test_write_stubs_defaults_do_not_require_source_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core as stubgen_module
    from core.stub_generation_options import StubGenerationOptions

    def _unexpected_constructor(*args: object, **kwargs: object) -> object:
        raise AssertionError("CSignatureExtractionVisitor should not be instantiated by default")

    monkeypatch.setattr(stubgen_module, "CSignatureExtractionVisitor", _unexpected_constructor)

    options = StubGenerationOptions()
    assert options.source_root is None
    stubgen_module.write_stubs("math", tmp_path, options=options)

    assert list(tmp_path.rglob("*.pyi"))


def test_write_stubs_logs_to_output_file_and_cleans_up_handler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core as stubgen_module
    from core.stub_generation_options import StubGenerationOptions

    def _emit_warning(self: Pipeline, module: IRModule) -> None:
        _ = (self, module)
        logging.getLogger("core.tests").warning("file log works")

    monkeypatch.setattr(stubgen_module.Pipeline, "run", _emit_warning)

    options = StubGenerationOptions(
        enable_docstring_signature_parser=False,
    )
    stubgen_module.write_stubs("math", tmp_path, options=options)

    log_file = tmp_path / "pcstubgen.log"
    assert log_file.exists()
    assert log_file.read_text(encoding="utf-8") == (
        "[WARNING] - core.tests\n"
        "file log works\n"
        "\n"
    )

    package_logger = logging.getLogger("core")
    assert all(
        not isinstance(handler, logging.FileHandler)
        or Path(handler.baseFilename) != log_file
        for handler in package_logger.handlers
    )


def test_write_stubs_logs_project_level_c_ast_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core as stubgen_module
    from core.stub_generation_options import StubGenerationOptions

    ir_module = IRModule(
        full_name=QualifiedName.from_str("pkg"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", args=_generic_signature())],
        classes=[
            IRClass(
                name="Builder",
                methods=[IRMethod(function=IRFunction(name="build", args=_generic_signature()), decorator=None)],
            )
        ],
        sub_modules=[
            IRModule(
                full_name=QualifiedName.from_str("pkg.child"),
                module_type=IRModuleType.EXTENSION,
                functions=[IRFunction(name="bar", args=_generic_signature())],
            )
        ],
    )
    monkeypatch.setattr(stubgen_module, "build_module", lambda path, module: ir_module)
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            name="pkg",
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    ml_flags=METH_VARARGS,
                    signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")])],
                )
            }
        ),
    )

    options = StubGenerationOptions(
        enable_docstring_signature_parser=False,
        source_root=tmp_path,
    )
    stubgen_module.write_stubs("math", tmp_path, options=options)

    log_text = (tmp_path / "pcstubgen.log").read_text(encoding="utf-8")
    assert (
        "C AST signature inference summary for pkg: "
        "total_generic=2, success=1, failed=1, no_candidates=1, "
        "empty_selected_signatures=0, empty_extract=0"
    ) in log_text


def test_write_stubs_logs_empty_extract_summary_with_per_item_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core as stubgen_module
    from core.stub_generation_options import StubGenerationOptions

    ir_module = IRModule(
        full_name=QualifiedName.from_str("pkg"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", args=_generic_signature())],
        classes=[
            IRClass(
                name="Builder",
                methods=[IRMethod(function=IRFunction(name="build", args=_generic_signature()), decorator=None)],
            )
        ],
    )
    monkeypatch.setattr(stubgen_module, "build_module", lambda path, module: ir_module)
    _patch_c_signature_extractor(monkeypatch, modules={})

    options = StubGenerationOptions(
        enable_docstring_signature_parser=False,
        source_root=tmp_path,
    )
    stubgen_module.write_stubs("math", tmp_path, options=options)

    log_text = (tmp_path / "pcstubgen.log").read_text(encoding="utf-8")
    assert (
        "Failed to rewrite generic signature for foo (is_method=False): "
        "C signature extraction returned no results"
    ) in log_text
    assert (
        "C AST signature inference summary for pkg: "
        "total_generic=1, success=0, failed=1, no_candidates=0, "
        "empty_selected_signatures=0, empty_extract=1"
    ) in log_text


def test_write_stubs_propagates_extract_errors_without_logging_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core as stubgen_module
    from core.stub_generation_options import StubGenerationOptions

    ir_module = IRModule(
        full_name=QualifiedName.from_str("pkg"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", args=_generic_signature())],
        classes=[
            IRClass(
                name="Builder",
                methods=[IRMethod(function=IRFunction(name="build", args=_generic_signature()), decorator=None)],
            )
        ],
        sub_modules=[
            IRModule(
                full_name=QualifiedName.from_str("pkg.child"),
                module_type=IRModuleType.EXTENSION,
                functions=[IRFunction(name="bar", args=_generic_signature())],
            )
        ],
    )
    monkeypatch.setattr(stubgen_module, "build_module", lambda path, module: ir_module)
    _patch_raising_c_signature_extractor(monkeypatch, RuntimeError("boom"))

    options = StubGenerationOptions(
        enable_docstring_signature_parser=False,
        source_root=tmp_path,
    )
    with pytest.raises(RuntimeError, match="boom"):
        stubgen_module.write_stubs("math", tmp_path, options=options)

    log_text = (tmp_path / "pcstubgen.log").read_text(encoding="utf-8")
    assert "Failed to extract C signatures: boom" not in log_text
    assert "C signature extraction failed" not in log_text
    assert "C AST signature inference summary for pkg:" not in log_text


def test_doc_parser_runs_before_c_ast_visitor_in_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[
            IRFunction(
                name="foo",
                args=_generic_signature(),
                doc="foo(a: int, b: int = 0) -> int",
            )
        ],
    )
    extractor = _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")])],
                )
            }
        ),
    )
    pipeline = Pipeline(
        [
            DocStringSignatureParserVisitor(),
            CSignatureExtractionVisitor(
                source_root=tmp_path,
            ),
        ]
    )
    pipeline.run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.args] == ["a", "b"]
    assert [arg.type_name for arg in parsed.args] == ["int", "int"]
    assert extractor.called == 1


def test_doc_parser_prevents_no_candidate_warning_after_signature_rewrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[
            IRFunction(
                name="cdist_minkowski",
                args=_generic_signature(),
                doc=(
                    "cdist_minkowski(x: object, y: object, w: object = None, "
                    "out: object = None, p: typing.SupportsFloat = 2.0) -> numpy.ndarray"
                ),
            )
        ],
    )
    extractor = _patch_c_signature_extractor(monkeypatch, modules={})

    with caplog.at_level(
        logging.WARNING,
        logger="core.node_visitors.c_signature_extraction.c_signature_extraction_visitor",
    ):
        Pipeline(
            [
                DocStringSignatureParserVisitor(),
                CSignatureExtractionVisitor(
                    source_root=tmp_path,
                ),
            ]
        ).run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.args] == ["x", "y", "w", "out", "p"]
    assert parsed.return_type_name == "numpy.ndarray"
    assert "Failed to rewrite generic signature for cdist_minkowski" not in caplog.text
    assert extractor.called == 1


def test_c_signature_extraction_engine_extract_modules_isolates_same_named_functions_per_module(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    first_source = tmp_path / "first.c"
    second_source = tmp_path / "second.c"
    for source, module_name, c_name in [
        (first_source, "first", "first_foo_impl"),
        (second_source, "second", "second_foo_impl"),
    ]:
        source.write_text(
            "\n".join(
                [
                    "typedef struct _object PyObject;",
                    "typedef struct PyMethodDef {",
                    "    const char* ml_name;",
                    "    void* ml_meth;",
                    "    int ml_flags;",
                    "    const char* ml_doc;",
                    "} PyMethodDef;",
                    "typedef struct PyModuleDef {",
                    "    int m_base;",
                    "    const char* m_name;",
                    "    const char* m_doc;",
                    "    int m_size;",
                    "    PyMethodDef* m_methods;",
                    "    void* m_slots;",
                    "    void* m_traverse;",
                    "    void* m_clear;",
                    "    void* m_free;",
                    "} PyModuleDef;",
                    "#define PyModuleDef_HEAD_INIT 0",
                    "#define METH_VARARGS 1",
                    "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                    f"static PyObject* {c_name}(PyObject* self, PyObject* args) {{",
                    "    int value = 0;",
                    "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                    "        return (PyObject*)0;",
                    "    }",
                    "    return (PyObject*)0;",
                    "}",
                    "static PyMethodDef Methods[] = {",
                    f"    {{\"foo\", {c_name}, METH_VARARGS, \"doc\"}},",
                    "    {0, 0, 0, 0}",
                    "};",
                    "static PyModuleDef moduledef = {",
                    "    PyModuleDef_HEAD_INIT,",
                    f"    \"{module_name}\",",
                    "    0,",
                    "    -1,",
                    "    Methods,",
                    "    0, 0, 0, 0",
                    "};",
                ]
            ),
            encoding="utf-8",
        )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    extracted = engine.extract_modules()

    assert set(extracted) == {"first", "second"}
    assert set(extracted["first"].functions) == {"foo"}
    assert set(extracted["second"].functions) == {"foo"}
    assert extracted["first"].functions["foo"].ml_flags == METH_VARARGS
    assert extracted["second"].functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_extract_modules_handles_multiple_moduledefs_in_one_file(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "multi_init_module.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* first_foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyObject* second_foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef FirstMethods[] = {",
                "    {\"foo\", first_foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyMethodDef SecondMethods[] = {",
                "    {\"foo\", second_foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef first_moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"first\",",
                "    0,",
                "    -1,",
                "    FirstMethods,",
                "    0, 0, 0, 0",
                "};",
                "static PyModuleDef second_moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"second\",",
                "    0,",
                "    -1,",
                "    SecondMethods,",
                "    0, 0, 0, 0",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    extracted = engine.extract_modules()

    assert set(extracted) == {"first", "second"}
    assert set(extracted["first"].functions) == {"foo"}
    assert set(extracted["second"].functions) == {"foo"}
    assert extracted["first"].functions["foo"].ml_flags == METH_VARARGS
    assert extracted["second"].functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_discards_duplicate_modules_across_files(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    first_source = tmp_path / "a_first.c"
    second_source = tmp_path / "b_second.c"
    for source, py_name, c_name in [
        (first_source, "foo", "first_foo_impl"),
        (second_source, "bar", "second_bar_impl"),
    ]:
        source.write_text(
            "\n".join(
                [
                    "typedef struct _object PyObject;",
                    "typedef struct PyMethodDef {",
                    "    const char* ml_name;",
                    "    void* ml_meth;",
                    "    int ml_flags;",
                    "    const char* ml_doc;",
                    "} PyMethodDef;",
                    "typedef struct PyModuleDef {",
                    "    int m_base;",
                    "    const char* m_name;",
                    "    const char* m_doc;",
                    "    int m_size;",
                    "    PyMethodDef* m_methods;",
                    "    void* m_slots;",
                    "    void* m_traverse;",
                    "    void* m_clear;",
                    "    void* m_free;",
                    "} PyModuleDef;",
                    "#define PyModuleDef_HEAD_INIT 0",
                    "#define METH_VARARGS 1",
                    "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                    f"static PyObject* {c_name}(PyObject* self, PyObject* args) {{",
                    "    int value = 0;",
                    "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                    "        return (PyObject*)0;",
                    "    }",
                    "    return (PyObject*)0;",
                    "}",
                    "static PyMethodDef Methods[] = {",
                    f"    {{\"{py_name}\", {c_name}, METH_VARARGS, \"doc\"}},",
                    "    {0, 0, 0, 0}",
                    "};",
                    "static PyModuleDef moduledef = {",
                    "    PyModuleDef_HEAD_INIT,",
                    "    \"dup.shared\",",
                    "    0,",
                    "    -1,",
                    "    Methods,",
                    "    0, 0, 0, 0",
                    "};",
                ]
            ),
            encoding="utf-8",
        )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    with caplog.at_level(logging.WARNING, logger="core"):
        extracted = engine.extract_modules()

    module = extracted["dup.shared"]
    assert set(module.functions) == {"foo"}
    assert module.functions["foo"].ml_flags == METH_VARARGS
    assert (
        "Discarded duplicate extracted module dup.shared: "
        "kept existing module, discarded incoming module"
    ) in caplog.text


def test_c_signature_extraction_engine_discards_duplicate_modules_in_one_file(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "duplicate_modules.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* first_foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyObject* second_bar_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef FirstMethods[] = {",
                "    {\"foo\", first_foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyMethodDef SecondMethods[] = {",
                "    {\"bar\", second_bar_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef first_moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"dup.same_file\",",
                "    0,",
                "    -1,",
                "    FirstMethods,",
                "    0, 0, 0, 0",
                "};",
                "static PyModuleDef second_moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"dup.same_file\",",
                "    0,",
                "    -1,",
                "    SecondMethods,",
                "    0, 0, 0, 0",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    with caplog.at_level(logging.WARNING, logger="core"):
        extracted = engine.extract_modules()

    module = extracted["dup.same_file"]
    assert set(module.functions) == {"foo"}
    assert module.functions["foo"].ml_flags == METH_VARARGS
    assert (
        "Discarded duplicate extracted module dup.same_file: "
        "kept existing module, discarded incoming module"
    ) in caplog.text


def test_c_signature_extraction_engine_warns_and_keeps_first_duplicate_in_same_method_table(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "duplicate_methods.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* first_foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyObject* second_foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", first_foo_impl, METH_VARARGS, \"doc\"},",
                "    {\"foo\", second_foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"dup.mod\",",
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    with caplog.at_level(logging.WARNING, logger="core"):
        extracted = engine.extract_modules()

    module = extracted["dup.mod"]
    assert module.functions["foo"].ml_flags == METH_VARARGS
    assert (
        "Discarded duplicate extracted function in module dup.mod for Python name foo: "
        "kept existing function, discarded incoming function"
    ) in caplog.text


def test_c_signature_extraction_engine_warns_and_discards_duplicate_module_across_files(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    first_source = tmp_path / "a_first.c"
    second_source = tmp_path / "b_second.c"
    for source, c_name in [
        (first_source, "first_foo_impl"),
        (second_source, "second_foo_impl"),
    ]:
        source.write_text(
            "\n".join(
                [
                    "typedef struct _object PyObject;",
                    "typedef struct PyMethodDef {",
                    "    const char* ml_name;",
                    "    void* ml_meth;",
                    "    int ml_flags;",
                    "    const char* ml_doc;",
                    "} PyMethodDef;",
                    "typedef struct PyModuleDef {",
                    "    int m_base;",
                    "    const char* m_name;",
                    "    const char* m_doc;",
                    "    int m_size;",
                    "    PyMethodDef* m_methods;",
                    "    void* m_slots;",
                    "    void* m_traverse;",
                    "    void* m_clear;",
                    "    void* m_free;",
                    "} PyModuleDef;",
                    "#define PyModuleDef_HEAD_INIT 0",
                    "#define METH_VARARGS 1",
                    "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                    f"static PyObject* {c_name}(PyObject* self, PyObject* args) {{",
                    "    int value = 0;",
                    "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                    "        return (PyObject*)0;",
                    "    }",
                    "    return (PyObject*)0;",
                    "}",
                    "static PyMethodDef Methods[] = {",
                    f"    {{\"foo\", {c_name}, METH_VARARGS, \"doc\"}},",
                    "    {0, 0, 0, 0}",
                    "};",
                    "static PyModuleDef moduledef = {",
                    "    PyModuleDef_HEAD_INIT,",
                    "    \"dup.shared\",",
                    "    0,",
                    "    -1,",
                    "    Methods,",
                    "    0, 0, 0, 0",
                    "};",
                ]
            ),
            encoding="utf-8",
        )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    with caplog.at_level(logging.WARNING, logger="core"):
        extracted = engine.extract_modules()

    module = extracted["dup.shared"]
    assert module.functions["foo"].ml_flags == METH_VARARGS
    assert (
        "Discarded duplicate extracted module dup.shared: "
        "kept existing module, discarded incoming module"
    ) in caplog.text


def test_c_signature_extraction_engine_extract_modules_ignores_registered_types_from_pymodule_addobject(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "module_with_type.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "typedef struct PyTypeObject {",
                "    const char* tp_name;",
                "    PyMethodDef* tp_methods;",
                "} PyTypeObject;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "PyObject* PyModule_Create(PyModuleDef* def);",
                "int PyModule_AddObject(PyObject* module, const char* name, PyObject* value);",
                "static PyObject* module_foo(PyObject* self, PyObject* args) {",
                "    int count = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &count)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyObject* point_foo(PyObject* self, PyObject* args) {",
                "    const char* label = 0;",
                "    if (!PyArg_ParseTuple(args, \"s\", &label)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef ModuleMethods[] = {",
                "    {\"foo\", module_foo, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyMethodDef PointMethods[] = {",
                "    {\"foo\", point_foo, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyTypeObject PointType = {",
                "    .tp_name = \"pkg.mod.Point\",",
                "    .tp_methods = PointMethods,",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"pkg.mod\",",
                "    0,",
                "    -1,",
                "    ModuleMethods,",
                "    0, 0, 0, 0",
                "};",
                "PyObject* PyInit_mod(void) {",
                "    PyObject* m = PyModule_Create(&moduledef);",
                "    PyModule_AddObject(m, \"Point\", (PyObject*)&PointType);",
                "    return m;",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    extracted = engine.extract_modules()

    module = extracted["pkg.mod"]
    assert module.functions["foo"].ml_name == "foo"
    assert module.functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_extract_modules_supports_pymodule_addobjectref(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "module_with_addobjectref_type.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "typedef struct PyTypeObject {",
                "    const char* tp_name;",
                "    PyMethodDef* tp_methods;",
                "} PyTypeObject;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "PyObject* PyModule_Create(PyModuleDef* def);",
                "int PyModule_AddObjectRef(PyObject* module, const char* name, PyObject* value);",
                "static PyObject* point_foo(PyObject* self, PyObject* args) {",
                "    const char* label = 0;",
                "    if (!PyArg_ParseTuple(args, \"s\", &label)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef ModuleMethods[] = {",
                "    {0, 0, 0, 0}",
                "};",
                "static PyMethodDef PointMethods[] = {",
                "    {\"foo\", point_foo, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyTypeObject PointType = {",
                "    .tp_name = \"pkg.mod.Point\",",
                "    .tp_methods = PointMethods,",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"pkg.mod\",",
                "    0,",
                "    -1,",
                "    ModuleMethods,",
                "    0, 0, 0, 0",
                "};",
                "PyObject* PyInit_mod(void) {",
                "    PyObject* m = PyModule_Create(&moduledef);",
                "    PyModule_AddObjectRef(m, \"Point\", (PyObject*)&PointType);",
                "    return m;",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    extracted = engine.extract_modules()

def test_c_signature_extraction_engine_extract_modules_supports_pymodule_addtype(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "module_with_addtype.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "typedef struct PyTypeObject {",
                "    const char* tp_name;",
                "    PyMethodDef* tp_methods;",
                "} PyTypeObject;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "PyObject* PyModule_Create(PyModuleDef* def);",
                "int PyModule_AddType(PyObject* module, PyTypeObject* type);",
                "static PyObject* point_foo(PyObject* self, PyObject* args) {",
                "    const char* label = 0;",
                "    if (!PyArg_ParseTuple(args, \"s\", &label)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef ModuleMethods[] = {",
                "    {0, 0, 0, 0}",
                "};",
                "static PyMethodDef PointMethods[] = {",
                "    {\"foo\", point_foo, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyTypeObject PointType = {",
                "    .tp_name = \"pkg.mod.Point\",",
                "    .tp_methods = PointMethods,",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"pkg.mod\",",
                "    0,",
                "    -1,",
                "    ModuleMethods,",
                "    0, 0, 0, 0",
                "};",
                "PyObject* PyInit_mod(void) {",
                "    PyObject* m = PyModule_Create(&moduledef);",
                "    PyModule_AddType(m, &PointType);",
                "    return m;",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    extracted = engine.extract_modules()

def test_c_signature_extraction_engine_extract_modules_supports_designated_moduledef_initializer(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "designated_module.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    .m_name = \"designated.mod\",",
                "    .m_doc = 0,",
                "    .m_size = -1,",
                "    .m_methods = Methods,",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    extracted = engine.extract_modules()

    assert "designated.mod" in extracted
    assert extracted["designated.mod"].functions["foo"].ml_name == "foo"
    assert extracted["designated.mod"].functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_extract_modules_supports_mixed_moduledef_initializer_styles(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "mixed_designated_module.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    .m_name = \"mixed.mod\",",
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    extracted = engine.extract_modules()

    assert "mixed.mod" in extracted
    assert extracted["mixed.mod"].functions["foo"].ml_name == "foo"
    assert extracted["mixed.mod"].functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_extract_modules_accepts_moduledefs_without_pyinit(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "unreachable_module.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"orphan.mod\",",
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    extracted = engine.extract_modules()

    assert extracted["orphan.mod"].functions["foo"].ml_name == "foo"
    assert extracted["orphan.mod"].functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_extract_modules_keeps_named_modules_without_methods(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "module_without_methods.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"empty.mod\",",
                "    0,",
                "    -1,",
                "    0,",
                "    0, 0, 0, 0",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    extracted = engine.extract_modules()

    assert "empty.mod" in extracted
    assert extracted["empty.mod"].functions == {}


def test_c_signature_extraction_engine_extract_modules_ignores_moduledefs_without_m_name(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "nameless_module.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    .m_doc = 0,",
                "    .m_size = -1,",
                "    .m_methods = Methods,",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    extracted = engine.extract_modules()

    assert extracted == {}


def test_c_signature_extraction_engine_does_not_extract_initializer_list_method_table_yet(tmp_path: Path) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "mini_ext.cpp"
    source.write_text(
        "\n".join(
            [
                "namespace std {",
                "template<class E> class initializer_list {",
                "public:",
                "    const E* begin() const;",
                "    const E* end() const;",
                "    unsigned long size() const;",
                "};",
                "}",
                "typedef struct _object PyObject;",
                "typedef PyObject* (*PyCFunction)(PyObject*, PyObject*);",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    PyCFunction ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* add_impl(PyObject* self, PyObject* args) {",
                "    int a = 0;",
                "    int b = 0;",
                "    if (!PyArg_ParseTuple(args, \"ii\", &a, &b)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static std::initializer_list<PyMethodDef> Methods = {",
                "    {\"add\", add_impl, METH_VARARGS, \"doc\"},",
                "    {nullptr, nullptr, 0, nullptr}",
                "};",
                "",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_cpp_std="c++17",
    )
    extracted = engine.extract_modules()

    assert extracted == {}


def test_c_ast_visitor_passes_clang_options_to_extractor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {
        "extract_calls": 0,
        "clang_include": None,
    }

    def _record_extract_c_signature_modules(
        source_root: Path,
        *,
        clang_include: list[str] = (),
        clang_include_directory: list[str] = (),
        clang_c_std: str = "c11",
        clang_cpp_std: str = "c++17",
    ) -> dict[str, ExtractedModule]:
        captured["extract_calls"] = int(captured["extract_calls"]) + 1
        captured["source_root"] = source_root
        captured["clang_include"] = list(clang_include)
        captured["clang_include_directory"] = list(clang_include_directory)
        captured["clang_c_std"] = clang_c_std
        captured["clang_cpp_std"] = clang_cpp_std
        return {}

    import core.node_visitors.c_signature_extraction.c_signature_extraction_visitor as visitor_module

    monkeypatch.setattr(
        visitor_module,
        "extract_c_signature_modules",
        _record_extract_c_signature_modules,
    )

    visitor = CSignatureExtractionVisitor(
        source_root=tmp_path,
        clang_include=["Python.h"],
        clang_include_directory=["C:/MyInclude"],
        clang_c_std="c99",
        clang_cpp_std="c++20",
    )

    assert captured["extract_calls"] == 0

    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
    )
    visitor.visit_module(module)
    visitor.visit_module(module)

    assert captured["extract_calls"] == 1
    assert captured["source_root"] == tmp_path
    assert captured["clang_include"] == ["Python.h"]
    assert captured["clang_include_directory"] == ["C:/MyInclude"]
    assert captured["clang_c_std"] == "c99"
    assert captured["clang_cpp_std"] == "c++20"


def test_c_signature_engine_extract_modules_runs_parse_build_infer_in_order(tmp_path: Path) -> None:
    source = tmp_path / "module.c"
    built_modules = {
        "pkg.mod": ExtractedModule(
            name="pkg.mod",
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                ),
            },
        )
    }
    calls: list[str] = []
    fake_translation_unit = SimpleNamespace(cursor=object())

    class _FakeIndexForPipeline:
        pass

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(translation_unit_module, "find_candidate_files", lambda source_root: [source])
    monkeypatch.setattr(
        module_table_module,
        "process_translation_unit",
        lambda cursor: (
            calls.append("build"),
            built_modules.values(),
        )[1]
        if calls == ["parse"]
        else pytest.fail("build should run after parse"),
    )
    monkeypatch.setattr(
        signature_rules_module,
        "inference_signature",
        lambda function: calls.append("infer")
        if calls == ["parse", "build"]
        else pytest.fail("infer should run after build"),
    )
    monkeypatch.setattr(
        translation_unit_module,
        "parse_translation_unit",
        lambda index, file_path, **kwargs: (
            calls.append("parse"),
            fake_translation_unit,
        )[1],
    )
    monkeypatch.setattr(c_signature_extraction_module.Index, "create", lambda: _FakeIndexForPipeline())
    try:
        extracted = extract_c_signature_modules(tmp_path)
    finally:
        monkeypatch.undo()

    assert calls == ["parse", "build", "infer"]
    assert extracted == built_modules


def test_c_signature_engine_extract_modules_skips_build_and_infer_when_parse_is_empty(tmp_path: Path) -> None:
    calls: list[str] = []

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        translation_unit_module,
        "find_candidate_files",
        lambda source_root: (
            calls.append("parse"),
            [],
        )[1],
    )
    monkeypatch.setattr(
        module_table_module,
        "process_translation_unit",
        lambda cursor: pytest.fail("build step should be skipped when parse result is empty"),
    )
    monkeypatch.setattr(
        signature_rules_module,
        "inference_signature",
        lambda function: pytest.fail("infer step should be skipped when parse result is empty"),
    )
    try:
        extracted = extract_c_signature_modules(tmp_path)
    finally:
        monkeypatch.undo()

    assert calls == ["parse"]
    assert extracted == {}


def test_c_signature_engine_builds_language_specific_std_args(tmp_path: Path) -> None:
    engine = CSignatureExtractor(source_root=tmp_path)
    assert engine._clang_include_directory is not None
    assert "-std=c11" not in engine._clang_include_directory
    assert (
        translation_unit_module.get_std_value_for_file(
            tmp_path / "module.c",
            clang_c_std=engine._clang_c_std,
            clang_cpp_std=engine._clang_cpp_std,
        )
        == "c11"
    )
    assert (
        translation_unit_module.get_std_value_for_file(
            tmp_path / "module.cxx",
            clang_c_std=engine._clang_c_std,
            clang_cpp_std=engine._clang_cpp_std,
        )
        == "c++17"
    )


def test_c_signature_engine_uses_configured_language_specific_std_args(tmp_path: Path) -> None:
    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_include_directory=["C:/MyInclude"],
        clang_c_std="-std=c99",
        clang_cpp_std="--std=c++20",
    )

    assert (
        translation_unit_module.get_std_value_for_file(
            tmp_path / "module.c",
            clang_c_std=engine._clang_c_std,
            clang_cpp_std=engine._clang_cpp_std,
        )
        == "-std=c99"
    )
    assert (
        translation_unit_module.get_std_value_for_file(
            tmp_path / "module.cxx",
            clang_c_std=engine._clang_c_std,
            clang_cpp_std=engine._clang_cpp_std,
        )
        == "--std=c++20"
    )
    assert (
        translation_unit_module.get_std_value_for_file(
            tmp_path / "module.hpp",
            clang_c_std=engine._clang_c_std,
            clang_cpp_std=engine._clang_cpp_std,
        )
        == "-std=c99"
    )


def test_c_signature_engine_extract_modules_keeps_external_include_options_and_injects_python_include_dirs(
    tmp_path: Path,
) -> None:
    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_include=["Python.h"],
        clang_include_directory=["C:/MyInclude"],
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(translation_unit_module, "find_candidate_files", lambda source_root: [])

    try:
        assert engine.extract_modules() == {}
        expected_include_dirs = ["C:/MyInclude"]
        for include_dir in [sysconfig.get_path("include"), sysconfig.get_path("platinclude")]:
            if not include_dir:
                continue
            if include_dir in expected_include_dirs:
                continue
            expected_include_dirs.append(include_dir)

        assert engine._clang_include == ["Python.h"]
        assert engine._clang_include_directory == expected_include_dirs
    finally:
        monkeypatch.undo()


def test_c_signature_engine_build_parse_args_uses_only_external_include_values(tmp_path: Path) -> None:
    engine = CSignatureExtractor(source_root=tmp_path, clang_c_std="c11")

    assert translation_unit_module.build_clang_parse_args(
        tmp_path / "module.c",
        clang_include=engine._clang_include,
        clang_include_directory=engine._clang_include_directory,
        clang_c_std=engine._clang_c_std,
        clang_cpp_std=engine._clang_cpp_std,
    ) == [
        "--std",
        "c11",
        *[
            item
            for include_dir in engine._clang_include_directory
            for item in ("--include-directory", include_dir)
        ],
    ]


def test_c_signature_engine_build_parse_args_places_include_before_include_directory(tmp_path: Path) -> None:
    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_include=["Python.h", "numpy/arrayobject.h"],
        clang_include_directory=["C:/MyInclude"],
        clang_c_std="c11",
    )

    assert translation_unit_module.build_clang_parse_args(
        tmp_path / "module.c",
        clang_include=engine._clang_include,
        clang_include_directory=engine._clang_include_directory,
        clang_c_std=engine._clang_c_std,
        clang_cpp_std=engine._clang_cpp_std,
    ) == [
        "--std",
        "c11",
        "--include",
        "Python.h",
        "--include",
        "numpy/arrayobject.h",
        "--include-directory",
        "C:/MyInclude",
        *[
            item
            for include_dir in engine._clang_include_directory[1:]
            for item in ("--include-directory", include_dir)
        ],
    ]


class _FakeToken:
    def __init__(self, kind: object, spelling: str) -> None:
        self.kind = kind
        self.spelling = spelling


class _FakeCursorLocation:
    def __init__(self, file: str | None = None) -> None:
        self.file = file


class _FakeNode:
    def __init__(
        self,
        *,
        kind: object,
        tokens: list[_FakeToken] | None = None,
        children: list[object] | None = None,
        spelling: str = "",
        location: object | None = None,
        referenced: object | None = None,
    ) -> None:
        self.kind = kind
        self._tokens = tokens or []
        self._children = children or []
        self.spelling = spelling
        self.location = location if location is not None else _FakeCursorLocation()
        self.referenced = referenced
        self.type = None

    def get_tokens(self) -> list[_FakeToken]:
        return self._tokens

    def get_children(self) -> Iterable[object]:
        return iter(self._children)

    def is_definition(self) -> bool:
        return False


def _fake_function_cursor(name: str = "fake_function") -> clang.cindex.Cursor:
    """构造可复用的假函数游标。"""
    return cast(
        clang.cindex.Cursor,
        _FakeNode(kind=clang.cindex.CursorKind.FUNCTION_DECL, spelling=name),
    )


def _int_literal(value: str = "0") -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.INTEGER_LITERAL,
        tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, value)],
    )


def _null_ptr_literal() -> _FakeNode:
    return _FakeNode(kind=clang.cindex.CursorKind.CXX_NULL_PTR_LITERAL_EXPR)


def _gnu_null_literal() -> _FakeNode:
    return _FakeNode(kind=clang.cindex.CursorKind.GNU_NULL_EXPR)


def _identifier_node(name: str) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.DECL_REF_EXPR,
        spelling=name,
        tokens=[_FakeToken(clang.cindex.TokenKind.IDENTIFIER, name)],
    )


def _wrap(kind: object, child: _FakeNode) -> _FakeNode:
    return _FakeNode(kind=kind, children=[child])


def _init_list(*children: _FakeNode) -> _FakeNode:
    return _FakeNode(kind=clang.cindex.CursorKind.INIT_LIST_EXPR, children=list(children))


def _designated_initializer(field_name: str, value: _FakeNode) -> _FakeNode:
    referenced = _FakeNode(kind=clang.cindex.CursorKind.FIELD_DECL, spelling=field_name)
    return _FakeNode(
        kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
        children=[
            _token_identifier_node(
                field_name,
                kind=clang.cindex.CursorKind.MEMBER_REF,
                referenced=referenced,
            ),
            value,
        ],
    )


def _string_literal(value: str) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.STRING_LITERAL,
        tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, f'"{value}"')],
        spelling=f'"{value}"',
    )


def _token_identifier_node(
    name: str,
    *,
    kind: object = clang.cindex.CursorKind.DECL_REF_EXPR,
    referenced: object | None = None,
) -> _FakeNode:
    return _FakeNode(
        kind=kind,
        spelling=name,
        tokens=[_FakeToken(clang.cindex.TokenKind.IDENTIFIER, name)],
        referenced=referenced,
    )


def _var_decl(name: str, initializer: _FakeNode | None = None) -> _FakeNode:
    children = [initializer] if initializer is not None else []
    return _FakeNode(
        kind=clang.cindex.CursorKind.VAR_DECL,
        spelling=name,
        children=children,
    )


def _address_of(name: str, *, referenced: object | None = None) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        children=[_token_identifier_node(name, referenced=referenced)],
    )


def _call_expr(name: str, *args: _FakeNode) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.CALL_EXPR,
        spelling=name,
        children=[
            _FakeNode(
                kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
                spelling=name,
                children=[_token_identifier_node(name)],
            ),
            *args,
        ],
    )


def _conditional_expr(condition: _FakeNode, when_true: _FakeNode, when_false: _FakeNode) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.CONDITIONAL_OPERATOR,
        children=[condition, when_true, when_false],
    )


def _return_stmt(expr: _FakeNode | None = None) -> _FakeNode:
    """构造 return 语句节点。"""
    children = [] if expr is None else [expr]
    return _FakeNode(kind=clang.cindex.CursorKind.RETURN_STMT, children=children)


def _macro_expr(name: str) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
        spelling=name,
        children=[_FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, spelling=name)],
    )


def _fake_function_cursor_with_children(
    *children: _FakeNode,
    name: str = "fake_function",
) -> clang.cindex.Cursor:
    """构造带子节点的假函数游标。"""
    return cast(
        clang.cindex.Cursor,
        _FakeNode(
            kind=clang.cindex.CursorKind.FUNCTION_DECL,
            spelling=name,
            children=list(children),
        ),
    )


def _ml_name_field(name: str) -> _FakeNode:
    return _wrap(clang.cindex.CursorKind.UNEXPOSED_EXPR, _wrap(clang.cindex.CursorKind.UNEXPOSED_EXPR, _string_literal(name)))


def _ml_meth_field(
    name: str,
    *,
    referenced_kind: object = clang.cindex.CursorKind.FUNCTION_DECL,
) -> _FakeNode:
    referenced = _FakeNode(kind=referenced_kind, spelling=name)
    return _FakeNode(
        kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
        spelling=name,
        children=[_token_identifier_node(name, referenced=referenced)],
    )


def _ml_meth_cast_field(name: str) -> _FakeNode:
    referenced = _FakeNode(kind=clang.cindex.CursorKind.FUNCTION_DECL, spelling=name)
    return _wrap(
        clang.cindex.CursorKind.UNEXPOSED_EXPR,
        _wrap(
            clang.cindex.CursorKind.PAREN_EXPR,
            _FakeNode(
                kind=clang.cindex.CursorKind.CSTYLE_CAST_EXPR,
                children=[_token_identifier_node(name, referenced=referenced)],
            ),
        ),
    )


def _ml_flags_identifier_field(*flags: str) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.BINARY_OPERATOR,
        children=[_token_identifier_node(flag) for flag in flags],
    )


@pytest.mark.parametrize(
    ("token_name", "expected"),
    [
        ("Py_None", "None"),
        ("Py_True", "bool"),
        ("Py_False", "bool"),
    ],
)
def test_infer_expr_type_detects_direct_object_returns(token_name: str, expected: str) -> None:
    inferred = signature_rules_module.infer_expr_type(_identifier_node(token_name))

    assert inferred == expected


@pytest.mark.parametrize(
    ("token_name", "expected"),
    [
        ("Py_RETURN_NONE", "None"),
        ("Py_RETURN_TRUE", "bool"),
        ("Py_RETURN_FALSE", "bool"),
        ("Py_RETURN_NAN", "float"),
        ("Py_RETURN_INF", "float"),
    ],
)
def test_infer_expr_type_detects_preserved_macro_tokens(token_name: str, expected: str) -> None:
    macro_expr = _macro_expr(token_name)

    inferred = signature_rules_module.infer_expr_type(macro_expr)

    assert inferred == expected


def test_infer_expr_type_returns_none_when_macro_name_is_not_exposed_by_ast() -> None:
    macro_expr = _FakeNode(
        kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
        children=[_FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR)],
    )

    inferred = signature_rules_module.infer_expr_type(macro_expr)

    assert inferred is None


@pytest.mark.parametrize(
    ("call_name", "expected"),
    [
        ("PyBool_FromLong", "bool"),
        ("PyLong_FromLong", "int"),
        ("PyFloat_FromDouble", "float"),
        ("PyComplex_FromDoubles", "complex"),
        ("PyUnicode_FromString", "str"),
        ("PyUnicode_AsUTF8String", "bytes"),
        ("PyByteArray_FromObject", "bytearray"),
        ("PySlice_New", "slice"),
        ("PyMemoryView_FromObject", "memoryview"),
        ("PyTuple_New", "tuple"),
        ("PyList_New", "list"),
        ("PyDict_New", "dict"),
        ("PySet_New", "set"),
        ("PyFrozenSet_New", "frozenset"),
        ("PyList_AsTuple", "tuple"),
        ("PyDict_Items", "list"),
    ],
)
def test_infer_expr_type_detects_exact_factory_mappings(call_name: str, expected: str) -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(call_name, _identifier_node("arg"))
    )

    assert inferred == expected


def test_infer_expr_type_parses_py_buildvalue() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            "Py_BuildValue",
            _string_literal("(is)"),
            _identifier_node("count"),
            _identifier_node("name"),
        )
    )

    assert inferred == "tuple[int, str | None]"


def test_infer_expr_type_resolves_py_buildvalue_object_slots() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            "Py_BuildValue",
            _string_literal("(O)"),
            _call_expr("PyLong_FromLong", _identifier_node("value")),
        )
    )

    assert inferred == "tuple[int,]"


def test_infer_expr_type_keeps_py_buildvalue_object_slots_as_any_when_unknown() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            "Py_BuildValue",
            _string_literal("(O)"),
            _call_expr("CustomFactory", _identifier_node("value")),
        )
    )

    assert inferred == "tuple[Any,]"


def test_infer_expr_type_unwraps_transparent_wrappers_and_casts() -> None:
    wrapped_expr = _wrap(
        clang.cindex.CursorKind.UNEXPOSED_EXPR,
        _wrap(
            clang.cindex.CursorKind.PAREN_EXPR,
            _FakeNode(
                kind=clang.cindex.CursorKind.CSTYLE_CAST_EXPR,
                children=[
                    _identifier_node("PyObject"),
                    _call_expr("PyUnicode_AsUTF8String", _identifier_node("value")),
                ],
            ),
        ),
    )

    inferred = signature_rules_module.infer_expr_type(wrapped_expr)

    assert inferred == "bytes"


def test_infer_expr_type_merges_conditional_branch_types() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _conditional_expr(
            _identifier_node("cond"),
            _call_expr("PyLong_FromLong", _identifier_node("left")),
            _call_expr("PyFloat_FromDouble", _identifier_node("right")),
        )
    )

    assert inferred == "int | float"


def test_infer_expr_type_deduplicates_conditional_branch_types() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _conditional_expr(
            _identifier_node("cond"),
            _call_expr("PyLong_FromLong", _identifier_node("left")),
            _call_expr("PyLong_FromLong", _identifier_node("right")),
        )
    )

    assert inferred == "int"


def test_infer_expr_type_keeps_known_conditional_branch_when_other_is_unknown() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _conditional_expr(
            _identifier_node("cond"),
            _call_expr("PyLong_FromLong", _identifier_node("value")),
            _call_expr("CustomFactory", _identifier_node("value")),
        )
    )

    assert inferred == "int"


def test_infer_expr_type_returns_none_when_conditional_branches_are_unknown() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _conditional_expr(
            _macro_expr("Py_RETURN_NONE"),
            _call_expr("CustomFactory", _identifier_node("left")),
            _identifier_node("UnknownToken"),
        )
    )

    assert inferred is None


def test_infer_expr_type_unwraps_wrapped_conditional_expr() -> None:
    wrapped_expr = _wrap(
        clang.cindex.CursorKind.UNEXPOSED_EXPR,
        _wrap(
            clang.cindex.CursorKind.PAREN_EXPR,
            _FakeNode(
                kind=clang.cindex.CursorKind.CSTYLE_CAST_EXPR,
                children=[
                    _identifier_node("PyObject"),
                    _conditional_expr(
                        _identifier_node("cond"),
                        _call_expr("PyUnicode_AsUTF8String", _identifier_node("value")),
                        _call_expr("PyBytes_FromString", _identifier_node("fallback")),
                    ),
                ],
            ),
        ),
    )

    inferred = signature_rules_module.infer_expr_type(wrapped_expr)

    assert inferred == "bytes"


def test_infer_expr_type_returns_none_for_unsupported_expr() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr("CustomFactory", _identifier_node("value"))
    )

    assert inferred is None


@pytest.mark.parametrize(
    ("token_name", "expected"),
    [
        ("Py_None", "None"),
        ("Py_True", "bool"),
        ("Py_False", "bool"),
    ],
)
def test_return_type_detects_direct_object_returns(token_name: str, expected: str) -> None:
    cursor = _fake_function_cursor_with_children(_return_stmt(_identifier_node(token_name)))

    inferred = signature_rules_module.inference_return_type(cursor)

    assert inferred == expected


@pytest.mark.parametrize(
    ("token_name", "expected"),
    [
        ("Py_RETURN_NONE", "None"),
        ("Py_RETURN_TRUE", "bool"),
        ("Py_RETURN_FALSE", "bool"),
        ("Py_RETURN_NAN", "float"),
        ("Py_RETURN_INF", "float"),
    ],
)
def test_return_type_detects_preserved_macro_tokens(token_name: str, expected: str) -> None:
    macro_expr = _macro_expr(token_name)
    cursor = _fake_function_cursor_with_children(_return_stmt(macro_expr))

    inferred = signature_rules_module.inference_return_type(cursor)

    assert inferred == expected


@pytest.mark.parametrize(
    ("call_name", "expected"),
    [
        ("PyBool_FromLong", "bool"),
        ("PyLong_FromLong", "int"),
        ("PyFloat_FromDouble", "float"),
        ("PyComplex_FromDoubles", "complex"),
        ("PyUnicode_FromString", "str"),
        ("PyUnicode_AsUTF8String", "bytes"),
        ("PyByteArray_FromObject", "bytearray"),
        ("PySlice_New", "slice"),
        ("PyMemoryView_FromObject", "memoryview"),
        ("PyTuple_New", "tuple"),
        ("PyList_New", "list"),
        ("PyDict_New", "dict"),
        ("PySet_New", "set"),
        ("PyFrozenSet_New", "frozenset"),
        ("PyList_AsTuple", "tuple"),
        ("PyDict_Items", "list"),
    ],
)
def test_return_type_detects_exact_factory_mappings(call_name: str, expected: str) -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr(call_name, _identifier_node("arg")))
    )

    inferred = signature_rules_module.inference_return_type(cursor)

    assert inferred == expected


def test_return_type_parses_py_buildvalue() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(
            _call_expr(
                "Py_BuildValue",
                _string_literal("(is)"),
                _identifier_node("count"),
                _identifier_node("name"),
            )
        )
    )

    inferred = signature_rules_module.inference_return_type(cursor)

    assert inferred == "tuple[int, str | None]"


def test_return_type_unwraps_transparent_wrappers_and_casts() -> None:
    wrapped_expr = _wrap(
        clang.cindex.CursorKind.UNEXPOSED_EXPR,
        _wrap(
            clang.cindex.CursorKind.PAREN_EXPR,
            _FakeNode(
                kind=clang.cindex.CursorKind.CSTYLE_CAST_EXPR,
                children=[
                    _identifier_node("PyObject"),
                    _call_expr("PyUnicode_AsUTF8String", _identifier_node("value")),
                ],
            ),
        ),
    )
    cursor = _fake_function_cursor_with_children(_return_stmt(wrapped_expr))

    inferred = signature_rules_module.inference_return_type(cursor)

    assert inferred == "bytes"


def test_return_type_deduplicates_and_preserves_order() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_identifier_node("Py_None")),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
        _return_stmt(
            _call_expr(
                "Py_BuildValue",
                _string_literal("i"),
                _identifier_node("value"),
            )
        ),
        _return_stmt(_identifier_node("Py_False")),
    )

    inferred = signature_rules_module.inference_return_type(cursor)

    assert inferred == "None | int | bool"


def test_return_type_detects_conditional_expr() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(
            _conditional_expr(
                _identifier_node("cond"),
                _call_expr("PyLong_FromLong", _identifier_node("value")),
                _call_expr("PyFloat_FromDouble", _identifier_node("value")),
            )
        )
    )

    inferred = signature_rules_module.inference_return_type(cursor)

    assert inferred == "int | float"


def test_return_type_returns_none_for_unsupported_returns() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("CustomFactory", _identifier_node("value"))),
        _return_stmt(),
    )

    inferred = signature_rules_module.inference_return_type(cursor)

    assert inferred is None


def test_return_type_returns_none_when_function_has_no_return() -> None:
    cursor = _fake_function_cursor_with_children()

    inferred = signature_rules_module.inference_return_type(cursor)

    assert inferred is None


def test_return_type_py_buildvalue_parser_is_importable_by_package_path() -> None:
    module = importlib.import_module(
        "core.node_visitors.c_signature_extraction.core.py_buildvalue_type_parser"
    )

    assert module.PyBuildValueTypeParser is not None


def _patch_fake_eval_int(monkeypatch: pytest.MonkeyPatch) -> None:
    original_eval_int = cursor_utils_module.clang_eval.eval_int
    method_flag_values = {
        "METH_VARARGS": METH_VARARGS,
        "METH_KEYWORDS": METH_KEYWORDS,
    }

    def _eval_int(cursor: object) -> int | None:
        if not isinstance(cursor, _FakeNode):
            return original_eval_int(cursor)
        if cursor.kind == clang.cindex.CursorKind.INTEGER_LITERAL:
            for token in cursor.get_tokens():
                if token.kind != clang.cindex.TokenKind.LITERAL:
                    continue
                text = str(token.spelling).strip()
                if not text:
                    continue
                try:
                    return int(text, 0)
                except ValueError:
                    continue
            return None
        if cursor.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
            return method_flag_values.get(cursor.spelling)
        if cursor.kind == clang.cindex.CursorKind.BINARY_OPERATOR:
            value = 0
            for child in cursor.get_children():
                child_value = _eval_int(child)
                if child_value is None:
                    return None
                value |= child_value
            return value
        return None

    monkeypatch.setattr(cursor_utils_module.clang_eval, "eval_int", _eval_int)

def test_c_signature_engine_resolve_init_list_expr_supports_positional_entries(tmp_path: Path) -> None:
    field_names = ("a", "b", "c")
    first = _string_literal("first")
    second = _int_literal("2")

    resolved = _resolve_INIT_LIST_EXPR(_init_list(first, second), field_names)

    assert resolved == {"a": first, "b": second}


def test_c_signature_engine_resolve_init_list_expr_supports_designated_entries(tmp_path: Path) -> None:
    field_names = ("a", "b", "c")
    second = _int_literal("2")
    third = _string_literal("third")

    resolved = _resolve_INIT_LIST_EXPR(
        _init_list(
            _designated_initializer("b", second),
            _designated_initializer("c", third),
        ),
        field_names,
    )

    assert resolved == {"b": second, "c": third}


def test_c_signature_engine_resolve_init_list_expr_supports_mixed_entries(tmp_path: Path) -> None:
    field_names = ("a", "b", "c", "d")
    first = _string_literal("first")
    third = _string_literal("third")
    fourth = _int_literal("4")

    resolved = _resolve_INIT_LIST_EXPR(
        _init_list(
            first,
            _designated_initializer("c", third),
            fourth,
        ),
        field_names,
    )

    assert resolved == {"a": first, "c": third, "d": fourth}


def test_c_signature_engine_resolve_init_list_expr_advances_positional_index_after_designated(
    tmp_path: Path,
) -> None:
    field_names = ("a", "b", "c", "d")
    second = _string_literal("second")
    third = _int_literal("3")
    fourth = _int_literal("4")

    resolved = _resolve_INIT_LIST_EXPR(
        _init_list(
            _designated_initializer("b", second),
            third,
            fourth,
        ),
        field_names,
    )

    assert resolved == {"b": second, "c": third, "d": fourth}


def test_c_signature_engine_resolve_init_list_expr_ignores_unknown_designated_field(tmp_path: Path) -> None:
    field_names = ("a", "b", "c")
    unknown = _string_literal("skip")
    first = _int_literal("1")

    resolved = _resolve_INIT_LIST_EXPR(
        _init_list(
            _designated_initializer("missing", unknown),
            first,
        ),
        field_names,
    )

    assert resolved == {"a": first}


def test_c_signature_engine_resolve_init_list_expr_last_value_wins_for_duplicate_field(tmp_path: Path) -> None:
    field_names = ("a", "b", "c")
    first = _int_literal("1")
    second = _int_literal("2")

    resolved = _resolve_INIT_LIST_EXPR(
        _init_list(
            first,
            _designated_initializer("a", second),
        ),
        field_names,
    )

    assert resolved == {"a": second}


def test_c_signature_engine_resolve_init_list_expr_keeps_nested_init_list_as_value(tmp_path: Path) -> None:
    field_names = ("a", "b")
    nested = _init_list(_int_literal("1"), _int_literal("2"))

    resolved = _resolve_INIT_LIST_EXPR(
        _init_list(_designated_initializer("b", nested)),
        field_names,
    )

    assert resolved == {"b": nested}


def test_c_signature_engine_extracts_pymethod_fields_from_ast_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_fake_eval_int(monkeypatch)
    is_sentinel, extracted = _extract_PyMethodDef_INIT_LIST_EXPR(
        init_list_expr=_init_list(
            _ml_name_field("add"),
            _ml_meth_field("simple_add"),
            _ml_flags_identifier_field("METH_VARARGS"),
            _string_literal("doc"),
        ),
    )

    assert is_sentinel is False
    assert extracted is not None
    assert extracted.ml_name == "add"
    assert extracted.ml_flags == METH_VARARGS
    assert extracted.function_cursor is not None
    assert extracted.signatures == []


def test_c_signature_engine_extracts_cast_wrapped_ml_meth_from_ast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_fake_eval_int(monkeypatch)
    is_sentinel, extracted = _extract_PyMethodDef_INIT_LIST_EXPR(
        init_list_expr=_init_list(
            _ml_name_field("distance"),
            _ml_meth_cast_field("Point_distance"),
            _ml_flags_identifier_field("METH_VARARGS"),
            _string_literal("doc"),
        ),
    )

    assert is_sentinel is False
    assert extracted is not None
    assert extracted.ml_name == "distance"
    assert extracted.ml_flags == METH_VARARGS
    assert extracted.function_cursor is not None


def test_c_signature_engine_extracts_combined_flags_from_ast_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_fake_eval_int(monkeypatch)
    is_sentinel, extracted = _extract_PyMethodDef_INIT_LIST_EXPR(
        init_list_expr=_init_list(
            _ml_name_field("kw"),
            _ml_meth_field("kw_impl"),
            _ml_flags_identifier_field("METH_VARARGS", "METH_KEYWORDS"),
            _string_literal("doc"),
        ),
    )

    assert is_sentinel is False
    assert extracted is not None
    assert extracted.ml_flags == (METH_VARARGS | METH_KEYWORDS)


def test_c_signature_engine_warns_and_keeps_empty_flags_when_ast_field_is_unparseable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    _patch_fake_eval_int(monkeypatch)
    with caplog.at_level(logging.WARNING):
        is_sentinel, extracted = _extract_PyMethodDef_INIT_LIST_EXPR(
            init_list_expr=_init_list(
                _ml_name_field("add"),
                _ml_meth_field("simple_add"),
                _identifier_node("flag_var"),
                _string_literal("doc"),
            ),
        )

    assert is_sentinel is False
    assert extracted is not None
    assert extracted.ml_flags == 0
    assert caplog.records == []


def test_c_signature_engine_extract_pymethoddef_init_list_expr_marks_sentinel(tmp_path: Path) -> None:
    is_sentinel, extracted = _extract_PyMethodDef_INIT_LIST_EXPR(
        init_list_expr=_init_list(_null_ptr_literal()),
    )

    assert is_sentinel is True
    assert extracted is None


def test_c_signature_engine_extract_pymethoddef_init_list_expr_discards_entry_without_function_cursor(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """验证缺失 `ml_meth` 引用时当前条目会被直接丢弃。"""
    _patch_fake_eval_int(monkeypatch)
    with caplog.at_level(logging.WARNING):
        is_sentinel, extracted = _extract_PyMethodDef_INIT_LIST_EXPR(
            init_list_expr=_init_list(
                _ml_name_field("missing"),
                _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR),
                _ml_flags_identifier_field("METH_VARARGS"),
                _string_literal("doc"),
            ),
        )

    assert is_sentinel is False
    assert extracted is None
    assert "cant find function cursor" in caplog.text


def test_c_signature_engine_extract_method_table_stops_at_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    method_1 = _init_list(
        _FakeNode(kind=clang.cindex.CursorKind.STRING_LITERAL, tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, '"a"')]),
        _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR),
        _int_literal("1"),
        _FakeNode(kind=clang.cindex.CursorKind.STRING_LITERAL, tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, '"doc"')]),
    )
    method_2 = _init_list(
        _FakeNode(kind=clang.cindex.CursorKind.STRING_LITERAL, tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, '"b"')]),
        _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR),
        _int_literal("1"),
        _FakeNode(kind=clang.cindex.CursorKind.STRING_LITERAL, tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, '"doc"')]),
    )
    supported_sentinel = _init_list(_null_ptr_literal())
    non_sentinel = _init_list(_identifier_node("nullptr"))
    calls: list[_FakeNode] = []

    def fake_extract(
        *,
        init_list_expr: _FakeNode,
    ) -> tuple[bool, SimpleNamespace | None]:
        calls.append(init_list_expr)
        if init_list_expr is supported_sentinel:
            return True, None
        return False, SimpleNamespace(ml_name=f"entry_{len(calls)}")

    monkeypatch.setattr(module_table_module, "extract_pymethoddef_init_list_expr", fake_extract)
    monkeypatch.setattr(module_table_module, "is_PyMethodDef_array_definition", lambda cursor: True)

    should_break_array = _FakeNode(
        kind=clang.cindex.CursorKind.VAR_DECL,
        children=[
            _FakeNode(
                kind=clang.cindex.CursorKind.INIT_LIST_EXPR,
                children=[method_1, supported_sentinel, method_2],
            ),
        ],
    )
    output = _extract_method_table(
        should_break_array,
        module_name="pkg.mod",
    )
    assert calls == [method_1, supported_sentinel]
    assert list(output) == ["entry_1"]

    calls.clear()
    output.clear()

    should_not_break_array = _FakeNode(
        kind=clang.cindex.CursorKind.VAR_DECL,
        children=[
            _FakeNode(
                kind=clang.cindex.CursorKind.INIT_LIST_EXPR,
                children=[method_1, non_sentinel, method_2],
            ),
        ],
    )
    output = _extract_method_table(
        should_not_break_array,
        module_name="pkg.mod",
    )
    assert calls == [method_1, non_sentinel, method_2]
    assert list(output) == ["entry_1", "entry_2", "entry_3"]



