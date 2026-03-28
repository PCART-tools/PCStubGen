from __future__ import annotations

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
from core.node_visitors.c_signature_extraction.core import signature_inference as signature_rules_module
from core.node_visitors.c_signature_extraction.core import module_table as module_table_module
from core.node_visitors.c_signature_extraction.core import translation_unit as translation_unit_module
from core.node_visitors.c_signature_extraction.core.py_build_value_type_nodes import (
    AnyTypeNode,
    ListTypeNode,
    NamedTypeNode,
    TupleTypeNode,
    UnionTypeNode,
)
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
    IRSignature,
    QualifiedName,
)
from core.node_visitors.c_signature_extraction.c_signature_extraction_visitor import (
    CSignatureExtractionVisitor,
)
from core.node_visitors.doc_string_signature_parser_visitor import (
    DocStringSignatureParserVisitor,
)
from core.pipeline import Pipeline
from core.stub_generation_options import StubGenerationOptions


def _signature(
    *,
    args: list[IRArgument] | None = None,
    return_type_name: str | None = None,
    doc: str | None = None,
) -> IRSignature:
    """构造测试用 IR 签名。"""
    return IRSignature(
        args=list(args or ()),
        return_type_name=return_type_name,
        doc=doc,
    )


def _unknown_function(name: str, *, doc: str | None = None) -> IRFunction:
    """构造签名未知的测试函数。"""
    return IRFunction(name=name, doc=doc)


def _module_fixture(
    *,
    name: str = "pkg.mod",
    functions: dict[str, ExtractedFunction] | None = None,
) -> dict[str, ExtractedModule]:
    return {
        name: ExtractedModule(
            name=name,
            functions=functions or {},
        )
    }


def _make_extraction_config(
    *,
    source_root: Path,
    include: list[str] = (),
    include_directory: list[Path] = (),
    c_std: str = "c11",
    cpp_std: str = "c++17",
) -> dict[str, object]:
    return {
        "source_root": source_root,
        "include": list(include),
        "include_directory": translation_unit_module.inject_python_include_directories(
            list(include_directory)
        ),
        "c_std": c_std,
        "cpp_std": cpp_std,
    }


class CSignatureExtractor:
    def __init__(
        self,
        source_root: Path,
        *,
        include: list[str] = (),
        include_directory: list[Path] = (),
        c_std: str = "c11",
        cpp_std: str = "c++17",
    ) -> None:
        self._source_root = source_root
        self._include = list(include)
        self._include_directory = translation_unit_module.inject_python_include_directories(
            list(include_directory)
        )
        self._c_std = c_std
        self._cpp_std = cpp_std

    def extract_modules(self) -> dict[str, ExtractedModule]:
        return extract_c_signature_modules(
            self._source_root,
            include=self._include,
            include_directory=self._include_directory,
            c_std=self._c_std,
            cpp_std=self._cpp_std,
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
        include: list[str] = (),
        include_directory: list[Path] = (),
        c_std: str = "c11",
        cpp_std: str = "c++17",
    ) -> dict[str, ExtractedModule]:
        _ = (source_root, include, include_directory, c_std, cpp_std)
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
        include: list[str] = (),
        include_directory: list[Path] = (),
        c_std: str = "c11",
        cpp_std: str = "c++17",
    ) -> dict[str, ExtractedModule]:
        _ = (source_root, include, include_directory, c_std, cpp_std)
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


def test_c_ast_visitor_rewrites_module_function_without_normalizing_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    func = _unknown_function("foo")
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
                                ExtractedArgument(
                                    name="flag",
                                    type_name="bool",
                                    default_value="False",
                                    has_default=True,
                                ),
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
    assert len(rewritten.signatures) == 1
    signature = rewritten.signatures[0]
    assert [arg.name for arg in signature.args] == ["self", "x", "flag"]
    assert signature.args[0].type_name == "object"
    assert signature.args[1].type_name == "int"
    assert signature.args[2].type_name == "bool"
    assert signature.args[2].default_value is not None
    assert signature.args[2].default_value == "False"
    assert signature.args[2].has_default is True
    assert signature.return_type_name is not None
    assert signature.return_type_name == "int"
    assert rewritten.c_inferred_source_comment is None
    assert visitor._stats.total_unknown_signatures == 1
    assert visitor._stats.success == 1


def test_c_ast_visitor_records_c_inferred_source_comment_when_enabled(
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

    func = _unknown_function("foo")
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
                    function_cursor=func_cursor,
                    ml_flags=METH_VARARGS,
                    signatures=[
                        ExtractedSignature(
                            arguments=[ExtractedArgument(name="value", type_name="int")]
                        )
                    ],
                )
            }
        ),
    )

    visitor = CSignatureExtractionVisitor(
        source_root=tmp_path,
        include_c_inferred_source_comment=True,
    )
    visitor.visit_module(module)

    assert module.functions[0].c_inferred_source_comment == snippet


def test_c_ast_visitor_skips_c_inferred_source_comment_when_extent_text_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    func = _unknown_function("foo")
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
                            arguments=[ExtractedArgument(name="value", type_name="int")]
                        )
                    ],
                )
            }
        ),
    )

    visitor = CSignatureExtractionVisitor(
        source_root=tmp_path,
        include_c_inferred_source_comment=True,
    )
    visitor.visit_module(module)

    assert module.functions[0].c_inferred_source_comment is None


def test_c_ast_visitor_preserves_has_default_without_default_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    func = _unknown_function("foo")
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
                                ExtractedArgument(
                                    name="flag",
                                    type_name="bool",
                                    has_default=True,
                                )
                            ]
                        )
                    ],
                )
            }
        ),
    )

    visitor = CSignatureExtractionVisitor(source_root=tmp_path)
    visitor.visit_module(module)

    signature = module.functions[0].signatures[0]
    assert signature.args[0].has_default is True
    assert signature.args[0].default_value is None


def test_c_ast_visitor_preserves_raw_argument_and_return_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    func = _unknown_function("foo")
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
                                ExtractedArgument(
                                    name="value",
                                    type_name="  int  ",
                                    default_value="  keep_raw()  ",
                                    has_default=True,
                                )
                            ],
                            return_type_name="  bool  ",
                        )
                    ],
                )
            }
        ),
    )

    visitor = CSignatureExtractionVisitor(source_root=tmp_path)
    visitor.visit_module(module)

    signature = module.functions[0].signatures[0]
    assert signature.args[0].type_name == "  int  "
    assert signature.args[0].default_value == "  keep_raw()  "
    assert signature.return_type_name == "  bool  "


def test_c_ast_visitor_preserves_extracted_argument_kinds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    func = _unknown_function("foo")
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
                    ml_flags=METH_VARARGS | METH_KEYWORDS,
                    signatures=[
                        ExtractedSignature(
                            arguments=[
                                ExtractedArgument(
                                    name="value",
                                    kind=IRArgumentKind.KEYWORD_ONLY,
                                ),
                                ExtractedArgument(
                                    name="args",
                                    kind=IRArgumentKind.VAR_POSITIONAL,
                                ),
                                ExtractedArgument(
                                    name="kwargs",
                                    kind=IRArgumentKind.VAR_KEYWORD,
                                ),
                            ]
                        )
                    ],
                )
            }
        ),
    )

    visitor = CSignatureExtractionVisitor(source_root=tmp_path)
    visitor.visit_module(module)

    signature = module.functions[0].signatures[0]
    assert [arg.kind for arg in signature.args] == [
        IRArgumentKind.KEYWORD_ONLY,
        IRArgumentKind.VAR_POSITIONAL,
        IRArgumentKind.VAR_KEYWORD,
    ]


def test_c_ast_visitor_keeps_known_function_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    visitor = CSignatureExtractionVisitor(
        source_root=tmp_path,
    )
    func = IRFunction(
        name="foo",
        signatures=[_signature(args=[IRArgument(name="x", kind=IRArgumentKind.POSITIONAL_OR_KEYWORD)])],
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
                    signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")])],
                )
            }
        ),
    )

    visitor.visit_module(module)

    assert module.functions[0] is func
    assert func.signatures[0].args[0].name == "x"
    assert func.signatures[0].args[0].type_name is None
    assert func.c_inferred_source_comment is None
    assert visitor._stats.total_unknown_signatures == 0
    assert visitor._stats.success == 0


def test_c_ast_visitor_records_missing_function_match_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "bar": ExtractedFunction(
                    ml_name="bar",
                    function_cursor=_fake_function_cursor("bar"),
                    ml_flags=METH_VARARGS,
                    signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")])],
                )
            }
        ),
    )

    visitor = CSignatureExtractionVisitor(source_root=tmp_path)
    visitor.visit_module(module)

    assert module.functions[0].signatures == []
    assert visitor._stats.total_unknown_signatures == 1
    assert visitor._stats.success == 0
    assert visitor._stats.missing_module_match == 0
    assert visitor._stats.missing_function_match == 1
    assert visitor._stats.matched_function_without_signatures == 0


def test_c_ast_visitor_records_matched_function_without_signatures_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    ml_flags=METH_VARARGS,
                    signatures=[],
                )
            }
        ),
    )

    visitor = CSignatureExtractionVisitor(source_root=tmp_path)
    visitor.visit_module(module)

    assert module.functions[0].signatures == []
    assert visitor._stats.total_unknown_signatures == 1
    assert visitor._stats.success == 0
    assert visitor._stats.missing_module_match == 0
    assert visitor._stats.missing_function_match == 0
    assert visitor._stats.matched_function_without_signatures == 1


def test_c_ast_visitor_records_empty_extraction_as_missing_module_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )
    _patch_c_signature_extractor(monkeypatch, modules={})

    visitor = CSignatureExtractionVisitor(source_root=tmp_path)
    visitor.visit_module(module)

    assert module.functions[0].signatures == []
    assert visitor._stats.total_unknown_signatures == 1
    assert visitor._stats.success == 0
    assert visitor._stats.missing_module_match == 1
    assert visitor._stats.missing_function_match == 0
    assert visitor._stats.matched_function_without_signatures == 0


def test_c_signature_engine_returns_translation_unit_when_error_present(tmp_path: Path) -> None:
    config = _make_extraction_config(source_root=tmp_path, c_std="c11")
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

    result = translation_unit_module.parse_translation_unit(
        index=_FakeIndex(translation_unit),
        file_path=source,
        source_root=config["source_root"],
        include=config["include"],
        include_directory=config["include_directory"],
        c_std=config["c_std"],
        cpp_std=config["cpp_std"],
    )

    assert result is translation_unit


def test_c_signature_engine_auto_adds_include_dir_for_nested_header_literal(tmp_path: Path) -> None:
    config = _make_extraction_config(source_root=tmp_path, c_std="c11")
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
        include=config["include"],
        include_directory=config["include_directory"],
        c_std=config["c_std"],
        cpp_std=config["cpp_std"],
    )

    assert result is second
    expected_include_root = header_path.parents[1]
    assert expected_include_root in config["include_directory"]
    assert header_path.parent not in config["include_directory"]
    assert len(index.calls) == 2
    assert _has_std_arg(index.calls[0][1], "c11")
    assert _has_std_arg(index.calls[1][1], "c11")


def test_c_signature_engine_retries_until_missing_includes_converge(tmp_path: Path) -> None:
    config = _make_extraction_config(source_root=tmp_path, c_std="c11")
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
        include=config["include"],
        include_directory=config["include_directory"],
        c_std=config["c_std"],
        cpp_std=config["cpp_std"],
    )

    assert result is third
    assert include_one in config["include_directory"]
    assert include_two in config["include_directory"]
    assert len(index.calls) == 3
    assert _has_std_arg(index.calls[0][1], "c11")
    assert _has_std_arg(index.calls[1][1], "c11")
    assert _has_std_arg(index.calls[2][1], "c11")
    assert not _has_include_directory_arg(index.calls[0][1], include_one)
    assert _has_include_directory_arg(index.calls[1][1], include_one)
    assert not _has_include_directory_arg(index.calls[1][1], include_two)
    assert _has_include_directory_arg(index.calls[2][1], include_one)
    assert _has_include_directory_arg(index.calls[2][1], include_two)


def test_c_signature_engine_does_not_retry_when_missing_header_is_unresolved(tmp_path: Path) -> None:
    config = _make_extraction_config(source_root=tmp_path, c_std="c11")
    source = tmp_path / "src" / "module.c"
    initial_include_dirs = list(config["include_directory"])

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
        include=config["include"],
        include_directory=config["include_directory"],
        c_std=config["c_std"],
        cpp_std=config["cpp_std"],
    )

    assert result is unresolved
    assert config["include_directory"] == initial_include_dirs
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
        functions=[_unknown_function("foo")],
    )
    second_module = IRModule(
        full_name=QualifiedName.from_str("pkg.second"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )

    visitor.visit_module(first_module)
    visitor.visit_module(second_module)

    assert [arg.name for arg in first_module.functions[0].signatures[0].args] == ["x"]
    assert first_module.functions[0].signatures[0].args[0].type_name == "int"
    assert [arg.name for arg in second_module.functions[0].signatures[0].args] == ["value"]
    assert second_module.functions[0].signatures[0].args[0].type_name == "float"


def test_c_ast_visitor_falls_back_to_unique_leaf_module_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_c_signature_extractor(
        monkeypatch,
        modules={
            "mod": ExtractedModule(
                name="mod",
                functions={
                    "foo": ExtractedFunction(
                        ml_name="foo",
                        function_cursor=_fake_function_cursor("foo"),
                        ml_flags=METH_VARARGS,
                        signatures=[
                            ExtractedSignature(
                                arguments=[ExtractedArgument(name="value", type_name="float")]
                            )
                        ],
                    )
                },
            ),
        },
    )
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )
    visitor = CSignatureExtractionVisitor(source_root=tmp_path)

    visitor.visit_module(module)

    assert [arg.name for arg in module.functions[0].signatures[0].args] == ["value"]
    assert module.functions[0].signatures[0].args[0].type_name == "float"
    assert visitor._stats.success == 1


def test_c_ast_visitor_rejects_ambiguous_leaf_module_match_without_global_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_c_signature_extractor(
        monkeypatch,
        modules={
            "one": ExtractedModule(
                name="mod",
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
                name="mod",
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
        functions=[_unknown_function("foo")],
    )
    visitor = CSignatureExtractionVisitor(
        source_root=tmp_path,
    )

    visitor.visit_module(module)

    assert module.functions[0].signatures == []
    assert visitor._stats.total_unknown_signatures == 1
    assert visitor._stats.missing_module_match == 1
    assert visitor._stats.missing_function_match == 0


def test_c_ast_visitor_overwrites_existing_return_with_raw_inferred_return(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    func = IRFunction(
        name="foo",
        signatures=[],
        doc="original doc",
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
    assert rewritten.signatures[0].return_type_name is not None
    assert rewritten.signatures[0].return_type_name == "typing.Optional[int]"


def test_c_ast_visitor_skips_python_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[_unknown_function("foo")],
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

    assert module.functions[0].signatures == []
    assert extractor.called == 0


def test_c_ast_visitor_propagates_signature_extraction_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )
    _patch_raising_c_signature_extractor(monkeypatch, RuntimeError("boom"))

    visitor = CSignatureExtractionVisitor(
        source_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="boom"):
        visitor.visit_module(module)


def test_write_stubs_propagates_extract_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core as stubgen_module
    from core.stub_generation_options import StubGenerationOptions

    ir_module = IRModule(
        full_name=QualifiedName.from_str("pkg"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
        classes=[
            IRClass(
                name="Builder",
                methods=[IRMethod(function=_unknown_function("build"), decorator=None)],
            )
        ],
        sub_modules=[
            IRModule(
                full_name=QualifiedName.from_str("pkg.child"),
                module_type=IRModuleType.EXTENSION,
                functions=[_unknown_function("bar")],
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


def test_write_stubs_passes_c_inferred_source_comment_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core as stubgen_module

    captured: dict[str, object] = {}
    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"))

    class FakeCSignatureExtractionVisitor:
        def __init__(
            self,
            *,
            source_root: Path,
            include: list[str] = (),
            include_directory: list[Path] = (),
            c_std: str = "c11",
            cpp_std: str = "c++17",
            include_c_inferred_source_comment: bool = False,
        ) -> None:
            captured["visitor_source_root"] = source_root
            captured["visitor_include"] = list(include)
            captured["visitor_include_directory"] = list(include_directory)
            captured["visitor_c_std"] = c_std
            captured["visitor_cpp_std"] = cpp_std
            captured["visitor_include_c_inferred_source_comment"] = (
                include_c_inferred_source_comment
            )

        def visit_module(self, node: IRModule) -> None:
            _ = node

        def visit_class(self, node: IRClass, module: IRModule) -> None:
            _ = (node, module)

        def visit_function(self, node: IRFunction, module: IRModule) -> None:
            _ = (node, module)

        def visit_method(self, node: IRMethod, module: IRModule) -> None:
            _ = (node, module)

        def log_summary(self) -> None:
            captured["visitor_log_summary_called"] = True

    class FakePrinterVisitor:
        def __init__(
            self,
            include_docstrings: bool = True,
            include_module_type_comment: bool = False,
            include_c_inferred_source_comment: bool = False,
        ) -> None:
            captured["printer_include_docstrings"] = include_docstrings
            captured["printer_include_module_type_comment"] = include_module_type_comment
            captured["printer_include_c_inferred_source_comment"] = (
                include_c_inferred_source_comment
            )

        def visit_module(self, node: IRModule) -> list[str]:
            _ = node
            return []

    class FakeWriter:
        def write(
            self,
            module: IRModule,
            printer: FakePrinterVisitor,
            to: Path,
        ) -> None:
            captured["written_module"] = module
            captured["written_printer"] = printer
            captured["written_to"] = to

    monkeypatch.setattr(stubgen_module, "build_module", lambda path, module: ir_module)
    monkeypatch.setattr(
        stubgen_module,
        "CSignatureExtractionVisitor",
        FakeCSignatureExtractionVisitor,
    )
    monkeypatch.setattr(stubgen_module, "PrinterVisitor", FakePrinterVisitor)

    options = StubGenerationOptions(
        source_root=tmp_path,
        include=["Python.h"],
        include_directory=[tmp_path / "include"],
        c_std="c99",
        cpp_std="c++20",
        include_docstrings=False,
        include_module_type_comment=True,
        include_c_inferred_source_comment=True,
    )

    stubgen_module.write_stubs(
        "math",
        tmp_path,
        options=options,
        _writer=FakeWriter(),
    )

    assert captured["visitor_source_root"] == tmp_path
    assert captured["visitor_include"] == ["Python.h"]
    assert captured["visitor_include_directory"] == [tmp_path / "include"]
    assert captured["visitor_c_std"] == "c99"
    assert captured["visitor_cpp_std"] == "c++20"
    assert captured["visitor_include_c_inferred_source_comment"] is True
    assert captured["visitor_log_summary_called"] is True
    assert captured["printer_include_docstrings"] is False
    assert captured["printer_include_module_type_comment"] is True
    assert captured["printer_include_c_inferred_source_comment"] is True
    assert captured["written_module"] is ir_module
    assert captured["written_to"] == tmp_path


def test_doc_parser_preserves_rewritten_signature_without_c_ast_candidates(
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
    extractor = _patch_c_signature_extractor(monkeypatch, modules={})

    Pipeline(
        [
            DocStringSignatureParserVisitor(),
            CSignatureExtractionVisitor(
                source_root=tmp_path,
            ),
        ]
    ).run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.signatures[0].args] == ["x", "y", "w", "out", "p"]
    assert parsed.signatures[0].return_type_name == "numpy.ndarray"
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
        c_std="c11",
    )
    extracted = engine.extract_modules()

    assert set(extracted) == {"first", "second"}
    assert set(extracted["first"].functions) == {"foo"}
    assert set(extracted["second"].functions) == {"foo"}
    assert extracted["first"].functions["foo"].ml_flags == METH_VARARGS
    assert extracted["second"].functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_extract_modules_populates_inferred_return_only_signature(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "return_only_module.c"
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
                "PyObject* PyLong_FromLong(long value);",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    return PyLong_FromLong(1);",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"return_only\",",
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
        c_std="c11",
    )
    extracted = engine.extract_modules()

    signatures = extracted["return_only"].functions["foo"].signatures
    assert len(signatures) == 1
    assert signatures[0].arguments == []
    assert signatures[0].return_type_name == "int"


def test_c_signature_extraction_engine_extract_modules_infers_parse_tuple_arguments(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "parse_tuple_args.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct _typeobject PyTypeObject;",
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
                "PyTypeObject PyList_Type;",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    int count = 0;",
                "    PyObject* items = (PyObject*)0;",
                "    if (!PyArg_ParseTuple(args, \"iO!\", &count, &PyList_Type, &items)) {",
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
                "    \"parse_tuple_args\",",
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
        c_std="c11",
    )
    extracted = engine.extract_modules()

    signatures = extracted["parse_tuple_args"].functions["foo"].signatures
    assert signatures == [
        ExtractedSignature(
            arguments=[
                ExtractedArgument(name="count", type_name="int"),
                ExtractedArgument(name="items", type_name="list"),
            ]
        )
    ]


def test_c_signature_extraction_engine_extract_modules_reads_object_type_from_extent_source_text(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "parse_tuple_extent_text.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct _typeobject PyTypeObject;",
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
                "PyTypeObject PyArray_Type;",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    /* 中文注释，验证 extent offset 按字节切片 */",
                "    PyObject* array = (PyObject*)0;",
                "    if (!PyArg_ParseTuple(args, \"O!\", &PyArray_Type, &array)) {",
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
                "    \"parse_tuple_extent_text\",",
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
        c_std="c11",
    )
    extracted = engine.extract_modules()

    signatures = extracted["parse_tuple_extent_text"].functions["foo"].signatures
    assert signatures == [
        ExtractedSignature(
            arguments=[
                ExtractedArgument(name="array", type_name="numpy.ndarray"),
            ]
        )
    ]


def test_c_signature_extraction_engine_extract_modules_emits_multiple_signatures_for_multiple_pyarg_calls(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "multiple_pyarg_signatures.c"
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
                "    const char* label = 0;",
                "    if (0) {",
                "        if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "            return (PyObject*)0;",
                "        }",
                "    }",
                "    if (!PyArg_ParseTuple(args, \"s\", &label)) {",
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
                "    \"multiple_pyarg_signatures\",",
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
        c_std="c11",
    )
    extracted = engine.extract_modules()

    signatures = extracted["multiple_pyarg_signatures"].functions["foo"].signatures
    assert signatures == [
        ExtractedSignature(arguments=[ExtractedArgument(name="value", type_name="int")]),
        ExtractedSignature(arguments=[ExtractedArgument(name="label", type_name="str")]),
    ]


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
        c_std="c11",
    )
    extracted = engine.extract_modules()

    assert set(extracted) == {"first", "second"}
    assert set(extracted["first"].functions) == {"foo"}
    assert set(extracted["second"].functions) == {"foo"}
    assert extracted["first"].functions["foo"].ml_flags == METH_VARARGS
    assert extracted["second"].functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_discards_duplicate_modules_across_files(
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
        c_std="c11",
    )
    extracted = engine.extract_modules()

    module = extracted["dup.shared"]
    assert set(module.functions) == {"foo"}
    assert module.functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_discards_duplicate_modules_in_one_file(
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
        c_std="c11",
    )
    extracted = engine.extract_modules()

    module = extracted["dup.same_file"]
    assert set(module.functions) == {"foo"}
    assert module.functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_warns_and_keeps_first_duplicate_in_same_method_table(
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
        c_std="c11",
    )
    extracted = engine.extract_modules()

    module = extracted["dup.mod"]
    assert module.functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_warns_and_discards_duplicate_module_across_files(
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
        c_std="c11",
    )
    extracted = engine.extract_modules()

    module = extracted["dup.shared"]
    assert module.functions["foo"].ml_flags == METH_VARARGS


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
        c_std="c11",
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
        c_std="c11",
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
        c_std="c11",
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
        c_std="c11",
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
        c_std="c11",
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
        c_std="c11",
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
        c_std="c11",
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
        c_std="c11",
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
        cpp_std="c++17",
    )
    extracted = engine.extract_modules()

    assert extracted == {}


def test_c_ast_visitor_passes_clang_options_to_extractor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {
        "extract_calls": 0,
        "include": None,
    }

    def _record_extract_c_signature_modules(
        source_root: Path,
        *,
        include: list[str] = (),
        include_directory: list[Path] = (),
        c_std: str = "c11",
        cpp_std: str = "c++17",
    ) -> dict[str, ExtractedModule]:
        captured["extract_calls"] = int(captured["extract_calls"]) + 1
        captured["source_root"] = source_root
        captured["include"] = list(include)
        captured["include_directory"] = list(include_directory)
        captured["c_std"] = c_std
        captured["cpp_std"] = cpp_std
        return {}

    import core.node_visitors.c_signature_extraction.c_signature_extraction_visitor as visitor_module

    monkeypatch.setattr(
        visitor_module,
        "extract_c_signature_modules",
        _record_extract_c_signature_modules,
    )

    visitor = CSignatureExtractionVisitor(
        source_root=tmp_path,
        include=["Python.h"],
        include_directory=[Path("C:/MyInclude")],
        c_std="c99",
        cpp_std="c++20",
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
    assert captured["include"] == ["Python.h"]
    assert captured["include_directory"] == [Path("C:/MyInclude")]
    assert captured["c_std"] == "c99"
    assert captured["cpp_std"] == "c++20"


def test_c_signature_engine_extract_modules_keeps_external_include_options_and_injects_python_include_dirs(
    tmp_path: Path,
) -> None:
    engine = CSignatureExtractor(
        source_root=tmp_path,
        include=["Python.h"],
        include_directory=[Path("C:/MyInclude")],
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(translation_unit_module, "find_candidate_files", lambda source_root: [])

    try:
        assert engine.extract_modules() == {}
        expected_include_dirs = [Path("C:/MyInclude")]
        for include_dir in [sysconfig.get_path("include"), sysconfig.get_path("platinclude")]:
            if not include_dir:
                continue
            include_path = Path(include_dir)
            if include_path in expected_include_dirs:
                continue
            expected_include_dirs.append(include_path)

        assert engine._include == ["Python.h"]
        assert engine._include_directory == expected_include_dirs
    finally:
        monkeypatch.undo()


def test_c_signature_engine_build_parse_args_uses_only_external_include_values(tmp_path: Path) -> None:
    engine = CSignatureExtractor(source_root=tmp_path, c_std="c11")

    assert translation_unit_module.build_clang_parse_args(
        tmp_path / "module.c",
        include=engine._include,
        include_directory=engine._include_directory,
        c_std=engine._c_std,
        cpp_std=engine._cpp_std,
    ) == [
        "--std",
        "c11",
        *[
            item
            for include_dir in engine._include_directory
            for item in ("--include-directory", str(include_dir))
        ],
    ]


def test_c_signature_engine_build_parse_args_places_include_before_include_directory(tmp_path: Path) -> None:
    engine = CSignatureExtractor(
        source_root=tmp_path,
        include=["Python.h", "numpy/arrayobject.h"],
        include_directory=[Path("C:/MyInclude")],
        c_std="c11",
    )

    assert translation_unit_module.build_clang_parse_args(
        tmp_path / "module.c",
        include=engine._include,
        include_directory=engine._include_directory,
        c_std=engine._c_std,
        cpp_std=engine._cpp_std,
    ) == [
        "--std",
        "c11",
        "--include",
        "Python.h",
        "--include",
        "numpy/arrayobject.h",
        "--include-directory",
        str(Path("C:/MyInclude")),
        *[
            item
            for include_dir in engine._include_directory[1:]
            for item in ("--include-directory", str(include_dir))
        ],
    ]


class _FakeToken:
    def __init__(self, kind: object, spelling: str) -> None:
        self.kind = kind
        self.spelling = spelling


class _FakeCursorLocation:
    def __init__(self, file: str | None = None, offset: int = 0) -> None:
        self.file = _FakeCursorFile(file) if file is not None else None
        self.offset = offset


class _FakeCursorFile:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSourceRange:
    def __init__(self, start: _FakeCursorLocation, end: _FakeCursorLocation) -> None:
        self.start = start
        self.end = end


class _FakeNode:
    def __init__(
        self,
        *,
        kind: object,
        tokens: list[_FakeToken] | None = None,
        children: list[object] | None = None,
        spelling: str = "",
        location: object | None = None,
        extent: object | None = None,
        referenced: object | None = None,
    ) -> None:
        self.kind = kind
        self._tokens = tokens or []
        self._children = children or []
        self.spelling = spelling
        self.location = location if location is not None else _FakeCursorLocation()
        self.extent = extent
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


def _float_literal(value: str) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.FLOATING_LITERAL,
        tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, value)],
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


def _extent_for_source_snippet(source_path: Path, snippet: str) -> _FakeSourceRange:
    source_bytes = source_path.read_bytes()
    snippet_bytes = snippet.encode("utf-8")
    start_offset = source_bytes.index(snippet_bytes)
    end_offset = start_offset + len(snippet_bytes)
    return _FakeSourceRange(
        _FakeCursorLocation(str(source_path), start_offset),
        _FakeCursorLocation(str(source_path), end_offset),
    )


def _signed_numeric_literal(sign: str, child: _FakeNode) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        tokens=[_FakeToken(clang.cindex.TokenKind.PUNCTUATION, sign)],
        children=[child],
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


def _kwlist_decl(name: str, *keywords: str) -> _FakeNode:
    return _var_decl(
        name,
        _init_list(
            *[_string_literal(keyword) for keyword in keywords],
            _FakeNode(kind=clang.cindex.CursorKind.GNU_NULL_EXPR),
        ),
    )


def _type_object_decl(name: str, tp_name: str) -> _FakeNode:
    return _var_decl(
        name,
        _init_list(_designated_initializer("tp_name", _string_literal(tp_name))),
    )


@pytest.mark.parametrize(
    ("token_name", "expected"),
    [
        ("Py_None", NamedTypeNode("None")),
        ("Py_True", NamedTypeNode("bool")),
        ("Py_False", NamedTypeNode("bool")),
    ],
)
def test_infer_expr_type_detects_direct_object_returns(token_name: str, expected: NamedTypeNode) -> None:
    inferred = signature_rules_module.infer_expr_type(_identifier_node(token_name))

    assert inferred == expected


@pytest.mark.parametrize(
    ("token_name", "expected"),
    [
        ("Py_RETURN_NONE", NamedTypeNode("None")),
        ("Py_RETURN_TRUE", NamedTypeNode("bool")),
        ("Py_RETURN_FALSE", NamedTypeNode("bool")),
        ("Py_RETURN_NAN", NamedTypeNode("float")),
        ("Py_RETURN_INF", NamedTypeNode("float")),
    ],
)
def test_infer_expr_type_detects_preserved_macro_tokens(token_name: str, expected: NamedTypeNode) -> None:
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
        ("PyBool_FromLong", NamedTypeNode("bool")),
        ("PyLong_FromLong", NamedTypeNode("int")),
        ("PyFloat_FromDouble", NamedTypeNode("float")),
        ("PyComplex_FromDoubles", NamedTypeNode("complex")),
        ("PyUnicode_FromString", NamedTypeNode("str")),
        ("PyUnicode_AsUTF8String", NamedTypeNode("bytes")),
        ("PyByteArray_FromObject", NamedTypeNode("bytearray")),
        ("PySlice_New", NamedTypeNode("slice")),
        ("PyMemoryView_FromObject", NamedTypeNode("memoryview")),
        ("PyTuple_New", NamedTypeNode("tuple")),
        ("PyList_New", NamedTypeNode("list")),
        ("PyDict_New", NamedTypeNode("dict")),
        ("PySet_New", NamedTypeNode("set")),
        ("PyFrozenSet_New", NamedTypeNode("frozenset")),
        ("PyList_AsTuple", NamedTypeNode("tuple")),
        ("PyDict_Items", NamedTypeNode("list")),
    ],
)
def test_infer_expr_type_detects_exact_factory_mappings(call_name: str, expected: NamedTypeNode) -> None:
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

    assert inferred == TupleTypeNode(
        (
            NamedTypeNode("int"),
            UnionTypeNode((NamedTypeNode("None"), NamedTypeNode("str"))),
        )
    )


def test_infer_expr_type_canonicalizes_py_buildvalue_container_unions() -> None:
    """`Py_BuildValue` 推断结果应在渲染前先规范化容器内部的联合类型。"""
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            "Py_BuildValue",
            _string_literal("[si]"),
            _identifier_node("name"),
            _identifier_node("count"),
        )
    )

    assert inferred == ListTypeNode(
        UnionTypeNode(
            (
                NamedTypeNode("None"),
                NamedTypeNode("int"),
                NamedTypeNode("str"),
            )
        )
    )


def test_infer_expr_type_resolves_py_buildvalue_object_slots() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            "Py_BuildValue",
            _string_literal("(O)"),
            _call_expr("PyLong_FromLong", _identifier_node("value")),
        )
    )

    assert inferred == TupleTypeNode((NamedTypeNode("int"),))


def test_infer_expr_type_keeps_py_buildvalue_object_slots_as_any_when_unknown() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            "Py_BuildValue",
            _string_literal("(O)"),
            _call_expr("CustomFactory", _identifier_node("value")),
        )
    )

    assert inferred == TupleTypeNode((AnyTypeNode(),))


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

    assert inferred == NamedTypeNode("bytes")


def test_infer_expr_type_returns_none_when_conditional_branches_are_unknown() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _conditional_expr(
            _macro_expr("Py_RETURN_NONE"),
            _call_expr("CustomFactory", _identifier_node("left")),
            _identifier_node("UnknownToken"),
        )
    )

    assert inferred is None


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

    inferred = signature_rules_module.infer_return_type(cursor)

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

    inferred = signature_rules_module.infer_return_type(cursor)

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

    inferred = signature_rules_module.infer_return_type(cursor)

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

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == "tuple[int, None | str]"


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

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == "bytes"


def test_return_type_deduplicates_and_canonicalizes_order() -> None:
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

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == "None | bool | int"


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

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == "float | int"


def test_return_type_deduplicates_members_already_present_in_union_return() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(
            _conditional_expr(
                _identifier_node("cond"),
                _call_expr("PyLong_FromLong", _identifier_node("value")),
                _call_expr("PyFloat_FromDouble", _identifier_node("value")),
            )
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == "float | int"


def test_return_type_returns_none_for_unsupported_returns() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("CustomFactory", _identifier_node("value"))),
        _return_stmt(),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is None


def test_return_type_returns_none_when_function_has_no_return() -> None:
    cursor = _fake_function_cursor_with_children()

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is None


def test_infer_argument_lists_parses_pyarg_parsetuple() -> None:
    count_decl = _var_decl("count", _int_literal("1"))
    label_decl = _var_decl("label", _identifier_node("Py_None"))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i|z"),
            _address_of("count", referenced=count_decl),
            _address_of("label", referenced=label_decl),
        )
    )

    inferred = signature_rules_module.infer_argument_lists(cursor)

    assert inferred == [
        [
            ExtractedArgument(name="count", type_name="int"),
            ExtractedArgument(
                name="label",
                type_name="str | None",
                default_value="None",
                has_default=True,
            ),
        ]
    ]


def test_resolve_object_type_for_pyarg_reads_name_from_extent_source_text(tmp_path: Path) -> None:
    source = tmp_path / "object_type_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O!\", (&PyUnicode_Type), &value);",
            ]
        ),
        encoding="utf-8",
    )
    cursor = _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        extent=_extent_for_source_snippet(source, "(&PyUnicode_Type)"),
    )

    inferred = signature_rules_module._resolve_object_type_for_pyarg(cursor)

    assert inferred == "str"


def test_extract_cursor_source_text_reads_text_from_extent(tmp_path: Path) -> None:
    source = tmp_path / "extent_text.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O!\", (&PyUnicode_Type), &value);",
            ]
        ),
        encoding="utf-8",
    )

    extracted = cursor_utils_module.source_range_get_text(
        _extent_for_source_snippet(source, "(&PyUnicode_Type)")
    )

    assert extracted == "(&PyUnicode_Type)"


def test_extract_cursor_source_text_returns_none_when_extent_start_file_is_missing() -> None:
    extracted = cursor_utils_module.source_range_get_text(
        _FakeSourceRange(
            _FakeCursorLocation(None, 0),
            _FakeCursorLocation("extent_text.c", 1),
        )
    )

    assert extracted is None


def test_extract_cursor_source_text_returns_none_for_cross_file_extent(tmp_path: Path) -> None:
    first = tmp_path / "first.c"
    second = tmp_path / "second.c"
    first.write_text("abc", encoding="utf-8")
    second.write_text("def", encoding="utf-8")

    extracted = cursor_utils_module.source_range_get_text(
        _FakeSourceRange(
            _FakeCursorLocation(str(first), 0),
            _FakeCursorLocation(str(second), 1),
        )
    )

    assert extracted is None


def test_extract_cursor_source_text_returns_none_when_file_read_fails(tmp_path: Path) -> None:
    missing = tmp_path / "missing_extent_text.c"

    extracted = cursor_utils_module.source_range_get_text(
        _FakeSourceRange(
            _FakeCursorLocation(str(missing), 0),
            _FakeCursorLocation(str(missing), 1),
        )
    )

    assert extracted is None


def test_infer_argument_lists_keeps_object_fallback_for_unknown_o_bang_type(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unknown_object_type_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O!\", (&UnknownRuntimeType), &value);",
            ]
        ),
        encoding="utf-8",
    )
    value_decl = _var_decl("value")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("O!"),
            _FakeNode(
                kind=clang.cindex.CursorKind.UNARY_OPERATOR,
                extent=_extent_for_source_snippet(source, "(&UnknownRuntimeType)"),
            ),
            _address_of("value", referenced=value_decl),
        )
    )

    inferred = signature_rules_module.infer_argument_lists(cursor)

    assert inferred == [
        [ExtractedArgument(name="value", type_name="object")]
    ]


def test_infer_argument_lists_joins_decl_ref_names_for_tuple_arguments() -> None:
    left_decl = _var_decl("left")
    right_decl = _var_decl("right")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("(ii)"),
            _address_of("left", referenced=left_decl),
            _address_of("right", referenced=right_decl),
        )
    )

    inferred = signature_rules_module.infer_argument_lists(cursor)

    assert inferred == [
        [ExtractedArgument(name="left_right", type_name="tuple[int, int]")]
    ]


def test_infer_argument_lists_skips_parse_tuple_and_keywords_without_valid_kwlist() -> None:
    invalid_kwlist_decl = _var_decl("kwlist", _init_list(_identifier_node("bad"), _null_ptr_literal()))
    x_decl = _var_decl("x", _float_literal("0.0"))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTupleAndKeywords",
            _identifier_node("args"),
            _identifier_node("kwds"),
            _string_literal("|d"),
            _token_identifier_node("kwlist", referenced=invalid_kwlist_decl),
            _address_of("x", referenced=x_decl),
        )
    )

    inferred = signature_rules_module.infer_argument_lists(cursor)

    assert inferred == []


def test_infer_argument_lists_keeps_matching_pyarg_calls() -> None:
    first_decl = _var_decl("value")
    second_decl = _var_decl("value")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            _address_of("value", referenced=first_decl),
        ),
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            _address_of("value", referenced=second_decl),
        ),
    )

    inferred = signature_rules_module.infer_argument_lists(cursor)

    assert inferred == [
        [ExtractedArgument(name="value", type_name="int")],
        [ExtractedArgument(name="value", type_name="int")],
    ]


def test_infer_signature_returns_signature_with_inferred_return_type() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(cursor)

    assert inferred == [
        ExtractedSignature(
            arguments=[],
            return_type_name="int",
        )
    ]


def test_infer_signature_returns_signature_with_inferred_arguments_when_return_is_unknown() -> None:
    value_decl = _var_decl("value", _int_literal("0"))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            _address_of("value", referenced=value_decl),
        ),
        _return_stmt(_call_expr("CustomFactory", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(cursor)

    assert inferred == [
        ExtractedSignature(arguments=[ExtractedArgument(name="value", type_name="int")])
    ]


def test_infer_signature_returns_empty_when_return_type_is_unknown() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("CustomFactory", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(cursor)

    assert inferred == []


def test_infer_signature_merges_inferred_arguments_and_return_type() -> None:
    value_decl = _var_decl("value", _int_literal("0"))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            _address_of("value", referenced=value_decl),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(cursor)

    assert inferred == [
        ExtractedSignature(
            arguments=[ExtractedArgument(name="value", type_name="int")],
            return_type_name="int",
        )
    ]


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


def test_c_signature_engine_keeps_empty_flags_when_ast_field_is_unparseable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_eval_int(monkeypatch)
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


def test_c_signature_engine_extract_pymethoddef_init_list_expr_marks_sentinel(tmp_path: Path) -> None:
    is_sentinel, extracted = _extract_PyMethodDef_INIT_LIST_EXPR(
        init_list_expr=_init_list(_null_ptr_literal()),
    )

    assert is_sentinel is True
    assert extracted is None


def test_c_signature_engine_extract_pymethoddef_init_list_expr_discards_entry_without_function_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证缺失 `ml_meth` 引用时当前条目会被直接丢弃。"""
    _patch_fake_eval_int(monkeypatch)
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

