from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Iterable

import clang.cindex
import pytest

c_signature_extractor_module = importlib.import_module(
    "pcstubgen2.NodeVisitors.CSignatureInference.CSignatureExtraction.CSignatureExtractor"
)
from pcstubgen2.NodeVisitors.CSignatureInference.CSignatureExtraction import CSignatureExtractor
from pcstubgen2.NodeVisitors.CSignatureInference.CSignatureExtraction.CSignatureExtractor import (
    _format_single_diagnostic,
    _get_diagnostic_severity_name,
    _is_initializer_list_PyMethodDef,
    _is_PyMethodDef_array,
    _is_PyMethodDef_sentinel,
)
from pcstubgen2.NodeVisitors.CSignatureInference.CSignatureExtraction.Models import (
    ExtractedArgument,
    ExtractedFunction,
    ExtractedSignature,
)
from pcstubgen2.ErrorCollector import ErrorCollector
from pcstubgen2.IR import (
    IRArgument,
    IRArgumentKind,
    IRClass,
    IRFunction,
    IRMethod,
    IRModule,
    IRModuleType,
    QualifiedName,
    ResolvedType,
)
from pcstubgen2.NodeVisitors.CSignatureInference.CAstSignatureInferenceVisitor import (
    CAstSignatureInferenceVisitor,
)
from pcstubgen2.NodeVisitors.DocStringSignatureParserVisitor import (
    DocStringSignatureParserVisitor,
)
from pcstubgen2.NodeVisitors.Fixes import InferMethodModifierVisitor
from pcstubgen2.Pipeline import Pipeline
from pcstubgen2.StubGenerationOptions import StubGenerationOptions


def _generic_signature() -> list[IRArgument]:
    return [
        IRArgument(name="args", kind=IRArgumentKind.VAR_POSITIONAL),
        IRArgument(name="kwargs", kind=IRArgumentKind.VAR_KEYWORD),
    ]


class _FakeExtractor:
    def __init__(self, data: dict[str, list[ExtractedFunction]]) -> None:
        self.data = data
        self.called = 0
        self._cached_result: dict[str, list[ExtractedFunction]] | None = None

    def extract(self) -> dict[str, list[ExtractedFunction]]:
        if self._cached_result is None:
            self.called += 1
            self._cached_result = self.data
        return self._cached_result


def _patch_c_signature_extractor(
    monkeypatch: pytest.MonkeyPatch,
    data: dict[str, list[ExtractedFunction]],
) -> _FakeExtractor:
    extractor = _FakeExtractor(data)

    class _PatchedExtractor:
        def __init__(
            self,
            source_root: Path,
            *,
            clang_parse_args: Iterable[str] = (),
            clang_c_std: str | None = None,
            clang_cpp_std: str | None = None,
        ) -> None:
            _ = (source_root, clang_parse_args, clang_c_std, clang_cpp_std)

        def extract(self) -> dict[str, list[ExtractedFunction]]:
            return extractor.extract()

    import pcstubgen2.NodeVisitors.CSignatureInference.CAstSignatureInferenceVisitor as visitor_module

    monkeypatch.setattr(visitor_module, "CSignatureExtractor", _PatchedExtractor)
    return extractor


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


class _DiagnosticlessTranslationUnit:
    pass


class _FakeIndex:
    def __init__(self, translation_unit: _FakeTranslationUnit) -> None:
        self.translation_unit = translation_unit

    def parse(self, filename: str, args: list[str]) -> _FakeTranslationUnit:
        return self.translation_unit


class _RaisingIndex:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def parse(self, filename: str, args: list[str]) -> None:
        raise self.error


def test_c_ast_visitor_rewrites_module_function_and_drops_self(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    func = IRFunction(name="foo", args=_generic_signature())
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
        functions=[func],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        {
            "foo": [
                ExtractedFunction(
                    py_name="foo",
                    c_name="c_foo",
                    method_flags=["METH_VARARGS"],
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
            ]
        }
    )

    visitor = CAstSignatureInferenceVisitor(
        error_collector=ErrorCollector(),
        c_source_root=tmp_path,
    )
    visitor.visit_module(module)

    rewritten = module.functions[0]
    assert [arg.name for arg in rewritten.args] == ["x", "flag"]
    assert str(rewritten.args[0].annotation) == "int"
    assert str(rewritten.args[1].annotation) == "bool"
    assert rewritten.args[1].default is not None
    assert rewritten.args[1].default.repr == "False"
    assert rewritten.return_annotation is not None
    assert str(rewritten.return_annotation) == "int"


def test_c_ast_visitor_logs_successful_generic_rewrite(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    func = IRFunction(name="foo", args=_generic_signature())
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
        functions=[func],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        {
            "foo": [
                ExtractedFunction(
                    py_name="foo",
                    c_name="c_foo",
                    method_flags=["METH_VARARGS"],
                    signatures=[
                        ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")]),
                    ],
                )
            ]
        },
    )

    visitor = CAstSignatureInferenceVisitor(
        error_collector=ErrorCollector(),
        c_source_root=tmp_path,
    )
    with caplog.at_level(logging.INFO, logger="pcstubgen2"):
        visitor.visit_module(module)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.INFO
    assert record.message == (
        "Rewrote generic signature for foo (is_method=False): "
        "selected_candidates=1, generated_signatures=1"
    )


def test_c_ast_visitor_logs_when_generic_function_has_no_candidates(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    visitor = CAstSignatureInferenceVisitor(
        error_collector=ErrorCollector(),
        c_source_root=tmp_path,
    )
    func = IRFunction(name="foo", args=_generic_signature())

    with caplog.at_level(logging.WARNING, logger="pcstubgen2"):
        rewritten = visitor._rewrite_function(func=func, signatures={}, is_method=False)

    assert rewritten == [func]
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert record.message == (
        "Failed to rewrite generic signature for foo (is_method=False): "
        "no C signature candidates found"
    )


def test_c_ast_visitor_logs_when_candidate_selection_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    visitor = CAstSignatureInferenceVisitor(
        error_collector=ErrorCollector(),
        c_source_root=tmp_path,
    )
    func = IRFunction(name="foo", args=_generic_signature())
    signatures = {
        "foo": [
            ExtractedFunction(
                py_name="foo",
                c_name="c_foo",
                method_flags=["METH_VARARGS"],
                signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")])],
            )
        ]
    }
    monkeypatch.setattr(visitor, "_select_candidate", lambda candidates, *, is_method: None)

    with caplog.at_level(logging.WARNING, logger="pcstubgen2"):
        rewritten = visitor._rewrite_function(func=func, signatures=signatures, is_method=False)

    assert rewritten == [func]
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert record.message == (
        "Failed to rewrite generic signature for foo (is_method=False): "
        "candidate selection failed"
    )


def test_c_ast_visitor_does_not_log_for_non_generic_function(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    visitor = CAstSignatureInferenceVisitor(
        error_collector=ErrorCollector(),
        c_source_root=tmp_path,
    )
    func = IRFunction(name="foo", args=[IRArgument(name="x", kind=IRArgumentKind.POSITIONAL_OR_KEYWORD)])
    signatures = {
        "foo": [
            ExtractedFunction(
                py_name="foo",
                c_name="c_foo",
                method_flags=["METH_VARARGS"],
                signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")])],
            )
        ]
    }

    with caplog.at_level(logging.INFO, logger="pcstubgen2"):
        rewritten = visitor._rewrite_function(func=func, signatures=signatures, is_method=False)

    assert rewritten == [func]
    assert caplog.records == []


def test_c_signature_engine_logs_parse_exception_details(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_parse_args=["-DMY_FLAG=1"],
    )
    source = tmp_path / "broken_module.cxx"

    with caplog.at_level(logging.WARNING, logger="pcstubgen2"):
        result = engine._parse_translation_unit(
            index=_RaisingIndex(RuntimeError("boom")),
            file_path=source,
        )

    assert result is None
    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert str(source) in message
    assert "suffix: .cxx" in message
    assert "parse_args: ['-std=c++17', '-DMY_FLAG=1']" in message
    assert "exception_type: RuntimeError" in message
    assert "exception: boom" in message


def test_c_signature_engine_logs_all_diagnostics_when_error_present(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    engine = CSignatureExtractor(source_root=tmp_path, clang_c_std="c11")
    engine._clang = _FakeClangWithDiagnostics
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

    with caplog.at_level(logging.WARNING, logger="pcstubgen2"):
        result = engine._parse_translation_unit(
            index=_FakeIndex(translation_unit),
            file_path=source,
        )

    assert result is translation_unit
    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert str(source) in message
    assert "suffix: .c" in message
    assert "parse_args: ['-std=c11']" in message
    assert f"[WARNING] {source}:3:1: warning detail" in message
    assert f"[ERROR] {source}:7:9: error detail" in message
    assert f"[FATAL] {source}:11:4: fatal detail" in message


def test_c_signature_engine_skips_logging_for_non_error_diagnostics(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    engine = CSignatureExtractor(source_root=tmp_path, clang_c_std="c11")
    engine._clang = _FakeClangWithDiagnostics
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

    with caplog.at_level(logging.WARNING, logger="pcstubgen2"):
        result = engine._parse_translation_unit(
            index=_FakeIndex(translation_unit),
            file_path=source,
        )

    assert result is translation_unit
    assert caplog.records == []


def test_c_signature_engine_raises_when_diagnostic_missing_required_field(tmp_path: Path) -> None:
    class _MissingSeverityDiagnostic:
        def __init__(self) -> None:
            self.spelling = "broken detail"
            self.location = _FakeDiagnosticLocation(
                file_name=str(tmp_path / "module.c"),
                line=1,
                column=2,
            )

    with pytest.raises(AttributeError):
        _format_single_diagnostic(_MissingSeverityDiagnostic())  # type: ignore[arg-type]


def test_c_signature_engine_raises_when_translation_unit_missing_diagnostics(tmp_path: Path) -> None:
    engine = CSignatureExtractor(source_root=tmp_path, clang_c_std="c11")
    source = tmp_path / "module.c"

    with pytest.raises(AttributeError):
        engine._parse_translation_unit(
            index=_FakeIndex(_DiagnosticlessTranslationUnit()),  # type: ignore[arg-type]
            file_path=source,
        )


def test_c_signature_engine_maps_all_builtin_severity_names(tmp_path: Path) -> None:
    assert _get_diagnostic_severity_name(clang.cindex.Diagnostic.Ignored) == "IGNORED"
    assert _get_diagnostic_severity_name(clang.cindex.Diagnostic.Note) == "NOTE"
    assert _get_diagnostic_severity_name(clang.cindex.Diagnostic.Warning) == "WARNING"
    assert _get_diagnostic_severity_name(clang.cindex.Diagnostic.Error) == "ERROR"
    assert _get_diagnostic_severity_name(clang.cindex.Diagnostic.Fatal) == "FATAL"


def test_c_ast_visitor_keeps_existing_return_when_inferred_return_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    func = IRFunction(
        name="foo",
        args=_generic_signature(),
        return_annotation=ResolvedType(name=QualifiedName.from_str("bytes")),
    )
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
        functions=[func],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        {
            "foo": [
                ExtractedFunction(
                    py_name="foo",
                    c_name="c_foo",
                    method_flags=["METH_VARARGS"],
                    signatures=[
                        ExtractedSignature(
                            arguments=[ExtractedArgument(name="x", type_name="int")],
                            return_type_name="typing.Optional[int]",
                        )
                    ],
                )
            ]
        },
    )

    visitor = CAstSignatureInferenceVisitor(
        error_collector=ErrorCollector(),
        c_source_root=tmp_path,
    )
    visitor.visit_module(module)

    rewritten = module.functions[0]
    assert rewritten.return_annotation is not None
    assert str(rewritten.return_annotation) == "bytes"


def test_c_ast_visitor_generates_overloads_for_methods(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    method = IRMethod(function=IRFunction(name="build", args=_generic_signature()), decorator=None)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
        classes=[IRClass(name="C", methods=[method])],
    )

    _patch_c_signature_extractor(
        monkeypatch,
        {
            "build": [
                ExtractedFunction(
                    py_name="build",
                    c_name="c_build",
                    method_flags=["METH_VARARGS"],
                    signatures=[
                        ExtractedSignature(
                            arguments=[
                                ExtractedArgument(name="self", type_name="object"),
                                ExtractedArgument(name="count", type_name="int"),
                            ]
                        ),
                        ExtractedSignature(
                            arguments=[
                                ExtractedArgument(name="self", type_name="object"),
                                ExtractedArgument(name="count", type_name="int"),
                                ExtractedArgument(name="scale", type_name="float", default_value="1.0"),
                            ]
                        ),
                    ],
                )
            ]
        },
    )

    visitor = CAstSignatureInferenceVisitor(
        error_collector=ErrorCollector(),
        c_source_root=tmp_path,
    )
    visitor.visit_module(module)

    methods = module.classes[0].methods
    assert len(methods) == 2
    assert [m.function.args[0].name for m in methods] == ["self", "self"]
    assert [str(m.function.args[0].annotation) for m in methods] == ["object", "object"]
    assert [str(m.function.args[1].annotation) for m in methods] == ["int", "int"]
    assert methods[1].function.args[2].annotation is not None
    assert str(methods[1].function.args[2].annotation) == "float"
    assert all(m.function.decorators == ["typing.overload"] for m in methods)
    assert methods[1].function.args[2].default is not None
    assert methods[1].function.args[2].default.repr == "1.0"


def test_c_ast_visitor_skips_python_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[IRFunction(name="foo", args=_generic_signature())],
    )
    extractor = _patch_c_signature_extractor(
        monkeypatch,
        {
            "foo": [
                ExtractedFunction(
                    py_name="foo",
                    c_name="c_foo",
                    signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")])],
                )
            ]
        },
    )
    visitor = CAstSignatureInferenceVisitor(
        error_collector=ErrorCollector(),
        c_source_root=tmp_path,
    )
    visitor.visit_module(module)

    assert module.functions[0].is_generic_signature()
    assert extractor.called == 0


def test_write_stubs_skips_c_ast_visitor_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pcstubgen2 as stubgen_module
    from pcstubgen2.StubGenerationOptions import StubGenerationOptions

    def _unexpected_constructor(*args: object, **kwargs: object) -> object:
        raise AssertionError("CAstSignatureInferenceVisitor should not be instantiated when disabled")

    monkeypatch.setattr(stubgen_module, "CAstSignatureInferenceVisitor", _unexpected_constructor)

    options = StubGenerationOptions(
        enable_docstring_signature_parser=False,
        enable_c_signature_inference=False,
    )
    stubgen_module.write_stubs("math", tmp_path, options=options)

    assert list(tmp_path.rglob("*.pyi"))


def test_write_stubs_raises_when_c_inference_enabled_without_source_root(tmp_path: Path) -> None:
    import pcstubgen2 as stubgen_module
    from pcstubgen2.StubGenerationOptions import StubGenerationOptions

    options = StubGenerationOptions(
        enable_docstring_signature_parser=False,
        enable_c_signature_inference=True,
        c_source_root=None,
    )

    with pytest.raises(
        ValueError,
        match="enable_c_signature_inference=True requires c_source_root",
    ):
        stubgen_module.write_stubs("math", tmp_path, options=options)


def test_write_stubs_defaults_do_not_require_c_source_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pcstubgen2 as stubgen_module
    from pcstubgen2.StubGenerationOptions import StubGenerationOptions

    def _unexpected_constructor(*args: object, **kwargs: object) -> object:
        raise AssertionError("CAstSignatureInferenceVisitor should not be instantiated by default")

    monkeypatch.setattr(stubgen_module, "CAstSignatureInferenceVisitor", _unexpected_constructor)

    options = StubGenerationOptions()
    assert options.enable_c_signature_inference is False
    stubgen_module.write_stubs("math", tmp_path, options=options)

    assert list(tmp_path.rglob("*.pyi"))


def test_write_stubs_adds_doc_parser_before_c_ast_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pcstubgen2 as stubgen_module
    from pcstubgen2.StubGenerationOptions import StubGenerationOptions

    captured_visitors: list[object] = []

    class _RecordingPipeline:
        def __init__(self, visitors):
            captured_visitors.extend(visitors)

        def run(self, module: IRModule) -> IRModule:
            return module

    monkeypatch.setattr(stubgen_module, "Pipeline", _RecordingPipeline)

    options = StubGenerationOptions(
        enable_docstring_signature_parser=True,
        enable_c_signature_inference=True,
        c_source_root=tmp_path,
    )
    stubgen_module.write_stubs("math", tmp_path, options=options)

    assert [type(visitor).__name__ for visitor in captured_visitors] == [
        "DocStringSignatureParserVisitor",
        "CAstSignatureInferenceVisitor",
        "InferMethodModifierVisitor",
    ]


def test_stub_generation_options_defaults_to_empty_clang_parse_args() -> None:
    first = StubGenerationOptions()
    second = StubGenerationOptions()

    assert first.clang_parse_args == []
    assert second.clang_parse_args == []
    assert first.clang_parse_args is not second.clang_parse_args


def test_c_ast_visitor_rejects_none_clang_parse_args(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        CAstSignatureInferenceVisitor(
            error_collector=ErrorCollector(),
            c_source_root=tmp_path,
            clang_parse_args=None,  # type: ignore[arg-type]
        )


def test_c_signature_engine_rejects_none_clang_parse_args(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        CSignatureExtractor(
            source_root=tmp_path,
            clang_parse_args=None,  # type: ignore[arg-type]
        )


def test_write_stubs_uses_multiline_logging_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pcstubgen2 as stubgen_module
    from pcstubgen2.StubGenerationOptions import StubGenerationOptions

    basic_config_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _record_basic_config(*args: object, **kwargs: object) -> None:
        basic_config_calls.append((args, kwargs))

    monkeypatch.setattr(stubgen_module.logging, "basicConfig", _record_basic_config)

    options = StubGenerationOptions(
        enable_docstring_signature_parser=False,
        enable_c_signature_inference=False,
    )
    stubgen_module.write_stubs("math", tmp_path, options=options)

    assert basic_config_calls == [
        (
            (),
            {
                "level": logging.INFO,
                "format": "[{levelname}] - {name}\n{message}\n",
                "style": "{",
            },
        )
    ]


def test_write_stubs_logs_to_output_file_and_cleans_up_handler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pcstubgen2 as stubgen_module
    from pcstubgen2.StubGenerationOptions import StubGenerationOptions

    def _emit_warning(self: Pipeline, module: IRModule) -> None:
        _ = (self, module)
        logging.getLogger("pcstubgen2.tests").warning("file log works")

    monkeypatch.setattr(stubgen_module.Pipeline, "run", _emit_warning)

    options = StubGenerationOptions(
        enable_docstring_signature_parser=False,
        enable_c_signature_inference=False,
    )
    stubgen_module.write_stubs("math", tmp_path, options=options)

    log_file = tmp_path / "pcstubgen2.log"
    assert log_file.exists()
    assert log_file.read_text(encoding="utf-8") == (
        "[WARNING] - pcstubgen2.tests\n"
        "file log works\n"
        "\n"
    )

    package_logger = logging.getLogger("pcstubgen2")
    assert all(
        not isinstance(handler, logging.FileHandler)
        or Path(handler.baseFilename) != log_file
        for handler in package_logger.handlers
    )


def test_doc_parser_runs_before_c_ast_visitor_in_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
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
        {
            "foo": [
                ExtractedFunction(
                    py_name="foo",
                    c_name="c_foo",
                    signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")])],
                )
            ]
        },
    )
    pipeline = Pipeline(
        [
            DocStringSignatureParserVisitor(error_collector=ErrorCollector()),
            CAstSignatureInferenceVisitor(
                error_collector=ErrorCollector(),
                c_source_root=tmp_path,
            ),
        ]
    )
    pipeline.run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.args] == ["a", "b"]
    assert [str(arg.annotation) for arg in parsed.args] == ["int", "int"]
    assert extractor.called == 1


def test_doc_parser_prevents_no_candidate_warning_after_signature_rewrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
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
    extractor = _patch_c_signature_extractor(monkeypatch, {})

    with caplog.at_level(
        logging.WARNING,
        logger="pcstubgen2.NodeVisitors.CSignatureInference.CAstSignatureInferenceVisitor",
    ):
        Pipeline(
            [
                DocStringSignatureParserVisitor(error_collector=ErrorCollector()),
                CAstSignatureInferenceVisitor(
                    error_collector=ErrorCollector(),
                    c_source_root=tmp_path,
                ),
            ]
        ).run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.args] == ["x", "y", "w", "out", "p"]
    assert str(parsed.return_annotation) == "numpy.ndarray"
    assert "Failed to rewrite generic signature for cdist_minkowski" not in caplog.text
    assert extractor.called == 1


def test_infer_method_modifier_after_c_ast_visitor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    method = IRMethod(
        function=IRFunction(name="make", args=_generic_signature()),
        decorator="staticmethod",
    )
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
        classes=[IRClass(name="Builder", methods=[method])],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        {
            "make": [
                ExtractedFunction(
                    py_name="make",
                    c_name="c_make",
                    method_flags=["METH_CLASS"],
                    signatures=[
                        ExtractedSignature(
                            arguments=[
                                ExtractedArgument(name="cls", type_name="type"),
                                ExtractedArgument(name="n", type_name="int"),
                            ]
                        )
                    ],
                )
            ]
        },
    )

    Pipeline(
        [
            CAstSignatureInferenceVisitor(
                error_collector=ErrorCollector(),
                c_source_root=tmp_path,
            ),
            InferMethodModifierVisitor(),
        ]
    ).run(module)

    rewritten = module.classes[0].methods[0]
    assert [arg.name for arg in rewritten.function.args] == ["cls", "n"]
    assert [str(arg.annotation) for arg in rewritten.function.args] == ["type", "int"]
    assert rewritten.decorator == "classmethod"


def test_c_signature_extraction_engine_parses_minimal_c_file(tmp_path: Path) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "mini_ext.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct {",
                "    const char* ml_name;",
                "    void* ml_meth;",
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
                "static PyMethodDef Methods[] = {",
                "    {\"add\", add_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    extracted = engine.extract()

    assert "add" in extracted
    add_candidates = extracted["add"]
    assert add_candidates
    first = add_candidates[0]
    assert first.py_name == "add"
    assert first.signatures
    assert [arg.name for arg in first.signatures[0].arguments] == ["self", "a", "b"]
    assert [arg.type_name for arg in first.signatures[0].arguments] == ["object", "int", "int"]
    assert first.signatures[0].return_type_name is None


def test_write_stubs_uses_doc_parser_for_pybind11_and_preserves_c_ast_results(
    tmp_path: Path,
) -> None:
    pytest.importorskip("scipy.spatial._distance_pybind")

    spatial_src_root = Path(r"C:\Things\third_package_source\scipy_scipy\scipy\spatial\src")
    if not spatial_src_root.exists():
        pytest.skip("SciPy spatial source tree is not available")

    import pcstubgen2 as stubgen_module

    pybind_output_dir = tmp_path / "pybind_stubs"
    wrap_output_dir = tmp_path / "wrap_stubs"
    options = StubGenerationOptions(
        include_docstrings=False,
        enable_docstring_signature_parser=True,
        enable_c_signature_inference=True,
        c_source_root=spatial_src_root,
    )

    stubgen_module.write_stubs("scipy.spatial._distance_pybind", pybind_output_dir, options=options)
    stubgen_module.write_stubs("scipy.spatial._distance_wrap", wrap_output_dir, options=options)

    pybind_stub = (pybind_output_dir / "_distance_pybind.pyi").read_text(encoding="utf-8")
    wrap_stub = (wrap_output_dir / "_distance_wrap.pyi").read_text(encoding="utf-8")
    pybind_log_text = (pybind_output_dir / "pcstubgen2.log").read_text(encoding="utf-8")

    assert (
        "def cdist_minkowski(x: object, y: object, w: object = None, "
        "out: object = None, p: typing.SupportsFloat = 2.0) -> numpy.ndarray:"
    ) in pybind_stub
    assert (
        "def pdist_minkowski(x: object, w: object = None, out: object = None, "
        "p: typing.SupportsFloat = 2.0) -> numpy.ndarray:"
    ) in pybind_stub
    assert (
        "def cdist_minkowski_double_wrap(XA_: object, XB_: object, dm_: object, p: object) -> float:"
    ) in wrap_stub
    assert "Failed to rewrite generic signature for cdist_minkowski" not in pybind_log_text
    assert "Failed to rewrite generic signature for pdist_minkowski" not in pybind_log_text


def test_c_signature_extraction_engine_parses_initializer_list_method_table(tmp_path: Path) -> None:
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
                "typedef struct {",
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
    extracted = engine.extract()

    assert "add" in extracted
    first = extracted["add"][0]
    assert first.py_name == "add"
    assert first.signatures
    assert [arg.name for arg in first.signatures[0].arguments] == ["self", "a", "b"]
    assert [arg.type_name for arg in first.signatures[0].arguments] == ["object", "int", "int"]


def test_c_signature_engine_infers_return_type_from_py_buildvalue(tmp_path: Path) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "mini_buildvalue_ext.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "PyObject* Py_BuildValue(const char* fmt, ...);",
                "static PyObject* make_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return Py_BuildValue(\"i\", value);",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"make\", make_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    extracted = engine.extract()

    assert "make" in extracted
    first = extracted["make"][0]
    assert first.signatures
    assert first.signatures[0].return_type_name == "int"


def test_c_signature_engine_falls_back_to_object_on_conflicting_return_types(tmp_path: Path) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "mini_conflict_return_ext.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "PyObject* PyLong_FromLong(long v);",
                "PyObject* PyBool_FromLong(long v);",
                "static PyObject* pick_impl(PyObject* self, PyObject* args) {",
                "    int flag = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &flag)) {",
                "        return (PyObject*)0;",
                "    }",
                "    if (flag) {",
                "        return PyLong_FromLong(1);",
                "    }",
                "    return PyBool_FromLong(0);",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"pick\", pick_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_c_std="c11",
    )
    extracted = engine.extract()

    assert "pick" in extracted
    first = extracted["pick"][0]
    assert first.signatures
    assert first.signatures[0].return_type_name == "object"


def test_c_ast_visitor_passes_clang_options_to_extractor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {
        "init_calls": 0,
        "extract_calls": 0,
    }

    class _RecorderExtractor:
        def __init__(
            self,
            source_root: Path,
            *,
            clang_parse_args: Iterable[str] = (),
            clang_c_std: str | None = None,
            clang_cpp_std: str | None = None,
        ) -> None:
            captured["init_calls"] = int(captured["init_calls"]) + 1
            captured["source_root"] = source_root
            captured["clang_parse_args"] = list(clang_parse_args)
            captured["clang_c_std"] = clang_c_std
            captured["clang_cpp_std"] = clang_cpp_std
            self._cached_result: dict[str, list[ExtractedFunction]] | None = None

        def extract(self) -> dict[str, list[ExtractedFunction]]:
            if self._cached_result is None:
                captured["extract_calls"] = int(captured["extract_calls"]) + 1
                self._cached_result = {}
            return self._cached_result

    import pcstubgen2.NodeVisitors.CSignatureInference.CAstSignatureInferenceVisitor as visitor_module

    monkeypatch.setattr(visitor_module, "CSignatureExtractor", _RecorderExtractor)

    visitor = CAstSignatureInferenceVisitor(
        error_collector=ErrorCollector(),
        c_source_root=tmp_path,
        clang_parse_args=["-DMY_FLAG=1"],
        clang_c_std="c99",
        clang_cpp_std="c++20",
    )

    assert captured["init_calls"] == 1
    assert captured["extract_calls"] == 0
    assert captured["source_root"] == tmp_path
    assert captured["clang_parse_args"] == ["-DMY_FLAG=1"]
    assert captured["clang_c_std"] == "c99"
    assert captured["clang_cpp_std"] == "c++20"

    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
    )
    visitor.visit_module(module)
    visitor.visit_module(module)

    assert captured["extract_calls"] == 1


def test_c_signature_engine_builds_language_specific_std_args(tmp_path: Path) -> None:
    class _FakeConfig:
        loaded = False
        configured_path: str | None = None

        @classmethod
        def set_library_file(cls, path: str) -> None:
            cls.configured_path = path

    class _FakeClang:
        Config = _FakeConfig

    engine = CSignatureExtractor(source_root=tmp_path)
    engine._clang = _FakeClang

    assert engine._ensure_clang_ready() is True
    assert engine._clang_parse_args is not None
    assert "-std=c11" not in engine._clang_parse_args
    assert engine._build_parse_args(tmp_path / "module.c")[0] == "-std=c11"
    assert engine._build_parse_args(tmp_path / "module.cxx")[0] == "-std=c++17"


def test_c_signature_engine_uses_configured_language_specific_std_args(tmp_path: Path) -> None:
    engine = CSignatureExtractor(
        source_root=tmp_path,
        clang_parse_args=["-DMY_FLAG=1"],
        clang_c_std="c99",
        clang_cpp_std="c++20",
    )

    assert engine._build_parse_args(tmp_path / "module.c") == ["-std=c99", "-DMY_FLAG=1"]
    assert engine._build_parse_args(tmp_path / "module.hpp") == ["-std=c++20", "-DMY_FLAG=1"]


def test_c_signature_engine_configures_packaged_libclang_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = CSignatureExtractor(source_root=tmp_path)
    configured_path: list[str] = []
    monkeypatch.setattr(clang.cindex.Config, "loaded", False, raising=False)
    monkeypatch.setattr(clang.cindex.Config, "set_library_file", lambda path: configured_path.append(path))
    monkeypatch.setattr(c_signature_extractor_module, "_get_packaged_libclang_path", lambda: "C:/fake/libclang.dll")

    assert engine._ensure_clang_ready() is True
    assert configured_path == ["C:/fake/libclang.dll"]


def test_c_signature_engine_skips_libclang_configuration_when_not_discovered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = CSignatureExtractor(source_root=tmp_path)
    configured_path: list[str] = []
    monkeypatch.setattr(clang.cindex.Config, "loaded", False, raising=False)
    monkeypatch.setattr(clang.cindex.Config, "set_library_file", lambda path: configured_path.append(path))
    monkeypatch.setattr(c_signature_extractor_module, "_get_packaged_libclang_path", lambda: None)

    assert engine._ensure_clang_ready() is True
    assert configured_path == []


def test_c_signature_engine_skips_non_parser_calls_in_token_params(tmp_path: Path) -> None:
    engine = CSignatureExtractor(source_root=tmp_path)

    assert (
        engine._set_token_params(
            func_cursor=object(),
            meth_flags=["METH_VARARGS"],
            token_list=["Py_BuildValue", '"i"', "value"],
        )
        is None
    )
    assert (
        engine._set_token_params(
            func_cursor=object(),
            meth_flags=["METH_VARARGS", "METH_KEYWORDS"],
            token_list=["PyArg_NoKeywords", "kwargs"],
        )
        is None
    )


def test_c_signature_engine_prefers_same_file_function_definition(tmp_path: Path) -> None:
    engine = CSignatureExtractor(source_root=tmp_path)
    preferred_file = str(tmp_path / "module_a.c")
    other_file = str(tmp_path / "module_b.c")

    class _FakeLocation:
        def __init__(self, file: str) -> None:
            self.file = file

    class _FakeFunctionCursor:
        def __init__(self, *, file: str, is_definition: bool) -> None:
            self.location = _FakeLocation(file=file)
            self._is_definition = is_definition

        def is_definition(self) -> bool:
            return self._is_definition

    from_other_file = _FakeFunctionCursor(file=other_file, is_definition=True)
    in_same_file_decl = _FakeFunctionCursor(file=preferred_file, is_definition=False)
    in_same_file_def = _FakeFunctionCursor(file=preferred_file, is_definition=True)

    selected = engine._select_function_cursor(
        [from_other_file, in_same_file_decl, in_same_file_def],
        preferred_file=preferred_file,
    )

    assert selected is in_same_file_def


def test_c_signature_engine_skips_non_array_types_before_reading_array_element() -> None:
    class _FakeType:
        kind = object()

        def get_array_element_type(self) -> object:
            raise AssertionError("non-array type should not read array element type")

    class _FakeNode:
        type = _FakeType()

    assert _is_PyMethodDef_array(_FakeNode()) is False


def test_c_signature_engine_detects_initializer_list_from_type_before_scanning_children() -> None:
    class _FakeType:
        spelling = "std::initializer_list<PyMethodDef>"

        def get_canonical(self) -> "_FakeType":
            return self

    class _FakeNode:
        type = _FakeType()

        def get_children(self) -> list[object]:
            raise AssertionError("initializer_list type match should not need AST child scan")

    assert _is_initializer_list_PyMethodDef(_FakeNode()) is True


def test_c_ast_visitor_drops_leading_self_for_static_method(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    method = IRMethod(function=IRFunction(name="build", args=_generic_signature()), decorator=None)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
        classes=[IRClass(name="Builder", methods=[method])],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        {
            "build": [
                ExtractedFunction(
                    py_name="build",
                    c_name="c_build",
                    method_flags=["METH_STATIC"],
                    signatures=[
                        ExtractedSignature(
                            arguments=[
                                ExtractedArgument(name="self", type_name="object"),
                                ExtractedArgument(name="count", type_name="int"),
                            ]
                        )
                    ],
                )
            ]
        },
    )

    Pipeline(
        [
            CAstSignatureInferenceVisitor(
                error_collector=ErrorCollector(),
                c_source_root=tmp_path,
            ),
            InferMethodModifierVisitor(),
        ]
    ).run(module)

    rewritten = module.classes[0].methods[0]
    assert [arg.name for arg in rewritten.function.args] == ["count"]
    assert str(rewritten.function.args[0].annotation) == "int"
    assert rewritten.decorator == "staticmethod"


@pytest.mark.parametrize(
    ("module_type", "expected_args", "expected_calls"),
    [
        (IRModuleType.C, ["count"], 1),
        (IRModuleType.PYTHON, ["args", "kwargs"], 0),
    ],
)
def test_c_ast_visitor_visit_class_uses_explicit_module_context_for_nested_classes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module_type: IRModuleType,
    expected_args: list[str],
    expected_calls: int,
) -> None:
    nested_method = IRMethod(
        function=IRFunction(name="build", args=_generic_signature()),
        decorator=None,
    )
    nested_class = IRClass(name="Inner", methods=[nested_method])
    outer_class = IRClass(name="Outer", classes=[nested_class])
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=module_type,
        classes=[outer_class],
    )
    extractor = _patch_c_signature_extractor(
        monkeypatch,
        {
            "build": [
                ExtractedFunction(
                    py_name="build",
                    c_name="c_build",
                    method_flags=["METH_STATIC"],
                    signatures=[
                        ExtractedSignature(
                            arguments=[
                                ExtractedArgument(name="self", type_name="object"),
                                ExtractedArgument(name="count", type_name="int"),
                            ]
                        )
                    ],
                )
            ]
        },
    )

    visitor = CAstSignatureInferenceVisitor(
        error_collector=ErrorCollector(),
        c_source_root=tmp_path,
    )
    visitor.visit_class(outer_class, module)

    rewritten_method = nested_class.methods[0]
    assert [arg.name for arg in rewritten_method.function.args] == expected_args
    assert extractor.called == expected_calls


def test_c_signature_engine_decodes_combined_numeric_method_flags(tmp_path: Path) -> None:
    engine = CSignatureExtractor(source_root=tmp_path)

    assert engine._decode_meth_literal_flags("3") == ["METH_VARARGS", "METH_KEYWORDS"]
    assert engine._decode_meth_literal_flags("0x21U") == ["METH_VARARGS", "METH_STATIC"]


class _FakeToken:
    def __init__(self, kind: object, spelling: str) -> None:
        self.kind = kind
        self.spelling = spelling


class _FakeNode:
    def __init__(
        self,
        *,
        kind: object,
        tokens: list[_FakeToken] | None = None,
        children: list[object] | None = None,
    ) -> None:
        self.kind = kind
        self._tokens = tokens or []
        self._children = children or []

    def get_tokens(self) -> list[_FakeToken]:
        return self._tokens

    def get_children(self) -> list[object]:
        return self._children


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
        tokens=[_FakeToken(clang.cindex.TokenKind.IDENTIFIER, name)],
    )


def _wrap(kind: object, child: _FakeNode) -> _FakeNode:
    return _FakeNode(kind=kind, children=[child])


def _init_list(*children: _FakeNode) -> _FakeNode:
    return _FakeNode(kind=clang.cindex.CursorKind.INIT_LIST_EXPR, children=list(children))


@pytest.mark.parametrize(
    "sentinel",
    [
        lambda c_null: _init_list(c_null, c_null, _int_literal("0"), c_null),
        lambda c_null: _init_list(_null_ptr_literal(), _null_ptr_literal(), _int_literal("0"), _null_ptr_literal()),
        lambda c_null: _init_list(_gnu_null_literal(), _gnu_null_literal(), _int_literal("0"), _gnu_null_literal()),
        lambda c_null: _init_list(),
        lambda c_null: _init_list(_int_literal("0")),
        lambda c_null: _init_list(_null_ptr_literal()),
        lambda c_null: _init_list(_gnu_null_literal()),
        lambda c_null: _init_list(_identifier_node("NULL")),
        lambda c_null: _init_list(_int_literal("0"), _int_literal("0"), _int_literal("1"), _int_literal("0")),
        lambda c_null: _init_list(_int_literal("0"), _int_literal("0"), _int_literal("0"), _int_literal("0")),
    ],
)
def test_c_signature_engine_array_end_accepts_supported_sentinel_forms(sentinel) -> None:
    c_null = _wrap(
        clang.cindex.CursorKind.UNEXPOSED_EXPR,
        _wrap(clang.cindex.CursorKind.PAREN_EXPR, _wrap(clang.cindex.CursorKind.CSTYLE_CAST_EXPR, _int_literal("0"))),
    )
    assert _is_PyMethodDef_sentinel(sentinel(c_null)) is True


@pytest.mark.parametrize(
    "non_sentinel",
    [
        _identifier_node("NULL"),
        _init_list(
            _FakeNode(
                kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
                children=[
                    _FakeNode(kind=clang.cindex.CursorKind.MEMBER_REF),
                    _int_literal("0"),
                ],
            )
        ),
        _init_list(_identifier_node("nullptr")),
        _init_list(_identifier_node("__null")),
        _init_list(_int_literal("1"), _int_literal("0"), _int_literal("0"), _int_literal("0")),
        _init_list(
            _FakeNode(kind=clang.cindex.CursorKind.STRING_LITERAL, tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, '"add"')]),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR),
            _int_literal("1"),
            _null_ptr_literal(),
        ),
    ],
)
def test_c_signature_engine_array_end_rejects_non_sentinel_forms(non_sentinel: _FakeNode) -> None:
    assert _is_PyMethodDef_sentinel(non_sentinel) is False


def test_c_signature_engine_accepts_single_NULL_token_via_fallback() -> None:
    assert _is_PyMethodDef_sentinel(_init_list(_identifier_node("NULL"))) is True


@pytest.mark.parametrize("name", ["nullptr", "__null"])
def test_c_signature_engine_fallback_rejects_non_NULL_identifiers(name: str) -> None:
    assert _is_PyMethodDef_sentinel(_init_list(_identifier_node(name))) is False


def test_c_signature_engine_iter_array_elements_breaks_only_on_supported_sentinel(tmp_path: Path) -> None:
    engine = CSignatureExtractor(source_root=tmp_path)

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

    should_break_array = _init_list(method_1, supported_sentinel, method_2)
    assert list(engine._iter_array_elements(should_break_array)) == [method_1]

    should_not_break_array = _init_list(method_1, non_sentinel, method_2)
    assert list(engine._iter_array_elements(should_not_break_array)) == [method_1, non_sentinel, method_2]


def test_c_signature_engine_parses_keywords_with_non_kwlist_name(tmp_path: Path) -> None:
    engine = CSignatureExtractor(source_root=tmp_path)

    args = engine._set_token_params(
        func_cursor=object(),
        meth_flags=["METH_VARARGS", "METH_KEYWORDS"],
        token_list=[
            "PyArg_ParseTupleAndKeywords",
            "args",
            "kwargs",
            '"iO!"',
            "keywords",
            "count",
            "expected_type",
            "value",
        ],
    )

    assert args is not None
    assert [arg.name for arg in args] == ["count", "expected_type", "value"]
    assert [arg.type_name for arg in args] == ["int", "object", "object"]
