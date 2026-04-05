from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import clang.cindex
from loguru import logger

from tests._c_extension_test_support import (
    ExtractedFunction,
    ExtractedModule,
    ExtractedSignature,
    RawType,
    _FakeDiagnostic,
    _FakeDiagnosticType,
    _FakeNode,
    _FakeTranslationUnit,
    c_signature_extraction_module,
    pytest,
    translation_unit_module,
)


def test_collect_modules_logs_parse_progress_and_effective_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample_command = translation_unit_module.CompilationCommand(
        file_path=tmp_path / "sample.c",
        working_directory=tmp_path,
        parse_args=["-Iinclude"],
    )

    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "list_compilation_commands",
        lambda compilation_database: [sample_command],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.Index,
        "create",
        staticmethod(lambda: object()),
    )
    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "build_effective_parse_args",
        lambda compilation_command: ["-Iinclude", "-resource-dir", "/opt/clang/resource"],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "parse",
        lambda *args, **kwargs: SimpleNamespace(
            cursor=_FakeNode(kind=clang.cindex.CursorKind.TRANSLATION_UNIT),
            diagnostics=[],
        ),
    )
    monkeypatch.setattr(
        c_signature_extraction_module.module_table,
        "collect_modules_from_translation_unit",
        lambda cursor, definition_index: [],
    )

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        extracted = c_signature_extraction_module.extract_c_signature_modules(
            tmp_path / "compile_commands.json"
        )
    finally:
        logger.remove(sink_id)

    log_text = log_output.getvalue()
    assert extracted == {}
    assert "开始Parse" in log_text
    assert "Parse成功" in log_text
    assert "开始推断签名" in log_text
    assert str(sample_command.file_path) in log_text
    assert "-resource-dir" in log_text


def test_collect_modules_logs_diagnostics_by_severity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "sample.c"
    sample_command = translation_unit_module.CompilationCommand(
        file_path=source_path,
        working_directory=tmp_path,
        parse_args=["-Iinclude"],
    )
    translation_unit = _FakeTranslationUnit(
        diagnostics=[
            _FakeDiagnostic(
                severity=_FakeDiagnosticType.Warning,
                message="warning detail",
                file_name=str(source_path),
                line=3,
                column=1,
            ),
            _FakeDiagnostic(
                severity=_FakeDiagnosticType.Error,
                message="error detail",
                file_name=str(source_path),
                line=7,
                column=9,
            ),
        ]
    )

    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "list_compilation_commands",
        lambda compilation_database: [sample_command],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.Index,
        "create",
        staticmethod(lambda: object()),
    )
    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "build_effective_parse_args",
        lambda compilation_command: ["-Iinclude"],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "parse",
        lambda *args, **kwargs: translation_unit,
    )
    monkeypatch.setattr(
        c_signature_extraction_module.module_table,
        "collect_modules_from_translation_unit",
        lambda cursor, definition_index: [],
    )

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        c_signature_extraction_module.extract_c_signature_modules(
            tmp_path / "compile_commands.json"
        )
    finally:
        logger.remove(sink_id)

    log_text = log_output.getvalue()
    assert "Parse诊断" in log_text
    assert "[WARNING]" in log_text
    assert "warning detail" in log_text
    assert "[ERROR]" in log_text
    assert "error detail" in log_text


def test_collect_modules_continues_after_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_command = translation_unit_module.CompilationCommand(
        file_path=tmp_path / "first.c",
        working_directory=tmp_path,
        parse_args=["-DFIRST"],
    )
    second_command = translation_unit_module.CompilationCommand(
        file_path=tmp_path / "second.c",
        working_directory=tmp_path,
        parse_args=["-DSECOND"],
    )
    parse_calls: list[object] = []

    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "list_compilation_commands",
        lambda compilation_database: [first_command, second_command],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.Index,
        "create",
        staticmethod(lambda: object()),
    )
    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "build_effective_parse_args",
        lambda compilation_command: list(compilation_command.parse_args),
    )

    def _parse(index: object, compilation_command: object, **kwargs: object) -> _FakeTranslationUnit:
        _ = (index, kwargs)
        parse_calls.append(compilation_command)
        if compilation_command is first_command:
            raise clang.cindex.TranslationUnitLoadError("broken parse")
        return _FakeTranslationUnit(diagnostics=[])

    monkeypatch.setattr(c_signature_extraction_module.clang_parser, "parse", _parse)
    monkeypatch.setattr(
        c_signature_extraction_module.module_table,
        "collect_modules_from_translation_unit",
        lambda cursor, definition_index: [],
    )

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        extracted = c_signature_extraction_module.extract_c_signature_modules(
            tmp_path / "compile_commands.json"
        )
    finally:
        logger.remove(sink_id)

    log_text = log_output.getvalue()
    assert extracted == {}
    assert parse_calls == [first_command, second_command]
    assert "Parse失败" in log_text
    assert str(first_command.file_path) in log_text
    assert str(second_command.file_path) in log_text
    assert "开始构建索引" in log_text


def test_collect_modules_logs_and_skips_signature_inference_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad_cursor = object()
    good_cursor = object()
    bad_function = ExtractedFunction(ml_name="bad", function_cursor=bad_cursor)
    good_function = ExtractedFunction(ml_name="good", function_cursor=good_cursor)
    logged_messages: list[str] = []
    sample_command = translation_unit_module.CompilationCommand(
        file_path=tmp_path / "sample.c",
        working_directory=tmp_path,
        parse_args=[],
    )

    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "list_compilation_commands",
        lambda compilation_database: [sample_command],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "build_effective_parse_args",
        lambda compilation_command: [],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.Index,
        "create",
        staticmethod(lambda: object()),
    )
    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "parse",
        lambda *args, **kwargs: SimpleNamespace(
            cursor=_FakeNode(kind=clang.cindex.CursorKind.TRANSLATION_UNIT),
            diagnostics=[],
        ),
    )
    monkeypatch.setattr(
        c_signature_extraction_module.module_table,
        "collect_modules_from_translation_unit",
        lambda cursor, definition_index: [
            ExtractedModule(
                name="pkg.mod",
                functions={"bad": bad_function, "good": good_function},
            )
        ],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.signature_inference,
        "infer_signature",
        lambda c_function: (_ for _ in ()).throw(RuntimeError("broken inference"))
        if getattr(c_function, "function_cursor", None) is bad_cursor
        else [ExtractedSignature(return_type=RawType("int"))],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.logger,
        "exception",
        lambda message, *args: logged_messages.append(message.format(*args)),
    )

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        extracted = c_signature_extraction_module.extract_c_signature_modules(
            tmp_path / "compile_commands.json"
        )
    finally:
        logger.remove(sink_id)

    assert extracted["pkg.mod"].functions["bad"].signatures == []
    assert extracted["pkg.mod"].functions["good"].signatures == [
        ExtractedSignature(return_type=RawType("int"))
    ]
    assert "签名推断进度" in log_output.getvalue()
    assert len(logged_messages) == 1
    assert "pkg.mod" in logged_messages[0]
    assert "bad" in logged_messages[0]
