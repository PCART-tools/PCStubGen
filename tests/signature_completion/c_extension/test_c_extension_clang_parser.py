from __future__ import annotations

from io import StringIO
import os
from types import SimpleNamespace

from loguru import logger

from tests._c_extension_test_support import (
    AnyType,
    CArgument,
    CExtensionSource,
    CFunction,
    CModule,
    CPP_SOURCE_SUFFIXES,
    CSignature,
    CSignatureExtractor,
    CSignatureResolver,
    DefinitionIndex,
    ExtractedArgument,
    ExtractedFunction,
    ExtractedModule,
    ExtractedSignature,
    IRArgument,
    IRArgumentKind,
    IRClass,
    IRFunction,
    IRMethod,
    IRModule,
    IRModuleType,
    IRSignature,
    Iterable,
    LinkageKind,
    ListType,
    METH_KEYWORDS,
    METH_NOARGS,
    METH_O,
    METH_VARARGS,
    NATIVE_SOURCE_SUFFIXES,
    Path,
    QualifiedName,
    RawType,
    SignatureCompleter,
    SignatureCompletionResult,
    SimpleNamespace,
    StubGenerationOptions,
    TupleType,
    Type,
    UnionType,
    _FakeClangWithDiagnostics,
    _FakeCursorFile,
    _FakeCursorLocation,
    _FakeDiagnostic,
    _FakeDiagnosticFile,
    _FakeDiagnosticLocation,
    _FakeDiagnosticType,
    _FakeExtractor,
    _FakeIndex,
    _FakeNode,
    _FakeSourceRange,
    _FakeToken,
    _FakeTranslationUnit,
    _SequentialIndex,
    _address_of,
    _arg,
    _build_definition_translation_unit,
    _call_expr,
    _collect_PyMethodDef_INIT_LIST_EXPR,
    _collect_PyMethodDef_INIT_LIST_EXPR_impl,
    _collect_method_table,
    _collect_method_table_impl,
    _conditional_expr,
    _definition_index,
    _designated_initializer,
    _extent_for_source_snippet,
    _extract_PyMethodDef_INIT_LIST_EXPR,
    _extract_method_table,
    _fake_function_cursor,
    _fake_function_cursor_with_children,
    _float_literal,
    _get_packaged_libclang_path,
    _gnu_null_literal,
    _has_include_directory_arg,
    _has_std_arg,
    _identifier_node,
    _init_list,
    _int_literal,
    _kwlist_decl,
    _macro_expr,
    _ml_flags_identifier_field,
    _ml_meth_cast_field,
    _ml_meth_field,
    _ml_name_field,
    _module_fixture,
    _null_ptr_literal,
    _patch_c_signature_extractor,
    _patch_fake_eval_int,
    _patch_raising_c_signature_extractor,
    _resolve_INIT_LIST_EXPR,
    _return_stmt,
    _signature,
    _signed_numeric_literal,
    _string_literal,
    _token_identifier_node,
    _type_object_decl,
    _unknown_function,
    _var_decl,
    _wrap,
    _write_compilation_database,
    annotations,
    c_extension_collect_module,
    c_extension_source_module,
    c_signature_extraction_module,
    cast,
    clang,
    collect_modules,
    cursor_utils_module,
    extract_c_signature_modules,
    json,
    module_collection_module,
    module_table_module,
    pytest,
    resolve_docstring_signatures,
    signature_rules_module,
    translation_unit_module,
)


class _FakeCompileCommand:
    def __init__(
        self,
        *,
        directory: Path,
        filename: str,
        arguments: list[str],
    ) -> None:
        self.directory = str(directory)
        self.filename = filename
        self.arguments = list(arguments)


class _FakeCompilationDatabase:
    def __init__(self, commands: list[_FakeCompileCommand]) -> None:
        self._commands = commands

    def getAllCompileCommands(self) -> list[_FakeCompileCommand]:
        return list(self._commands)


class _RecordingIndex:
    def __init__(self, translation_unit: _FakeTranslationUnit) -> None:
        self.translation_unit = translation_unit
        self.calls: list[tuple[str, list[str], str]] = []

    def parse(self, filename: str, args: list[str]) -> _FakeTranslationUnit:
        self.calls.append((filename, list(args), os.getcwd()))
        return self.translation_unit


def test_validate_compilation_database_path_requires_compile_commands_json(tmp_path: Path) -> None:
    wrong_file = tmp_path / "commands.json"
    wrong_file.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="文件名必须为 compile_commands.json"):
        translation_unit_module.validate_compilation_database_path(wrong_file)


def test_sanitize_compile_command_arguments_removes_driver_output_and_source_operands(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "module.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    command = _FakeCompileCommand(
        directory=tmp_path,
        filename="src/module.c",
        arguments=[
            "cc",
            "-Iinclude",
            "-DMODE=1",
            "-c",
            "-o",
            "build/module.o",
            "-MF",
            "build/module.d",
            "-MD",
            "src/module.c",
            str(source),
        ],
    )

    parse_args = translation_unit_module.sanitize_compile_command_arguments(command)

    assert parse_args == [
        "-Iinclude",
        "-DMODE=1",
    ]


def test_list_compilation_commands_keeps_first_command_per_source_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_source = tmp_path / "src" / "module.c"
    shared_source.parent.mkdir(parents=True, exist_ok=True)
    shared_source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    header = tmp_path / "src" / "module.h"
    header.write_text("int demo(void);\n", encoding="utf-8")

    commands = [
        _FakeCompileCommand(
            directory=tmp_path,
            filename="src/module.c",
            arguments=["cc", "-DFIRST", "-c", "src/module.c"],
        ),
        _FakeCompileCommand(
            directory=tmp_path,
            filename="src/module.c",
            arguments=["cc", "-DSECOND", "-c", "src/module.c"],
        ),
        _FakeCompileCommand(
            directory=tmp_path,
            filename="src/module.h",
            arguments=["cc", "-c", "src/module.h"],
        ),
    ]

    monkeypatch.setattr(
        translation_unit_module,
        "load_compilation_database",
        lambda compilation_database: _FakeCompilationDatabase(commands),
    )

    result = translation_unit_module.list_compilation_commands(
        tmp_path / "compile_commands.json"
    )

    assert len(result) == 1
    assert result[0].file_path == shared_source.resolve()
    assert result[0].parse_args == ["-DFIRST"]


def test_list_compilation_commands_skips_subproject_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept_source = tmp_path / "src" / "module.c"
    kept_source.parent.mkdir(parents=True, exist_ok=True)
    kept_source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    skipped_source = tmp_path / "subprojects" / "pkg" / "module.c"
    skipped_source.parent.mkdir(parents=True, exist_ok=True)
    skipped_source.write_text("int skipped(void) { return 0; }\n", encoding="utf-8")

    commands = [
        _FakeCompileCommand(
            directory=tmp_path,
            filename="subprojects/pkg/module.c",
            arguments=["cc", "-DSUBPROJECT", "-c", "subprojects/pkg/module.c"],
        ),
        _FakeCompileCommand(
            directory=tmp_path,
            filename="src/module.c",
            arguments=["cc", "-DKEPT", "-c", "src/module.c"],
        ),
    ]

    monkeypatch.setattr(
        translation_unit_module,
        "load_compilation_database",
        lambda compilation_database: _FakeCompilationDatabase(commands),
    )

    result = translation_unit_module.list_compilation_commands(
        tmp_path / "compile_commands.json"
    )

    assert len(result) == 1
    assert result[0].file_path == kept_source.resolve()
    assert result[0].parse_args == ["-DKEPT"]


def test_list_compilation_commands_skips_absolute_subproject_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    absolute_subproject_source = tmp_path / "vendor" / "subprojects" / "pkg" / "module.cpp"
    absolute_subproject_source.parent.mkdir(parents=True, exist_ok=True)
    absolute_subproject_source.write_text("int skipped(void) { return 0; }\n", encoding="utf-8")
    kept_source = tmp_path / "src" / "module.cpp"
    kept_source.parent.mkdir(parents=True, exist_ok=True)
    kept_source.write_text("int kept(void) { return 0; }\n", encoding="utf-8")

    commands = [
        _FakeCompileCommand(
            directory=tmp_path,
            filename=str(absolute_subproject_source),
            arguments=["c++", "-DSUBPROJECT", "-c", str(absolute_subproject_source)],
        ),
        _FakeCompileCommand(
            directory=tmp_path,
            filename="src/module.cpp",
            arguments=["c++", "-DKEPT", "-c", "src/module.cpp"],
        ),
    ]

    monkeypatch.setattr(
        translation_unit_module,
        "load_compilation_database",
        lambda compilation_database: _FakeCompilationDatabase(commands),
    )

    result = translation_unit_module.list_compilation_commands(
        tmp_path / "compile_commands.json"
    )

    assert len(result) == 1
    assert result[0].file_path == kept_source.resolve()
    assert result[0].parse_args == ["-DKEPT"]


def test_list_compilation_commands_skips_third_party_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept_source = tmp_path / "src" / "module.c"
    kept_source.parent.mkdir(parents=True, exist_ok=True)
    kept_source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    skipped_source = tmp_path / "third_party" / "pkg" / "module.c"
    skipped_source.parent.mkdir(parents=True, exist_ok=True)
    skipped_source.write_text("int skipped(void) { return 0; }\n", encoding="utf-8")

    commands = [
        _FakeCompileCommand(
            directory=tmp_path,
            filename="third_party/pkg/module.c",
            arguments=["cc", "-DTHIRD_PARTY", "-c", "third_party/pkg/module.c"],
        ),
        _FakeCompileCommand(
            directory=tmp_path,
            filename="src/module.c",
            arguments=["cc", "-DKEPT", "-c", "src/module.c"],
        ),
    ]

    monkeypatch.setattr(
        translation_unit_module,
        "load_compilation_database",
        lambda compilation_database: _FakeCompilationDatabase(commands),
    )

    result = translation_unit_module.list_compilation_commands(
        tmp_path / "compile_commands.json"
    )

    assert len(result) == 1
    assert result[0].file_path == kept_source.resolve()
    assert result[0].parse_args == ["-DKEPT"]


def test_list_compilation_commands_skips_absolute_third_party_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    absolute_third_party_source = tmp_path / "vendor" / "third_party" / "pkg" / "module.cpp"
    absolute_third_party_source.parent.mkdir(parents=True, exist_ok=True)
    absolute_third_party_source.write_text("int skipped(void) { return 0; }\n", encoding="utf-8")
    kept_source = tmp_path / "src" / "module.cpp"
    kept_source.parent.mkdir(parents=True, exist_ok=True)
    kept_source.write_text("int kept(void) { return 0; }\n", encoding="utf-8")

    commands = [
        _FakeCompileCommand(
            directory=tmp_path,
            filename=str(absolute_third_party_source),
            arguments=["c++", "-DTHIRD_PARTY", "-c", str(absolute_third_party_source)],
        ),
        _FakeCompileCommand(
            directory=tmp_path,
            filename="src/module.cpp",
            arguments=["c++", "-DKEPT", "-c", "src/module.cpp"],
        ),
    ]

    monkeypatch.setattr(
        translation_unit_module,
        "load_compilation_database",
        lambda compilation_database: _FakeCompilationDatabase(commands),
    )

    result = translation_unit_module.list_compilation_commands(
        tmp_path / "compile_commands.json"
    )

    assert len(result) == 1
    assert result[0].file_path == kept_source.resolve()
    assert result[0].parse_args == ["-DKEPT"]


def test_parse_uses_compile_command_working_directory_and_preserves_translation_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation_unit_module.detect_clang_resource_dir.cache_clear()
    monkeypatch.setattr(
        translation_unit_module,
        "detect_clang_resource_dir",
        lambda: None,
    )
    working_directory = tmp_path / "build"
    working_directory.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "src" / "module.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
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
        ]
    )
    index = _RecordingIndex(translation_unit)
    command = translation_unit_module.CompilationCommand(
        file_path=source.resolve(),
        working_directory=working_directory.resolve(),
        parse_args=["-I../include", "-DMODE=1"],
    )

    result = translation_unit_module.parse(index, command)

    assert result is translation_unit
    assert index.calls == [
        (
            str(source.resolve()),
            ["-I../include", "-DMODE=1"],
            str(working_directory.resolve()),
        )
    ]


def test_parse_appends_detected_clang_resource_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation_unit_module.detect_clang_resource_dir.cache_clear()
    monkeypatch.setattr(
        translation_unit_module,
        "detect_clang_resource_dir",
        lambda: "/opt/clang/resource",
    )
    working_directory = tmp_path / "build"
    working_directory.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "src" / "module.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    translation_unit = _FakeTranslationUnit(diagnostics=[])
    index = _RecordingIndex(translation_unit)
    command = translation_unit_module.CompilationCommand(
        file_path=source.resolve(),
        working_directory=working_directory.resolve(),
        parse_args=["-I../include", "-DMODE=1"],
    )

    translation_unit_module.parse(index, command)

    assert index.calls == [
        (
            str(source.resolve()),
            ["-I../include", "-DMODE=1", "-resource-dir", "/opt/clang/resource"],
            str(working_directory.resolve()),
        )
    ]


def test_build_effective_parse_args_appends_detected_clang_resource_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation_unit_module.detect_clang_resource_dir.cache_clear()
    monkeypatch.setattr(
        translation_unit_module,
        "detect_clang_resource_dir",
        lambda: "/opt/clang/resource",
    )
    command = translation_unit_module.CompilationCommand(
        file_path=tmp_path / "module.c",
        working_directory=tmp_path,
        parse_args=["-I../include", "-DMODE=1"],
    )

    assert translation_unit_module.build_effective_parse_args(command) == [
        "-I../include",
        "-DMODE=1",
        "-resource-dir",
        "/opt/clang/resource",
    ]


def test_parse_does_not_emit_parse_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation_unit_module.detect_clang_resource_dir.cache_clear()
    monkeypatch.setattr(
        translation_unit_module,
        "detect_clang_resource_dir",
        lambda: None,
    )
    working_directory = tmp_path / "build"
    working_directory.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "src" / "module.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    index = _RecordingIndex(_FakeTranslationUnit(diagnostics=[]))
    command = translation_unit_module.CompilationCommand(
        file_path=source.resolve(),
        working_directory=working_directory.resolve(),
        parse_args=["-I../include"],
    )

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        translation_unit_module.parse(index, command)
    finally:
        logger.remove(sink_id)

    assert log_output.getvalue() == ""


def test_detect_clang_resource_dir_returns_none_when_command_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation_unit_module.detect_clang_resource_dir.cache_clear()
    captured_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        captured_calls.append((args, kwargs))
        return SimpleNamespace(returncode=1, stdout="ignored", stderr="boom")

    monkeypatch.setattr(translation_unit_module.subprocess, "run", fake_run)

    assert translation_unit_module.detect_clang_resource_dir() is None
    assert captured_calls == [
        (
            (["clang", "-print-resource-dir"],),
            {
                "capture_output": True,
                "text": True,
                "check": False,
            },
        )
    ]


def test_detect_clang_resource_dir_returns_none_when_stdout_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation_unit_module.detect_clang_resource_dir.cache_clear()

    monkeypatch.setattr(
        translation_unit_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="   \n",
            stderr="",
        ),
    )

    assert translation_unit_module.detect_clang_resource_dir() is None


def test_detect_clang_resource_dir_uses_cached_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation_unit_module.detect_clang_resource_dir.cache_clear()
    run_call_count = 0

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal run_call_count
        run_call_count += 1
        return SimpleNamespace(returncode=0, stdout="/opt/clang/resource\n", stderr="")

    monkeypatch.setattr(translation_unit_module.subprocess, "run", fake_run)

    assert translation_unit_module.detect_clang_resource_dir() == "/opt/clang/resource"
    assert translation_unit_module.detect_clang_resource_dir() == "/opt/clang/resource"
    assert run_call_count == 1
