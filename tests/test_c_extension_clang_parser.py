from __future__ import annotations

from io import StringIO
import os
from types import SimpleNamespace

from loguru import logger

from tests._c_extension_test_support import *


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
