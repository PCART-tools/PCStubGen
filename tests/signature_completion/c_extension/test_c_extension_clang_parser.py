from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from pcstubgen.signature_completion.c_extension.clang import compilation_database as compilation_database_module
from pcstubgen.signature_completion.c_extension.clang import translation_unit as translation_unit_module
from tests._c_extension_test_support import (
    Path,
    _FakeDiagnostic,
    _FakeDiagnosticType,
    _FakeTranslationUnit,
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

    def getCompileCommands(self, filename: str) -> list[_FakeCompileCommand]:
        return [
            command
            for command in self._commands
            if compilation_database_module.resolve_compile_command_file_path(command)
            == Path(filename).resolve()
        ]


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
        compilation_database_module.validate_compilation_database_path(wrong_file)


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

    parse_args = compilation_database_module.sanitize_compile_command_arguments(command)

    assert parse_args == [
        "-Iinclude",
        "-DMODE=1",
    ]


def test_resolve_compilation_command_keeps_first_command_per_source_file(
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
        compilation_database_module,
        "load_compilation_database",
        lambda compilation_database: _FakeCompilationDatabase(commands),
    )

    result = compilation_database_module.resolve_compilation_command(
        compilation_database_module.load_compilation_database(
            tmp_path / "compile_commands.json"
        ),
        shared_source,
    )

    assert result.file_path == shared_source.resolve()
    assert result.parse_args == ["-DFIRST"]


def test_resolve_compilation_command_raises_when_source_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "src" / "module.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")

    monkeypatch.setattr(
        compilation_database_module,
        "load_compilation_database",
        lambda compilation_database: _FakeCompilationDatabase([]),
    )

    with pytest.raises(RuntimeError, match="未在编译数据库中定位到编译单元"):
        compilation_database_module.resolve_compilation_command(
            compilation_database_module.load_compilation_database(
                tmp_path / "compile_commands.json"
            ),
            source,
        )

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
    command = compilation_database_module.CompilationCommand(
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
    command = compilation_database_module.CompilationCommand(
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
    command = compilation_database_module.CompilationCommand(
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
