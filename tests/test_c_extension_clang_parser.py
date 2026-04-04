from __future__ import annotations

import os

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
) -> None:
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
