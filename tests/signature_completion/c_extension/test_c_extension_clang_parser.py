from __future__ import annotations

import os
from typing import Any, cast

import pytest
from clang.cindex import CompileCommand, Index

from pcstubgen.signature_completion.c_extension.clang import compilation_database as compilation_database_module
from pcstubgen.signature_completion.c_extension.clang import libclang_parse as libclang_parse_module
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
            if compilation_database_module.MyCompileCommand(
                cast(CompileCommand, cast(object, command))
            ).filename
            == Path(filename).resolve()
        ]


class _PointerHolder:
    def __init__(self) -> None:
        self.value: object | None = None

    def __bool__(self) -> bool:
        return self.value is not None


def test_validate_compilation_database_path_requires_compile_commands_json(tmp_path: Path) -> None:
    wrong_file = tmp_path / "commands.json"
    wrong_file.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="文件名必须为 compile_commands.json"):
        compilation_database_module.validate_compilation_database_path(wrong_file)


def test_resolve_compile_command_preserves_full_arguments(
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

    result = compilation_database_module.resolve_compile_command(
        compilation_database_module.load_compilation_database(
            tmp_path / "compile_commands.json"
        ),
        shared_source,
    )

    assert result.filename == shared_source.resolve()
    assert result.directory == tmp_path.resolve()
    assert result.arguments == [
        "cc",
        "-DFIRST",
        "-c",
        "src/module.c",
    ]


def test_resolve_compile_command_raises_when_source_is_missing(
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
        compilation_database_module.resolve_compile_command(
        compilation_database_module.load_compilation_database(
                tmp_path / "compile_commands.json"
            ),
            source,
        )


def test_parse_translation_unit_full_argv_preserves_all_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_translation_unit = object()
    index = Index.create()
    calls: list[tuple[Index, str | None, list[str], int]] = []

    def _fake_parse_translation_unit2_full_argv(
        received_index: Index,
        source_filename: str | None,
        arguments_array: Any,
        arguments_count: int,
        unsaved_files: object,
        unsaved_files_count: int,
        options: int,
        out_translation_unit: _PointerHolder,
    ) -> int:
        calls.append(
            (
                received_index,
                source_filename,
                [
                    arguments_array[arg_index].decode("utf-8")
                    for arg_index in range(arguments_count)
                ],
                options,
            )
        )
        assert unsaved_files is None
        assert unsaved_files_count == 0
        out_translation_unit.value = raw_translation_unit
        return 0

    monkeypatch.setattr(
        libclang_parse_module,
        "_get_parse_translation_unit2_full_argv",
        lambda: _fake_parse_translation_unit2_full_argv,
    )
    monkeypatch.setattr(
        libclang_parse_module.clang.cindex,
        "c_object_p",
        _PointerHolder,
    )
    monkeypatch.setattr(libclang_parse_module, "byref", lambda pointer: pointer)
    def _fake_translation_unit(pointer: _PointerHolder, **kwargs: Any) -> tuple[object, Index]:
        return pointer.value, kwargs["index"]

    monkeypatch.setattr(
        libclang_parse_module,
        "TranslationUnit",
        _fake_translation_unit,
    )

    result = libclang_parse_module.parse_translation_unit_full_argv(
        index,
        ["clang", "-Iinclude", "-c", "src/module.c", "-o", "build/module.o"],
    )

    assert result == (raw_translation_unit, index)
    assert calls == [
        (
            index,
            None,
            ["clang", "-Iinclude", "-c", "src/module.c", "-o", "build/module.o"],
            0,
        )
    ]


def test_parse_translation_unit_full_argv_raises_on_libclang_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = Index.create()

    def _fake_parse_translation_unit2_full_argv(
        received_index: Index,
        source_filename: str | None,
        arguments_array: Any,
        arguments_count: int,
        unsaved_files: object,
        unsaved_files_count: int,
        options: int,
        out_translation_unit: _PointerHolder,
    ) -> int:
        _ = (
            received_index,
            source_filename,
            arguments_array,
            arguments_count,
            unsaved_files,
            unsaved_files_count,
            options,
            out_translation_unit,
        )
        return 4

    monkeypatch.setattr(
        libclang_parse_module,
        "_get_parse_translation_unit2_full_argv",
        lambda: _fake_parse_translation_unit2_full_argv,
    )
    monkeypatch.setattr(
        libclang_parse_module.clang.cindex,
        "c_object_p",
        _PointerHolder,
    )
    monkeypatch.setattr(libclang_parse_module, "byref", lambda pointer: pointer)

    with pytest.raises(
        libclang_parse_module.TranslationUnitLoadError,
        match="libclang error code: 4",
    ):
        libclang_parse_module.parse_translation_unit_full_argv(
            index,
            ["clang", "src/module.c"],
        )


def test_try_resolve_clang_resource_dir_returns_valid_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_dir = tmp_path / "clang-resource"
    (resource_dir / "include").mkdir(parents=True)
    monkeypatch.setattr(
        translation_unit_module.subprocess,
        "check_output",
        lambda command, *, text: f"{resource_dir}\n",
    )

    assert translation_unit_module.try_get_clang_resource_dir() == resource_dir


def test_try_resolve_clang_resource_dir_returns_none_when_command_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        translation_unit_module.subprocess,
        "check_output",
        lambda command, *, text: (_ for _ in ()).throw(FileNotFoundError()),
    )

    assert translation_unit_module.try_get_clang_resource_dir() is None


def test_parse_uses_compile_command_working_directory_and_injects_resource_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    index = Index.create()
    calls: list[tuple[Index, list[str], str]] = []
    resource_dir = tmp_path / "clang-resource"
    resource_dir.mkdir()

    def _fake_parse_translation_unit_full_argv(
        received_index: Index,
        arguments: list[str],
    ) -> _FakeTranslationUnit:
        calls.append((received_index, list(arguments), os.getcwd()))
        return translation_unit

    monkeypatch.setattr(
        translation_unit_module,
        "parse_translation_unit_full_argv",
        _fake_parse_translation_unit_full_argv,
    )
    monkeypatch.setattr(
        translation_unit_module,
        "try_resolve_clang_resource_dir",
        lambda: resource_dir,
    )
    compile_command = compilation_database_module.MyCompileCommand(
        cast(
            CompileCommand,
            cast(
                object,
                _FakeCompileCommand(
                    directory=working_directory.resolve(),
                    filename=str(source.resolve()),
                    arguments=["clang", "-I../include", "-DMODE=1", "src/module.c"],
                ),
            ),
        )
    )

    result = translation_unit_module.parse(index, compile_command)

    assert result is translation_unit
    assert calls == [
        (
            index,
            [
                "clang",
                "-I../include",
                "-DMODE=1",
                "src/module.c",
                "-resource-dir",
                str(resource_dir),
            ],
            str(working_directory.resolve()),
        )
    ]


def test_parse_skips_resource_dir_when_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_directory = tmp_path / "build"
    working_directory.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "src" / "module.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    translation_unit = _FakeTranslationUnit(diagnostics=[])
    index = Index.create()
    calls: list[list[str]] = []

    def _fake_parse_translation_unit_full_argv(
        received_index: Index,
        arguments: list[str],
    ) -> _FakeTranslationUnit:
        _ = received_index
        calls.append(list(arguments))
        return translation_unit

    monkeypatch.setattr(
        translation_unit_module,
        "parse_translation_unit_full_argv",
        _fake_parse_translation_unit_full_argv,
    )
    monkeypatch.setattr(
        translation_unit_module,
        "try_resolve_clang_resource_dir",
        lambda: None,
    )

    compile_command = compilation_database_module.MyCompileCommand(
        cast(
            CompileCommand,
            cast(
                object,
                _FakeCompileCommand(
                    directory=working_directory.resolve(),
                    filename=str(source.resolve()),
                    arguments=["clang", "-I../include", "-DMODE=1", "src/module.c"],
                ),
            ),
        )
    )

    result = translation_unit_module.parse(index, compile_command)

    assert result is translation_unit
    assert calls == [["clang", "-I../include", "-DMODE=1", "src/module.c"]]
