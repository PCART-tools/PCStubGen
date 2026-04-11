from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import clang.cindex
import pytest
from clang.cindex import CompileCommand, Index, TranslationUnitLoadError

from pcstubgen.signature_completion.c_extension.clang import parser as parser_module
from tests._c_extension_test_support import _FakeDiagnostic, _FakeTranslationUnit


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
        self._arguments = list(arguments)

    @property
    def arguments(self) -> list[str]:
        return list(self._arguments)


class _FakeCompilationDatabase:
    def __init__(self, commands: list[_FakeCompileCommand]) -> None:
        self._commands = commands

    def getCompileCommands(self, filename: str) -> list[_FakeCompileCommand]:
        source_path = Path(filename).resolve()
        return [
            command
            for command in self._commands
            if _compile_command_filename(command) == source_path
        ]


def _compile_command_filename(command: _FakeCompileCommand) -> Path:
    file_path = Path(command.filename)
    if not file_path.is_absolute():
        file_path = Path(command.directory) / file_path
    return file_path.resolve()


def _make_compile_command(
    *,
    working_directory: Path,
    source: Path,
    arguments: list[str],
) -> CompileCommand:
    return cast(
        CompileCommand,
        cast(
            object,
            _FakeCompileCommand(
                directory=working_directory.resolve(),
                filename=str(source.resolve()),
                arguments=arguments,
            ),
        ),
    )


def _make_parser(
    *,
    database: _FakeCompilationDatabase | None = None,
    index: object | None = None,
) -> parser_module.ClangParser:
    parser = object.__new__(parser_module.ClangParser)
    parser._compilation_database = database if database is not None else _FakeCompilationDatabase([])
    parser._translation_units = {}
    parser._index = index if index is not None else object()
    return parser


def test_try_get_clang_resource_dir_returns_valid_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_dir = tmp_path / "clang-resource"
    (resource_dir / "include").mkdir(parents=True)
    parser_module.try_get_clang_resource_dir.cache_clear()
    monkeypatch.setattr(
        parser_module.subprocess,
        "check_output",
        lambda command, *, text: f"{resource_dir}\n",
    )

    assert parser_module.try_get_clang_resource_dir() == resource_dir
    parser_module.try_get_clang_resource_dir.cache_clear()


def test_try_get_clang_resource_dir_returns_none_when_command_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser_module.try_get_clang_resource_dir.cache_clear()
    monkeypatch.setattr(
        parser_module.subprocess,
        "check_output",
        lambda command, *, text: (_ for _ in ()).throw(FileNotFoundError()),
    )

    assert parser_module.try_get_clang_resource_dir() is None
    parser_module.try_get_clang_resource_dir.cache_clear()


def test_parse_translation_unit_appends_resource_dir_and_uses_compile_command_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_directory = tmp_path / "build"
    working_directory.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "src" / "module.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    translation_unit = object()
    index = Index.create()
    observed_arguments: list[list[str]] = []
    observed_cwds: list[Path] = []
    resource_dir = tmp_path / "clang-resource"
    resource_dir.mkdir()

    def _fake_parse_translation_unit_full_argv(
        received_index: Index,
        arguments: list[str],
    ) -> object:
        assert received_index is index
        observed_arguments.append(list(arguments))
        observed_cwds.append(Path.cwd())
        return translation_unit

    parser = _make_parser(index=index)
    monkeypatch.setattr(
        parser_module,
        "parse_translation_unit_full_argv",
        _fake_parse_translation_unit_full_argv,
    )
    monkeypatch.setattr(
        parser_module,
        "try_get_clang_resource_dir",
        lambda: resource_dir,
    )

    compile_command = _make_compile_command(
        working_directory=working_directory,
        source=source,
        arguments=["clang", "-I../include", "-DMODE=1", "src/module.c"],
    )

    result = parser._parse_translation_unit(compile_command)

    assert result is translation_unit
    assert observed_arguments == [
        ["clang", "-I../include", "-DMODE=1", "src/module.c", "-resource-dir", str(resource_dir)]
    ]
    assert observed_cwds == [working_directory.resolve()]


def test_parse_translation_unit_keeps_original_arguments_when_resource_dir_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_directory = tmp_path / "build"
    working_directory.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "src" / "module.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    translation_unit = object()
    index = Index.create()
    observed_arguments: list[list[str]] = []

    def _fake_parse_translation_unit_full_argv(
        received_index: Index,
        arguments: list[str],
    ) -> object:
        assert received_index is index
        observed_arguments.append(list(arguments))
        return translation_unit

    parser = _make_parser(index=index)
    monkeypatch.setattr(
        parser_module,
        "parse_translation_unit_full_argv",
        _fake_parse_translation_unit_full_argv,
    )
    monkeypatch.setattr(
        parser_module,
        "try_get_clang_resource_dir",
        lambda: None,
    )

    compile_command = _make_compile_command(
        working_directory=working_directory,
        source=source,
        arguments=["clang", "-I../include", "-DMODE=1", "src/module.c"],
    )

    result = parser._parse_translation_unit(compile_command)

    assert result is translation_unit
    assert observed_arguments == [["clang", "-I../include", "-DMODE=1", "src/module.c"]]


def test_get_translation_unit_caches_successful_result_by_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "src" / "module.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    compile_command = _FakeCompileCommand(
        directory=tmp_path,
        filename="src/module.c",
        arguments=["clang", "src/module.c"],
    )
    translation_unit = SimpleNamespace(diagnostics=[])
    parser = _make_parser(database=_FakeCompilationDatabase([compile_command]))
    calls: list[CompileCommand] = []

    monkeypatch.setattr(
        parser,
        "_parse_translation_unit",
        lambda command: calls.append(command) or translation_unit,
    )

    first = parser.get_translation_unit(source)
    second = parser.get_translation_unit(tmp_path / "src" / "." / "module.c")

    assert first is translation_unit
    assert second is translation_unit
    assert calls == [compile_command]


def test_get_translation_unit_does_not_cache_parse_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "src" / "module.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    compile_command = _FakeCompileCommand(
        directory=tmp_path,
        filename="src/module.c",
        arguments=["clang", "src/module.c"],
    )
    parser = _make_parser(database=_FakeCompilationDatabase([compile_command]))
    calls: list[CompileCommand] = []

    def _raise_parse_error(command: CompileCommand) -> object:
        calls.append(command)
        raise TranslationUnitLoadError("boom")

    monkeypatch.setattr(
        parser,
        "_parse_translation_unit",
        _raise_parse_error,
    )

    with pytest.raises(RuntimeError, match="Parse失败"):
        parser.get_translation_unit(source)

    with pytest.raises(RuntimeError, match="Parse失败"):
        parser.get_translation_unit(source)

    assert calls == [compile_command, compile_command]


def test_get_translation_unit_logs_error_diagnostics_and_returns_translation_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "src" / "module.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    compile_command = _FakeCompileCommand(
        directory=tmp_path,
        filename="src/module.c",
        arguments=["clang", "src/module.c"],
    )
    diagnostic = _FakeDiagnostic(
        severity=clang.cindex.Diagnostic.Error,
        message="broken",
        file_name=str(source),
        line=1,
        column=1,
    )
    translation_unit = _FakeTranslationUnit([diagnostic])
    parser = _make_parser(database=_FakeCompilationDatabase([compile_command]))
    warnings: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        parser,
        "_parse_translation_unit",
        lambda command: translation_unit,
    )
    monkeypatch.setattr(
        parser_module.logger,
        "warning",
        lambda *args: warnings.append(args),
    )

    result = parser.get_translation_unit(source)

    assert result is translation_unit
    assert len(warnings) == 1
    assert str(source.resolve()) in str(warnings[0][1])
