from __future__ import annotations

from pathlib import Path
from typing import cast

from pcstubgen.signature_completion.c_extension.clang import parser as parser_module


class FakeCompileCommand:
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


class FakeCompilationDatabase:
    def __init__(self, commands: list[FakeCompileCommand]) -> None:
        self._commands = commands

    def getCompileCommands(self, filename: str) -> list[FakeCompileCommand]:
        source_path = Path(filename).resolve()
        return [
            command
            for command in self._commands
            if compile_command_filename(command) == source_path
        ]


def compile_command_filename(command: FakeCompileCommand) -> Path:
    file_path = Path(command.filename)
    if not file_path.is_absolute():
        file_path = Path(command.directory) / file_path
    return file_path.resolve()


def make_parser(
    *,
    database: FakeCompilationDatabase | None = None,
    index: object | None = None,
) -> parser_module.ClangParser:
    parser = object.__new__(parser_module.ClangParser)
    parser._compilation_database = database if database is not None else FakeCompilationDatabase([])
    parser._translation_units = {}
    parser._index = index if index is not None else object()
    return parser
