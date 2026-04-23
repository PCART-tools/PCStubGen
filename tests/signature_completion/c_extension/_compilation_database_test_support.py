from __future__ import annotations

import functools
from pathlib import Path

from pcstubgen.signature_completion.c_extension.libclang import parser as parser_module


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


def make_locator(
    *,
    database: FakeCompilationDatabase | None = None,
    index: object | None = None,
) -> parser_module.ClangFunctionLocator:
    locator = object.__new__(parser_module.ClangFunctionLocator)
    locator._compilation_database = database if database is not None else FakeCompilationDatabase([])
    locator._index = index if index is not None else object()
    locator._resource_dir = None
    locator._get_parsed_source = functools.lru_cache(maxsize=8)(locator._build_parsed_source)
    return locator
