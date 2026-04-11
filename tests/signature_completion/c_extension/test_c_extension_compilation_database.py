from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from clang.cindex import CompileCommand

from pcstubgen.signature_completion.c_extension.clang import compilation_database as compilation_database_module


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


def test_validate_compilation_database_path_requires_compile_commands_json(tmp_path: Path) -> None:
    wrong_file = tmp_path / "commands.json"
    wrong_file.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="文件名必须为 compile_commands.json"):
        compilation_database_module.validate_compilation_database_path(wrong_file)


def test_get_compile_command_preserves_full_arguments(tmp_path: Path) -> None:
    shared_source = tmp_path / "src" / "module.c"
    shared_source.parent.mkdir(parents=True, exist_ok=True)
    shared_source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    header = tmp_path / "src" / "module.h"
    header.write_text("int demo(void);\n", encoding="utf-8")

    database = _FakeCompilationDatabase(
        [
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
    )

    result = compilation_database_module.get_compile_command(database, shared_source)

    assert result.filename == shared_source.resolve()
    assert result.directory == tmp_path.resolve()
    assert result.arguments == ["cc", "-DFIRST", "-c", "src/module.c"]


def test_get_compile_command_raises_when_source_is_missing(tmp_path: Path) -> None:
    source = tmp_path / "src" / "module.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="未在编译数据库中定位到编译单元"):
        compilation_database_module.get_compile_command(
            _FakeCompilationDatabase([]),
            source,
        )
