from __future__ import annotations

from pathlib import Path

import pytest

from pcstubgen.signature_completion.c_extension.libclang import function_cursor_locator as parser_module
from tests.signature_completion.c_extension._compilation_database_test_support import (
    FakeCompilationDatabase,
    FakeCompileCommand,
    compile_command_filename,
    make_locator,
)


def test_validate_compilation_database_path_requires_compile_commands_json(tmp_path: Path) -> None:
    wrong_file = tmp_path / "commands.json"
    wrong_file.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="文件名必须为 compile_commands.json"):
        parser_module.validate_compilation_database_path(wrong_file)


def test_get_compile_command_returns_first_matching_command(tmp_path: Path) -> None:
    shared_source = tmp_path / "src" / "module.c"
    shared_source.parent.mkdir(parents=True, exist_ok=True)
    shared_source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    header = tmp_path / "src" / "module.h"
    header.write_text("int demo(void);\n", encoding="utf-8")

    database = FakeCompilationDatabase(
        [
            FakeCompileCommand(
                directory=tmp_path,
                filename="src/module.c",
                arguments=["cc", "-DFIRST", "-c", "src/module.c"],
            ),
            FakeCompileCommand(
                directory=tmp_path,
                filename="src/module.c",
                arguments=["cc", "-DSECOND", "-c", "src/module.c"],
            ),
            FakeCompileCommand(
                directory=tmp_path,
                filename="src/module.h",
                arguments=["cc", "-c", "src/module.h"],
            ),
        ]
    )

    locator = make_locator(database=database)
    result = locator._get_compile_command(shared_source.resolve())

    assert compile_command_filename(result) == shared_source.resolve()
    assert Path(str(result.directory)).resolve() == tmp_path.resolve()
    assert list(result.arguments) == ["cc", "-DFIRST", "-c", "src/module.c"]


def test_get_compile_command_raises_when_source_is_missing(tmp_path: Path) -> None:
    source = tmp_path / "src" / "module.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")

    locator = make_locator(database=FakeCompilationDatabase([]))

    with pytest.raises(RuntimeError, match="未在编译数据库中定位到编译单元"):
        locator._get_compile_command(source.resolve())
