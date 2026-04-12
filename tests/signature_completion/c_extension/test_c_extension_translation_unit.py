from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from clang.cindex import CompileCommand, TranslationUnitLoadError

from tests.signature_completion.c_extension._compilation_database_test_support import (
    FakeCompilationDatabase,
    FakeCompileCommand,
    make_parser,
)


def test_get_translation_unit_caches_successful_result_by_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "src" / "module.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    compile_command = FakeCompileCommand(
        directory=tmp_path,
        filename="src/module.c",
        arguments=["clang", "src/module.c"],
    )
    translation_unit = SimpleNamespace(diagnostics=[])
    parser = make_parser(database=FakeCompilationDatabase([compile_command]))
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
    compile_command = FakeCompileCommand(
        directory=tmp_path,
        filename="src/module.c",
        arguments=["clang", "src/module.c"],
    )
    parser = make_parser(database=FakeCompilationDatabase([compile_command]))
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
