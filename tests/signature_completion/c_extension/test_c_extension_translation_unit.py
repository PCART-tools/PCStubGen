from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import clang.cindex
import pytest
from clang.cindex import CompileCommand, TranslationUnitLoadError

from tests._c_extension_test_support import _FakeNode
from tests.signature_completion.c_extension._compilation_database_test_support import (
    FakeCompilationDatabase,
    FakeCompileCommand,
    make_locator,
)


def test_get_function_cursor_caches_successful_result_by_absolute_path(
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
    translation_unit = _translation_unit_with_children(
        [
            _function_cursor("demo"),
        ]
    )
    locator = make_locator(database=FakeCompilationDatabase([compile_command]))
    calls: list[CompileCommand] = []

    monkeypatch.setattr(
        locator,
        "_parse_translation_unit",
        lambda command: calls.append(command) or translation_unit,
    )

    first = locator.get_function_cursor(source, "demo", None)
    second = locator.get_function_cursor(tmp_path / "src" / "." / "module.c", "demo", None)

    assert first is second
    assert calls == [compile_command]


def test_get_function_cursor_does_not_cache_parse_failure(
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
    locator = make_locator(database=FakeCompilationDatabase([compile_command]))
    calls: list[CompileCommand] = []

    def _raise_parse_error(command: CompileCommand) -> object:
        calls.append(command)
        raise TranslationUnitLoadError("boom")

    monkeypatch.setattr(
        locator,
        "_parse_translation_unit",
        _raise_parse_error,
    )

    with pytest.raises(RuntimeError, match="Parse失败"):
        locator.get_function_cursor(source, "demo", None)

    with pytest.raises(RuntimeError, match="Parse失败"):
        locator.get_function_cursor(source, "demo", None)

    assert calls == [compile_command, compile_command]


def test_get_function_cursor_matches_linkage_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "src" / "module.cpp"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int foo(int value) { return value; }\n", encoding="utf-8")
    compile_command = FakeCompileCommand(
        directory=tmp_path,
        filename="src/module.cpp",
        arguments=["clang++", "src/module.cpp"],
    )
    first_cursor = _function_cursor("foo", mangled_name="_Z3fooi")
    second_cursor = _function_cursor("foo", mangled_name="_Z3food")
    translation_unit = _translation_unit_with_children([first_cursor, second_cursor])
    locator = make_locator(database=FakeCompilationDatabase([compile_command]))

    monkeypatch.setattr(locator, "_parse_translation_unit", lambda command: translation_unit)

    matched = locator.get_function_cursor(source, "foo", "_Z3fooi")

    assert matched is first_cursor


def test_get_function_cursor_matches_nested_definition_by_spelling_first_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "src" / "module.cpp"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("int foo_impl(int value) { return value; }\n", encoding="utf-8")
    compile_command = FakeCompileCommand(
        directory=tmp_path,
        filename="src/module.cpp",
        arguments=["clang++", "src/module.cpp"],
    )
    first_cursor = _function_cursor("foo_impl", mangled_name="_Z8foo_impli")
    second_cursor = _function_cursor("foo_impl", mangled_name="_Z8foo_impld")
    translation_unit = _translation_unit_with_children(
        [
            _FakeNode(
                kind=clang.cindex.CursorKind.NAMESPACE,
                children=[
                    _FakeNode(
                        kind=clang.cindex.CursorKind.LINKAGE_SPEC,
                        children=[first_cursor],
                    ),
                    second_cursor,
                ],
            )
        ]
    )
    locator = make_locator(database=FakeCompilationDatabase([compile_command]))

    monkeypatch.setattr(locator, "_parse_translation_unit", lambda command: translation_unit)

    matched = locator.get_function_cursor(source, "foo_impl", None)

    assert matched is first_cursor


def test_get_function_cursor_reparses_after_lru_eviction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[FakeCompileCommand] = []
    for index in range(9):
        source = tmp_path / "src" / f"module_{index}.c"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
        commands.append(
            FakeCompileCommand(
                directory=tmp_path,
                filename=str(Path("src") / f"module_{index}.c"),
                arguments=["clang", str(Path("src") / f"module_{index}.c")],
            )
        )

    locator = make_locator(database=FakeCompilationDatabase(commands))
    calls: list[Path] = []

    def _parse_translation_unit(command: CompileCommand) -> SimpleNamespace:
        command_path = Path(str(command.directory)) / Path(str(command.filename))
        source_path = command_path.resolve()
        calls.append(source_path)
        return _translation_unit_with_children([_function_cursor("demo")])

    monkeypatch.setattr(locator, "_parse_translation_unit", _parse_translation_unit)

    for index in range(9):
        locator.get_function_cursor(tmp_path / "src" / f"module_{index}.c", "demo", None)

    first_source = (tmp_path / "src" / "module_0.c").resolve()
    locator.get_function_cursor(first_source, "demo", None)

    assert calls.count(first_source) == 2


def _translation_unit_with_children(children: list[_FakeNode]) -> SimpleNamespace:
    return SimpleNamespace(
        cursor=_FakeNode(
            kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
            children=children,
            location="translation_unit:1:1",
        ),
        diagnostics=[],
    )


def _function_cursor(spelling: str, *, mangled_name: str | None = None) -> _FakeNode:
    cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling=spelling,
        is_definition=True,
        location=f"{spelling}:1:1",
    )
    if mangled_name is not None:
        cursor.mangled_name = mangled_name
    else:
        cursor.mangled_name = ""
    return cursor
