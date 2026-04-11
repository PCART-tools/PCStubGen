from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from clang.cindex import CompileCommand, Index

from pcstubgen.signature_completion.c_extension.clang import translation_unit as translation_unit_module


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


def test_try_get_clang_resource_dir_returns_valid_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_dir = tmp_path / "clang-resource"
    (resource_dir / "include").mkdir(parents=True)
    translation_unit_module.try_get_clang_resource_dir.cache_clear()
    monkeypatch.setattr(
        translation_unit_module.subprocess,
        "check_output",
        lambda command, *, text: f"{resource_dir}\n",
    )

    assert translation_unit_module.try_get_clang_resource_dir() == resource_dir
    translation_unit_module.try_get_clang_resource_dir.cache_clear()


def test_try_get_clang_resource_dir_returns_none_when_command_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation_unit_module.try_get_clang_resource_dir.cache_clear()
    monkeypatch.setattr(
        translation_unit_module.subprocess,
        "check_output",
        lambda command, *, text: (_ for _ in ()).throw(FileNotFoundError()),
    )

    assert translation_unit_module.try_get_clang_resource_dir() is None
    translation_unit_module.try_get_clang_resource_dir.cache_clear()


def test_parse_appends_resource_dir_and_uses_compile_command_directory(
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

    monkeypatch.setattr(
        translation_unit_module,
        "parse_translation_unit_full_argv",
        _fake_parse_translation_unit_full_argv,
    )
    monkeypatch.setattr(
        translation_unit_module,
        "try_get_clang_resource_dir",
        lambda: resource_dir,
    )

    compile_command = _make_compile_command(
        working_directory=working_directory,
        source=source,
        arguments=["clang", "-I../include", "-DMODE=1", "src/module.c"],
    )

    result = translation_unit_module.parse(index, compile_command)

    assert result is translation_unit
    assert observed_arguments == [
        ["clang", "-I../include", "-DMODE=1", "src/module.c", "-resource-dir", str(resource_dir)]
    ]
    assert observed_cwds == [working_directory.resolve()]


def test_parse_keeps_original_arguments_when_resource_dir_is_unavailable(
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

    monkeypatch.setattr(
        translation_unit_module,
        "parse_translation_unit_full_argv",
        _fake_parse_translation_unit_full_argv,
    )
    monkeypatch.setattr(
        translation_unit_module,
        "try_get_clang_resource_dir",
        lambda: None,
    )

    compile_command = _make_compile_command(
        working_directory=working_directory,
        source=source,
        arguments=["clang", "-I../include", "-DMODE=1", "src/module.c"],
    )

    result = translation_unit_module.parse(index, compile_command)

    assert result is translation_unit
    assert observed_arguments == [["clang", "-I../include", "-DMODE=1", "src/module.c"]]
