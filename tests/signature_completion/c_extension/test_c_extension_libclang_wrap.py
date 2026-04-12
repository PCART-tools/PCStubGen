from __future__ import annotations

from pathlib import Path
from typing import Any

import clang.cindex
import pytest
from clang.cindex import Index

from pcstubgen.signature_completion.c_extension.clang import libclang_wrap as libclang_wrap_module


class _PointerHolder:
    def __init__(self) -> None:
        self.value: object | None = None

    def __bool__(self) -> bool:
        return self.value is not None


def test_parse_translation_unit_full_argv_passes_full_argv_to_libclang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_translation_unit = object()
    index = Index.create()
    received_arguments: list[str] = []
    received_options: list[int] = []

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
        assert received_index is index
        assert source_filename is None
        assert unsaved_files is None
        assert unsaved_files_count == 0
        received_arguments.extend(
            arguments_array[arg_index].decode("utf-8")
            for arg_index in range(arguments_count)
        )
        received_options.append(options)
        out_translation_unit.value = raw_translation_unit
        return 0

    monkeypatch.setattr(
        libclang_wrap_module,
        "_parse_translation_unit2_full_argv",
        _fake_parse_translation_unit2_full_argv,
    )
    monkeypatch.setattr(
        libclang_wrap_module.clang.cindex,
        "c_object_p",
        _PointerHolder,
    )
    monkeypatch.setattr(libclang_wrap_module, "byref", lambda pointer: pointer)
    monkeypatch.setattr(
        libclang_wrap_module,
        "TranslationUnit",
        lambda pointer, **kwargs: (pointer.value, kwargs["index"]),
    )

    result = libclang_wrap_module.parse_translation_unit_full_argv(
        index,
        ["clang", "-Iinclude", "-c", "src/module.c", "-o", "build/module.o"],
    )

    assert result == (raw_translation_unit, index)
    assert received_arguments == [
        "clang",
        "-Iinclude",
        "-c",
        "src/module.c",
        "-o",
        "build/module.o",
    ]
    assert received_options == [0]


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
        libclang_wrap_module,
        "_parse_translation_unit2_full_argv",
        _fake_parse_translation_unit2_full_argv,
    )
    monkeypatch.setattr(
        libclang_wrap_module.clang.cindex,
        "c_object_p",
        _PointerHolder,
    )
    monkeypatch.setattr(libclang_wrap_module, "byref", lambda pointer: pointer)

    with pytest.raises(
        libclang_wrap_module.TranslationUnitLoadError,
        match="libclang error code: 4",
    ):
        libclang_wrap_module.parse_translation_unit_full_argv(
            index,
            ["clang", "src/module.c"],
        )


def test_get_file_location_returns_file_and_offset(tmp_path: Path) -> None:
    source = tmp_path / "file_location.c"
    source.write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    index = clang.cindex.Index.create()
    translation_unit = clang.cindex.TranslationUnit.from_source(str(source), index=index)
    function_cursor = next(
        cursor
        for cursor in translation_unit.cursor.get_children()
        if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL
    )

    file, line, column, offset = libclang_wrap_module.get_file_location(function_cursor.extent.start)

    assert file is not None
    assert Path(file.name) == source
    assert (line, column, offset) == (1, 1, 0)


def test_get_file_contents_returns_loaded_buffer(tmp_path: Path) -> None:
    source = tmp_path / "file_contents.c"
    content = "int demo(void) { return 0; }\n"
    source.write_text(content, encoding="utf-8")
    index = clang.cindex.Index.create()
    translation_unit = clang.cindex.TranslationUnit.from_source(str(source), index=index)
    file = clang.cindex.File.from_name(translation_unit, str(source))

    buffer = libclang_wrap_module.get_file_contents(translation_unit, file)

    assert buffer == content.encode("utf-8")
