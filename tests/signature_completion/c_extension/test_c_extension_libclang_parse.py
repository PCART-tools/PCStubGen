from __future__ import annotations

from typing import Any

import pytest
from clang.cindex import Index

from pcstubgen.signature_completion.c_extension.clang import libclang_parse as libclang_parse_module


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
    monkeypatch.setattr(
        libclang_parse_module,
        "TranslationUnit",
        lambda pointer, **kwargs: (pointer.value, kwargs["index"]),
    )

    result = libclang_parse_module.parse_translation_unit_full_argv(
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
