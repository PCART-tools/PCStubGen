from __future__ import annotations

from typing import Any

import pytest
from clang.cindex import Index

from pcstubgen.signature_completion.c_extension.clang import libclang_wrap as libclang_wrap_module


class _PointerHolder:
    def __init__(self) -> None:
        self.value: object | None = None

    def __bool__(self) -> bool:
        return self.value is not None


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
