from __future__ import annotations

import contextlib
from pathlib import Path

import clang.cindex

from tests._c_extension_test_support import ast_utils_module


def _parse_first_function_cursor(source_arg: str, *, cwd: Path | None = None) -> clang.cindex.Cursor:
    index = clang.cindex.Index.create()
    with contextlib.chdir(cwd) if cwd is not None else contextlib.nullcontext():
        translation_unit = clang.cindex.TranslationUnit.from_source(source_arg, index=index)

    return next(
        cursor
        for cursor in translation_unit.cursor.get_children()
        if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL
    )


def test_extract_cursor_source_text_reads_text_from_translation_unit_buffer(tmp_path: Path) -> None:
    source = tmp_path / "extent_text.c"
    snippet = "PyArg_ParseTuple(args, \"O!\", (&PyUnicode_Type), &value);"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "int demo(PyObject* args, PyObject* value) {",
                f"    {snippet}",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    function_cursor = _parse_first_function_cursor(str(source))

    extracted = ast_utils_module.cursor_get_text(function_cursor)

    assert snippet in extracted


def test_extract_cursor_source_text_reads_relative_path_extent_from_translation_unit_buffer(
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "build"
    source_dir = tmp_path / "src"
    build_dir.mkdir()
    source_dir.mkdir()
    source = source_dir / "relative_extent.c"
    snippet = "return value + 1;"
    source.write_text(
        "\n".join(
            [
                "int demo(int value) {",
                f"    {snippet}",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    function_cursor = _parse_first_function_cursor("../src/relative_extent.c", cwd=build_dir)

    extracted = ast_utils_module.cursor_get_text(function_cursor)

    assert snippet in extracted
