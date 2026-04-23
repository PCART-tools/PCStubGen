from __future__ import annotations

import contextlib
from pathlib import Path
import re

import clang.cindex
import pytest

from pcstubgen.signature_completion.c_extension.libclang import ast_utils as ast_utils_module
from tests._c_extension_test_support import (
    _FakeCursorLocation,
    _FakeNode,
    _FakeSourceRange,
    _identifier_node,
)


def _location_text(text: str) -> object:
    class _Location:
        def __str__(self) -> str:
            return text

    return _Location()


def _parse_first_function_cursor(source_arg: str, *, cwd: Path | None = None) -> clang.cindex.Cursor:
    index = clang.cindex.Index.create()
    with contextlib.chdir(cwd) if cwd is not None else contextlib.nullcontext():
        translation_unit = clang.cindex.TranslationUnit.from_source(source_arg, index=index)

    return next(
        cursor
        for cursor in translation_unit.cursor.get_children()
        if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL
    )


def _parse_first_call_cursor(source_arg: str) -> clang.cindex.Cursor:
    index = clang.cindex.Index.create()
    translation_unit = clang.cindex.TranslationUnit.from_source(source_arg, index=index)
    return next(
        cursor
        for cursor in translation_unit.cursor.walk_preorder()
        if cursor.kind == clang.cindex.CursorKind.CALL_EXPR
    )


@pytest.mark.libclang
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

    extracted = ast_utils_module.get_cursor_source_text(function_cursor)

    assert snippet in extracted


@pytest.mark.libclang
def test_get_call_expr_source_name_reads_direct_call_start_token(tmp_path: Path) -> None:
    source = tmp_path / "direct_call_name.c"
    source.write_text(
        "\n".join(
            [
                "int PyLong_FromLong(int value);",
                "int demo(int value) {",
                "    return PyLong_FromLong(value);",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    call_cursor = _parse_first_call_cursor(str(source))

    call_name = ast_utils_module.get_first_token_str(call_cursor)

    assert call_name == "PyLong_FromLong"


@pytest.mark.libclang
def test_get_call_expr_source_name_reads_function_like_macro_start_token(tmp_path: Path) -> None:
    source = tmp_path / "macro_call_name.c"
    source.write_text(
        "\n".join(
            [
                "int PyLong_FromLong(int value);",
                "#define PyArray_ContiguousFromObject(op, type, min_depth, max_depth) PyLong_FromLong(op)",
                "int demo(int value) {",
                "    return PyArray_ContiguousFromObject(value, 0, 0, 0);",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    call_cursor = _parse_first_call_cursor(str(source))

    call_name = ast_utils_module.get_first_token_str(call_cursor)

    assert call_name == "PyArray_ContiguousFromObject"


@pytest.mark.libclang
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

    extracted = ast_utils_module.get_cursor_source_text(function_cursor)

    assert snippet in extracted


def test_var_decl_to_init_list_expr_raises_with_cursor_location() -> None:
    cursor = _identifier_node("value")
    cursor.location = _location_text("ast_utils.c:3:5")

    with pytest.raises(AssertionError):
        ast_utils_module.var_decl_to_init_list_expr(cursor)


def test_cursor_get_text_raises_with_cursor_location_when_extent_lacks_file_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        location=_location_text("ast_utils.c:8:2"),
        extent=_FakeSourceRange(_FakeCursorLocation(), _FakeCursorLocation()),
    )
    monkeypatch.setattr(
        ast_utils_module,
        "get_file_location",
        lambda location: (None, 0, 0, 0),
    )

    with pytest.raises(
        RuntimeError,
        match=rf"源码范围缺少起止文件信息.*{re.escape('ast_utils.c:8:2')}",
    ):
        ast_utils_module.get_cursor_source_text(cursor)


def test_extract_string_literal_raises_with_cursor_location() -> None:
    cursor = _identifier_node("value")
    cursor.location = _location_text("ast_utils.c:12:7")

    with pytest.raises(
        RuntimeError,
        match=rf"节点不是字符串字面量.*{re.escape('ast_utils.c:12:7')}",
    ):
        ast_utils_module.get_string_literal(cursor)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, True),
        (1, False),
    ],
)
def test_is_integer_literal_zero_only_accepts_int_zero(
    monkeypatch: pytest.MonkeyPatch,
    value: int,
    expected: bool,
) -> None:
    cursor = _FakeNode(kind=clang.cindex.CursorKind.INTEGER_LITERAL)
    monkeypatch.setattr(ast_utils_module, "evaluate_cursor", lambda _: value)

    assert ast_utils_module.is_nullptr_or_zero(cursor) is expected
