from __future__ import annotations

from typing import Any

import clang.cindex
import pytest
from clang.cindex import Index

from pcstubgen.signature_completion.c_extension.libclang import libclang_wrap as libclang_wrap_module


pytestmark = pytest.mark.libclang


class _PointerHolder:
    def __init__(self) -> None:
        self.value: object | None = None

    def __bool__(self) -> bool:
        return self.value is not None


def _parse_eval_translation_unit(source: str) -> clang.cindex.TranslationUnit:
    """解析 evaluate 测试专用 translation unit。"""
    index = Index.create()
    return index.parse(
        "eval_cursor_test.c",
        args=["-x", "c", "-std=c11"],
        unsaved_files=[("eval_cursor_test.c", source)],
    )


def _find_cursor(
    translation_unit: clang.cindex.TranslationUnit,
    kind: clang.cindex.CursorKind,
    spelling: str,
) -> clang.cindex.Cursor:
    """按 kind 与 spelling 定位测试 cursor。"""
    return next(
        cursor
        for cursor in translation_unit.cursor.walk_preorder()
        if cursor.kind == kind and cursor.spelling == spelling
    )


def _find_binary_operator_cursor(
    translation_unit: clang.cindex.TranslationUnit,
    token_spellings: list[str],
) -> clang.cindex.Cursor:
    """按 token 文本定位测试用二元操作符 cursor。"""
    return next(
        cursor
        for cursor in translation_unit.cursor.walk_preorder()
        if cursor.kind == clang.cindex.CursorKind.BINARY_OPERATOR
        and [token.spelling for token in cursor.get_tokens()] == token_spellings
    )


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
            ["libclang", "src/module.c"],
        )


def test_evaluate_cursor_reads_signed_integer_result() -> None:
    translation_unit = _parse_eval_translation_unit(
        "\n".join(
            [
                "const int signed_value = -1;",
                "void demo(void) {}",
            ]
        )
    )

    cursor = _find_cursor(
        translation_unit,
        clang.cindex.CursorKind.VAR_DECL,
        "signed_value",
    )

    assert libclang_wrap_module.evaluate_cursor(cursor) == -1


def test_evaluate_cursor_reads_unsigned_integer_result() -> None:
    translation_unit = _parse_eval_translation_unit(
        "const unsigned long long unsigned_value = 18446744073709551615ULL;"
    )

    cursor = _find_cursor(
        translation_unit,
        clang.cindex.CursorKind.VAR_DECL,
        "unsigned_value",
    )

    assert libclang_wrap_module.evaluate_cursor(cursor) == 18446744073709551615


def test_evaluate_cursor_reads_float_result() -> None:
    translation_unit = _parse_eval_translation_unit("const double float_value = 1.25;")

    cursor = _find_cursor(
        translation_unit,
        clang.cindex.CursorKind.VAR_DECL,
        "float_value",
    )

    assert libclang_wrap_module.evaluate_cursor(cursor) == pytest.approx(1.25)


def test_evaluate_cursor_reads_string_result() -> None:
    translation_unit = _parse_eval_translation_unit('const char *string_value = "abc";')

    cursor = _find_cursor(
        translation_unit,
        clang.cindex.CursorKind.VAR_DECL,
        "string_value",
    )

    assert libclang_wrap_module.evaluate_cursor(cursor) == "abc"


def test_evaluate_cursor_raises_for_non_evaluable_cursor() -> None:
    translation_unit = _parse_eval_translation_unit("void demo(void) {}")

    cursor = _find_cursor(
        translation_unit,
        clang.cindex.CursorKind.FUNCTION_DECL,
        "demo",
    )

    with pytest.raises(RuntimeError, match="无法求值"):
        libclang_wrap_module.evaluate_cursor(cursor)


def test_get_cursor_binary_operator_kind_distinguishes_assignment_from_equality() -> None:
    translation_unit = _parse_eval_translation_unit(
        "\n".join(
            [
                "void demo(int a, int b) {",
                "    a = b;",
                "    if (a == b) {}",
                "}",
            ]
        )
    )
    assignment_cursor = _find_binary_operator_cursor(
        translation_unit,
        ["a", "=", "b"],
    )
    equality_cursor = _find_binary_operator_cursor(
        translation_unit,
        ["a", "==", "b"],
    )

    assert (
        libclang_wrap_module.get_cursor_binary_operator_kind(assignment_cursor)
        == libclang_wrap_module.CX_BINARY_OPERATOR_ASSIGN
    )
    assert (
        libclang_wrap_module.get_cursor_binary_operator_kind(equality_cursor)
        != libclang_wrap_module.CX_BINARY_OPERATOR_ASSIGN
    )
