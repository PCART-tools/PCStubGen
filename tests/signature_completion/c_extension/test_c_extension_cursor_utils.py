from __future__ import annotations

from pathlib import Path

from tests._c_extension_test_support import (
    _FakeCursorLocation,
    _FakeSourceRange,
    _extent_for_source_snippet,
    cursor_utils_module,
)


def test_extract_cursor_source_text_reads_text_from_extent(tmp_path: Path) -> None:
    source = tmp_path / "extent_text.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O!\", (&PyUnicode_Type), &value);",
            ]
        ),
        encoding="utf-8",
    )

    extracted = cursor_utils_module.source_range_get_text(
        _extent_for_source_snippet(source, "(&PyUnicode_Type)")
    )

    assert extracted == "(&PyUnicode_Type)"


def test_extract_cursor_source_text_returns_none_when_extent_start_file_is_missing() -> None:
    extracted = cursor_utils_module.source_range_get_text(
        _FakeSourceRange(
            _FakeCursorLocation(None, 0),
            _FakeCursorLocation("extent_text.c", 1),
        )
    )

    assert extracted is None


def test_extract_cursor_source_text_returns_none_for_cross_file_extent(tmp_path: Path) -> None:
    first = tmp_path / "first.c"
    second = tmp_path / "second.c"
    first.write_text("abc", encoding="utf-8")
    second.write_text("def", encoding="utf-8")

    extracted = cursor_utils_module.source_range_get_text(
        _FakeSourceRange(
            _FakeCursorLocation(str(first), 0),
            _FakeCursorLocation(str(second), 1),
        )
    )

    assert extracted is None


def test_extract_cursor_source_text_returns_none_when_file_read_fails(tmp_path: Path) -> None:
    missing = tmp_path / "missing_extent_text.c"

    extracted = cursor_utils_module.source_range_get_text(
        _FakeSourceRange(
            _FakeCursorLocation(str(missing), 0),
            _FakeCursorLocation(str(missing), 1),
        )
    )

    assert extracted is None
