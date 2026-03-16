from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import clang.cindex
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tool import libclang_ast


def test_parse_args_accepts_clang_include_and_include_directory() -> None:
    args = libclang_ast.parse_args(
        [
            "sample.c",
            "--clang-include",
            "Python.h",
            "--clang-include=numpy/arrayobject.h",
            "--clang-include-directory",
            "C:/IncludeA",
            "--clang-include-directory=C:/IncludeB",
        ]
    )

    assert args.clang_include == ["Python.h", "numpy/arrayobject.h"]
    assert args.clang_include_directory == ["C:/IncludeA", "C:/IncludeB"]


def test_build_parse_args_places_include_before_include_directory() -> None:
    parse_args = libclang_ast._build_parse_args(
        source_path=Path("sample.c"),
        clang_args=[],
        include_headers=["Python.h", "numpy/arrayobject.h"],
        include_paths=["C:/IncludeA"],
        clang_c_std="c11",
        clang_cpp_std="c++17",
    )

    assert parse_args == [
        "--std",
        "c11",
        "--include",
        "Python.h",
        "--include",
        "numpy/arrayobject.h",
        "--include-directory",
        "C:/IncludeA",
    ]


def test_normalize_include_headers_rejects_option_like_values() -> None:
    with pytest.raises(ValueError, match="option-like"):
        libclang_ast._normalize_include_headers(["-Winvalid"])


class _FakeLocation:
    def __init__(self, *, file: str | None = None, line: int = 0, column: int = 0) -> None:
        self.file = file
        self.line = line
        self.column = column


class _FakeType:
    def __init__(self, spelling: str) -> None:
        self.spelling = spelling


class _FakeToken:
    def __init__(self, *, kind: object, spelling: str) -> None:
        self.kind = kind
        self.spelling = spelling


class _FakeCursor:
    def __init__(
        self,
        *,
        kind: object,
        spelling: str = "",
        displayname: str = "",
        type_spelling: str | None = None,
        tokens: list[_FakeToken] | None = None,
        children: list["_FakeCursor"] | None = None,
        file: str | None = None,
    ) -> None:
        self.kind = kind
        self.spelling = spelling
        self.displayname = displayname
        self.type = _FakeType(type_spelling) if type_spelling is not None else None
        self.location = _FakeLocation(file=file)
        self._tokens = tokens or []
        self._children = children or []

    def get_tokens(self) -> list[_FakeToken]:
        return self._tokens

    def get_children(self) -> list["_FakeCursor"]:
        return list(self._children)


class _FakeDiagnostic:
    def __init__(
        self,
        *,
        severity: int,
        spelling: str,
        file: str | None = None,
        line: int = 0,
        column: int = 0,
    ) -> None:
        self.severity = severity
        self.spelling = spelling
        self.location = _FakeLocation(file=file, line=line, column=column)


def test_resolve_output_path_defaults_to_ast_txt() -> None:
    output_path = libclang_ast.resolve_output_path(Path("sample.c"), None)
    assert output_path == libclang_ast.DEFAULT_OUTPUT_DIR / "sample.ast.txt"


def test_build_ast_payload_renders_tree_and_filters_external_children() -> None:
    source_path = Path("C:/project/sample.c").resolve()
    output_path = Path("C:/project/out/sample.ast.txt").resolve()
    literal_token = _FakeToken(kind=clang.cindex.TokenKind.LITERAL, spelling="3")
    external_child = _FakeCursor(
        kind=clang.cindex.CursorKind.DECL_REF_EXPR,
        spelling="PyModule_Create2",
        type_spelling="PyObject *",
        file=str(Path("C:/python/include/Python.h").resolve()),
    )
    root = _FakeCursor(
        kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
        spelling=str(source_path),
        children=[
            _FakeCursor(
                kind=clang.cindex.CursorKind.CALL_EXPR,
                spelling="PyModule_Create2",
                type_spelling="PyObject *",
                file=str(source_path),
                children=[
                    _FakeCursor(
                        kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
                        type_spelling="PyObject *",
                        file=str(source_path),
                        children=[
                            _FakeCursor(
                                kind=clang.cindex.CursorKind.DECL_REF_EXPR,
                                spelling="PyModule_Create2",
                                type_spelling="PyObject *",
                                file=str(source_path),
                            )
                        ],
                    ),
                    _FakeCursor(
                        kind=clang.cindex.CursorKind.PAREN_EXPR,
                        type_spelling="PyModuleDef *",
                        file=str(source_path),
                        children=[
                            _FakeCursor(
                                kind=clang.cindex.CursorKind.UNARY_OPERATOR,
                                type_spelling="PyModuleDef *",
                                file=str(source_path),
                                children=[
                                    _FakeCursor(
                                        kind=clang.cindex.CursorKind.DECL_REF_EXPR,
                                        spelling="defs",
                                        type_spelling="PyModuleDef",
                                        file=str(source_path),
                                    )
                                ],
                            )
                        ],
                    ),
                    _FakeCursor(
                        kind=clang.cindex.CursorKind.INTEGER_LITERAL,
                        type_spelling="int",
                        tokens=[literal_token],
                        file=str(source_path),
                    ),
                    external_child,
                ],
            )
        ],
    )
    translation_unit = SimpleNamespace(cursor=root, diagnostics=[])

    output = libclang_ast.build_ast_payload(
        translation_unit,
        source_path=source_path,
        output_path=output_path,
        clang_args=["--std", "c11"],
    )

    assert f"source_file: {source_path}" in output
    assert f"output_file: {output_path}" in output
    assert 'parse_args: ["--std", "c11"]' in output
    assert "diagnostics:\n- <none>\nast:\n" in output
    assert "TRANSLATION_UNIT spelling=" in output
    assert "└─ CALL_EXPR spelling=PyModule_Create2 type=PyObject *" in output
    assert "   ├─ UNEXPOSED_EXPR type=PyObject *" in output
    assert "   │  └─ DECL_REF_EXPR spelling=PyModule_Create2 type=PyObject *" in output
    assert "   ├─ PAREN_EXPR type=PyModuleDef *" in output
    assert "   │  └─ UNARY_OPERATOR type=PyModuleDef *" in output
    assert "   │     └─ DECL_REF_EXPR spelling=defs type=PyModuleDef" in output
    assert "   └─ INTEGER_LITERAL type=int literal=3" in output
    assert "Python.h" not in output


def test_build_ast_payload_formats_diagnostics() -> None:
    source_path = Path("C:/project/sample.c").resolve()
    output_path = Path("C:/project/out/sample.ast.txt").resolve()
    root = _FakeCursor(
        kind=clang.cindex.CursorKind.TRANSLATION_UNIT,
        spelling=str(source_path),
    )
    diagnostic = _FakeDiagnostic(
        severity=clang.cindex.Diagnostic.Warning,
        spelling="unused value",
        file=str(source_path),
        line=12,
        column=8,
    )
    translation_unit = SimpleNamespace(cursor=root, diagnostics=[diagnostic])

    output = libclang_ast.build_ast_payload(
        translation_unit,
        source_path=source_path,
        output_path=output_path,
        clang_args=[],
    )

    assert f"- [WARNING] {source_path}:12:8: unused value" in output
