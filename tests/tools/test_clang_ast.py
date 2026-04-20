from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import clang.cindex
import pytest

from tools import clang_ast


def test_normalize_include_headers_rejects_option_like_values() -> None:
    with pytest.raises(ValueError, match="option-like"):
        clang_ast._normalize_include_headers(["-Winvalid"])


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


def test_build_ast_payload_includes_metadata_and_filters_external_children() -> None:
    source_path = Path("C:/project/sample.c").resolve()
    output_path = Path("C:/project/out/sample.libclang.txt").resolve()
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
                tokens=[
                    _FakeToken(kind=clang.cindex.TokenKind.LITERAL, spelling='"a"'),
                    _FakeToken(kind=clang.cindex.TokenKind.IDENTIFIER, spelling=r"C:\tmp"),
                ],
                file=str(source_path),
                children=[
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

    output = clang_ast.build_ast_payload(
        translation_unit,
        source_path=source_path,
        output_path=output_path,
        clang_args=["--std", "c11"],
    )

    assert f"source_file: {source_path}" in output
    assert f"output_file: {output_path}" in output
    assert 'parse_args: ["--std", "c11"]' in output
    assert "CALL_EXPR spelling=PyModule_Create2" in output
    assert 'tokens=["\\"a\\"", "C:\\\\tmp"]' in output
    assert 'INTEGER_LITERAL type=int literal=3 tokens=["3"]' in output
    assert "Python.h" not in output


def test_build_ast_payload_formats_diagnostics() -> None:
    source_path = Path("C:/project/sample.c").resolve()
    output_path = Path("C:/project/out/sample.libclang.txt").resolve()
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

    output = clang_ast.build_ast_payload(
        translation_unit,
        source_path=source_path,
        output_path=output_path,
        clang_args=[],
    )

    assert "[WARNING]" in output
    assert "unused value" in output
    assert f"{source_path}:12:8" in output


def _run_ast_export_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    parse_translation_unit_impl,
    build_ast_payload_impl=None,
    clang_dump_result: clang_ast.ClangAstDumpResult,
) -> tuple[list[str], Path, Path]:
    source_path = tmp_path / "sample.c"
    output_dir = tmp_path / "ast_output"
    source_path.write_text("int sample(void) { return 0; }\n", encoding="utf-8")

    monkeypatch.setattr(clang_ast, "DEFAULT_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(clang_ast, "parse_translation_unit", parse_translation_unit_impl)
    if build_ast_payload_impl is not None:
        monkeypatch.setattr(clang_ast, "build_ast_payload", build_ast_payload_impl)
    monkeypatch.setattr(clang_ast, "run_clang_ast_dump", lambda *args, **kwargs: clang_dump_result)

    errors = clang_ast.run_ast_export(
        source_path=source_path,
        include=[],
        include_directory=[],
        c_std=None,
        cpp_std=None,
        clang_library_path=None,
    )
    libclang_path, clang_path = clang_ast.build_output_paths(source_path.resolve())
    return errors, libclang_path, clang_path


def test_build_output_paths_separate_libclang_and_clang_outputs() -> None:
    source_path = Path("C:/project/sample.c").resolve()

    libclang_path, clang_path = clang_ast.build_output_paths(source_path)

    assert libclang_path != clang_path
    assert libclang_path.name == "sample.libclang.txt"
    assert clang_path.name == "sample.clang.txt"


def test_run_ast_export_writes_available_outputs_even_when_clang_reports_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    errors, libclang_path, clang_path = _run_ast_export_case(
        tmp_path,
        monkeypatch,
        parse_translation_unit_impl=lambda *args, **kwargs: SimpleNamespace(cursor=object(), diagnostics=[]),
        build_ast_payload_impl=lambda *args, **kwargs: "libclang payload",
        clang_dump_result=clang_ast.ClangAstDumpResult(
            stdout="partial ast\n",
            stderr="syntax error",
            returncode=1,
        ),
    )
    captured = capsys.readouterr()

    assert len(errors) == 1
    assert libclang_path.read_text(encoding="utf-8") == "libclang payload\n"
    assert clang_path.read_text(encoding="utf-8") == "partial ast\n"
    assert "clang AST export failed" in captured.err


def test_run_ast_export_allows_partial_success_when_libclang_export_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    errors, libclang_path, clang_path = _run_ast_export_case(
        tmp_path,
        monkeypatch,
        parse_translation_unit_impl=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("libclang unavailable")),
        clang_dump_result=clang_ast.ClangAstDumpResult(stdout="clang payload\n", stderr="", returncode=0),
    )
    captured = capsys.readouterr()

    assert len(errors) == 1
    assert not libclang_path.exists()
    assert clang_path.read_text(encoding="utf-8") == "clang payload\n"
    assert "libclang AST export failed" in captured.err


def test_run_ast_export_writes_both_outputs_when_exports_succeed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors, libclang_path, clang_path = _run_ast_export_case(
        tmp_path,
        monkeypatch,
        parse_translation_unit_impl=lambda *args, **kwargs: SimpleNamespace(cursor=object(), diagnostics=[]),
        build_ast_payload_impl=lambda *args, **kwargs: "libclang payload",
        clang_dump_result=clang_ast.ClangAstDumpResult(stdout="clang payload\n", stderr="", returncode=0),
    )

    assert errors == []
    assert libclang_path.read_text(encoding="utf-8") == "libclang payload\n"
    assert clang_path.read_text(encoding="utf-8") == "clang payload\n"
