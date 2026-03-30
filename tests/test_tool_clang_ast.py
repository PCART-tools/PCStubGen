from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import clang.cindex
import pytest
from typer.testing import CliRunner

from tools import clang_ast

RUNNER = CliRunner()


def test_cli_accepts_include_and_include_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_kwargs: dict[str, object] = {}
    source_path = tmp_path / "sample.c"
    source_path.write_text("int sample(void) { return 0; }\n", encoding="utf-8")

    def fake_run_ast_export(**kwargs: object) -> list[Exception]:
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(clang_ast, "run_ast_export", fake_run_ast_export)

    result = RUNNER.invoke(
        clang_ast.app,
        [
            str(source_path),
            "--include",
            "Python.h",
            "--include=numpy/arrayobject.h",
            "--include-directory",
            "C:/IncludeA",
            "--include-directory=C:/IncludeB",
            "--c-std",
            "c99",
            "--cpp-std",
            "c++20",
        ],
        prog_name="clang_ast",
    )

    assert result.exit_code == 0
    assert captured_kwargs["source_path"] == source_path.resolve()
    assert captured_kwargs["include"] == ["Python.h", "numpy/arrayobject.h"]
    assert captured_kwargs["include_directory"] == [
        Path("C:/IncludeA"),
        Path("C:/IncludeB"),
    ]
    assert captured_kwargs["c_std"] == "c99"
    assert captured_kwargs["cpp_std"] == "c++20"
def test_cli_invalid_include_reports_bad_parameter(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.c"
    source_path.write_text("int sample(void) { return 0; }\n", encoding="utf-8")

    result = RUNNER.invoke(
        clang_ast.app,
        [str(source_path), "--include=-bad"],
        prog_name="clang_ast",
    )

    assert result.exit_code == 2
    assert "Invalid value for '--include'" in result.stderr
    assert "'-bad'" in result.stderr


def test_cli_invalid_include_directory_reports_bad_parameter(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.c"
    source_path.write_text("int sample(void) { return 0; }\n", encoding="utf-8")

    result = RUNNER.invoke(
        clang_ast.app,
        [str(source_path), "--include-directory=-bad"],
        prog_name="clang_ast",
    )

    assert result.exit_code == 2
    assert "Invalid value for '--include-directory'" in result.stderr
    assert "'-bad'" in result.stderr


def test_normalize_include_headers_rejects_option_like_values() -> None:
    with pytest.raises(ValueError, match="option-like"):
        clang_ast._normalize_include_headers(["-Winvalid"])


def test_validate_include_preserves_message() -> None:
    with pytest.raises(clang_ast.typer.BadParameter) as ex:
        clang_ast._validate_include(["-bad"])

    message = str(ex.value)
    assert "include" in message
    assert "-bad" in message


def test_validate_include_directory_preserves_message() -> None:
    with pytest.raises(clang_ast.typer.BadParameter) as ex:
        clang_ast._validate_include_directory([Path("-bad")])

    message = str(ex.value)
    assert "include_directory" in message
    assert "-bad" in message


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
def test_build_ast_payload_renders_tree_and_filters_external_children() -> None:
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

    output = clang_ast.build_ast_payload(
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
    assert "tokens=[]" in output
    assert "└─ CALL_EXPR spelling=PyModule_Create2 type=PyObject * tokens=[]" in output
    assert "   ├─ UNEXPOSED_EXPR type=PyObject * tokens=[]" in output
    assert "   │  └─ DECL_REF_EXPR spelling=PyModule_Create2 type=PyObject * tokens=[]" in output
    assert "   ├─ PAREN_EXPR type=PyModuleDef * tokens=[]" in output
    assert "   │  └─ UNARY_OPERATOR type=PyModuleDef * tokens=[]" in output
    assert "   │     └─ DECL_REF_EXPR spelling=defs type=PyModuleDef tokens=[]" in output
    assert "   └─ INTEGER_LITERAL type=int literal=3 tokens=[\"3\"]" in output
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

    assert f"- [WARNING] {source_path}:12:8: unused value" in output


def test_format_cursor_line_appends_empty_tokens_for_spelling_only_cursor() -> None:
    cursor = _FakeCursor(
        kind=clang.cindex.CursorKind.DECL_REF_EXPR,
        spelling="PyModule_Create2",
        type_spelling="PyObject *",
    )

    output = clang_ast._format_cursor_line(cursor)

    assert output == "DECL_REF_EXPR spelling=PyModule_Create2 type=PyObject * tokens=[]"


def test_format_cursor_line_renders_tokens_without_spelling() -> None:
    cursor = _FakeCursor(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        type_spelling="int",
        tokens=[
            _FakeToken(kind=clang.cindex.TokenKind.PUNCTUATION, spelling="-"),
            _FakeToken(kind=clang.cindex.TokenKind.LITERAL, spelling="1"),
        ],
    )

    output = clang_ast._format_cursor_line(cursor)

    assert output == "UNARY_OPERATOR type=int tokens=[\"-\", \"1\"]"


def test_format_cursor_line_escapes_token_spellings() -> None:
    cursor = _FakeCursor(
        kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
        type_spelling="const char *",
        tokens=[
            _FakeToken(kind=clang.cindex.TokenKind.LITERAL, spelling='"a"'),
            _FakeToken(kind=clang.cindex.TokenKind.IDENTIFIER, spelling=r"C:\tmp"),
        ],
    )

    output = clang_ast._format_cursor_line(cursor)

    assert output == (
        "UNEXPOSED_EXPR type=const char * "
        "tokens=[\"\\\"a\\\"\", \"C:\\\\tmp\"]"
    )


def test_run_clang_ast_dump_returns_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_command: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured_command.extend(command)
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(command, 0, stdout="AST\n", stderr="")

    monkeypatch.setattr(clang_ast.subprocess, "run", fake_run)

    output = clang_ast.run_clang_ast_dump(
        source_path=Path("C:/project/sample.c"),
        clang_args=["--std", "c11"],
    )

    assert output == clang_ast.ClangAstDumpResult(stdout="AST\n", stderr="", returncode=0)
    assert captured_command[:4] == ["clang", "-Xclang", "-ast-dump-all", "-fsyntax-only"]


def test_run_ast_export_writes_clang_stdout_even_when_clang_returns_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_path = tmp_path / "sample.c"
    output_dir = tmp_path / "ast_output"
    source_path.write_text("int sample(void) { return 0; }\n", encoding="utf-8")

    monkeypatch.setattr(clang_ast, "DEFAULT_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(clang_ast, "parse_translation_unit", lambda *args, **kwargs: SimpleNamespace(cursor=object(), diagnostics=[]))
    monkeypatch.setattr(clang_ast, "build_ast_payload", lambda *args, **kwargs: "libclang payload")
    monkeypatch.setattr(
        clang_ast,
        "run_clang_ast_dump",
        lambda *args, **kwargs: clang_ast.ClangAstDumpResult(
            stdout="partial ast\n",
            stderr="syntax error",
            returncode=1,
        ),
    )

    errors = clang_ast.run_ast_export(
        source_path=source_path,
        include=[],
        include_directory=[],
        c_std=None,
        cpp_std=None,
        clang_library_path=None,
    )
    libclang_path, clang_path = clang_ast.resolve_output_paths(source_path.resolve())
    captured = capsys.readouterr()

    assert len(errors) == 1
    assert libclang_path.read_text(encoding="utf-8") == "libclang payload\n"
    assert clang_path.read_text(encoding="utf-8") == "partial ast\n"
    assert "clang AST export failed: syntax error" in captured.err


def test_run_ast_export_partial_success_when_clang_export_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_path = tmp_path / "sample.c"
    output_dir = tmp_path / "ast_output"
    source_path.write_text("int sample(void) { return 0; }\n", encoding="utf-8")

    monkeypatch.setattr(clang_ast, "DEFAULT_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(clang_ast, "parse_translation_unit", lambda *args, **kwargs: SimpleNamespace(cursor=object(), diagnostics=[]))
    monkeypatch.setattr(clang_ast, "build_ast_payload", lambda *args, **kwargs: "libclang payload")

    def raise_clang_failure(*args: object, **kwargs: object) -> clang_ast.ClangAstDumpResult:
        raise RuntimeError("clang missing")

    monkeypatch.setattr(clang_ast, "run_clang_ast_dump", raise_clang_failure)

    errors = clang_ast.run_ast_export(
        source_path=source_path,
        include=[],
        include_directory=[],
        c_std=None,
        cpp_std=None,
        clang_library_path=None,
    )
    libclang_path, clang_path = clang_ast.resolve_output_paths(source_path.resolve())
    captured = capsys.readouterr()

    assert len(errors) == 1
    assert libclang_path.read_text(encoding="utf-8") == "libclang payload\n"
    assert not clang_path.exists()
    assert "clang AST export failed: clang missing" in captured.err


def test_run_ast_export_partial_success_when_libclang_export_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_path = tmp_path / "sample.c"
    output_dir = tmp_path / "ast_output"
    source_path.write_text("int sample(void) { return 0; }\n", encoding="utf-8")

    monkeypatch.setattr(clang_ast, "DEFAULT_OUTPUT_DIR", output_dir)
    def raise_libclang_failure(*args: object, **kwargs: object) -> SimpleNamespace:
        raise RuntimeError("libclang unavailable")

    monkeypatch.setattr(clang_ast, "parse_translation_unit", raise_libclang_failure)
    monkeypatch.setattr(
        clang_ast,
        "run_clang_ast_dump",
        lambda *args, **kwargs: clang_ast.ClangAstDumpResult(stdout="clang payload", stderr="", returncode=0),
    )

    errors = clang_ast.run_ast_export(
        source_path=source_path,
        include=[],
        include_directory=[],
        c_std=None,
        cpp_std=None,
        clang_library_path=None,
    )
    libclang_path, clang_path = clang_ast.resolve_output_paths(source_path.resolve())
    captured = capsys.readouterr()

    assert len(errors) == 1
    assert not libclang_path.exists()
    assert clang_path.read_text(encoding="utf-8") == "clang payload\n"
    assert "libclang AST export failed: libclang unavailable" in captured.err


def test_run_ast_export_writes_both_outputs_when_exports_succeed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "sample.c"
    output_dir = tmp_path / "ast_output"
    source_path.write_text("int sample(void) { return 0; }\n", encoding="utf-8")

    monkeypatch.setattr(clang_ast, "DEFAULT_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(clang_ast, "parse_translation_unit", lambda *args, **kwargs: SimpleNamespace(cursor=object(), diagnostics=[]))
    monkeypatch.setattr(clang_ast, "build_ast_payload", lambda *args, **kwargs: "libclang payload")
    monkeypatch.setattr(
        clang_ast,
        "run_clang_ast_dump",
        lambda *args, **kwargs: clang_ast.ClangAstDumpResult(stdout="clang payload\n", stderr="", returncode=0),
    )

    errors = clang_ast.run_ast_export(
        source_path=source_path,
        include=[],
        include_directory=[],
        c_std=None,
        cpp_std=None,
        clang_library_path=None,
    )
    libclang_path, clang_path = clang_ast.resolve_output_paths(source_path.resolve())

    assert errors == []
    assert libclang_path.read_text(encoding="utf-8") == "libclang payload\n"
    assert clang_path.read_text(encoding="utf-8") == "clang payload\n"
