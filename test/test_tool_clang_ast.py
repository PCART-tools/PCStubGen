from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import clang.cindex
import pytest
from typer.testing import CliRunner

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tool import clang_ast

RUNNER = CliRunner()


def test_cli_accepts_clang_include_and_include_directory(
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
            "--clang-include",
            "Python.h",
            "--clang-include=numpy/arrayobject.h",
            "--clang-include-directory",
            "C:/IncludeA",
            "--clang-include-directory=C:/IncludeB",
        ],
        prog_name="clang_ast",
    )

    assert result.exit_code == 0
    assert captured_kwargs["source_path"] == source_path.resolve()
    assert captured_kwargs["clang_include"] == ["Python.h", "numpy/arrayobject.h"]
    assert captured_kwargs["clang_include_directory"] == [
        Path("C:/IncludeA"),
        Path("C:/IncludeB"),
    ]


def test_cli_rejects_removed_output_option(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.c"
    source_path.write_text("int sample(void) { return 0; }\n", encoding="utf-8")

    result = RUNNER.invoke(
        clang_ast.app,
        [str(source_path), "--output", "out.txt"],
        prog_name="clang_ast",
    )

    assert result.exit_code == 2
    assert "No such option: --output" in result.stderr


def test_cli_help_contains_chinese_text() -> None:
    result = RUNNER.invoke(clang_ast.app, ["--help"], prog_name="clang_ast")

    assert result.exit_code == 0
    assert "使用 libclang 和 clang 导出单个 C/C++ 源文件的 AST 文本。" in result.stdout
    assert "--clang-include" in result.stdout
    assert "追加 include 头文件，可重复传入。" in result.stdout


def test_cli_invalid_clang_include_reports_bad_parameter(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.c"
    source_path.write_text("int sample(void) { return 0; }\n", encoding="utf-8")

    result = RUNNER.invoke(
        clang_ast.app,
        [str(source_path), "--clang-include=-bad"],
        prog_name="clang_ast",
    )

    assert result.exit_code == 2
    assert "Invalid value for '--clang-include'" in result.stderr
    assert "'-bad'" in result.stderr


def test_cli_invalid_clang_include_directory_reports_bad_parameter(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.c"
    source_path.write_text("int sample(void) { return 0; }\n", encoding="utf-8")

    result = RUNNER.invoke(
        clang_ast.app,
        [str(source_path), "--clang-include-directory=-bad"],
        prog_name="clang_ast",
    )

    assert result.exit_code == 2
    assert "Invalid value for '--clang-include-directory'" in result.stderr
    assert "'-bad'" in result.stderr


def test_cli_missing_source_path_reports_typer_path_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.c"

    result = RUNNER.invoke(
        clang_ast.app,
        [str(missing_path)],
        prog_name="clang_ast",
    )

    assert result.exit_code == 2
    assert "SOURCE_PATH" in result.stderr
    assert "does not exist" in result.stderr


def test_build_parse_args_places_include_before_include_directory() -> None:
    parse_args = clang_ast._build_parse_args(
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
        clang_ast._normalize_include_headers(["-Winvalid"])


def test_validate_clang_include_preserves_message() -> None:
    with pytest.raises(clang_ast.typer.BadParameter) as ex:
        clang_ast._validate_clang_include(["-bad"])

    assert str(ex.value) == "clang_include entry must be a header, got option-like value: '-bad'"


def test_validate_clang_include_directory_preserves_message() -> None:
    with pytest.raises(clang_ast.typer.BadParameter) as ex:
        clang_ast._validate_clang_include_directory([Path("-bad")])

    assert str(ex.value) == "clang_include_directory entry must be a path, got option-like value: '-bad'"


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


def test_resolve_output_paths_follow_dynamic_script_name() -> None:
    source_path = Path("sample.c")
    libclang_path, clang_path = clang_ast.resolve_output_paths(source_path)

    assert clang_ast.DEFAULT_OUTPUT_DIR == clang_ast.SCRIPT_DIR / f"{clang_ast.SCRIPT_PATH.stem}_output"
    assert libclang_path == clang_ast.DEFAULT_OUTPUT_DIR / "sample.libclang.txt"
    assert clang_path == clang_ast.DEFAULT_OUTPUT_DIR / "sample.clang.txt"


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


def test_build_clang_ast_dump_command_uses_expected_flags() -> None:
    source_path = Path("C:/project/sample.c")
    command = clang_ast.build_clang_ast_dump_command(
        source_path=source_path,
        clang_args=["--std", "c11", "--include", "Python.h"],
    )

    assert command == [
        "clang",
        "-Xclang",
        "-ast-dump-all",
        "-fsyntax-only",
        "--std",
        "c11",
        "--include",
        "Python.h",
        str(source_path),
    ]


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
        clang_include=[],
        clang_include_directory=[],
        clang_c_std=None,
        clang_cpp_std=None,
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
        clang_include=[],
        clang_include_directory=[],
        clang_c_std=None,
        clang_cpp_std=None,
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
        clang_include=[],
        clang_include_directory=[],
        clang_c_std=None,
        clang_cpp_std=None,
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
        clang_include=[],
        clang_include_directory=[],
        clang_c_std=None,
        clang_cpp_std=None,
        clang_library_path=None,
    )
    libclang_path, clang_path = clang_ast.resolve_output_paths(source_path.resolve())

    assert errors == []
    assert libclang_path.read_text(encoding="utf-8") == "libclang payload\n"
    assert clang_path.read_text(encoding="utf-8") == "clang payload\n"
