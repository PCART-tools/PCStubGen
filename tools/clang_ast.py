from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import typer
from clang.cindex import Cursor, CursorKind, Diagnostic, TranslationUnit

import clang.cindex as clang_cindex


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / f"{SCRIPT_PATH.stem}_output"
LIBCLANG_OUTPUT_EXTENSION = ".libclang.txt"
CLANG_OUTPUT_EXTENSION = ".clang.txt"

EXIT_ERROR = 1

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class ClangAstDumpResult:
    stdout: str
    stderr: str
    returncode: int


def _safe_str(value: str | None) -> str | None:
    if value is None:
        return None
    return value if value else None


def _kind_name(kind: CursorKind) -> str:
    if kind.name:
        return kind.name
    text = str(kind)
    return text.rsplit(".", 1)[-1]


def _normalize_path_for_compare(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


CPP_SOURCE_SUFFIXES = {".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx"}


def _normalize_std_value(std_value: str | None) -> str | None:
    if std_value is None:
        return None
    normalized = std_value.strip()
    if not normalized:
        return None
    if normalized.startswith("--std="):
        return normalized.partition("=")[2]
    if normalized.startswith("-std="):
        return normalized.partition("=")[2]
    return normalized


def _normalize_include_paths(include_paths: Sequence[str]) -> list[str]:
    normalized_paths: list[str] = []
    for raw_path in include_paths:
        if raw_path is None:
            raise TypeError("include_directory entries must be non-empty include paths")
        include_path = str(raw_path).strip()
        if not include_path:
            raise ValueError("include_directory entries must be non-empty include paths")
        if include_path.startswith("-"):
            raise ValueError(f"include_directory entry must be a path, got option-like value: {include_path!r}")
        if include_path not in normalized_paths:
            normalized_paths.append(include_path)
    return normalized_paths


def _normalize_include_headers(include_headers: Sequence[str]) -> list[str]:
    normalized_headers: list[str] = []
    for raw_header in include_headers:
        if raw_header is None:
            raise TypeError("include entries must be non-empty include headers")
        include_header = str(raw_header).strip()
        if not include_header:
            raise ValueError("include entries must be non-empty include headers")
        if include_header.startswith("-"):
            raise ValueError(f"include entry must be a header, got option-like value: {include_header!r}")
        if include_header not in normalized_headers:
            normalized_headers.append(include_header)
    return normalized_headers


def _validate_include(value: list[str] | None) -> list[str]:
    values = [] if value is None else list(value)
    try:
        _normalize_include_headers(values)
    except (TypeError, ValueError) as ex:
        raise typer.BadParameter(str(ex)) from ex
    return values


def _validate_include_directory(value: list[Path] | None) -> list[Path]:
    values = [] if value is None else list(value)
    try:
        _normalize_include_paths([str(path) for path in values])
    except (TypeError, ValueError) as ex:
        raise typer.BadParameter(str(ex)) from ex
    return values


def build_clang_args(
    *,
    source_path: Path,
) -> list[str]:
    _ = source_path
    clang_args: list[str] = []
    return clang_args


def _resolve_std_value_for_source(
    *,
    source_path: Path,
    c_std: str | None,
    cpp_std: str | None,
) -> str:
    suffix = source_path.suffix.lower()
    if suffix in CPP_SOURCE_SUFFIXES:
        return _normalize_std_value(cpp_std or "c++17") or "c++17"
    return _normalize_std_value(c_std or "c11") or "c11"


def _build_parse_args(
    *,
    source_path: Path,
    clang_args: Sequence[str],
    include_headers: Sequence[str],
    include_paths: Sequence[str],
    c_std: str | None,
    cpp_std: str | None,
) -> list[str]:
    parse_args = list(clang_args)
    std_value = _resolve_std_value_for_source(
        source_path=source_path,
        c_std=c_std,
        cpp_std=cpp_std,
    )
    parse_args.extend(["--std", std_value])
    for include_header in include_headers:
        parse_args.extend(["--include", include_header])
    for include_path in include_paths:
        parse_args.extend(["--include-directory", include_path])
    return parse_args


def resolve_output_paths(source_path: Path) -> tuple[Path, Path]:
    return (
        DEFAULT_OUTPUT_DIR / f"{source_path.stem}{LIBCLANG_OUTPUT_EXTENSION}",
        DEFAULT_OUTPUT_DIR / f"{source_path.stem}{CLANG_OUTPUT_EXTENSION}",
    )


def _diagnostic_severity_name(severity: int) -> str:
    diagnostic_type = clang_cindex.Diagnostic
    mapping = {
        diagnostic_type.Ignored: "IGNORED",
        diagnostic_type.Note: "NOTE",
        diagnostic_type.Warning: "WARNING",
        diagnostic_type.Error: "ERROR",
        diagnostic_type.Fatal: "FATAL",
    }
    return mapping.get(severity, str(severity))


def _format_string_list(values: Sequence[str]) -> str:
    escaped_values = [str(value).replace("\\", "\\\\").replace('"', '\\"') for value in values]
    return "[" + ", ".join(f'"{value}"' for value in escaped_values) + "]"


def _format_diagnostic(diagnostic: Diagnostic) -> str:
    severity_name = _diagnostic_severity_name(diagnostic.severity)
    message = _safe_str(diagnostic.spelling) or "<unknown>"
    location = diagnostic.location
    file = location.file
    if file is None:
        return f"[{severity_name}] {message}"
    return f"[{severity_name}] {file}:{location.line}:{location.column}: {message}"


def _is_cursor_from_source(cursor: Cursor, *, normalized_source_path: str) -> bool:
    location = getattr(cursor, "location", None)
    file = getattr(location, "file", None)
    if file is None:
        return True
    cursor_path = str(file)
    if not cursor_path:
        return True
    return _normalize_path_for_compare(cursor_path) == normalized_source_path


def _cursor_type_spelling(cursor: Cursor) -> str | None:
    cursor_type = cursor.type
    if cursor_type is None:
        return None
    return _safe_str(cursor_type.spelling)


def _cursor_spelling_for_display(cursor: Cursor) -> str | None:
    spelling = _safe_str(cursor.spelling)
    if spelling is not None:
        return spelling
    return _safe_str(cursor.displayname)


def _cursor_literal_text(cursor: Cursor) -> str | None:
    kind_name = _kind_name(cursor.kind)
    if kind_name not in {"INTEGER_LITERAL", "STRING_LITERAL"}:
        return None
    if _safe_str(cursor.spelling) is not None:
        return None

    literal_kind = clang_cindex.TokenKind.LITERAL
    literal_tokens = [str(token.spelling) for token in cursor.get_tokens() if token.kind == literal_kind]
    if not literal_tokens:
        return None
    if kind_name == "INTEGER_LITERAL":
        return literal_tokens[0]
    return " ".join(literal_tokens)


def _format_cursor_tokens(cursor: Cursor) -> str:
    token_spellings = [str(token.spelling) for token in cursor.get_tokens()]
    return f"tokens={_format_string_list(token_spellings)}"


def _format_cursor_line(cursor: Cursor) -> str:
    parts = [_kind_name(cursor.kind)]
    spelling = _cursor_spelling_for_display(cursor)
    if spelling is not None:
        parts.append(f"spelling={spelling}")
    type_spelling = _cursor_type_spelling(cursor)
    if type_spelling is not None:
        parts.append(f"type={type_spelling}")
    literal = _cursor_literal_text(cursor)
    if literal is not None:
        parts.append(f"literal={literal}")
    parts.append(_format_cursor_tokens(cursor))
    return " ".join(parts)


def _render_cursor_subtree(
    cursor: Cursor,
    *,
    normalized_source_path: str,
    prefix: str,
    is_last: bool,
) -> list[str]:
    if not _is_cursor_from_source(cursor, normalized_source_path=normalized_source_path):
        return []

    connector = "└─ " if is_last else "├─ "
    child_prefix = prefix + ("   " if is_last else "│  ")
    lines = [f"{prefix}{connector}{_format_cursor_line(cursor)}"]
    children = [
        child
        for child in cursor.get_children()
        if _is_cursor_from_source(child, normalized_source_path=normalized_source_path)
    ]
    for index, child in enumerate(children):
        lines.extend(
            _render_cursor_subtree(
                child,
                normalized_source_path=normalized_source_path,
                prefix=child_prefix,
                is_last=index == len(children) - 1,
            )
        )
    return lines


def _render_cursor_tree(cursor: Cursor, *, source_path: Path) -> list[str]:
    normalized_source_path = _normalize_path_for_compare(str(source_path))
    if not _is_cursor_from_source(cursor, normalized_source_path=normalized_source_path):
        return []

    lines = [_format_cursor_line(cursor)]
    children = [
        child
        for child in cursor.get_children()
        if _is_cursor_from_source(child, normalized_source_path=normalized_source_path)
    ]
    for index, child in enumerate(children):
        lines.extend(
            _render_cursor_subtree(
                child,
                normalized_source_path=normalized_source_path,
                prefix="",
                is_last=index == len(children) - 1,
            )
        )
    return lines


def initialize_clang(library_path: str | None) -> ModuleType:
    if library_path and not clang_cindex.Config.loaded:
        clang_cindex.Config.set_library_file(library_path)
    return clang_cindex


def parse_translation_unit(
    source_path: Path,
    *,
    library_path: str | None,
    parse_args: Sequence[str],
) -> TranslationUnit:
    cindex = initialize_clang(library_path)
    index = cindex.Index.create()
    return index.parse(str(source_path), args=list(parse_args), options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)


def build_ast_payload(
    translation_unit: TranslationUnit,
    *,
    source_path: Path,
    output_path: Path,
    clang_args: Sequence[str],
) -> str:
    ast_lines = _render_cursor_tree(
        translation_unit.cursor,
        source_path=source_path,
    )
    if not ast_lines:
        raise RuntimeError(f"Failed to serialize root cursor for source file: {source_path}")
    diagnostic_lines = [_format_diagnostic(diagnostic) for diagnostic in translation_unit.diagnostics]
    if not diagnostic_lines:
        diagnostic_lines = ["<none>"]
    lines = [
        f"source_file: {source_path.resolve()}",
        f"output_file: {output_path.resolve()}",
        f"parse_args: {_format_string_list([str(arg) for arg in clang_args])}",
        "diagnostics:",
    ]
    lines.extend(f"- {line}" for line in diagnostic_lines)
    lines.append("ast:")
    lines.extend(ast_lines)
    return "\n".join(lines)


def build_clang_ast_dump_command(*, source_path: Path, clang_args: Sequence[str]) -> list[str]:
    return [
        "clang",
        "-Xclang",
        "-ast-dump-all",
        "-fsyntax-only",
        *clang_args,
        str(source_path),
    ]


def run_clang_ast_dump(*, source_path: Path, clang_args: Sequence[str]) -> ClangAstDumpResult:
    command = build_clang_ast_dump_command(source_path=source_path, clang_args=clang_args)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return ClangAstDumpResult(
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )


def _write_output(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")


def _report_failure(label: str, error: Exception) -> None:
    print(f"{label} failed: {error}", file=sys.stderr)


def _build_clang_failure(result: ClangAstDumpResult) -> RuntimeError | None:
    if result.returncode == 0:
        return None
    stderr_text = result.stderr.strip()
    stdout_text = result.stdout.strip()
    details = stderr_text or stdout_text or f"clang exited with code {result.returncode}"
    return RuntimeError(details)


def run_ast_export(
    *,
    source_path: Path,
    include: Sequence[str],
    include_directory: Sequence[Path],
    c_std: str | None,
    cpp_std: str | None,
    clang_library_path: str | None,
) -> list[Exception]:
    """
    导出单个源文件的 libclang 与 clang AST，并返回执行过程中收集到的失败。
    """
    source_path = source_path.resolve()

    libclang_output_path, clang_output_path = resolve_output_paths(source_path)
    library_path = _safe_str(clang_library_path)
    normalized_include = _normalize_include_headers([str(header) for header in include])
    normalized_include_directory = _normalize_include_paths(
        [str(path) for path in include_directory]
    )
    normalized_c_std = _safe_str(c_std)
    normalized_cpp_std = _safe_str(cpp_std)
    clang_args = build_clang_args(
        source_path=source_path,
    )
    clang_parse_args = _build_parse_args(
        source_path=source_path,
        clang_args=clang_args,
        include_headers=normalized_include,
        include_paths=normalized_include_directory,
        c_std=normalized_c_std,
        cpp_std=normalized_cpp_std,
    )

    errors: list[Exception] = []

    try:
        translation_unit = parse_translation_unit(
            source_path,
            library_path=library_path,
            parse_args=clang_parse_args,
        )
        libclang_payload = build_ast_payload(
            translation_unit,
            source_path=source_path,
            output_path=libclang_output_path,
            clang_args=clang_parse_args,
        )
        _write_output(libclang_output_path, libclang_payload)
    except Exception as error:
        errors.append(error)
        _report_failure("libclang AST export", error)

    try:
        clang_result = run_clang_ast_dump(
            source_path=source_path,
            clang_args=clang_parse_args,
        )
        if clang_result.stdout:
            _write_output(clang_output_path, clang_result.stdout.rstrip("\n"))
        clang_error = _build_clang_failure(clang_result)
        if clang_error is not None:
            errors.append(clang_error)
            _report_failure("clang AST export", clang_error)
    except Exception as error:
        errors.append(error)
        _report_failure("clang AST export", error)

    return errors


@app.command(help="使用 libclang 和 clang 导出单个 C/C++ 源文件的 AST 文本。")
def command(
    source_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        metavar="SOURCE_PATH",
        help="待解析的 C/C++ 源文件路径。",
    ),
    include: list[str] | None = typer.Option(
        None,
        "--include",
        callback=_validate_include,
        help="追加 include 头文件，可重复传入。",
    ),
    include_directory: list[Path] | None = typer.Option(
        None,
        "--include-directory",
        callback=_validate_include_directory,
        help="追加 include 目录路径，可重复传入。",
    ),
    c_std: str | None = typer.Option(
        None,
        "--c-std",
        help="指定 C 标准（如 c11 或 -std=c11）。",
    ),
    cpp_std: str | None = typer.Option(
        None,
        "--cpp-std",
        help="指定 C++ 标准（如 c++17 或 -std=c++17）。",
    ),
    clang_library_path: str | None = typer.Option(
        None,
        "--clang-library-path",
        help="显式指定 libclang 动态库路径。",
    ),
) -> None:
    errors = run_ast_export(
        source_path=source_path,
        include=include or [],
        include_directory=include_directory or [],
        c_std=c_std,
        cpp_std=cpp_std,
        clang_library_path=clang_library_path,
    )
    if errors:
        raise typer.Exit(code=EXIT_ERROR)


if __name__ == "__main__":
    app()
