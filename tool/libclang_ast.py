from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import TypedDict

from clang.cindex import Cursor, CursorKind, Diagnostic, TranslationUnit

import clang.cindex as clang_cindex


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / SCRIPT_PATH.stem
DEFAULT_OUTPUT_EXTENSION = ".ast.json"

EXIT_OK = 0


class SerializedDiagnostic(TypedDict):
    severity: int
    severity_name: str | None
    spelling: str | None


class SerializedCursor(TypedDict):
    kind: str
    spelling: str | None
    displayname: str | None
    type_spelling: str | None
    children: list[SerializedCursor]


class AstPayload(TypedDict):
    source_file: str
    output_file: str
    parse_args: list[str]
    diagnostics: list[SerializedDiagnostic]
    ast: SerializedCursor


def configure_output_encoding() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            pass


def _normalize_clang_include_tokens(argv: Sequence[str] | None) -> list[str] | None:
    if argv is None:
        return None

    normalized: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token != "--clang-include":
            normalized.append(token)
            index += 1
            continue

        if index + 1 >= len(argv):
            normalized.append(token)
            break

        normalized.append(f"--clang-include={argv[index + 1]}")
        index += 2

    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 libclang 解析单个 C/C++ 源文件，并将完整 AST 导出为 JSON。"
    )
    parser.add_argument("source_path", help="待解析的 C/C++ 源文件路径。")
    parser.add_argument(
        "--output",
        help="输出 JSON 文件路径；若传入目录，则自动生成 `<source_stem>.ast.json`。",
    )
    parser.add_argument(
        "--clang-include",
        action="append",
        default=[],
        help="追加 include 路径（不要带 -I 前缀），可重复传入。",
    )
    parser.add_argument("--clang-c-std", help="指定 C 标准（如 c11 或 -std=c11）。")
    parser.add_argument("--clang-cpp-std", help="指定 C++ 标准（如 c++17 或 -std=c++17）。")
    parser.add_argument(
        "--clang-library-path",
        help="显式指定 libclang 动态库路径。",
    )
    return parser.parse_args(_normalize_clang_include_tokens(argv))


def _safe_str(value: str | None) -> str | None:
    if value is None:
        return None
    return value if value else None


def _kind_name(kind: CursorKind) -> str:
    if kind.name:
        return kind.name
    text = str(kind)
    return text.rsplit(".", 1)[-1]


def _cursor_file_path(cursor: Cursor) -> Path | None:
    location = getattr(cursor, "location", None)
    file = getattr(location, "file", None)
    if file is None:
        return None

    try:
        return Path(str(file)).resolve()
    except OSError:
        return Path(str(file))


DEFAULT_CLANG_C_STD = "c11"
DEFAULT_CLANG_CPP_STD = "c++17"
CPP_SOURCE_SUFFIXES = {".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx"}


def _normalize_std_arg(std_value: str | None) -> str | None:
    if std_value is None:
        return None
    normalized = std_value.strip()
    if not normalized:
        return None
    if normalized.startswith("-std="):
        return normalized
    return f"-std={normalized}"


def _build_include_args(include_paths: Sequence[str]) -> list[str]:
    include_args: list[str] = []
    for raw_path in include_paths:
        if raw_path is None:
            raise TypeError("clang_include entries must be non-empty include paths")
        include_path = str(raw_path).strip()
        if not include_path:
            raise ValueError("clang_include entries must be non-empty include paths")
        if include_path.startswith("-I"):
            raise ValueError("clang_include entries must not include '-I' prefix")
        if include_path.startswith("-"):
            raise ValueError(f"clang_include entry must be a path, got option-like value: {include_path!r}")
        include_arg = f"-I{include_path}"
        if include_arg not in include_args:
            include_args.append(include_arg)
    return include_args


def build_clang_args(
    *,
    source_path: Path,
    include_paths: Sequence[str],
    clang_c_std: str | None,
    clang_cpp_std: str | None,
) -> list[str]:
    suffix = source_path.suffix.lower()
    if suffix in CPP_SOURCE_SUFFIXES:
        std_arg = _normalize_std_arg(clang_cpp_std or DEFAULT_CLANG_CPP_STD)
    else:
        std_arg = _normalize_std_arg(clang_c_std or DEFAULT_CLANG_C_STD)

    clang_args: list[str] = []
    if std_arg is not None:
        clang_args.append(std_arg)
    clang_args.extend(_build_include_args(include_paths))
    return clang_args


def resolve_output_path(source_path: Path, output_arg: str | None) -> Path:
    if output_arg is None:
        return DEFAULT_OUTPUT_DIR / f"{source_path.stem}{DEFAULT_OUTPUT_EXTENSION}"

    output_path = Path(output_arg)
    if output_path.exists() and output_path.is_dir():
        return output_path / f"{source_path.stem}{DEFAULT_OUTPUT_EXTENSION}"
    if output_path.suffix.lower() == ".json":
        return output_path
    return output_path / f"{source_path.stem}{DEFAULT_OUTPUT_EXTENSION}"

def _serialize_diagnostic(diagnostic: Diagnostic) -> SerializedDiagnostic:
    return {
        "severity": diagnostic.severity,
        "severity_name": _diagnostic_severity_name(diagnostic.severity),
        "spelling": _safe_str(diagnostic.spelling),
    }


def _serialize_cursor(cursor: Cursor, *, source_path: Path) -> SerializedCursor | None:
    cursor_path = _cursor_file_path(cursor)
    if cursor_path is not None and cursor_path != source_path:
        return None

    type_spelling = _safe_str(cursor.type.spelling)
    children = [
        serialized_child
        for child in cursor.get_children()
        if (serialized_child := _serialize_cursor(child, source_path=source_path)) is not None
    ]
    serialized: SerializedCursor = {
        "kind": _kind_name(cursor.kind),
        "spelling": _safe_str(cursor.spelling),
        "displayname": _safe_str(cursor.displayname),
        "type_spelling": type_spelling,
        "children": children,
    }
    return serialized


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


def initialize_clang(library_path: str | None) -> ModuleType:
    if library_path and not clang_cindex.Config.loaded:
        clang_cindex.Config.set_library_file(library_path)
    return clang_cindex


def parse_translation_unit(
    source_path: Path,
    *,
    library_path: str | None,
    clang_args: Sequence[str],
) -> TranslationUnit:
    cindex = initialize_clang(library_path)
    index = cindex.Index.create()
    return index.parse(str(source_path), args=list(clang_args), options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)


def build_ast_payload(
    translation_unit: TranslationUnit,
    *,
    source_path: Path,
    output_path: Path,
    clang_args: Sequence[str],
) -> AstPayload:
    diagnostics = [_serialize_diagnostic(diagnostic) for diagnostic in translation_unit.diagnostics]
    ast = _serialize_cursor(translation_unit.cursor, source_path=source_path)
    if ast is None:
        raise RuntimeError(f"Failed to serialize root cursor for source file: {source_path}")
    return {
        "source_file": str(source_path.resolve()),
        "output_file": str(output_path.resolve()),
        "parse_args": [str(arg) for arg in clang_args],
        "diagnostics": diagnostics,
        "ast": ast,
    }


def main(argv: Sequence[str] | None = None) -> int:
    configure_output_encoding()
    args = parse_args(argv)

    source_path = Path(args.source_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    output_path = resolve_output_path(source_path, args.output).resolve()
    library_path: str | None = _safe_str(args.clang_library_path)
    clang_include = [str(path) for path in args.clang_include]
    clang_c_std: str | None = _safe_str(args.clang_c_std)
    clang_cpp_std: str | None = _safe_str(args.clang_cpp_std)
    clang_args = build_clang_args(
        source_path=source_path,
        include_paths=clang_include,
        clang_c_std=clang_c_std,
        clang_cpp_std=clang_cpp_std,
    )

    initialize_clang(library_path)
    translation_unit = parse_translation_unit(
        source_path,
        library_path=library_path,
        clang_args=clang_args,
    )

    payload = build_ast_payload(
        translation_unit,
        source_path=source_path,
        output_path=output_path,
        clang_args=clang_args,
    )
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_text + "\n", encoding="utf-8")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
