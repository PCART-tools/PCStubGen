from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from clang.cindex import Cursor, CursorKind, Diagnostic, SourceLocation, TranslationUnit
else:
    Cursor = Any
    CursorKind = Any
    Diagnostic = Any
    SourceLocation = Any
    TranslationUnit = Any

try:
    import clang.cindex as clang_cindex
except Exception as exc:  # pragma: no cover - exercised via init failure path
    clang_cindex = None
    _CLANG_IMPORT_ERROR: Exception | None = exc
else:
    _CLANG_IMPORT_ERROR = None


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config.toml"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / SCRIPT_PATH.stem
DEFAULT_OUTPUT_EXTENSION = ".ast.json"

EXIT_OK = 0
EXIT_INVALID_INPUT = 1
EXIT_CLANG_INIT_FAILED = 2
EXIT_PARSE_FAILED = 3


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
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _normalize_clang_arg_tokens(argv: Sequence[str] | None) -> list[str] | None:
    if argv is None:
        return None

    normalized: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token != "--clang-arg":
            normalized.append(token)
            index += 1
            continue

        if index + 1 >= len(argv):
            normalized.append(token)
            break

        normalized.append(f"--clang-arg={argv[index + 1]}")
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
        "--clang-arg",
        action="append",
        default=[],
        help="追加传给 libclang 的解析参数，可重复传入。",
    )
    parser.add_argument(
        "--clang-library-path",
        help="显式覆盖 config.toml 中的 clang_library_path。",
    )
    return parser.parse_args(_normalize_clang_arg_tokens(argv))


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


def load_clang_settings(
    config_path: Path = CONFIG_PATH,
    *,
    override_library_path: str | None = None,
    extra_parse_args: Sequence[str] = (),
) -> tuple[str | None, list[str]]:
    config_library_path: str | None = None
    config_parse_args: list[str] = []

    if config_path.exists():
        with config_path.open("rb") as file:
            config = tomllib.load(file)
        config_library_path = _safe_str(config.get("clang_library_path"))
        raw_parse_args = config.get("clang_parse_args", [])
        if isinstance(raw_parse_args, list):
            config_parse_args = [str(arg) for arg in raw_parse_args]

    library_path = override_library_path or config_library_path
    parse_args = [*config_parse_args, *[str(arg) for arg in extra_parse_args]]
    return library_path, parse_args


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
    return {
        "kind": _kind_name(cursor.kind),
        "spelling": _safe_str(cursor.spelling),
        "displayname": _safe_str(cursor.displayname),
        "type_spelling": type_spelling,
        "children": children,
    }


def _diagnostic_severity_name(severity: int) -> str:
    if clang_cindex is None:
        return str(severity)

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
    if clang_cindex is None:
        raise RuntimeError(f"Failed to import clang.cindex: {_CLANG_IMPORT_ERROR}")

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
    return index.parse(str(source_path), args=list(parse_args))


def build_ast_payload(
    translation_unit: TranslationUnit,
    *,
    source_path: Path,
    output_path: Path,
    parse_args: Sequence[str],
) -> AstPayload:
    diagnostics = [_serialize_diagnostic(diagnostic) for diagnostic in translation_unit.diagnostics]
    ast = _serialize_cursor(translation_unit.cursor, source_path=source_path)
    return {
        "source_file": str(source_path.resolve()),
        "output_file": str(output_path.resolve()),
        "parse_args": [str(arg) for arg in parse_args],
        "diagnostics": diagnostics,
        "ast": ast if ast is not None else _serialize_cursor(translation_unit.cursor, source_path=source_path),
    }


def main(argv: Sequence[str] | None = None, *, config_path: Path = CONFIG_PATH) -> int:
    configure_output_encoding()
    args = parse_args(argv)

    source_path = Path(args.source_path).resolve()
    if not source_path.is_file():
        return EXIT_INVALID_INPUT

    output_path = resolve_output_path(source_path, args.output).resolve()
    library_path, clang_parse_args = load_clang_settings(
        config_path,
        override_library_path=args.clang_library_path,
        extra_parse_args=args.clang_arg,
    )

    try:
        initialize_clang(library_path)
    except Exception:
        return EXIT_CLANG_INIT_FAILED

    try:
        translation_unit = parse_translation_unit(
            source_path,
            library_path=library_path,
            parse_args=clang_parse_args,
        )
    except Exception:
        return EXIT_PARSE_FAILED

    payload = build_ast_payload(
        translation_unit,
        source_path=source_path,
        output_path=output_path,
        parse_args=clang_parse_args,
    )
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_text + "\n", encoding="utf-8")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
