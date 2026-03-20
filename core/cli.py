from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from . import write_stubs
from .stub_generation_options import StubGenerationOptions

EXIT_OK = 0
EXIT_ERROR = 1


def _regex(pattern_str: str) -> re.Pattern:
    try:
        return re.compile(pattern_str)
    except re.error as ex:
        raise argparse.ArgumentTypeError(f"Invalid REGEX pattern: {ex}") from ex


def _regex_colon_path(regex_path: str) -> tuple[re.Pattern, str]:
    if ":" not in regex_path:
        raise argparse.ArgumentTypeError(
            "Invalid enum class location, expected REGEX:PATH format"
        )

    pattern_str, path = regex_path.rsplit(":", maxsplit=1)
    if any(not part.isidentifier() for part in path.split(".")):
        raise argparse.ArgumentTypeError(f"Invalid PATH: {path}")
    return _regex(pattern_str), path


def _normalize_clang_include_directory(include_paths: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for raw_path in include_paths:
        if raw_path is None:
            raise TypeError("clang_include_directory entries must be non-empty include paths")

        include_path = str(raw_path).strip()
        if not include_path:
            raise ValueError("clang_include_directory entries must be non-empty include paths")
        if include_path.startswith("-"):
            raise ValueError(
                f"clang_include_directory entry must be a path, got option-like value: {include_path!r}"
            )
        if include_path not in normalized:
            normalized.append(include_path)
    return normalized


def _normalize_clang_include(includes: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for raw_include in includes:
        if raw_include is None:
            raise TypeError("clang_include entries must be non-empty include headers")

        include = str(raw_include).strip()
        if not include:
            raise ValueError("clang_include entries must be non-empty include headers")
        if include.startswith("-"):
            raise ValueError(
                f"clang_include entry must be a header, got option-like value: {include!r}"
            )
        if include not in normalized:
            normalized.append(include)
    return normalized


def _normalize_source_root(raw_source_root: str | None) -> Path | None:
    if raw_source_root is None:
        return None

    source_root = str(raw_source_root).strip()
    if not source_root:
        return None
    return Path(source_root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcstubgen",
        description="Generate Python stubs for a module with pcstubgen.",
        allow_abbrev=False,
    )
    parser.add_argument("module_name", metavar="MODULE_NAME", help="module name")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="./stubs",
        help="The root directory for output stubs",
    )

    parser.add_argument(
        "--ignore-invalid-expressions",
        metavar="REGEX",
        default=None,
        type=_regex,
        help="Ignore invalid expressions matching REGEX",
    )
    parser.add_argument(
        "--ignore-all-errors",
        default=False,
        action="store_true",
        help="Ignore all errors during module parsing",
    )
    parser.add_argument(
        "--enum-class-locations",
        dest="enum_class_locations",
        metavar="REGEX:LOC",
        action="append",
        default=[],
        type=_regex_colon_path,
        help="Locations of enum classes in <enum-class-name-regex>:<path-to-class> format",
    )

    parser.add_argument(
        "--no-docstring-signature-parser",
        action="store_false",
        dest="enable_docstring_signature_parser",
        default=True,
        help="Disable parsing signatures from docstrings",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="C/C++ source root used for C signature inference",
    )
    parser.add_argument(
        "--clang-include",
        action="append",
        default=[],
        help="Additional clang include header, can be repeated",
    )
    parser.add_argument(
        "--clang-include-directory",
        action="append",
        default=[],
        help="Additional clang include directory path, can be repeated",
    )
    parser.add_argument(
        "--clang-c-std",
        default=None,
        help="C standard passed to clang, e.g. c11",
    )
    parser.add_argument(
        "--clang-cpp-std",
        default=None,
        help="C++ standard passed to clang, e.g. c++17",
    )

    parser.add_argument(
        "--print-invalid-expressions-as-is",
        default=False,
        action="store_true",
        help="Print invalid expressions as-is instead of replacing with ...",
    )
    parser.add_argument(
        "--no-docstrings",
        action="store_false",
        dest="include_docstrings",
        default=True,
        help="Do not include docstrings in generated stubs",
    )
    parser.add_argument(
        "--include-module-type-comment",
        default=False,
        action="store_true",
        help="Include module type comment in generated stubs",
    )
    parser.add_argument(
        "--stub-extension",
        type=str,
        default="pyi",
        metavar="EXT",
        choices=["pyi", "py"],
        help="The extension of generated stubs: pyi (default) or py",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        args.clang_include = _normalize_clang_include(args.clang_include)
        args.clang_include_directory = _normalize_clang_include_directory(args.clang_include_directory)
    except (TypeError, ValueError) as ex:
        parser.error(str(ex))

    return args


def _build_options(args: argparse.Namespace) -> StubGenerationOptions:
    default_options = StubGenerationOptions()
    source_root = _normalize_source_root(args.source_root)

    return StubGenerationOptions(
        ignore_invalid_expressions=args.ignore_invalid_expressions,
        ignore_all_errors=args.ignore_all_errors,
        enum_class_locations=list(args.enum_class_locations),
        enable_docstring_signature_parser=args.enable_docstring_signature_parser,
        source_root=source_root,
        clang_c_std=args.clang_c_std or default_options.clang_c_std,
        clang_cpp_std=args.clang_cpp_std or default_options.clang_cpp_std,
        clang_include=list(args.clang_include),
        clang_include_directory=list(args.clang_include_directory),
        print_invalid_expressions_as_is=args.print_invalid_expressions_as_is,
        include_docstrings=args.include_docstrings,
        include_module_type_comment=args.include_module_type_comment,
        stub_extension=args.stub_extension,
    )


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)

    write_stubs(
        module_name=args.module_name,
        output_dir=Path(args.output_dir),
        options=_build_options(args),
    )


if __name__ == "__main__":
    main()
