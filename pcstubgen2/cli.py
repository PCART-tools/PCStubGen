from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from . import write_stubs
from .StubGenerationOptions import StubGenerationOptions

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


def _normalize_clang_include(include_paths: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for raw_path in include_paths:
        if raw_path is None:
            raise TypeError("clang_include entries must be non-empty include paths")

        include_path = str(raw_path).strip()
        if not include_path:
            raise ValueError("clang_include entries must be non-empty include paths")
        if include_path.startswith("-I"):
            raise ValueError("clang_include entries must not include '-I' prefix")
        if include_path.startswith("-"):
            raise ValueError(
                f"clang_include entry must be a path, got option-like value: {include_path!r}"
            )
        if include_path not in normalized:
            normalized.append(include_path)
    return normalized


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcstubgen2",
        description="Generate Python stubs for a module with pcstubgen2.",
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
        "--enable-c-signature-inference",
        default=False,
        action="store_true",
        help="Enable C AST based signature inference",
    )
    parser.add_argument(
        "--c-source-root",
        default=None,
        help="C/C++ source root used for C signature inference",
    )
    parser.add_argument(
        "--clang-include",
        action="append",
        default=[],
        help="Additional clang include path (no -I prefix), can be repeated",
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

    if args.enable_c_signature_inference and args.c_source_root is None:
        parser.error(
            "--c-source-root is required when --enable-c-signature-inference is set"
        )

    try:
        args.clang_include = _normalize_clang_include(args.clang_include)
    except (TypeError, ValueError) as ex:
        parser.error(str(ex))

    return args


def _build_options(args: argparse.Namespace) -> StubGenerationOptions:
    default_options = StubGenerationOptions()
    c_source_root = Path(args.c_source_root) if args.c_source_root is not None else None

    return StubGenerationOptions(
        ignore_invalid_expressions=args.ignore_invalid_expressions,
        ignore_all_errors=args.ignore_all_errors,
        enum_class_locations=list(args.enum_class_locations),
        enable_docstring_signature_parser=args.enable_docstring_signature_parser,
        enable_c_signature_inference=args.enable_c_signature_inference,
        c_source_root=c_source_root,
        clang_c_std=args.clang_c_std or default_options.clang_c_std,
        clang_cpp_std=args.clang_cpp_std or default_options.clang_cpp_std,
        clang_include=list(args.clang_include),
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
