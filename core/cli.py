from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from loguru import logger

from . import write_stubs
from .stub_generation_options import StubGenerationOptions

MY_LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | \n"
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n"
    "<level>{message}</level>\n"
)


def _regex(pattern_str: str) -> re.Pattern:
    try:
        return re.compile(pattern_str)
    except re.error as ex:
        raise argparse.ArgumentTypeError(f"无效的 REGEX pattern: {ex}") from ex


def _regex_colon_path(regex_path: str) -> tuple[re.Pattern, str]:
    if ":" not in regex_path:
        raise argparse.ArgumentTypeError(
            "无效的 enum class 位置，期望格式为 REGEX:PATH"
        )

    pattern_str, path = regex_path.rsplit(":", maxsplit=1)
    if any(not part.isidentifier() for part in path.split(".")):
        raise argparse.ArgumentTypeError(f"无效的 PATH: {path}")
    return _regex(pattern_str), path


def _normalize_clang_include_directory(include_paths: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for raw_path in include_paths:
        if raw_path is None:
            raise TypeError("clang_include_directory 条目必须是非空的 include path")

        include_path = str(raw_path).strip()
        if not include_path:
            raise ValueError("clang_include_directory 条目必须是非空的 include path")
        if include_path.startswith("-"):
            raise ValueError(
                f"clang_include_directory 条目必须是 path，不能是类似选项的值: {include_path!r}"
            )
        if include_path not in normalized:
            normalized.append(include_path)
    return normalized


def _normalize_clang_include(includes: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for raw_include in includes:
        if raw_include is None:
            raise TypeError("clang_include 条目必须是非空的 include header")

        include = str(raw_include).strip()
        if not include:
            raise ValueError("clang_include 条目必须是非空的 include header")
        if include.startswith("-"):
            raise ValueError(
                f"clang_include 条目必须是 header，不能是类似选项的值: {include!r}"
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
        description="使用 pcstubgen 为模块生成 Python stub。",
        allow_abbrev=False,
    )
    parser.add_argument("module_name", metavar="MODULE_NAME", help="模块名")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="./stubs",
        help="输出 stub 的根目录",
    )

    parser.add_argument(
        "--enum-class-locations",
        dest="enum_class_locations",
        metavar="REGEX:LOC",
        action="append",
        default=[],
        type=_regex_colon_path,
        help="enum class 位置，格式为 <enum-class-name-regex>:<path-to-class>",
    )

    parser.add_argument(
        "--no-docstring-signature-parser",
        action="store_false",
        dest="enable_docstring_signature_parser",
        default=True,
        help="禁用从 docstring 解析签名",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="用于 C signature inference 的 C/C++ 源码根目录",
    )
    parser.add_argument(
        "--clang-include",
        action="append",
        default=[],
        help="额外的 clang include 头文件，可重复指定",
    )
    parser.add_argument(
        "--clang-include-directory",
        action="append",
        default=[],
        help="额外的 clang include 目录路径，可重复指定",
    )
    parser.add_argument(
        "--clang-c-std",
        default=None,
        help="传给 clang 的 C standard，例如 c11",
    )
    parser.add_argument(
        "--clang-cpp-std",
        default=None,
        help="传给 clang 的 C++ standard，例如 c++17",
    )

    parser.add_argument(
        "--no-docstrings",
        action="store_false",
        dest="include_docstrings",
        default=True,
        help="生成 stub 时不包含 docstring",
    )
    parser.add_argument(
        "--include-module-type-comment",
        default=False,
        action="store_true",
        help="在生成的 stub 中包含 module type comment",
    )
    parser.add_argument(
        "--stub-extension",
        type=str,
        default="pyi",
        metavar="EXT",
        choices=["pyi", "py"],
        help="生成 stub 的扩展名: pyi (默认) 或 py",
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
        enum_class_locations=list(args.enum_class_locations),
        enable_docstring_signature_parser=args.enable_docstring_signature_parser,
        source_root=source_root,
        clang_c_std=args.clang_c_std or default_options.clang_c_std,
        clang_cpp_std=args.clang_cpp_std or default_options.clang_cpp_std,
        clang_include=list(args.clang_include),
        clang_include_directory=list(args.clang_include_directory),
        include_docstrings=args.include_docstrings,
        include_module_type_comment=args.include_module_type_comment,
        stub_extension=args.stub_extension,
    )


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    console_sink_id = logger.add(sys.stderr, format=MY_LOGURU_FORMAT)
    sink_id = logger.add(
        output_dir / "pcstubgen.log",
        mode="w",
        encoding="utf-8",
        catch=False,
        backtrace=False,
        diagnose=False,
        format=MY_LOGURU_FORMAT,
    )
    try:
        write_stubs(
            module_name=args.module_name,
            output_dir=output_dir,
            options=_build_options(args),
        )
    finally:
        logger.remove(console_sink_id)
        logger.remove(sink_id)
        logger.add(sys.stderr)


if __name__ == "__main__":
    main()
