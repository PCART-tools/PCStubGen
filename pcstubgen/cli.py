from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger

from . import write_stubs
from .stub_generation_options import StubGenerationOptions

MY_LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | \n"
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n"
    "<level>{message}</level>\n"
)

app = typer.Typer(add_completion=False)


def _normalize_include(includes: list[str]) -> list[str]:
    normalized: list[str] = []
    for raw_include in includes:
        if raw_include is None:
            raise TypeError("include 条目必须是非空的 include header")

        include = str(raw_include).strip()
        if not include:
            raise ValueError("include 条目必须是非空的 include header")
        if include.startswith("-"):
            raise ValueError(
                f"include 条目必须是 header，不能是类似选项的值: {include!r}"
            )
        if include not in normalized:
            normalized.append(include)
    return normalized


def _validate_include(value: list[str] | None) -> list[str]:
    values = [] if value is None else list(value)
    try:
        _normalize_include(values)
    except (TypeError, ValueError) as ex:
        raise typer.BadParameter(str(ex)) from ex
    return values


def _build_options(
    *,
    enable_docstring_signature_parser: bool,
    source_root: Path | None,
    include: list[str] | None,
    include_directory: list[Path] | None,
    c_std: str | None,
    cpp_std: str | None,
    include_docstrings: bool,
    include_module_type_comment: bool,
    include_c_inferred_source_comment: bool,
) -> StubGenerationOptions:
    """
    将 CLI 入参归一化并转换为 stub 生成配置。
    """
    default_options = StubGenerationOptions()

    return StubGenerationOptions(
        enable_docstring_signature_parser=enable_docstring_signature_parser,
        source_root=source_root,
        c_std=c_std or default_options.c_std,
        cpp_std=cpp_std or default_options.cpp_std,
        include=_normalize_include(include or []),
        include_directory=list(include_directory or []),
        include_docstrings=include_docstrings,
        include_module_type_comment=include_module_type_comment,
        include_c_inferred_source_comment=include_c_inferred_source_comment,
    )


@app.command(help="使用 pcstubgen 为模块生成 Python stub。")
def main(
    module_name: str = typer.Argument(..., metavar="MODULE_NAME", help="模块名"),
    output_dir: Path = typer.Option(
        Path("./stubs"),
        "--output-dir",
        "-o",
        help="输出 stub 的根目录",
    ),
    enable_docstring_signature_parser: bool = typer.Option(
        True,
        "--no-docstring-signature-parser",
        help="禁用从 docstring 解析签名",
    ),
    source_root: Path | None = typer.Option(
        None,
        "--source-root",
        help="用于 C signature inference 的 C/C++ 源码根目录",
    ),
    include: list[str] | None = typer.Option(
        None,
        "--include",
        callback=_validate_include,
        help="额外的 include 头文件，可重复指定",
    ),
    include_directory: list[Path] | None = typer.Option(
        None,
        "--include-directory",
        help="额外的 include 目录路径，可重复指定",
    ),
    c_std: str | None = typer.Option(
        None,
        "--c-std",
        help="传给 clang 的 C standard，例如 c11",
    ),
    cpp_std: str | None = typer.Option(
        None,
        "--cpp-std",
        help="传给 clang 的 C++ standard，例如 c++17",
    ),
    include_docstrings: bool = typer.Option(
        True,
        "--no-docstrings",
        help="生成 stub 时不包含 docstring",
    ),
    include_module_type_comment: bool = typer.Option(
        False,
        "--include-module-type-comment",
        help="在生成的 stub 中包含 module type comment",
    ),
    include_c_inferred_source_comment: bool = typer.Option(
        False,
        "--include-c-inferred-source-comment",
        help="在函数 stub 后包含 C AST 推断签名对应的源码注释",
    ),
) -> None:
    """
    使用 pcstubgen 为模块生成 Python stub。
    """
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
            module_name=module_name,
            output_dir=output_dir,
            options=_build_options(
                enable_docstring_signature_parser=enable_docstring_signature_parser,
                source_root=source_root,
                include=include,
                include_directory=include_directory,
                c_std=c_std,
                cpp_std=cpp_std,
                include_docstrings=include_docstrings,
                include_module_type_comment=include_module_type_comment,
                include_c_inferred_source_comment=include_c_inferred_source_comment,
            ),
        )
    finally:
        logger.remove(console_sink_id)
        logger.remove(sink_id)
        logger.add(sys.stderr)


if __name__ == "__main__":
    app()
