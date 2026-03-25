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


def _normalize_clang_include(includes: list[str]) -> list[str]:
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


def _normalize_stub_extension(stub_extension: str) -> str:
    if stub_extension not in {"pyi", "py"}:
        raise ValueError("生成 stub 的扩展名必须是 pyi 或 py")
    return stub_extension


def _validate_clang_include(value: list[str] | None) -> list[str]:
    values = [] if value is None else list(value)
    try:
        _normalize_clang_include(values)
    except (TypeError, ValueError) as ex:
        raise typer.BadParameter(str(ex)) from ex
    return values


def _validate_stub_extension(value: str) -> str:
    try:
        return _normalize_stub_extension(value)
    except ValueError as ex:
        raise typer.BadParameter(str(ex)) from ex


def _build_options(
    *,
    enable_docstring_signature_parser: bool,
    source_root: Path | None,
    clang_include: list[str] | None,
    clang_include_directory: list[Path] | None,
    clang_c_std: str | None,
    clang_cpp_std: str | None,
    include_docstrings: bool,
    include_module_type_comment: bool,
    stub_extension: str,
) -> StubGenerationOptions:
    """
    将 CLI 入参归一化并转换为 stub 生成配置。
    """
    default_options = StubGenerationOptions()

    return StubGenerationOptions(
        enable_docstring_signature_parser=enable_docstring_signature_parser,
        source_root=source_root,
        clang_c_std=clang_c_std or default_options.clang_c_std,
        clang_cpp_std=clang_cpp_std or default_options.clang_cpp_std,
        clang_include=_normalize_clang_include(clang_include or []),
        clang_include_directory=[
            str(include_dir) for include_dir in (clang_include_directory or [])
        ],
        include_docstrings=include_docstrings,
        include_module_type_comment=include_module_type_comment,
        stub_extension=_normalize_stub_extension(stub_extension),
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
    clang_include: list[str] | None = typer.Option(
        None,
        "--clang-include",
        callback=_validate_clang_include,
        help="额外的 clang include 头文件，可重复指定",
    ),
    clang_include_directory: list[Path] | None = typer.Option(
        None,
        "--clang-include-directory",
        help="额外的 clang include 目录路径，可重复指定",
    ),
    clang_c_std: str | None = typer.Option(
        None,
        "--clang-c-std",
        help="传给 clang 的 C standard，例如 c11",
    ),
    clang_cpp_std: str | None = typer.Option(
        None,
        "--clang-cpp-std",
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
    stub_extension: str = typer.Option(
        "pyi",
        "--stub-extension",
        metavar="EXT",
        callback=_validate_stub_extension,
        help="生成 stub 的扩展名: pyi (默认) 或 py",
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
                clang_include=clang_include,
                clang_include_directory=clang_include_directory,
                clang_c_std=clang_c_std,
                clang_cpp_std=clang_cpp_std,
                include_docstrings=include_docstrings,
                include_module_type_comment=include_module_type_comment,
                stub_extension=stub_extension,
            ),
        )
    finally:
        logger.remove(console_sink_id)
        logger.remove(sink_id)
        logger.add(sys.stderr)


if __name__ == "__main__":
    app()
