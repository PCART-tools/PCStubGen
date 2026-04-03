from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import typer
from loguru import logger

from .api import write_stubs
from .stub_generation_options import StubGenerationOptions

MY_LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | \n"
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n"
    "<level>{message}</level>\n"
)

app = typer.Typer(add_completion=False)


def _build_log_file_name(module_name: str, now: datetime | None = None) -> str:
    """
    为本次运行生成日志文件名。
    """
    if now is None:
        now = datetime.now()

    leaf_module_name = module_name.rsplit(".", maxsplit=1)[-1]
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return f"pcstubgen_{leaf_module_name}_{timestamp}.log"


@app.command(help="使用 pcstubgen 为模块生成 Python stub。")
def main(
    module_name: str = typer.Argument(..., metavar="MODULE_NAME", help="模块名"),
    output: Path = typer.Option(
        Path("./stubs"),
        "--output",
        help="输出 stub 的根目录",
    ),
    source: Path | None = typer.Option(
        None,
        "--source",
        help="用于 C signature inference 的 C/C++ 源码根目录",
    ),
    include: list[str] = typer.Option(
        [],
        "--include",
        help="额外的 include 头文件，可重复指定",
    ),
    include_directory: list[Path] = typer.Option(
        [],
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
        False,
        "--include-docstrings",
        help="生成 stub 时包含 docstring",
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
    default_options = StubGenerationOptions()
    output.mkdir(parents=True, exist_ok=True)

    logger.remove()
    console_sink_id = logger.add(sys.stderr, format=MY_LOGURU_FORMAT)
    file_sink_id = logger.add(
        output / _build_log_file_name(module_name),
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
            output=output,
            options=StubGenerationOptions(
                source=source,
                c_std=c_std or default_options.c_std,
                cpp_std=cpp_std or default_options.cpp_std,
                include=list(include),
                include_directory=list(include_directory),
                include_docstrings=include_docstrings,
                include_module_type_comment=include_module_type_comment,
                include_c_inferred_source_comment=include_c_inferred_source_comment,
            ),
        )
    finally:
        logger.remove(console_sink_id)
        logger.remove(file_sink_id)
        logger.add(sys.stderr)


if __name__ == "__main__":
    app()
