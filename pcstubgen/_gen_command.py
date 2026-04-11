from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import typer
from loguru import logger

from .api import gen_stubs
from .stub_output import JsonWriter

MY_LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | \n"
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n"
    "<level>{message}</level>\n"
)


def _build_log_file_name(module_name: str, now: datetime | None = None) -> str:
    """
    为本次运行生成日志文件名。
    """
    if now is None:
        now = datetime.now()

    leaf_module_name = module_name.rsplit(".", maxsplit=1)[-1]
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return f"pcstubgen_{leaf_module_name}_{timestamp}.log"


def _format_path_for_log(path: Path | None) -> str:
    """
    将路径格式化为稳定的日志输出。
    """
    if path is None:
        return "None"
    return str(path)


def _log_cli_arguments(
    *,
    module_name: str,
    output: Path,
    compilation_database: Path,
    include_docstrings: bool,
    json: bool,
) -> None:
    """
    记录本次 CLI 解析后的参数。
    """
    logger.info(
        "CLI参数: module_name={}, output={}, compilation_database={}, include_docstrings={}, json={}",
        module_name,
        _format_path_for_log(output),
        _format_path_for_log(compilation_database),
        include_docstrings,
        json,
    )


def _gen_command(
    module_name: str = typer.Argument(..., metavar="MODULE_NAME", help="模块名"),
    output: Path = typer.Option(
        Path("./stubs"),
        "--output",
        help="输出 stub 的根目录",
    ),
    compilation_database: Path = typer.Option(
        ...,
        "--compilation-database",
        help="必填的 compile_commands.json 文件路径",
    ),
    include_docstrings: bool = typer.Option(
        False,
        "--include-docstrings",
        help="生成 stub 时包含 docstring",
    ),
    json: bool = typer.Option(
        False,
        "--json",
        help="输出 JSON 格式的函数记录而不是 .pyi 文件",
    ),
) -> None:
    """
    使用 pcstubgen 为模块生成 Python stub。
    """
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
        _log_cli_arguments(
            module_name=module_name,
            output=output,
            compilation_database=compilation_database,
            include_docstrings=include_docstrings,
            json=json,
        )
        gen_stubs(
            module_name=module_name,
            output=output,
            compilation_database=compilation_database,
            include_docstrings=include_docstrings,
            writer=JsonWriter() if json else None,
        )
    finally:
        logger.remove(console_sink_id)
        logger.remove(file_sink_id)
        logger.add(sys.stderr)
