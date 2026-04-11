from __future__ import annotations

import importlib
from pathlib import Path

from loguru import logger

from .ir_modules import QualifiedName
from .module_collect import collect_module
from .signature_completion import SignatureCompleter
from .stub_output import StubRenderer, StubWriter

__all__ = ["gen_stubs"]


def gen_stubs(
    module_name: str,
    output: Path,
    compilation_database: Path,
    *,
    include_docstrings: bool = False,
    writer: StubWriter | None = None,
) -> None:
    """
    为模块生成 stub 并写入输出目录。
    """
    effective_writer = writer if writer is not None else StubWriter()

    output.mkdir(parents=True, exist_ok=True)

    module = importlib.import_module(module_name)
    ir_module = collect_module(
        QualifiedName.from_str(module_name),
        module,
    )

    result = SignatureCompleter(compilation_database).run(ir_module)
    logger.info("{}", result)

    renderer = StubRenderer(include_docstrings=include_docstrings)
    effective_writer.write(ir_module, renderer, to=output)
