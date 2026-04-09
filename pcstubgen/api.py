from __future__ import annotations

import importlib
from pathlib import Path

from loguru import logger

from .module_collect import collect_module
from .ir_modules import QualifiedName
from .signature_completion import SignatureCompleter
from .stub_generation_options import StubGenerationOptions
from .stub_output import StubRenderer, StubWriter

__all__ = ["write_stubs"]


def write_stubs(
    module_name: str,
    output: Path,
    options: StubGenerationOptions | None = None,
    writer: StubWriter | None = None,
) -> None:
    """
    生成存根并写入文件。
    """
    effective_options = options if options is not None else StubGenerationOptions()
    effective_writer = writer if writer is not None else StubWriter()

    output.mkdir(parents=True, exist_ok=True)

    module = importlib.import_module(module_name)
    ir_module = collect_module(
        QualifiedName.from_str(module_name),
        module,
    )

    result = SignatureCompleter(effective_options).run(ir_module)
    logger.info("{}", result)

    renderer = StubRenderer(
        include_docstrings=effective_options.include_docstrings,
        include_c_inferred_source_comment=effective_options.include_c_inferred_source_comment,
    )
    effective_writer.write(ir_module, renderer, to=output)
