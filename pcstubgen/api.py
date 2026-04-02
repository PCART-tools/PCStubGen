from __future__ import annotations

import importlib
from pathlib import Path

from loguru import logger

from .module_build import build_module
from .stub_generation_options import StubGenerationOptions
from .ir_modules import QualifiedName
from .signature_completion import SignatureCompleter
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
    if options is None:
        options = StubGenerationOptions()
    if writer is None:
        writer = StubWriter()

    output.mkdir(parents=True, exist_ok=True)

    module = importlib.import_module(module_name)
    ir_module = build_module(
        QualifiedName.from_str(module_name),
        module,
    )

    result = SignatureCompleter(options).run(ir_module)
    logger.info("{}", result)

    renderer = StubRenderer(
        include_docstrings=options.include_docstrings,
        include_module_type_comment=options.include_module_type_comment,
        include_c_inferred_source_comment=options.include_c_inferred_source_comment,
    )
    writer.write(ir_module, renderer, to=output)
