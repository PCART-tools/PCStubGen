from __future__ import annotations

import importlib
from pathlib import Path

from loguru import logger

from .module_collect import collect_module
from .ir_modules import QualifiedName
from .ir_modules import IRModule, IRModuleType
from .signature_completion import SignatureCompleter
from .signature_completion.c_extension.llvm_symbolizer import require_llvm_symbolizer
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
    _ensure_extension_dependencies(ir_module)

    result = SignatureCompleter(effective_options).run(ir_module)
    logger.info("{}", result)

    renderer = StubRenderer(
        include_docstrings=effective_options.include_docstrings,
        include_c_inferred_source_comment=effective_options.include_c_inferred_source_comment,
    )
    effective_writer.write(ir_module, renderer, to=output)


def _ensure_extension_dependencies(module: IRModule) -> None:
    if _contains_extension_module(module):
        require_llvm_symbolizer()


def _contains_extension_module(module: IRModule) -> bool:
    if module.module_type is IRModuleType.EXTENSION:
        return True
    return any(_contains_extension_module(sub_module) for sub_module in module.sub_modules)
