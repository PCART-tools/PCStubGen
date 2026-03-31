from __future__ import annotations

import importlib
from pathlib import Path

from .module_builder import build_module
from .stub_generation_options import StubGenerationOptions
from .ir import QualifiedName
from .supplementer import supplement_signatures
from .stub_renderer import StubRenderer
from .stub_writer import StubWriter
from .checks import check

__all__ = ["write_stubs"]


def write_stubs(
    module_name: str,
    output_dir: Path,
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

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 导入模块
    module = importlib.import_module(module_name)

    # 2. 构建原始 IR
    ir_module = build_module(
        QualifiedName.from_str(module_name),
        module,
    )

    # 3. 统一补全签名
    summary = supplement_signatures(ir_module, options)
    summary.log_summary()

    renderer = StubRenderer(
        include_docstrings=options.include_docstrings,
        include_module_type_comment=options.include_module_type_comment,
        include_c_inferred_source_comment=options.include_c_inferred_source_comment,
    )
    writer.write(ir_module, renderer, to=output_dir)
