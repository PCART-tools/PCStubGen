from __future__ import annotations

import importlib
from pathlib import Path

from .module_builder import build_module
from .stub_generation_options import StubGenerationOptions
from .ir import QualifiedName
from .visitor_runner import run_visitors
from .visitors.node_visitor import NodeVisitor
from .visitors.docstring_signature_visitor import DocstringSignatureVisitor
from .visitors.c_signature_visitor import CSignatureVisitor
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

    # 3. 设置 visitor runner
    visitors: list[NodeVisitor] = []
    c_signature_visitor: CSignatureVisitor | None = None

    if options.source_root is not None:
        c_signature_visitor = CSignatureVisitor(
            source_root=options.source_root,
            include=options.include,
            include_directory=options.include_directory,
            c_std=options.c_std,
            cpp_std=options.cpp_std,
            include_c_inferred_source_comment=options.include_c_inferred_source_comment,
        )
        visitors.append(c_signature_visitor)

    docstring_signature_visitor = DocstringSignatureVisitor()
    visitors.append(docstring_signature_visitor)

    # 4. 运行 visitor runner
    run_visitors(ir_module, visitors)
    if c_signature_visitor is not None:
        c_signature_visitor.log_summary()
    docstring_signature_visitor.log_summary()

    renderer = StubRenderer(
        include_docstrings=options.include_docstrings,
        include_module_type_comment=options.include_module_type_comment,
        include_c_inferred_source_comment=options.include_c_inferred_source_comment,
    )
    writer.write(ir_module, renderer, to=output_dir)
