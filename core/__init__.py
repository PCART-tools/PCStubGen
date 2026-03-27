from __future__ import annotations

import importlib
from pathlib import Path

from .module_builder import build_module
from .stub_generation_options import StubGenerationOptions
from .ir import QualifiedName
from .pipeline import Pipeline
from .node_visitors.node_visitor import NodeVisitor
from .node_visitors.doc_string_signature_parser_visitor import DocStringSignatureParserVisitor
from .node_visitors.c_signature_extraction.c_signature_extraction_visitor import (
    CSignatureExtractionVisitor,
)
from .printer_visitor import PrinterVisitor
from .writer import Writer

__all__ = ["write_stubs"]


def write_stubs(
    module_name: str,
    output_dir: Path,
    options: StubGenerationOptions | None = None,
    _writer: Writer | None = None,
) -> None:
    """
    生成存根并写入文件。
    """
    if options is None:
        options = StubGenerationOptions()

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 导入模块
    module = importlib.import_module(module_name)

    # 2. 构建原始 IR
    ir_module = build_module(
        QualifiedName.from_str(module_name),
        module,
    )

    # 3. 设置管道
    visitors: list[NodeVisitor] = []

    # 核心签名解析与类型修复 visitor（仅覆盖模块树 / 函数 / 类方法主链路）
    if options.enable_docstring_signature_parser:
        visitors.append(DocStringSignatureParserVisitor())

    c_ast_visitor: CSignatureExtractionVisitor | None = None

    if options.source_root is not None:
        c_ast_visitor = CSignatureExtractionVisitor(
            source_root=options.source_root,
            include=options.include,
            include_directory=options.include_directory,
            c_std=options.c_std,
            cpp_std=options.cpp_std,
        )
        visitors.append(c_ast_visitor)

    _pipeline = Pipeline(visitors)

    # 4. 运行管道
    _pipeline.run(ir_module)
    if c_ast_visitor is not None:
        c_ast_visitor.log_summary()

    if _writer is None:
        _writer = Writer()
    printer = PrinterVisitor(
        include_docstrings=options.include_docstrings,
        include_module_type_comment=options.include_module_type_comment,
    )
    _writer.write(ir_module, printer, to=output_dir)
