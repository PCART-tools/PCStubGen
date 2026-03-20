from __future__ import annotations

import importlib
import logging
from pathlib import Path

from .module_builder import build_module
from .stub_generation_options import StubGenerationOptions
from .ir import QualifiedName
from .pipeline import Pipeline
from .node_visitors.NodeVisitor import NodeVisitor
from .node_visitors.DocStringSignatureParserVisitor import DocStringSignatureParserVisitor
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

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="[{levelname}]: {message}\nat {filename}:{lineno} (in {funcName}())\n",
        style="{",
    )

    file_handler = logging.FileHandler(output_dir / "pcstubgen.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[{levelname}] - {name}\n{message}\n", style="{"))
    package_logger = logging.getLogger(__name__)
    previous_package_logger_level = package_logger.level
    package_logger.setLevel(logging.INFO)
    package_logger.addHandler(file_handler)
    try:
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
            visitors.append(
                DocStringSignatureParserVisitor(
                    enum_class_locations=dict(options.enum_class_locations),
                )
            )

        c_ast_visitor: CSignatureExtractionVisitor | None = None

        if options.source_root is not None:
            c_ast_visitor = CSignatureExtractionVisitor(
                source_root=options.source_root,
                clang_include=options.clang_include,
                clang_include_directory=options.clang_include_directory,
                clang_c_std=options.clang_c_std,
                clang_cpp_std=options.clang_cpp_std,
            )
            visitors.append(c_ast_visitor)

        _pipeline = Pipeline(visitors)

        # 4. 运行管道
        _pipeline.run(ir_module)
        if c_ast_visitor is not None:
            c_ast_visitor.log_summary(str(ir_module.full_name))

        ext = options.stub_extension if options.stub_extension else "pyi"
        if _writer is None:
            _writer = Writer(stub_extension=ext)
        else:
            _writer.stub_extension = ext
        printer = PrinterVisitor(
            include_docstrings=options.include_docstrings,
            include_module_type_comment=options.include_module_type_comment,
        )
        _writer.write(ir_module, printer, to=output_dir)
    finally:
        package_logger.removeHandler(file_handler)
        package_logger.setLevel(previous_package_logger_level)
        file_handler.close()
