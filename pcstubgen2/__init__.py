from __future__ import annotations

import importlib
import logging
from pathlib import Path

from .ErrorCollector import ErrorCollector
from .ModuleBuilder import ModuleBuilder
from .StubGenerationOptions import StubGenerationOptions
from .IR import QualifiedName
from .Pipeline import Pipeline
from .NodeVisitors.NodeVisitor import NodeVisitor
from .NodeVisitors.DocStringSignatureParserVisitor import DocStringSignatureParserVisitor
from .NodeVisitors.Filters import (
    FilterClassMembersVisitor,
    FilterInvalidIdentifierVisitor,
    FilterPybind11ViewClassesVisitor,
    FilterPybindInternalsVisitor,
    FilterTypingModuleAttributesVisitor,
)
from .NodeVisitors.Fixes import (
    ImportResolverVisitor,
    FixMissingFutureAnnotationsVisitor,
    FixMissingAllVisitor,
    RemoveSelfAnnotationVisitor,
    FixBuiltinTypesVisitor,
    FixTypingTypeNamesVisitor,
    FixRedundantMethodsFromBuiltinObjectVisitor,
    FixNumpyDtypeVisitor,
    FixScipyTypeArgumentsVisitor,
    FixNumpyArrayFlagsVisitor,
    FixNumpyArrayRemoveParametersVisitor,
    OverridePrintSafeValuesVisitor,
    RewritePybind11EnumValueReprVisitor,
    FixPybind11EnumStrDocVisitor,
    FixMissingEnumMembersAnnotationVisitor,
    FixPEP585CollectionNamesVisitor,
    FixValueReprRandomAddressVisitor,
    FixRedundantBuiltinsAnnotationVisitor,
    FixMissingNoneHashFieldAnnotationVisitor,
    FixCurrentModulePrefixInTypeNamesVisitor,
    ReplaceReadWritePropertyWithFieldVisitor,
    FixMissingFixedSizeImportVisitor,
    FixNumpyArrayDimAnnotationVisitor,
    FixNumpyArrayDimTypeVarVisitor,
)
from .PrinterVisitor import PrinterVisitor
from .Writer import Writer

__all__ = ["write_stubs"]


def write_stubs(
    module_name: str,
    output_dir: str | Path,
    options: StubGenerationOptions | None = None,
    writer: Writer | None = None,
) -> None:
    """
    生成存根并写入文件。
    """
    if options is None:
        options = StubGenerationOptions()

    # 设置日志
    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s - [%(levelname)7s] %(message)s",
    )

    # 为本次运行创建错误收集器
    error_collector = ErrorCollector()
    error_collector.ignore_invalid_expressions = options.ignore_invalid_expressions
    error_collector.ignore_invalid_identifiers = options.ignore_invalid_identifiers
    error_collector.ignore_unresolved_names = options.ignore_unresolved_names
    error_collector.ignore_all_errors = options.ignore_all_errors

    # 1. 导入模块
    module = importlib.import_module(module_name)

    # 2. 构建原始 IR
    builder = ModuleBuilder(error_collector=error_collector)
    ir_module = builder.build_module(
        QualifiedName.from_str(module_name),
        module,
    )

    # 3. 设置管道
    visitors: list[NodeVisitor] = []

    # 基础导入与 __all__
    visitors.extend([
        FixMissingFutureAnnotationsVisitor(),
        FixMissingAllVisitor(),
        FixMissingNoneHashFieldAnnotationVisitor(),
        FilterTypingModuleAttributesVisitor(),
        DocStringSignatureParserVisitor(
            error_collector=error_collector,
            enum_class_locations=dict(options.enum_class_locations),
        ),
        FixPybind11EnumStrDocVisitor(),
        FixPEP585CollectionNamesVisitor(),
        FixTypingTypeNamesVisitor(),
        FixScipyTypeArgumentsVisitor(),
        FixMissingFixedSizeImportVisitor(),
        FixMissingEnumMembersAnnotationVisitor(),
        OverridePrintSafeValuesVisitor(pattern=options.print_safe_value_reprs),
    ])

    # 与 NumPy 数组相关的修复（按条件启用）
    if options.numpy_array_wrap_with_annotated:
        visitors.append(FixNumpyArrayDimAnnotationVisitor())
    if options.numpy_array_use_type_var:
        visitors.append(FixNumpyArrayDimTypeVarVisitor())
    if options.numpy_array_remove_parameters:
        visitors.append(FixNumpyArrayRemoveParametersVisitor())

    visitors.extend([
        FixNumpyDtypeVisitor(),
        FixNumpyArrayFlagsVisitor(),
        FixCurrentModulePrefixInTypeNamesVisitor(),
        FixBuiltinTypesVisitor(),
        RewritePybind11EnumValueReprVisitor(
            enum_class_locations=dict(options.enum_class_locations)
        ),
        FilterClassMembersVisitor(),
        ReplaceReadWritePropertyWithFieldVisitor(),
        FilterInvalidIdentifierVisitor(error_collector=error_collector),
        FixValueReprRandomAddressVisitor(),
        FixRedundantBuiltinsAnnotationVisitor(),
        FilterPybindInternalsVisitor(),
        FilterPybind11ViewClassesVisitor(),
        FixRedundantMethodsFromBuiltinObjectVisitor(),
        RemoveSelfAnnotationVisitor(),
        ImportResolverVisitor(error_collector=error_collector),
    ])

    pipeline = Pipeline(visitors)

    # 4. 运行管道
    pipeline.run(ir_module)

    # 5. 完成错误处理（发出警告等）
    error_collector.finalize()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = options.stub_extension if options.stub_extension else "pyi"
    if writer is None:
        writer = Writer(stub_extension=ext)
    else:
        writer.stub_extension = ext
    printer = PrinterVisitor(
        invalid_expr_as_ellipses=not options.print_invalid_expressions_as_is
    )
    writer.write(ir_module, printer, to=output_dir)
