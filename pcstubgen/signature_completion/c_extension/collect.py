from __future__ import annotations

from pathlib import Path

from clang.cindex import Index, TranslationUnitLoadError
from loguru import logger

from .definition_index import DefinitionIndex
from .models import CModule
from .clang import parser as clang_parser
from .modules import collect_modules as module_collection
from .signatures import inference


def collect_modules(
    compilation_database: Path,
) -> dict[str, CModule]:
    """
    基于 libclang 提取模块级 C 签名。

    该流程从 translation unit 顶层 `PyInit_*` 定义出发，反查对应的
    `PyModuleDef` 与 `m_methods`，还原模块级 `PyMethodDef`，再结合
    `PyArg_*` 调用和格式串规则推断 Python 侧参数信息。
    """
    compilation_commands = clang_parser.list_compilation_commands(compilation_database)
    total_compilation_commands = len(compilation_commands)

    logger.info(
        "阶段进度 [1/4] 开始Parse, 文件数: {}",
        total_compilation_commands,
    )

    index = Index.create()
    translation_units = []
    for command_index, compilation_command in enumerate(compilation_commands, start=1):
        logger.info(
            "Parse进度 [{}/{}], 文件: {}",
            command_index,
            total_compilation_commands,
            compilation_command.file_path,
        )
        effective_parse_args = clang_parser.build_effective_parse_args(compilation_command)
        try:
            tu = clang_parser.parse(
                index,
                compilation_command,
                effective_parse_args=effective_parse_args,
            )
        except TranslationUnitLoadError:
            logger.warning(
                "Parse失败, 跳过文件\n"
                "文件路径: {}\n"
                "工作目录: {}\n"
                "解析参数: {}\n",
                compilation_command.file_path,
                compilation_command.working_directory,
                " ".join(effective_parse_args),
            )
            continue
        diagnostics = tu.diagnostics
        if clang_parser.has_error_diagnostics(diagnostics):
            logger.warning(
                "Parse诊断\n"
                "文件路径: {}\n"
                "工作目录: {}\n"
                "解析参数: {}\n"
                "诊断: \n"
                "{}",
                compilation_command.file_path,
                compilation_command.working_directory,
                " ".join(effective_parse_args),
                "\n".join(
                    f"- {clang_parser.diagnostic_to_str(diagnostic)}"
                    for diagnostic in diagnostics
                ),
            )
        else:
            logger.info(
                "Parse成功\n"
                "文件路径: {}\n"
                "工作目录: {}\n"
                "解析参数: {}\n",
                compilation_command.file_path,
                compilation_command.working_directory,
                " ".join(effective_parse_args),
            )
        translation_units.append(tu)

    logger.info(
        "阶段进度 [2/4] 开始构建索引, TU数: {}",
        len(translation_units),
    )
    definition_index = DefinitionIndex(translation_units)

    logger.info(
        "阶段进度 [3/4] 开始收集模块, TU数: {}",
        len(translation_units),
    )
    result: dict[str, CModule] = {}
    for tu in translation_units:
        try:
            modules = module_collection.collect_modules_from_translation_unit(
                tu,
                definition_index=definition_index,
            )
        except AssertionError:
            logger.exception("处理 translation unit 时触发 AssertionError")
            continue
        for module in modules:
            existing = result.get(module.name)
            if existing is not None:
                logger.warning("模块重复, 丢弃新模块, module: {}", existing.name)
                continue
            result[module.name] = module

    total_functions = sum(len(module.functions) for module in result.values())
    logger.info(
        "阶段进度 [4/4] 开始推断签名, 模块数: {}, 函数数: {}",
        len(result),
        total_functions,
    )
    inferred_function_index = 0
    for module in result.values():
        for function in module.functions.values():
            inferred_function_index += 1
            logger.info(
                "签名推断进度 [{}/{}], module_name: {}, func_name: {}",
                inferred_function_index,
                total_functions,
                module.name,
                function.ml_name,
            )
            try:
                function.signatures = inference.infer_signature(function)
            except Exception:
                logger.exception(
                    "推断 C 函数签名失败, module_name: {}, func_name: {}",
                    module.name,
                    function.ml_name,
                )

    return result
