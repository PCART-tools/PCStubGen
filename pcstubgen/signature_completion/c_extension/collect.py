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

    index = Index.create()
    translation_units = []
    for compilation_command in compilation_commands:
        try:
            tu = clang_parser.parse(index, compilation_command)
        except TranslationUnitLoadError:
            logger.warning(
                "Parse失败, 跳过文件\n"
                "文件路径: {}\n"
                "工作目录: {}\n"
                "解析参数: {}\n",
                compilation_command.file_path,
                compilation_command.working_directory,
                ' '.join(compilation_command.parse_args),
            )
            continue
        translation_units.append(tu)

    definition_index = DefinitionIndex(translation_units)

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

    for module in result.values():
        for function in module.functions.values():
            try:
                function.signatures = inference.infer_signature(function)
            except Exception:
                logger.exception(
                    "推断 C 函数签名失败, module_name: {}, func_name: {}",
                    module.name,
                    function.ml_name,
                )

    return result
