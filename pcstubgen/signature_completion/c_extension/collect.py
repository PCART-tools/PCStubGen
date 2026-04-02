from __future__ import annotations

from pathlib import Path

from clang.cindex import Index
from loguru import logger

from ...checks import check
from .definition_index import DefinitionIndex
from .models import CModule
from .clang import parser as clang_parser
from .modules import collect_modules as module_collection
from .signatures import inference


def collect_modules(
    source: Path,
    *,
    include: list[str] = (),
    include_directory: list[Path] = (),
    c_std: str = "c11",
    cpp_std: str = "c++17",
) -> dict[str, CModule]:
    """
    基于 libclang 提取模块级 C 签名。

    该流程从 translation unit 顶层 `PyInit_*` 定义出发，反查对应的
    `PyModuleDef` 与 `m_methods`，还原模块级 `PyMethodDef`，再结合
    `PyArg_*` 调用和格式串规则推断 Python 侧参数信息。
    """
    check(source.exists())

    normalized_include_dirs = clang_parser.inject_python_include_directories(include_directory)

    source_files = clang_parser.list_files(source)

    index = Index.create()
    translation_units = []
    for file_path in source_files:
        tu = clang_parser.parse(
            index,
            file_path,
            source=source,
            include=include,
            include_directory=normalized_include_dirs,
            c_std=c_std,
            cpp_std=cpp_std,
        )
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
                function.signatures = inference.infer_signature(
                    function.function_cursor
                )
            except Exception:
                logger.exception(
                    "推断 C 函数签名失败, module_name: {}, func_name: {}",
                    module.name,
                    function.ml_name,
                )

    return result
