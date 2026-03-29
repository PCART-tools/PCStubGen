from __future__ import annotations

from pathlib import Path

from clang.cindex import Index
from loguru import logger

from .._checks import check
from .models import ExtractedModule
from . import signature_inference
from . import module_table as module_table
from . import translation_unit as translation_unit


def extract_c_signature_modules(
    source_root: Path,
    *,
    include: list[str] = (),
    include_directory: list[Path] = (),
    c_std: str = "c11",
    cpp_std: str = "c++17",
) -> dict[str, ExtractedModule]:
    """
    基于 libclang 提取模块级 C 签名。

    该流程从 `PyModuleDef` 变量定义出发，读取 `m_name` / `m_methods`
    还原模块级 `PyMethodDef`，再结合 `PyArg_*` 调用和格式串规则推断
    Python 侧参数信息。
    """
    check(source_root.exists())

    normalized_include_dirs = translation_unit.inject_python_include_directories(include_directory)

    source_files = translation_unit.find_candidate_files(source_root)

    index = Index.create()
    translation_units = []
    for file_path in source_files:
        tu = translation_unit.parse_translation_unit(
            index,
            file_path,
            source_root=source_root,
            include=include,
            include_directory=normalized_include_dirs,
            c_std=c_std,
            cpp_std=cpp_std,
        )
        translation_units.append(tu)

    result: dict[str, ExtractedModule] = {}
    for tu in translation_units:
        try:
            modules = module_table.process_translation_unit(tu.cursor)
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
                function.signatures = signature_inference.infer_signature(
                    function.function_cursor
                )
            except Exception:
                logger.exception(
                    "推断 C 函数签名失败, module_name: {}, func_name: {}",
                    module.name,
                    function.ml_name,
                )

    return result
