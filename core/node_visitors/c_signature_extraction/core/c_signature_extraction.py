from __future__ import annotations

from pathlib import Path

from clang.cindex import Index
from loguru import logger

from .models import ExtractedModule
from . import signature_inference
from . import module_table as module_table
from . import translation_unit as translation_unit


def _check(condition: bool, message: str = "前置条件检查失败。") -> None:
    """在核心前置条件不满足时抛出显式异常。"""
    if not condition:
        raise RuntimeError(message)


def extract_c_signature_modules(
    source_root: Path,
    *,
    clang_include: list[str] = (),
    clang_include_directory: list[Path] = (),
    clang_c_std: str = "c11",
    clang_cpp_std: str = "c++17",
) -> dict[str, ExtractedModule]:
    """
    基于 libclang 提取模块级 C 签名。

    该流程从 `PyModuleDef` 变量定义出发，读取 `m_name` / `m_methods`
    还原模块级 `PyMethodDef`，再结合 `PyArg_*` 调用和格式串规则推断
    Python 侧参数信息。
    """
    _check(source_root.exists())

    normalized_include_dirs = translation_unit.inject_python_include_directories(clang_include_directory)

    source_files = translation_unit.find_candidate_files(source_root)

    index = Index.create()
    translation_units = []
    for file_path in source_files:
        tu = translation_unit.parse_translation_unit(
            index,
            file_path,
            source_root=source_root,
            clang_include=clang_include,
            clang_include_directory=normalized_include_dirs,
            clang_c_std=clang_c_std,
            clang_cpp_std=clang_cpp_std,
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
            signature_inference.infer_signature(function)

    return result
