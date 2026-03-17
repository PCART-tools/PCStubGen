from __future__ import annotations

import logging
from pathlib import Path

from clang.cindex import Index

from .Models import ExtractedModule
from . import _module_table as module_table
from . import _translation_unit as translation_unit

logger = logging.getLogger(__name__)


def _check(condition: bool, message: str = "check failed!") -> None:
    """在核心前置条件不满足时抛出显式异常。"""
    if not condition:
        raise RuntimeError(message)


class CSignatureExtractor:
    """
    基于 libclang 的 C 签名提取引擎。

    该引擎从 `PyModuleDef` 变量定义出发，读取 `m_name` / `m_methods`
    还原模块级 `PyMethodDef`，再结合 `PyArg_*` 调用和格式串规则推断
    Python 侧参数信息。
    """

    def __init__(
        self,
        source_root: Path,
        *,
        clang_include: list[str] = (),
        clang_include_directory: list[str] = (),
        clang_c_std: str = "c11",
        clang_cpp_std: str = "c++17",
    ) -> None:
        """初始化提取器、clang 参数和模块级缓存。"""
        self._source_root = source_root
        self._clang_include = list(clang_include)
        self._clang_include_directory = translation_unit.inject_python_include_directories(
            list(clang_include_directory)
        )
        self._clang_c_std = clang_c_std
        self._clang_cpp_std = clang_cpp_std
        self._cache_result: dict[str, ExtractedModule] | None = None

    def extract_modules(self) -> dict[str, ExtractedModule]:
        """执行模块级签名提取主流程。"""
        if self._cache_result is not None:
            return self._cache_result

        _check(self._source_root.exists())

        source_files = translation_unit.find_candidate_files(self._source_root)
        if not source_files:
            self._cache_result = {}
            return self._cache_result

        index = Index.create()
        translation_units = [
            translation_unit.parse_translation_unit(
                index,
                file_path,
                source_root=self._source_root,
                clang_include=self._clang_include,
                clang_include_directory=self._clang_include_directory,
                clang_c_std=self._clang_c_std,
                clang_cpp_std=self._clang_cpp_std,
            )
            for file_path in source_files
        ]

        result: dict[str, ExtractedModule] = {}
        for unit in translation_units:
            try:
                modules = module_table.process_translation_unit(unit.cursor)
            except AssertionError as ex:
                logger.exception("AssertionError", exc_info=ex)
                continue
            for module in modules:
                existing = result.get(module.name)
                if existing is not None:
                    logger.warning(
                        "Discarded duplicate extracted module %s: kept existing module, discarded incoming module",
                        existing.name,
                    )
                    continue
                result[module.name] = module

        self._cache_result = result
        return self._cache_result
