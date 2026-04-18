from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ..models import Class, Function, Module
from ..runtime import is_cpython_builtin, is_pybind11_builtin
from .producers import (
    CExtensionSignatureProducer,
    DocstringSignatureProducer,
    MinimalSignatureProducer,
)


@dataclass
class SignatureCompletionResult:
    total_functions: int = 0
    c_completed: int = 0
    docstring_completed: int = 0
    uncompleted: int = 0

    def __str__(self) -> str:
        return (
            "签名补全结果: "
            f"函数总数={self.total_functions}, "
            f"C源码补全={self.c_completed}, "
            f"文档字符串补全={self.docstring_completed}, "
            f"未补全={self.uncompleted}"
        )


class SignatureCompleter:
    def __init__(self, compilation_database: Path) -> None:
        self._c_producer = CExtensionSignatureProducer(compilation_database)
        self._docstring_producer = DocstringSignatureProducer()
        self._minimal_producer = MinimalSignatureProducer()
        self._result = SignatureCompletionResult()

    def run(self, module: Module) -> SignatureCompletionResult:
        self._result = SignatureCompletionResult()
        self._complete_module(module)

        logger.info("{}", self._result)
        return self._result

    def _complete_module(
        self,
        module: Module,
    ) -> None:
        for sub_module in module.sub_modules:
            self._complete_module(sub_module)

        for cls in module.classes:
            self._complete_class(cls, module)

        for func in module.functions:
            self._complete_function(func, module, is_method=False)

    def _complete_class(
        self,
        node: Class,
        module: Module,
    ) -> None:
        for nested_cls in node.classes:
            self._complete_class(nested_cls, module)

        for method in node.methods:
            self._complete_function(method, module, is_method=True)

    def _complete_function(
        self,
        func: Function,
        module: Module,
        *,
        is_method: bool,
    ) -> None:
        """按函数来源分支执行签名补全。"""
        self._result.total_functions += 1
        if func.signatures:
            logger.info(
                "跳过补全, module: {}, func: {}, is_method: {}, reason: 已存在签名",
                module.full_name,
                func.name,
                is_method,
            )
            return

        branch = "unsupported"
        reason = "函数不属于受支持的签名补全来源。"

        logger.info("开始补全, module: {}, func: {}, is_method: {}", module.full_name, func.name, is_method)
        try:
            if is_cpython_builtin(func.runtime_handle):
                branch = "c_builtin"
                production_result = self._c_producer.produce(
                    module,
                    func,
                    is_method=is_method,
                )
                func.signatures = production_result.signatures
                func.comment = production_result.comment
                self._result.c_completed += 1
                logger.info(
                    "补全成功, branch: c_builtin, module: {}, func: {}, is_method: {}",
                    module.full_name,
                    func.name,
                    is_method,
                )
                return

            if is_pybind11_builtin(func.runtime_handle):
                branch = "pybind11_builtin"
                production_result = self._docstring_producer.produce(
                    module,
                    func,
                    is_method=is_method,
                )
                func.signatures = production_result.signatures
                func.comment = production_result.comment
                self._result.docstring_completed += 1
                logger.info(
                    "补全成功, branch: pybind11_builtin, module: {}, func: {}, is_method: {}",
                    module.full_name,
                    func.name,
                    is_method,
                )
                return
        except Exception as ex:
            reason = f"{type(ex).__name__}: {ex}"

        minimal_result = self._minimal_producer.produce(
            module,
            func,
            is_method=is_method,
        )
        func.signatures = minimal_result.signatures
        func.comment = minimal_result.comment
        self._result.uncompleted += 1
        logger.warning(
            "补全失败, branch: {}, module: {}, func: {}, is_method: {}, reason: {}, fallback: minimal",
            branch,
            module.full_name,
            func.name,
            is_method,
            reason,
        )
