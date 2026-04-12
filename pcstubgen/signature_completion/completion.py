from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ..models import Class, Function, Method, Module
from .c_extension import CExtensionSource
from .c_extension.runtime import supports_builtin_function_inference
from .docstring_source import parse_docstring_signatures


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
        self._c_source = CExtensionSource(compilation_database)
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

        # for cls in module.classes:
        #     self._complete_class(cls, module)

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
            self._complete_method(method, module)

    def _complete_method(
        self,
        method: Method,
        module: Module,
    ) -> None:
        self._complete_function(method.function, module, is_method=True)

    def _complete_function(
        self,
        func: Function,
        module: Module,
        *,
        is_method: bool,
    ) -> None:
        """按函数来源分支执行签名补全。"""
        self._result.total_functions += 1

        branch = "unsupported"
        reason = "函数不属于受支持的签名补全来源。"

        try:
            if supports_builtin_function_inference(func.runtime_handle):
                branch = "c_builtin"
                signatures, c_inferred_source_comment = self._c_source.infer_function_signatures(
                    module,
                    func,
                )
                func.signatures = signatures
                func.c_inferred_source_comment = c_inferred_source_comment
                self._result.c_completed += 1
                logger.info(
                    "通过C源码补全成功, branch: c_builtin, module: {}, func: {}, is_method: {}",
                    module.full_name,
                    func.name,
                    is_method,
                )
                return

            if self._is_pybind11_builtin(func.runtime_handle):
                branch = "pybind11_builtin"
                func.signatures = parse_docstring_signatures(module, func)
                self._result.docstring_completed += 1
                logger.info(
                    "通过docstring补全成功, branch: pybind11_builtin, module: {}, func: {}, is_method: {}",
                    module.full_name,
                    func.name,
                    is_method,
                )
                return
        except Exception as ex:
            reason = f"{type(ex).__name__}: {ex}"

        self._result.uncompleted += 1
        logger.warning(
            "补全失败, branch: {}, module: {}, func: {}, is_method: {}, reason: {}",
            branch,
            module.full_name,
            func.name,
            is_method,
            reason,
        )

    @staticmethod
    def _is_pybind11_builtin(handle: object) -> bool:
        """判断运行时函数句柄是否为 pybind11 绑定函数。"""
        handle_type = type(handle)
        if (
            handle_type.__module__ != "builtins"
            or handle_type.__name__ != "builtin_function_or_method"
        ):
            return False

        self_obj = getattr(handle, "__self__", None)
        if self_obj is None:
            return False

        return type(self_obj).__module__.startswith("pybind11_builtins")
