from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from loguru import logger

from ..ir_modules import IRClass, IRFunction, IRMethod, IRModule
from .c_extension import CExtensionSource
from .c_extension.runtime import supports_builtin_function_inference
from .docstring_source import resolve_docstring_signatures


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
        self._c_source = CExtensionSource(compilation_database=compilation_database)
        self._result = SignatureCompletionResult()

    def run(self, module: IRModule) -> SignatureCompletionResult:
        self._result = SignatureCompletionResult()
        self._complete_module(module)
        return self._result

    def _complete_module(
        self,
        module: IRModule,
    ) -> None:
        for sub_module in module.sub_modules:
            self._complete_module(sub_module)

        # for cls in module.classes:
        #     self._complete_class(cls, module)

        for func in module.functions:
            self._complete_function(func, module, is_method=False)

    def _complete_class(
        self,
        node: IRClass,
        module: IRModule,
    ) -> None:
        for nested_cls in node.classes:
            self._complete_class(nested_cls, module)

        for method in node.methods:
            self._complete_method(method, module)

    def _complete_method(
        self,
        method: IRMethod,
        module: IRModule,
    ) -> None:
        self._complete_function(method.function, module, is_method=True)

    def _complete_function(
        self,
        func: IRFunction,
        module: IRModule,
        *,
        is_method: bool,
    ) -> None:
        """按函数来源分支执行签名补全。"""
        self._result.total_functions += 1

        if supports_builtin_function_inference(func.runtime_handle):
            branch = "c_builtin"
            try:
                signatures, c_inferred_source_comment = self._c_source.infer_function_signatures(
                    module,
                    func,
                )
            except BaseException as ex:
                reason = f"{type(ex).__name__}: {ex}"
            else:
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
        elif self._is_pybind11_builtin(func.runtime_handle):
            branch = "pybind11_builtin"
            try:
                docstring_result = resolve_docstring_signatures(module, func)
            except BaseException as ex:
                reason = f"{type(ex).__name__}: {ex}"
            else:
                if docstring_result is not None:
                    func.signatures = docstring_result
                    self._result.docstring_completed += 1
                    logger.info(
                        "通过docstring补全成功, branch: pybind11_builtin, module: {}, func: {}, is_method: {}",
                        module.full_name,
                        func.name,
                        is_method,
                    )
                    return
                reason = self._describe_docstring_failure(func)
        else:
            branch = "unsupported"
            reason = "函数不属于受支持的签名补全来源。"

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

    @staticmethod
    def _describe_docstring_failure(func: IRFunction) -> str:
        """描述 docstring 分支未能产出签名的原因。"""
        doc = func.doc
        if not doc:
            return "docstring为空或缺失，无法解析签名。"

        doc_lines = doc.splitlines()
        if len(doc_lines) == 0:
            return "docstring为空或缺失，无法解析签名。"

        top_signature_regex = re.compile(
            rf"^{re.escape(func.name)}\((?P<args>.*)\)\s*(->\s*(?P<returns>.+))?$"
        )
        if top_signature_regex.match(doc_lines[0]) is None:
            return "docstring首行不是可解析的签名声明。"

        return "docstring未解析出可用签名。"
