from __future__ import annotations

from dataclasses import dataclass
import re

from loguru import logger

from .c_extension import CExtensionSource
from .docstring_source import resolve_docstring_signatures
from ..ir_modules import IRClass, IRFunction, IRMethod, IRModule, IRModuleType
from ..stub_generation_options import StubGenerationOptions


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
    def __init__(self, options: StubGenerationOptions) -> None:
        self._options = options
        self._c_source = CExtensionSource(
            compilation_database=options.compilation_database,
        )
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
        self._result.total_functions += 1
        c_reason = "未启用C扩展补全。"

        try:
            c_result = self._c_source.resolve_function(
                module,
                func,
            )
        except RuntimeError as ex:
            c_reason = str(ex)
        else:
            signatures, c_inferred_source_comment = c_result
            func.signatures = signatures
            if self._options.include_c_inferred_source_comment:
                func.c_inferred_source_comment = c_inferred_source_comment
            self._result.c_completed += 1
            logger.info(
                "通过C源码补全成功, module: {}, func: {}, is_method: {}",
                module.full_name,
                func.name,
                is_method,
            )
            return

        try:
            docstring_result = resolve_docstring_signatures(module, func)
        except ValueError as ex:
            docstring_reason = f"{type(ex).__name__}: {ex}"
        else:
            if docstring_result is not None:
                func.signatures = docstring_result
                self._result.docstring_completed += 1
                logger.info("通过docstring补全成功, module: {}, func: {}", module.full_name, func.name)
                return
            docstring_reason = self._describe_docstring_failure(func)

        self._result.uncompleted += 1
        logger.warning(
            "补全失败, module: {}, func: {}, c_reason: {}, docstring_reason: {}",
            module.full_name,
            func.name,
            c_reason,
            docstring_reason,
        )

    @staticmethod
    def _describe_docstring_failure(func: IRFunction) -> str:
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
