from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from .c_extension import CExtensionSource
from .docstring_source import resolve_docstring_signatures
from ..ir import IRClass, IRFunction, IRMethod, IRModule, IRModuleType
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
        if options.source_root is None:
            self._c_source = None
        else:
            self._c_source = CExtensionSource(
                source_root=options.source_root,
                include=options.include,
                include_directory=options.include_directory,
                c_std=options.c_std,
                cpp_std=options.cpp_std,
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

        for cls in module.classes:
            self._complete_class(cls, module)

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

        if self._c_source is not None:
            c_result = self._c_source.resolve_function(
                module,
                func,
                is_method,
            )
            if c_result is not None:
                signatures, c_inferred_source_comment = c_result
                func.signatures = signatures
                if self._options.include_c_inferred_source_comment:
                    func.c_inferred_source_comment = c_inferred_source_comment
                self._result.c_completed += 1
                logger.info("通过C源码补全成功, module: {}, func: {}", module.full_name, func.name)
                return

        try:
            docstring_result = resolve_docstring_signatures(module, func)
        except ValueError as ex:
            logger.warning(
                "解析 docstring 签名失败, module_name: {}, func_name: {}, error_type: {}, error: {}",
                str(module.full_name),
                func.name,
                type(ex).__name__,
                ex,
            )
        else:
            if docstring_result is not None:
                func.signatures = docstring_result
                self._result.docstring_completed += 1
                logger.info("通过docstring补全成功, module: {}, func: {}", module.full_name, func.name)
                return

        self._result.uncompleted += 1
        logger.warning("补全失败, module: {}, func: {}", module.full_name, func.name)