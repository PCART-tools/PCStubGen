from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from .c_extension import CExtensionSource
from .docstring_source import resolve_docstring_signatures
from .inspect_source import resolve_inspect_signatures
from ..ir import IRClass, IRFunction, IRMethod, IRModule
from ..stub_generation_options import StubGenerationOptions


@dataclass
class SignatureCompletionSummary:
    total_functions: int = 0
    skipped_known_signatures: int = 0
    c_resolved: int = 0
    docstring_resolved: int = 0
    inspect_resolved: int = 0
    unresolved: int = 0

    def __str__(self) -> str:
        return (
            "签名补全汇总: "
            f"函数总数={self.total_functions}, "
            f"跳过已有签名={self.skipped_known_signatures}, "
            f"C源码补全={self.c_resolved}, "
            f"文档字符串补全={self.docstring_resolved}, "
            f"运行时反射补全={self.inspect_resolved}, "
            f"未补全={self.unresolved}"
        )


class SignatureCompleter:
    def __init__(self, options: StubGenerationOptions) -> None:
        self._options = options
        self._c_source = self._build_c_source(options)
        self._summary = SignatureCompletionSummary()

    def run(self, module: IRModule) -> SignatureCompletionSummary:
        self._summary = SignatureCompletionSummary()
        self._complete_module(module)
        return self._summary

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
        self._summary.total_functions += 1
        if func.signatures:
            self._summary.skipped_known_signatures += 1
            return

        if self._c_source is not None:
            c_result = self._c_source.resolve_function(
                module=module,
                func=func,
                is_method=is_method,
            )
            if c_result is not None:
                func.signatures = c_result.signatures
                if self._options.include_c_inferred_source_comment:
                    func.c_inferred_source_comment = c_result.c_inferred_source_comment
                self._summary.c_resolved += 1
                return

        try:
            docstring_result = resolve_docstring_signatures(func_name=func.name, doc=func.doc)
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
                func.signatures = docstring_result.signatures
                self._summary.docstring_resolved += 1
                return

        inspect_result = resolve_inspect_signatures(
            func.runtime_function,
            module_type=module.module_type,
        )
        if inspect_result is not None:
            func.signatures = inspect_result.signatures
            self._summary.inspect_resolved += 1
            return

        self._summary.unresolved += 1

    @staticmethod
    def _build_c_source(options: StubGenerationOptions) -> CExtensionSource | None:
        if options.source_root is None:
            return None
        return CExtensionSource(
            source_root=options.source_root,
            include=options.include,
            include_directory=options.include_directory,
            c_std=options.c_std,
            cpp_std=options.cpp_std,
        )
