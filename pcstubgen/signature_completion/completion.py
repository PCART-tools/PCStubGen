from __future__ import annotations

import dataclasses

from loguru import logger

from .c_extension import CExtensionSource
from .docstring_source import resolve_docstring_signatures
from .inspect_source import resolve_inspect_signatures
from ..ir import IRArgument, IRClass, IRFunction, IRMethod, IRModule, IRSignature
from .models import ResolvedArgument, ResolvedFunctionSignatures, ResolvedSignature
from ..stub_generation_options import StubGenerationOptions


@dataclasses.dataclass
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
            f"total_functions={self.total_functions}, "
            f"skipped_known_signatures={self.skipped_known_signatures}, "
            f"c_resolved={self.c_resolved}, "
            f"docstring_resolved={self.docstring_resolved}, "
            f"inspect_resolved={self.inspect_resolved}, "
            f"unresolved={self.unresolved}"
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

        resolved = self._resolve_function(func=func, module=module, is_method=is_method)
        if resolved is None:
            self._summary.unresolved += 1
            return

        source, result = resolved
        func.signatures = [self._build_ir_signature(sig, fallback_doc=func.doc) for sig in result.signatures]
        if source == "c" and self._options.include_c_inferred_source_comment:
            func.c_inferred_source_comment = result.c_inferred_source_comment

        match source:
            case "c":
                self._summary.c_resolved += 1
            case "docstring":
                self._summary.docstring_resolved += 1
            case "inspect":
                self._summary.inspect_resolved += 1

    def _resolve_function(
        self,
        *,
        func: IRFunction,
        module: IRModule,
        is_method: bool,
    ) -> tuple[str, ResolvedFunctionSignatures] | None:
        if self._c_source is not None:
            c_result = self._c_source.resolve_function(
                module=module,
                func=func,
                is_method=is_method,
            )
            if c_result is not None:
                return ("c", c_result)

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
                return ("docstring", docstring_result)

        inspect_result = resolve_inspect_signatures(
            func.runtime_function,
            module_type=module.module_type,
        )
        if inspect_result is not None:
            return ("inspect", inspect_result)
        return None

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

    @staticmethod
    def _build_ir_signature(
        signature: ResolvedSignature,
        *,
        fallback_doc: str | None,
    ) -> IRSignature:
        return IRSignature(
            args=[SignatureCompleter._build_ir_argument(arg) for arg in signature.arguments],
            return_type=signature.return_type,
            doc=signature.doc if signature.doc is not None else fallback_doc,
        )

    @staticmethod
    def _build_ir_argument(argument: ResolvedArgument) -> IRArgument:
        return IRArgument(
            name=argument.name,
            type=argument.type,
            default_value=argument.default_value,
            has_default=argument.has_default,
            kind=argument.kind,
        )
