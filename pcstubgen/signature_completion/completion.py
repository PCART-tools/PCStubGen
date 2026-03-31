from __future__ import annotations

import dataclasses

from loguru import logger

from .c_extensions import CSignatureResolver
from .docstring_source import resolve_docstring_signatures
from .inspect_source import resolve_inspect_signatures
from ..ir import IRArgument, IRClass, IRFunction, IRMethod, IRModule, IRSignature
from .models import ResolvedArgument, ResolvedFunctionSignatures, ResolvedSignature
from ..stub_generation_options import StubGenerationOptions


@dataclasses.dataclass
class SignatureSupplementSummary:
    total_functions: int = 0
    skipped_known_signatures: int = 0
    c_resolved: int = 0
    docstring_resolved: int = 0
    inspect_resolved: int = 0
    unresolved: int = 0

    def log_summary(self) -> None:
        if self.total_functions <= 0:
            return
        logger.info(
            "签名补全汇总: total_functions={}, skipped_known_signatures={}, c_resolved={}, "
            "docstring_resolved={}, inspect_resolved={}, unresolved={}",
            self.total_functions,
            self.skipped_known_signatures,
            self.c_resolved,
            self.docstring_resolved,
            self.inspect_resolved,
            self.unresolved,
        )


class SignatureSupplementer:
    def __init__(self, options: StubGenerationOptions) -> None:
        self._options = options
        self._c_resolver = self._build_c_resolver(options)

    def supplement(self, module: IRModule) -> SignatureSupplementSummary:
        summary = SignatureSupplementSummary()
        self._supplement_module(module, summary)
        return summary

    def _supplement_module(
        self,
        module: IRModule,
        summary: SignatureSupplementSummary,
    ) -> None:
        for sub_module in module.sub_modules:
            self._supplement_module(sub_module, summary)

        for cls in module.classes:
            self._supplement_class(cls, module, summary)

        for func in module.functions:
            self._supplement_function(func, module, summary, is_method=False)

    def _supplement_class(
        self,
        node: IRClass,
        module: IRModule,
        summary: SignatureSupplementSummary,
    ) -> None:
        for nested_cls in node.classes:
            self._supplement_class(nested_cls, module, summary)

        for method in node.methods:
            self._supplement_method(method, module, summary)

    def _supplement_method(
        self,
        method: IRMethod,
        module: IRModule,
        summary: SignatureSupplementSummary,
    ) -> None:
        self._supplement_function(method.function, module, summary, is_method=True)

    def _supplement_function(
        self,
        func: IRFunction,
        module: IRModule,
        summary: SignatureSupplementSummary,
        *,
        is_method: bool,
    ) -> None:
        summary.total_functions += 1
        if func.signatures:
            summary.skipped_known_signatures += 1
            return

        resolved = self._resolve_function(func=func, module=module, is_method=is_method)
        if resolved is None:
            summary.unresolved += 1
            return

        source, result = resolved
        func.signatures = [self._build_ir_signature(sig, fallback_doc=func.doc) for sig in result.signatures]
        if source == "c" and self._options.include_c_inferred_source_comment:
            func.c_inferred_source_comment = result.c_inferred_source_comment

        match source:
            case "c":
                summary.c_resolved += 1
            case "docstring":
                summary.docstring_resolved += 1
            case "inspect":
                summary.inspect_resolved += 1

    def _resolve_function(
        self,
        *,
        func: IRFunction,
        module: IRModule,
        is_method: bool,
    ) -> tuple[str, ResolvedFunctionSignatures] | None:
        if self._c_resolver is not None:
            c_result = self._c_resolver.resolve_function(
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
    def _build_c_resolver(options: StubGenerationOptions) -> CSignatureResolver | None:
        if options.source_root is None:
            return None
        return CSignatureResolver(
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
            args=[SignatureSupplementer._build_ir_argument(arg) for arg in signature.arguments],
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


def supplement_signatures(
    module: IRModule,
    options: StubGenerationOptions,
) -> SignatureSupplementSummary:
    return SignatureSupplementer(options).supplement(module)
