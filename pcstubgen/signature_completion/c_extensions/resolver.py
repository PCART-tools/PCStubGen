from __future__ import annotations

from pathlib import Path

from ...ir import IRFunction, IRModule, IRModuleType
from ..models import (
    ResolvedArgument,
    ResolvedFunctionSignatures,
    ResolvedSignature,
)
from .c_signature_extraction import extract_c_signature_modules
from .cursor_utils import source_range_get_text
from .models import ExtractedArgument, ExtractedFunction, ExtractedModule


class CSignatureResolver:
    def __init__(
        self,
        *,
        source_root: Path,
        include: list[str] = (),
        include_directory: list[Path] = (),
        c_std: str = "c11",
        cpp_std: str = "c++17",
    ) -> None:
        self._modules = extract_c_signature_modules(
            source_root,
            include=list(include),
            include_directory=list(include_directory),
            c_std=c_std,
            cpp_std=cpp_std,
        )

    def resolve_function(
        self,
        *,
        module: IRModule,
        func: IRFunction,
        is_method: bool = False,
    ) -> ResolvedFunctionSignatures | None:
        if is_method:
            return None
        if module.module_type is not IRModuleType.EXTENSION:
            return None

        extracted_module = self._match_extracted_module(module, self._modules)
        if extracted_module is None:
            return None

        selected = extracted_module.functions.get(func.name)
        if selected is None or not selected.signatures:
            return None

        signatures = [
            ResolvedSignature(
                arguments=[self._build_argument(arg) for arg in sig.arguments],
                return_type=sig.return_type,
            )
            for sig in selected.signatures
        ]

        return ResolvedFunctionSignatures(
            signatures=signatures,
            c_inferred_source_comment=self._get_source_comment(selected),
        )

    @staticmethod
    def _build_argument(argument: ExtractedArgument) -> ResolvedArgument:
        return ResolvedArgument(
            name=argument.name,
            type=argument.type,
            default_value=argument.default_value,
            has_default=argument.has_default,
            kind=argument.kind,
        )

    @staticmethod
    def _get_source_comment(extracted_function: ExtractedFunction) -> str | None:
        extent = extracted_function.function_cursor.extent
        if extent is None:
            return None

        source_text = source_range_get_text(extent)
        if not source_text:
            return None
        return source_text

    @staticmethod
    def _match_extracted_module(
        node: IRModule,
        modules: dict[str, ExtractedModule],
    ) -> ExtractedModule | None:
        full_name = str(node.full_name)
        exact_matches = [
            module
            for module in modules.values()
            if module.name == full_name
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]

        leaf_name = node.full_name.name
        leaf_matches = [
            module
            for module in modules.values()
            if module.name == leaf_name
        ]
        if len(leaf_matches) == 1:
            return leaf_matches[0]
        return None
