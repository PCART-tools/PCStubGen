from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from ...ir import IRArgument, IRFunction, IRModule, IRModuleType, IRSignature
from .collect import collect_modules
from .clang.cursor_utils import source_range_get_text
from .models import CArgument, CFunction, CModule

ResolvedCExtensionFunction: TypeAlias = tuple[list[IRSignature], str | None]


class CExtensionSource:
    def __init__(
        self,
        *,
        source_root: Path,
        include: list[str] = (),
        include_directory: list[Path] = (),
        c_std: str = "c11",
        cpp_std: str = "c++17",
    ) -> None:
        self._modules = collect_modules(
            source_root,
            include=list(include),
            include_directory=list(include_directory),
            c_std=c_std,
            cpp_std=cpp_std,
        )

    def resolve_function(
        self,
        irmodule: IRModule,
        irfunction: IRFunction,
        is_method: bool = False,
    ) -> ResolvedCExtensionFunction | None:
        if is_method:
            return None

        c_module = self._match_c_module(irmodule, self._modules)
        if c_module is None:
            return None

        selected = c_module.functions.get(irfunction.name)
        if selected is None or not selected.signatures:
            return None

        signatures = [
            IRSignature(
                args=[self._build_argument(arg) for arg in sig.arguments],
                return_type=sig.return_type,
            )
            for sig in selected.signatures
        ]

        return signatures, self._get_source_comment(selected)

    @staticmethod
    def _build_argument(argument: CArgument) -> IRArgument:
        return IRArgument(
            name=argument.name,
            type=argument.type,
            default_value=argument.default_value,
            has_default=argument.has_default,
            kind=argument.kind,
        )

    @staticmethod
    def _get_source_comment(c_function: CFunction) -> str | None:
        extent = c_function.function_cursor.extent
        if extent is None:
            return None

        source_text = source_range_get_text(extent)
        if not source_text:
            return None
        return source_text

    @staticmethod
    def _match_c_module(
        node: IRModule,
        modules: dict[str, CModule],
    ) -> CModule | None:
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
