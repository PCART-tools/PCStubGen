from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from ...ir_modules import IRArgument, IRFunction, IRModule, IRModuleType, IRSignature
from .collect import collect_modules
from .clang.cursor_utils import source_range_get_text
from .models import CArgument, CFunction, CModule

ResolvedCExtensionFunction: TypeAlias = tuple[list[IRSignature], str | None]


class CExtensionSource:
    def __init__(
        self,
        *,
        compilation_database: Path,
    ) -> None:
        self._modules = collect_modules(
            compilation_database,
        )

    def resolve_function(
        self,
        irmodule: IRModule,
        irfunction: IRFunction,
        is_method: bool = False,
    ) -> ResolvedCExtensionFunction:
        if is_method:
            raise RuntimeError("C源码补全暂不支持方法。")

        if irmodule.module_type is not IRModuleType.EXTENSION:
            raise RuntimeError(f"模块 {irmodule.full_name} 不是扩展模块。")

        c_module = self._match_c_module(irmodule, self._modules)
        if c_module is None:
            raise RuntimeError(f"未匹配到唯一C模块: {irmodule.full_name}")

        selected = c_module.functions.get(irfunction.name)
        if selected is None:
            raise RuntimeError(f"C模块 {c_module.name} 中未找到函数 {irfunction.name}")
        if not selected.signatures:
            raise RuntimeError(f"C函数 {c_module.name}.{irfunction.name} 没有可用签名")

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
        return modules.get(node.full_name.name)
