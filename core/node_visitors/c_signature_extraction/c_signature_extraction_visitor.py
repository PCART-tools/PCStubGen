from __future__ import annotations

import dataclasses
from pathlib import Path
from loguru import logger

from .core import (
    ExtractedArgument,
    ExtractedFunction,
    ExtractedModule,
    extract_c_signature_modules,
)
from .core.constants import METH_CLASS, METH_STATIC
from ..node_visitor import NodeVisitor
from ...ir import (
    IRArgument,
    IRFunction,
    IRModule,
    IRModuleType,
    IRSignature,
)


@dataclasses.dataclass
class _InferenceStats:
    total_unknown_signatures: int = 0
    success: int = 0
    missing_module_match: int = 0
    missing_function_match: int = 0
    matched_function_without_signatures: int = 0


class CSignatureExtractionVisitor(NodeVisitor):
    """
    使用 C AST 提取结果补全未知函数签名。

    执行顺序设计为：
    1) DocStringSignatureParserVisitor
    2) CAstSignatureInferenceVisitor（本 visitor）
    3) InferMethodModifierVisitor
    """

    def __init__(
        self,
        *,
        source_root: Path,
        clang_include: list[str] = (),
        clang_include_directory: list[Path] = (),
        clang_c_std: str = "c11",
        clang_cpp_std: str = "c++17",
    ) -> None:
        """初始化 Visitor 运行配置与提取结果缓存。"""
        self._source_root = source_root
        self._clang_include = list(clang_include)
        self._clang_include_directory = list(clang_include_directory)
        self._clang_c_std = clang_c_std
        self._clang_cpp_std = clang_cpp_std
        self._stats = _InferenceStats()
        self._signature_modules: dict[str, ExtractedModule] | None = None

    def visit_module(self, node: IRModule) -> None:
        """按模块粒度决定是否启用 C AST 签名补全。"""
        if node.module_type is IRModuleType.EXTENSION:
            module_full_name = str(node.full_name)
            modules = self._get_signature_modules()
            extracted_module = self._match_extracted_module(node, modules)
            if extracted_module is None:
                self._record_missing_module_match(
                    funcs=node.functions,
                    module_full_name=module_full_name,
                )
            else:
                for func in node.functions:
                    self._rewrite_function(
                        func=func,
                        signatures=extracted_module.functions,
                        is_method=False,
                        module_full_name=module_full_name,
                    )

        super().visit_module(node)

    def log_summary(self) -> None:
        """输出一次项目级 C AST 签名补全统计。"""
        if self._stats.total_unknown_signatures <= 0:
            self._reset_stats()
            return

        logger.info(
            "C AST 签名推断汇总: "
            "total_unknown_signatures={}, success={}, missing_module_match={}, missing_function_match={}, "
            "matched_function_without_signatures={}",
            self._stats.total_unknown_signatures,
            self._stats.success,
            self._stats.missing_module_match,
            self._stats.missing_function_match,
            self._stats.matched_function_without_signatures,
        )
        self._reset_stats()

    def _rewrite_function(
        self,
        *,
        func: IRFunction,
        signatures: dict[str, ExtractedFunction],
        is_method: bool,
        module_full_name: str,
    ) -> None:
        """
        用提取结果原地重写单个函数签名并记录统计。

        仅处理仍缺失签名的函数，避免覆盖前序 visitor 已解析出的精确信息。
        """
        if not self._has_unknown_signatures(func):
            return

        self._stats.total_unknown_signatures += 1

        selected = signatures.get(func.name)
        if selected is None:
            logger.warning(
                "重写签名失败, 未找到 C 函数, module_name: {}, func_name: {}, is_method: {}",
                module_full_name,
                func.name,
                is_method,
            )
            self._stats.missing_function_match += 1
            return

        if not selected.signatures:
            logger.warning(
                "重写签名失败, 选中的 candidate 不包含 signatures, module_name: {}, func_name: {}, is_method: {}",
                module_full_name,
                func.name,
                is_method,
            )
            self._stats.matched_function_without_signatures += 1
            return

        rewritten_signatures: list[IRSignature] = []
        for sig in selected.signatures:
            args = self._build_ir_arguments(
                arguments=sig.arguments,
                is_method=is_method,
                ml_flags=selected.ml_flags,
            )
            rewritten_signatures.append(
                IRSignature(
                    args=args,
                    return_type_name=self._build_annotation(sig.return_type_name),
                    doc=func.doc,
                )
            )

        func.signatures = rewritten_signatures
        logger.info(
            "重写签名成功, func_name: {}, is_method: {}: generated_signatures={}",
            func.name,
            is_method,
            len(rewritten_signatures),
        )
        self._stats.success += 1

    def _build_ir_arguments(
        self,
        *,
        arguments: list[ExtractedArgument],
        is_method: bool,
        ml_flags: int,
    ) -> list[IRArgument]:
        """
        将提取参数转换为 IR 参数，并修正方法首参语义。

        对模块函数会剔除误带的 `self/cls`，避免生成错误 API。
        """
        normalized = list(arguments)

        if is_method:
            if ml_flags & METH_STATIC:
                while normalized and normalized[0].name in {"self", "cls"}:
                    normalized.pop(0)
            if not normalized:
                if ml_flags & METH_STATIC:
                    normalized = []
                elif ml_flags & METH_CLASS:
                    normalized = [ExtractedArgument(name="cls", type_name="type")]
                else:
                    normalized = [ExtractedArgument(name="self", type_name="object")]
        else:
            while normalized and normalized[0].name in {"self", "cls"}:
                normalized.pop(0)

        result: list[IRArgument] = []
        for arg in normalized:
            ir_arg = IRArgument(name=arg.name, kind=arg.kind)
            ir_arg.type_name = self._build_annotation(arg.type_name)
            ir_arg.default_value = self._build_default_value(arg.default_value)
            result.append(ir_arg)
        return result

    @staticmethod
    def _build_annotation(type_name: str | None) -> str | None:
        """清理提取结果里的注解文本，仅过滤空白值。"""
        if type_name is None:
            return None
        text = type_name.strip()
        if not text:
            return None
        return text

    @staticmethod
    def _build_default_value(default_value: str | None) -> str | None:
        """清理提取结果里的默认值文本，仅过滤空白值。"""
        if default_value is None:
            return None
        expr = default_value.strip()
        if expr == "":
            return None
        return expr

    def _match_extracted_module(
        self,
        node: IRModule,
        modules: dict[str, ExtractedModule],
    ) -> ExtractedModule | None:
        """在提取结果里匹配当前模块节点。"""
        # 先按全名查找C模块，再按全名的最后一节查找
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

    def _get_signature_modules(self) -> dict[str, ExtractedModule]:
        """
        按需提取 C AST 结果；空结果也缓存到 visitor。

        提取失败时直接向上传播异常，由调用方决定是否中断主流程。
        """
        if self._signature_modules is not None:
            return self._signature_modules

        self._signature_modules = extract_c_signature_modules(
            self._source_root,
            clang_include=self._clang_include,
            clang_include_directory=self._clang_include_directory,
            clang_c_std=self._clang_c_std,
            clang_cpp_std=self._clang_cpp_std,
        )
        return self._signature_modules

    def _record_missing_module_match(
        self,
        *,
        funcs: list[IRFunction],
        module_full_name: str,
    ) -> None:
        """记录未匹配到提取模块时仍缺失签名的函数。"""
        unknown_count = 0
        for func in funcs:
            if not self._has_unknown_signatures(func):
                continue
            unknown_count += 1
            logger.warning(
                "重写签名失败, 未找到 C 模块, module_name: {}, func_name: {}",
                module_full_name,
                func.name,
            )

        if unknown_count > 0:
            self._stats.total_unknown_signatures += unknown_count
            self._stats.missing_module_match += unknown_count

    @staticmethod
    def _has_unknown_signatures(func: IRFunction) -> bool:
        """判断函数是否仍缺失已解析签名。"""
        return len(func.signatures) == 0

    def _reset_stats(self) -> None:
        """重置项目级统计。"""
        self._stats = _InferenceStats()
