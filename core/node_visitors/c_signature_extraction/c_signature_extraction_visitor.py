from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Literal

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
    IRArgumentKind,
    IRFunction,
    IRModule,
    IRModuleType,
    IRSignature,
)

logger = logging.getLogger(__name__)

SignatureLoadStatus = Literal["nonempty", "empty"]
RewriteOutcome = Literal[
    "success",
    "no_candidates",
    "empty_selected_signatures",
]


@dataclasses.dataclass
class _InferenceStats:
    total_unknown_signatures: int = 0
    success: int = 0
    failed: int = 0
    no_candidates: int = 0
    empty_selected_signatures: int = 0
    empty_extract: int = 0


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
        clang_include_directory: list[str] = (),
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
        self._signature_load_status: SignatureLoadStatus | None = None

    def visit_module(self, node: IRModule) -> None:
        """按模块粒度决定是否启用 C AST 签名补全。"""
        if node.module_type is IRModuleType.EXTENSION:
            modules, load_status = self._get_signature_modules()
            extracted_module = self._match_extracted_module(node, modules)
            self._rewrite_module_functions(
                funcs=node.functions,
                extracted_module=extracted_module,
                load_status=load_status,
            )

        super().visit_module(node)

    def log_summary(self, project_name: str) -> None:
        """输出一次项目级 C AST 签名补全统计。"""
        if self._stats.total_unknown_signatures <= 0:
            self._reset_stats()
            return

        logger.info(
            "C AST signature inference summary for %s: "
            "total_unknown_signatures=%d, success=%d, failed=%d, no_candidates=%d, "
            "empty_selected_signatures=%d, empty_extract=%d",
            project_name,
            self._stats.total_unknown_signatures,
            self._stats.success,
            self._stats.failed,
            self._stats.no_candidates,
            self._stats.empty_selected_signatures,
            self._stats.empty_extract,
        )
        self._reset_stats()

    def _rewrite_module_functions(
        self,
        funcs: list[IRFunction],
        extracted_module: ExtractedModule | None,
        *,
        load_status: SignatureLoadStatus,
    ) -> None:
        """批量重写模块级函数。"""
        if load_status != "nonempty":
            self._record_unavailable_extract(
                funcs=[(func, False) for func in funcs],
                failure_key="empty_extract",
            )
            return

        signatures = extracted_module.functions if extracted_module is not None else {}
        for func in funcs:
            had_unknown_signature = self._has_unknown_signatures(func)
            outcome = self._rewrite_function_with_outcome(
                func=func,
                signatures=signatures,
                is_method=False,
            )
            self._record_outcome(
                had_unknown_signature=had_unknown_signature,
                outcome=outcome,
            )

    def _rewrite_function(
        self,
        *,
        func: IRFunction,
        signatures: dict[str, ExtractedFunction],
        is_method: bool,
    ) -> IRFunction:
        """改写单个函数并返回同一节点。"""
        self._rewrite_function_with_outcome(
            func=func,
            signatures=signatures,
            is_method=is_method,
        )
        return func

    def _rewrite_function_with_outcome(
        self,
        *,
        func: IRFunction,
        signatures: dict[str, ExtractedFunction],
        is_method: bool,
    ) -> RewriteOutcome | None:
        """
        用提取结果重写单个函数签名。

        仅处理仍缺失签名的函数，避免覆盖前序 visitor 已解析出的精确信息。
        """
        if not self._has_unknown_signatures(func):
            return None

        selected = signatures.get(func.name)
        if selected is None:
            logger.warning(
                "Failed to rewrite unknown signature for %s (is_method=%s): no C signature candidates found",
                func.name,
                is_method,
            )
            return "no_candidates"

        if not selected.signatures:
            logger.warning(
                "Failed to rewrite unknown signature for %s (is_method=%s): selected candidate has no signatures",
                func.name,
                is_method,
            )
            return "empty_selected_signatures"

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
            "Rewrote unknown signature for %s (is_method=%s): generated_signatures=%d",
            func.name,
            is_method,
            len(rewritten_signatures),
        )
        return "success"

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
            kind = {
                "keyword_only": IRArgumentKind.KEYWORD_ONLY,
                "var_positional": IRArgumentKind.VAR_POSITIONAL,
                "var_keyword": IRArgumentKind.VAR_KEYWORD,
            }.get(arg.kind, IRArgumentKind.POSITIONAL_OR_KEYWORD)

            ir_arg = IRArgument(name=arg.name, kind=kind)
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

    def _build_default_value(self, default_value: str | None) -> str | None:
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
        if not modules:
            return None

        full_name = str(node.full_name)
        exact_matches = [
            module
            for module in modules.values()
            if full_name in module.lookup_names
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            return None

        leaf_name = node.full_name.name
        leaf_matches = [
            module
            for module in modules.values()
            if leaf_name in module.lookup_names
        ]
        if len(leaf_matches) == 1:
            return leaf_matches[0]
        return None

    def _get_signature_modules(self) -> tuple[dict[str, ExtractedModule], SignatureLoadStatus]:
        """
        按需提取 C AST 结果；空结果也缓存到 visitor。

        提取失败时直接向上传播异常，由调用方决定是否中断主流程。
        """
        if self._signature_modules is not None and self._signature_load_status is not None:
            return self._signature_modules, self._signature_load_status

        self._signature_modules = extract_c_signature_modules(
            self._source_root,
            clang_include=self._clang_include,
            clang_include_directory=self._clang_include_directory,
            clang_c_std=self._clang_c_std,
            clang_cpp_std=self._clang_cpp_std,
        )
        self._signature_load_status = "nonempty" if self._signature_modules else "empty"
        return self._signature_modules, self._signature_load_status

    def _record_outcome(
        self,
        *,
        had_unknown_signature: bool,
        outcome: RewriteOutcome | None,
    ) -> None:
        """记录单个函数的改写结果。"""
        if not had_unknown_signature or outcome is None:
            return

        self._stats.total_unknown_signatures += 1
        if outcome == "success":
            self._stats.success += 1
            return

        self._stats.failed += 1
        setattr(self._stats, outcome, getattr(self._stats, outcome) + 1)

    def _record_unavailable_extract(
        self,
        *,
        funcs: list[tuple[IRFunction, bool]],
        failure_key: Literal["empty_extract"],
    ) -> None:
        """记录提取结果整体不可用时的逐项失败。"""
        unknown_count = 0
        reason = "C signature extraction returned no results"
        for func, is_method in funcs:
            if not self._has_unknown_signatures(func):
                continue
            unknown_count += 1
            logger.warning(
                "Failed to rewrite unknown signature for %s (is_method=%s): %s",
                func.name,
                is_method,
                reason,
            )

        if unknown_count > 0:
            self._stats.total_unknown_signatures += unknown_count
            self._stats.failed += unknown_count
            setattr(self._stats, failure_key, getattr(self._stats, failure_key) + unknown_count)

    @staticmethod
    def _has_unknown_signatures(func: IRFunction) -> bool:
        """判断函数是否仍缺失已解析签名。"""
        return len(func.signatures) == 0

    def _reset_stats(self) -> None:
        """重置项目级统计。"""
        self._stats = _InferenceStats()
