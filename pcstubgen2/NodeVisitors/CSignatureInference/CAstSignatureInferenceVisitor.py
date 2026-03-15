from __future__ import annotations

import ast
import dataclasses
import logging
import re
from pathlib import Path
from typing import Literal

from .CSignatureExtraction import CSignatureExtractor, ExtractedArgument, ExtractedFunction
from ..NodeVisitor import NodeVisitor
from ...ErrorCollector import ErrorCollector
from ...Errors import InvalidExpressionError
from ...IR import (
    IRArgument,
    IRArgumentKind,
    IRClass,
    IRFunction,
    InvalidExpression,
    IRMethod,
    IRModule,
    IRModuleType,
    IRValue,
    QualifiedName,
    ResolvedType,
)

logger = logging.getLogger(__name__)

SignatureLoadStatus = Literal["nonempty", "empty"]
RewriteOutcome = Literal[
    "success",
    "no_candidates",
    "candidate_selection_failed",
    "empty_selected_signatures",
]


@dataclasses.dataclass
class _InferenceStats:
    total_generic: int = 0
    success: int = 0
    failed: int = 0
    no_candidates: int = 0
    candidate_selection_failed: int = 0
    empty_selected_signatures: int = 0
    empty_extract: int = 0


class CAstSignatureInferenceVisitor(NodeVisitor):
    """
    使用 C AST 提取结果补全 generic signature。

    执行顺序设计为：
    1) DocStringSignatureParserVisitor
    2) CAstSignatureInferenceVisitor（本 visitor）
    3) InferMethodModifierVisitor
    """

    def __init__(
        self,
        *,
        error_collector: ErrorCollector,
        source_root: Path,
        clang_include: list[str] = (),
        clang_include_directory: list[str] = (),
        clang_c_std: str = "c11",
        clang_cpp_std: str = "c++17",
    ) -> None:
        """初始化 Visitor 运行配置与 C 签名提取器。"""
        self.error_collector = error_collector
        self._extractor = CSignatureExtractor(
            source_root=source_root,
            clang_include=clang_include,
            clang_include_directory=clang_include_directory,
            clang_c_std=clang_c_std,
            clang_cpp_std=clang_cpp_std,
        )
        self._stats = _InferenceStats()
        self._signatures: dict[str, list[ExtractedFunction]] | None = None
        self._signature_load_status: SignatureLoadStatus | None = None

    def visit_module(self, node: IRModule) -> None:
        """按模块粒度决定是否启用 C AST 签名补全。"""

        if node.module_type is IRModuleType.EXTENSION:
            signatures, load_status = self._get_signatures()
            node.functions = self._rewrite_module_functions(
                node.functions,
                signatures,
                load_status=load_status,
            )

        super().visit_module(node)

    def visit_class(self, node: IRClass, module: IRModule) -> None:
        """在模块启用时重写类方法签名。"""

        if module.module_type is IRModuleType.EXTENSION:
            signatures, load_status = self._get_signatures()
            node.methods = self._rewrite_class_methods(
                node.methods,
                signatures,
                load_status=load_status,
            )

        super().visit_class(node, module)

    def log_summary(self, project_name: str) -> None:
        if self._stats.total_generic <= 0:
            self._reset_stats()
            return

        logger.info(
            "C AST signature inference summary for %s: "
            "total_generic=%d, success=%d, failed=%d, no_candidates=%d, "
            "candidate_selection_failed=%d, empty_selected_signatures=%d, "
            "empty_extract=%d",
            project_name,
            self._stats.total_generic,
            self._stats.success,
            self._stats.failed,
            self._stats.no_candidates,
            self._stats.candidate_selection_failed,
            self._stats.empty_selected_signatures,
            self._stats.empty_extract,
        )
        self._reset_stats()

    def _rewrite_module_functions(
        self,
        funcs: list[IRFunction],
        signatures: dict[str, list[ExtractedFunction]],
        *,
        load_status: SignatureLoadStatus,
    ) -> list[IRFunction]:
        """批量重写模块级函数。"""
        if load_status != "nonempty":
            self._record_unavailable_extract(
                funcs=[(func, False) for func in funcs],
                failure_key="empty_extract",
            )
            return funcs

        new_funcs: list[IRFunction] = []
        for func in funcs:
            rewritten, outcome = self._rewrite_function_with_outcome(
                func=func,
                signatures=signatures,
                is_method=False,
            )
            self._record_outcome(func=func, outcome=outcome)
            new_funcs.extend(rewritten)
        return new_funcs

    def _rewrite_class_methods(
        self,
        methods: list[IRMethod],
        signatures: dict[str, list[ExtractedFunction]],
        *,
        load_status: SignatureLoadStatus,
    ) -> list[IRMethod]:
        """批量重写类方法，并保留原有 decorator 封装。"""
        if load_status != "nonempty":
            self._record_unavailable_extract(
                funcs=[(method.function, True) for method in methods],
                failure_key="empty_extract",
            )
            return methods

        new_methods: list[IRMethod] = []
        for method in methods:
            rewritten, outcome = self._rewrite_function_with_outcome(
                func=method.function,
                signatures=signatures,
                is_method=True,
            )
            self._record_outcome(func=method.function, outcome=outcome)
            if len(rewritten) == 1 and rewritten[0] is method.function:
                # 未发生替换时直接复用原对象，减少不必要重建。
                new_methods.append(method)
                continue
            for func in rewritten:
                new_methods.append(IRMethod(function=func, decorator=method.decorator))
        return new_methods

    def _rewrite_function(
        self,
        *,
        func: IRFunction,
        signatures: dict[str, list[ExtractedFunction]],
        is_method: bool,
    ) -> list[IRFunction]:
        rewritten, _ = self._rewrite_function_with_outcome(
            func=func,
            signatures=signatures,
            is_method=is_method,
        )
        return rewritten

    def _rewrite_function_with_outcome(
        self,
        *,
        func: IRFunction,
        signatures: dict[str, list[ExtractedFunction]],
        is_method: bool,
    ) -> tuple[list[IRFunction], RewriteOutcome | None]:
        """
        用提取结果重写单个函数签名。

        仅处理仍为 generic 占位签名的函数，避免覆盖前序 visitor 已解析出的精确信息。
        """
        if not func.is_generic_signature():
            return [func], None

        candidates = signatures.get(func.name)
        if not candidates:
            logger.warning(
                "Failed to rewrite generic signature for %s (is_method=%s): no C signature candidates found",
                func.name,
                is_method,
            )
            return [func], "no_candidates"

        selected = self._select_candidate(candidates, is_method=is_method)
        if selected is None:
            logger.warning(
                "Failed to rewrite generic signature for %s (is_method=%s): candidate selection failed",
                func.name,
                is_method,
            )
            return [func], "candidate_selection_failed"

        if not selected.signatures:
            logger.warning(
                "Failed to rewrite generic signature for %s (is_method=%s): selected candidate has no signatures",
                func.name,
                is_method,
            )
            return [func], "empty_selected_signatures"

        overload = len(selected.signatures) > 1
        rewritten: list[IRFunction] = []
        for sig in selected.signatures:
            args = self._build_ir_arguments(
                arguments=sig.arguments,
                is_method=is_method,
                method_flags=selected.method_flags,
            )
            inferred_return = self._build_annotation(sig.return_type_name)
            return_annotation = inferred_return if inferred_return is not None else func.return_annotation
            rewritten.append(
                IRFunction(
                    name=func.name,
                    args=args,
                    return_annotation=return_annotation,
                    # 多签名时转为 overload，避免单条 doc 误导到某个具体重载。
                    doc=func.doc if not overload else None,
                    decorators=["typing.overload"] if overload else list(func.decorators),
                )
            )
        if rewritten:
            logger.info(
                "Rewrote generic signature for %s (is_method=%s): selected_candidates=%d, generated_signatures=%d",
                func.name,
                is_method,
                len(candidates),
                len(rewritten),
            )
        return (rewritten if rewritten else [func]), "success"

    def _build_ir_arguments(
        self,
        *,
        arguments: list[ExtractedArgument],
        is_method: bool,
        method_flags: list[str],
    ) -> list[IRArgument]:
        """
        将提取参数转换为 IR 参数，并修正方法首参语义。

        对模块函数会剔除误带的 `self/cls`，避免生成错误 API。
        """
        normalized = list(arguments)

        if is_method:
            if "METH_STATIC" in method_flags:
                while normalized and normalized[0].name in {"self", "cls"}:
                    normalized.pop(0)
            if not normalized:
                if "METH_STATIC" in method_flags:
                    normalized = []
                elif "METH_CLASS" in method_flags:
                    normalized = [ExtractedArgument(name="cls", type_name="type")]
                else:
                    normalized = [ExtractedArgument(name="self", type_name="object")]
        else:
            # 某些提取样本会错误携带实例首参，这里统一清洗。
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
            ir_arg.annotation = self._build_annotation(arg.type_name)
            ir_arg.default = self._build_default_value(arg.default_value)
            result.append(ir_arg)
        return result

    def _build_annotation(self, type_name: str | None) -> ResolvedType | None:
        """把字符串类型名转换为 `ResolvedType`，仅接受合法 dotted name。"""
        if type_name is None:
            return None
        text = type_name.strip()
        if not text:
            return None
        if not re.match(r"^[_A-Za-z]\w*(\.[_A-Za-z]\w*)*$", text):
            return None
        return ResolvedType(name=QualifiedName.from_str(text))

    def _build_default_value(self, default_value: str | None) -> IRValue | InvalidExpression | None:
        """
        构建默认值表达式节点。

        `ast.parse` 用于语法合法性检查；`literal_eval` 仅用于判断是否可安全打印。
        """
        if default_value is None:
            return None
        expr = default_value.strip()
        if expr == "":
            return None

        try:
            ast.parse(expr)
        except SyntaxError:
            self.error_collector.report_error(InvalidExpressionError(expr))
            return InvalidExpression(expr)

        is_print_safe = False
        try:
            ast.literal_eval(expr)
            is_print_safe = True
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            pass
        return IRValue(repr=expr, is_print_safe=is_print_safe)

    def _select_candidate(
        self,
        candidates: list[ExtractedFunction],
        *,
        is_method: bool,
    ) -> ExtractedFunction | None:
        """
        从同名候选中选择最匹配的提取结果。

        评分策略为启发式：优先匹配方法/函数首参特征，再结合方法标志与重载数量。
        """
        if not candidates:
            return None

        def candidate_score(item: ExtractedFunction) -> int:
            first_arg = None
            if item.signatures and item.signatures[0].arguments:
                first_arg = item.signatures[0].arguments[0].name

            score = 0
            if is_method:
                if first_arg in {"self", "cls"}:
                    score += 3
                if "METH_CLASS" in item.method_flags:
                    score += 1
            else:
                if first_arg in {"self", "cls"}:
                    score -= 3
                else:
                    score += 1
                if "METH_STATIC" in item.method_flags:
                    score += 1

            score += len(item.signatures)
            return score

        return max(candidates, key=candidate_score)

    def _get_signatures(self) -> tuple[dict[str, list[ExtractedFunction]], SignatureLoadStatus]:
        """
        按需提取 C AST 结果；缓存由 extraction engine 负责。

        提取失败时直接向上传播异常，由调用方决定是否中断主流程。
        """
        if self._signatures is not None and self._signature_load_status is not None:
            return self._signatures, self._signature_load_status

        self._signatures = self._extractor.extract()
        self._signature_load_status = "nonempty" if self._signatures else "empty"
        return self._signatures, self._signature_load_status

    def _record_outcome(self, *, func: IRFunction, outcome: RewriteOutcome | None) -> None:
        if not func.is_generic_signature() or outcome is None:
            return

        self._stats.total_generic += 1
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
        generic_count = 0
        reason = "C signature extraction returned no results"
        for func, is_method in funcs:
            if not func.is_generic_signature():
                continue
            generic_count += 1
            logger.warning(
                "Failed to rewrite generic signature for %s (is_method=%s): %s",
                func.name,
                is_method,
                reason,
            )

        if generic_count > 0:
            self._stats.total_generic += generic_count
            self._stats.failed += generic_count
            setattr(self._stats, failure_key, getattr(self._stats, failure_key) + generic_count)

    def _reset_stats(self) -> None:
        self._stats = _InferenceStats()
