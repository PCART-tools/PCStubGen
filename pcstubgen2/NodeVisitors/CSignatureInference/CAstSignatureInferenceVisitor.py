from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .CSignatureExtraction import CSignatureExtractionEngine, ExtractedArgument, ExtractedFunction
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
from ..NodeVisitor import NodeVisitor

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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
        c_source_root: Path,
        clang_parse_args: Iterable[str] = (),
        clang_c_std: str | None = None,
        clang_cpp_std: str | None = None,
    ) -> None:
        """初始化 Visitor 运行配置与 C 签名提取器。"""
        self.error_collector = error_collector
        self.c_source_root = c_source_root
        self.clang_parse_args = list(clang_parse_args)
        self.clang_c_std = clang_c_std
        self.clang_cpp_std = clang_cpp_std
        self._extractor = CSignatureExtractionEngine(
            source_root=self.c_source_root,
            clang_parse_args=self.clang_parse_args,
            clang_c_std=self.clang_c_std,
            clang_cpp_std=self.clang_cpp_std,
        )

    def visit_module(self, node: IRModule) -> None:
        """按模块粒度决定是否启用 C AST 签名补全。"""

        if node.module_type is IRModuleType.C:
            signatures = self._get_signatures()
            if signatures:
                node.functions = self._rewrite_module_functions(node.functions, signatures)

        super().visit_module(node)

    def visit_class(self, node: IRClass, module: IRModule) -> None:
        """在模块启用时重写类方法签名。"""

        if module.module_type is IRModuleType.C:
            signatures = self._get_signatures()
            if signatures:
                node.methods = self._rewrite_class_methods(node.methods, signatures)

        super().visit_class(node, module)

    def _rewrite_module_functions(
        self,
        funcs: list[IRFunction],
        signatures: dict[str, list[ExtractedFunction]],
    ) -> list[IRFunction]:
        """批量重写模块级函数。"""
        new_funcs: list[IRFunction] = []
        for func in funcs:
            new_funcs.extend(self._rewrite_function(func=func, signatures=signatures, is_method=False))
        return new_funcs

    def _rewrite_class_methods(
        self,
        methods: list[IRMethod],
        signatures: dict[str, list[ExtractedFunction]],
    ) -> list[IRMethod]:
        """批量重写类方法，并保留原有 decorator 封装。"""
        new_methods: list[IRMethod] = []
        for method in methods:
            rewritten = self._rewrite_function(
                func=method.function,
                signatures=signatures,
                is_method=True,
            )
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
        """
        用提取结果重写单个函数签名。

        仅处理仍为 generic 占位签名的函数，避免覆盖前序 visitor 已解析出的精确信息。
        """
        if not func.is_generic_signature():
            return [func]

        candidates = signatures.get(func.name)
        if not candidates:
            return [func]

        selected = self._select_candidate(candidates, is_method=is_method)
        if selected is None:
            return [func]

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
        return rewritten if rewritten else [func]

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

    def _get_signatures(self) -> dict[str, list[ExtractedFunction]]:
        """
        按需提取 C AST 结果；缓存由 extraction engine 负责。

        任何提取失败都降级为空结果，保证 stub 生成主流程可持续执行。
        """
        try:
            return self._extractor.extract()
        except Exception as ex:  # pragma: no cover - 防御性分支
            # 提取阶段异常不应阻断整体生成流程。
            logger.warning("Failed to extract C signatures: %s", ex)
            return {}

