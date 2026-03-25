from __future__ import annotations

import re

from ..ir import (
    IRArgument,
    IRArgumentKind,
    IRClass,
    IRFunction,
    IRModule,
    IRSignature,
)
from .node_visitor import NodeVisitor


class DocStringSignatureParserVisitor(NodeVisitor):
    """
    解析文档字符串中的函数和方法签名。
    """

    _arg_star_name_regex = re.compile(
        r"^\s*(?P<stars>\*{1,2})?" r"\s*(?P<name>\w+)\s*$"
    )

    def visit_module(self, node: IRModule) -> None:
        """在模块层原地补全文档字符串签名。"""
        for func in node.functions:
            self._parse_function(func)

        super().visit_module(node)

    def visit_class(self, node: IRClass, module: IRModule) -> None:
        """在类层原地补全文档字符串签名。"""
        for method in node.methods:
            self._parse_function(method.function)

        super().visit_class(node, module)

    def _parse_function(self, func: IRFunction) -> None:
        """就地解析仍缺失签名的函数节点。"""
        if func.signatures:
            return

        if not func.doc:
            return

        parsed_signatures = self.parse_function_docstring(
            func_name=func.name,
            doc_lines=func.doc.splitlines(),
        )
        if parsed_signatures:
            func.signatures = parsed_signatures

    def parse_function_docstring(
        self, func_name: str, doc_lines: list[str]
    ) -> list[IRSignature]:
        """
        解析函数文档字符串中的签名。

        Example（单签名）:
            add(a: int, b: int = 0) -> int
            Return a + b.

        Example（pybind11 重载）:
            add(*args, **kwargs)
            Overloaded function.
            1. add(a: int, b: int) -> int
            2. add(a: float, b: float) -> float
        """
        if len(doc_lines) == 0:
            return []

        top_signature_regex = re.compile(
            rf"^{re.escape(func_name)}\((?P<args>.*)\)\s*(->\s*(?P<returns>.+))?$"
        )

        match = top_signature_regex.match(doc_lines[0])
        if match is None:
            return []

        if len(doc_lines) < 2 or doc_lines[1].strip() != "Overloaded function.":
            args = self.parse_args_str(match.group("args"))
            if args is None:
                return []

            returns = self.parse_annotation_str(match.group("returns") or "")
            return [
                IRSignature(
                    args=args,
                    doc=self._strip_empty_lines(doc_lines[1:]),
                    return_type_name=returns,
                )
            ]

        overload_signature_regex = re.compile(
            rf"^(\s*(?P<overload_number>\d+).\s*)"
            rf"{re.escape(func_name)}\((?P<args>.*)\)\s*->\s*(?P<returns>.+)$"
        )

        doc_start = 0
        overloads: list[IRSignature] = []

        for i in range(2, len(doc_lines)):
            match = overload_signature_regex.match(doc_lines[i])
            if match is None:
                continue

            if match.group("overload_number") != f"{len(overloads) + 1}":
                continue

            if overloads:
                overloads[-1].doc = self._strip_empty_lines(doc_lines[doc_start:i])

            args = self.parse_args_str(match.group("args"))
            if args is None:
                return []

            overloads.append(
                IRSignature(
                    args=args,
                    return_type_name=self.parse_annotation_str(match.group("returns")),
                )
            )
            doc_start = i + 1

        if not overloads:
            return []

        overloads[-1].doc = self._strip_empty_lines(doc_lines[doc_start:])
        return overloads

    def parse_args_str(self, args_str: str) -> list[IRArgument] | None:
        """解析签名中的参数串。"""
        split_args = self._split_args_str(args_str)
        if split_args is None:
            return None

        result: list[IRArgument] = []
        kw_only_section = False
        for arg_str, annotation_str, default_str in split_args:
            if arg_str.strip() == "/":
                for arg in result:
                    if arg.kind is IRArgumentKind.POSITIONAL_OR_KEYWORD:
                        arg.kind = IRArgumentKind.POSITIONAL_ONLY
                continue
            if arg_str.strip() == "*":
                kw_only_section = True
                continue
            match = self._arg_star_name_regex.match(arg_str)
            if match is None:
                return None
            name = match.group("name")

            stars = match.group("stars")
            if stars == "*":
                kind = IRArgumentKind.VAR_POSITIONAL
                kw_only_section = True
            elif stars == "**":
                kind = IRArgumentKind.VAR_KEYWORD
            elif kw_only_section:
                kind = IRArgumentKind.KEYWORD_ONLY
            else:
                kind = IRArgumentKind.POSITIONAL_OR_KEYWORD

            annotation = None
            if annotation_str is not None:
                annotation = self.parse_annotation_str(annotation_str)

            default = None
            if default_str is not None:
                default = self.parse_value_str(default_str)

            result.append(
                IRArgument(
                    name=name,
                    default_value=default,
                    type_name=annotation,
                    kind=kind,
                )
            )
        return result

    @staticmethod
    def parse_annotation_str(annotation_str: str) -> str | None:
        """清理注解文本中的空白。"""
        text = annotation_str.strip()
        return text or None

    def parse_value_str(self, value: str) -> str | None:
        """解析参数默认值文本。"""
        strip_expr = value.strip()
        if not strip_expr:
            return None
        return strip_expr

    def _split_args_str(
        self, args_str: str
    ) -> list[tuple[str, str | None, str | None]] | None:
        """按顶层逗号分割参数串。"""
        result = []
        closing = {"(": ")", "{": "}", "[": "]"}
        stack = []
        i = 0
        arg_begin = 0
        semicolon_pos: int | None = None
        eq_sign_pos: int | None = None

        def add_arg() -> None:
            nonlocal semicolon_pos
            nonlocal eq_sign_pos
            annotation = None
            default = None

            arg_end = i

            if eq_sign_pos is not None:
                arg_end = eq_sign_pos
                default = args_str[eq_sign_pos + 1 : i]

            if semicolon_pos is not None:
                annotation = args_str[semicolon_pos + 1 : arg_end]
                arg_end = semicolon_pos

            name = args_str[arg_begin:arg_end]
            result.append((name, annotation, default))
            semicolon_pos = None
            eq_sign_pos = None

        while i < len(args_str):
            c = args_str[i]
            if c in "\"'":
                str_end = self._find_str_end(args_str, i)
                if str_end is None:
                    return None
                i = str_end
            elif c in closing:
                stack.append(closing[c])
            elif len(stack) == 0:
                if c == ",":
                    add_arg()
                    arg_begin = i + 1
                elif c == ":" and semicolon_pos is None:
                    semicolon_pos = i
                elif c == "=" and args_str[i : i + 2] != "==":
                    eq_sign_pos = i
            elif stack[-1] == c:
                stack.pop()
            i += 1

        if len(stack) != 0:
            return None

        if len(args_str[arg_begin:i].strip()) != 0:
            add_arg()

        return result

    def _find_str_end(self, s: str, start: int) -> int | None:
        """查找字符串字面量的结束位置。"""
        for i in range(start + 1, len(s)):
            c = s[i]
            if c == "\\":
                continue
            if c == s[start]:
                return i
        return None

    def _strip_empty_lines(self, doc_lines: list[str]) -> str | None:
        """去掉文档前后的空行。"""
        if not doc_lines:
            return None
        start = 0
        for start in range(0, len(doc_lines)):
            if len(doc_lines[start].strip()) > 0:
                break
        end = len(doc_lines) - 1
        for end in range(len(doc_lines) - 1, 0, -1):
            if len(doc_lines[end].strip()) > 0:
                break
        if start > end:
            return None

        result = "\n".join(doc_lines[start : end + 1])
        if len(result) == 0:
            return None
        return result
