from __future__ import annotations

import keyword
import re
from enum import Enum, auto

from ..ir import (
    IRArgument,
    IRArgumentKind,
    IRClass,
    IRFunction,
    IRModule,
    IRSignature,
)
from .node_visitor import NodeVisitor


class _ArgsParseState(Enum):
    """参数头解析阶段。"""

    POSITIONAL = auto()
    POSITIONAL_OR_KEYWORD = auto()
    KEYWORD_ONLY = auto()
    FINISHED = auto()


class DocStringSignatureParserVisitor(NodeVisitor):
    """
    解析文档字符串中的函数和方法签名。
    """

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

        # 单条
        if len(doc_lines) < 2 or doc_lines[1].strip() != "Overloaded function.":
            args = self.parse_args_str(match.group("args"))
            if args is None:
                return []

            returns = (match.group("returns") or "").strip('"')
            return [
                IRSignature(
                    args=args,
                    doc=self._strip_empty_lines(doc_lines[1:]),
                    return_type_name=returns,
                )
            ]

        # 多条overload
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
        state = _ArgsParseState.POSITIONAL

        for arg_decl, type_name, default_str in split_args:
            if state is _ArgsParseState.FINISHED:
                return None

            if arg_decl == "/":
                if (
                    state is not _ArgsParseState.POSITIONAL
                    or type_name is not None
                    or default_str is not None
                ):
                    return None

                # 不允许 f(/, a)
                if not any(
                    arg.kind is IRArgumentKind.POSITIONAL_OR_KEYWORD for arg in result
                ):
                    return None

                for arg in result:
                    if arg.kind is IRArgumentKind.POSITIONAL_OR_KEYWORD:
                        arg.kind = IRArgumentKind.POSITIONAL_ONLY
                state = _ArgsParseState.POSITIONAL_OR_KEYWORD
                continue

            if arg_decl == "*":
                if (
                    state not in (_ArgsParseState.POSITIONAL, _ArgsParseState.POSITIONAL_OR_KEYWORD)
                    or type_name is not None
                    or default_str is not None
                ):
                    return None

                state = _ArgsParseState.KEYWORD_ONLY
                continue

            if arg_decl.startswith("**"):
                if default_str is not None:
                    return None

                name = arg_decl[2:]
                if name is None:
                    return None

                kind = IRArgumentKind.VAR_KEYWORD
                state = _ArgsParseState.FINISHED
            elif arg_decl.startswith("*"):
                if state not in (_ArgsParseState.POSITIONAL, _ArgsParseState.POSITIONAL_OR_KEYWORD):
                    return None
                if default_str is not None:
                    return None

                name = arg_decl[1:]
                if name is None:
                    return None

                kind = IRArgumentKind.VAR_POSITIONAL
                state = _ArgsParseState.KEYWORD_ONLY
            else:
                name = arg_decl
                if name is None:
                    return None

                if state is _ArgsParseState.KEYWORD_ONLY:
                    kind = IRArgumentKind.KEYWORD_ONLY
                else:
                    kind = IRArgumentKind.POSITIONAL_OR_KEYWORD

            result.append(
                IRArgument(
                    name=name,
                    default_value=default_str,
                    type_name=type_name,
                    kind=kind,
                )
            )
        return result

    @staticmethod
    def parse_annotation_str(annotation_str: str) -> str | None:
        """清理注解文本中的空白。"""
        text = annotation_str.strip()
        return text or None

    def _split_args_str(
        self, args_str: str
    ) -> list[tuple[str, str | None, str | None]] | None:
        """按顶层逗号拆参数，再分阶段解析注解和默认值。"""
        if not args_str.strip():
            return []

        arg_blocks = self._split_top_level(args_str, ",")
        if arg_blocks is None:
            return None

        result: list[tuple[str, str | None, str | None]] = []
        for arg_block in arg_blocks:
            if not arg_block.strip():
                return None

            # 先拆=号，再拆:
            nametype_default_parts = self._split_top_level(arg_block, "=")
            if nametype_default_parts is None or len(nametype_default_parts) > 2:
                return None

            nametype = nametype_default_parts[0]
            default = nametype_default_parts[1] if len(nametype_default_parts) == 2 else None

            name_type_parts = self._split_top_level(nametype, ":")
            if name_type_parts is None or len(name_type_parts) > 2:
                return None

            name = name_type_parts[0].strip()
            type_ = name_type_parts[1] if len(name_type_parts) == 2 else None
            result.append((name, type_, default))

        return result

    def _split_top_level(self, text: str, delim: str) -> list[str] | None:
        """仅在顶层按单字符分隔，忽略括号和字符串内部的分隔符。"""
        if len(delim) != 1:
            raise ValueError("delim must be a single character")

        left_to_right = {"(": ")", "{": "}", "[": "]"}
        rights = left_to_right.values()
        stack: list[str] = []
        parts: list[str] = []
        start = 0
        idx = 0

        while idx < len(text):
            ch = text[idx]
            if ch in "\"'":
                str_end = self._find_str_end(text, idx)
                if str_end is None:
                    return None
                idx = str_end + 1
                continue

            if ch in left_to_right:
                stack.append(left_to_right[ch]) # 进右侧符号
            elif ch in rights:
                # 栈为空或ch不为栈顶
                if not stack or ch != stack[-1]:
                    return None
                stack.pop()
            elif not stack and ch == delim:
                parts.append(text[start:idx])
                start = idx + 1
            idx += 1

        if stack:
            return None

        parts.append(text[start:])
        return parts

    def _find_str_end(self, s: str, start: int) -> int | None:
        """查找字符串字面量的结束位置。"""
        quote = s[start]
        i = start + 1
        while i < len(s):
            if s[i] == "\\":
                i += 2
                continue
            if s[i] == quote:
                return i
            i += 1
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
