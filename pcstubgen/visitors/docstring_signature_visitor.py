from __future__ import annotations

import re
from enum import Enum, auto

from loguru import logger

from ..ir import (
    IRArgument,
    IRArgumentKind,
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


class DocstringSignatureVisitor(NodeVisitor):
    """
    解析文档字符串中的函数和方法签名。
    """

    def visit_function(self, node: IRFunction, module: IRModule) -> None:
        """在函数层原地补全文档字符串签名。"""
        if node.signatures:
            return

        if not node.doc:
            return

        try:
            parsed_signatures = self.parse_function_docstring(
                func_name=node.name,
                doc_lines=node.doc.splitlines(),
            )
        except ValueError as ex:
            logger.warning(
                "解析 docstring 签名失败, module_name: {}, func_name: {}, error_type: {}, error: {}",
                str(module.full_name),
                node.name,
                type(ex).__name__,
                ex,
            )
            return

        if parsed_signatures:
            node.signatures = parsed_signatures

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
            overloads.append(
                IRSignature(
                    args=args,
                    return_type_name=self.parse_annotation_str(match.group("returns")),
                )
            )
            doc_start = i + 1

        if not overloads:
            raise ValueError("Overloaded function. 之后未找到有效重载签名。")

        overloads[-1].doc = self._strip_empty_lines(doc_lines[doc_start:])
        return overloads

    def parse_args_str(self, args_str: str) -> list[IRArgument]:
        """解析签名中的参数串。"""
        split_args = self._split_args_str(args_str)

        result: list[IRArgument] = []
        state = _ArgsParseState.POSITIONAL

        for arg_decl, type_name, default_str in split_args:
            if state is _ArgsParseState.FINISHED:
                raise ValueError("可变关键字参数之后不允许再出现其他参数。")

            if arg_decl == "/":
                if (
                    state is not _ArgsParseState.POSITIONAL
                    or type_name is not None
                    or default_str is not None
                ):
                    raise ValueError("位置参数分隔符 '/' 位置非法。")

                # 不允许 f(/, a)
                if not any(
                    arg.kind is IRArgumentKind.POSITIONAL_OR_KEYWORD for arg in result
                ):
                    raise ValueError("位置参数分隔符 '/' 前必须至少有一个普通参数。")

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
                    raise ValueError("关键字专用分隔符 '*' 位置非法。")

                state = _ArgsParseState.KEYWORD_ONLY
                continue

            if arg_decl.startswith("**"):
                if default_str is not None:
                    raise ValueError("可变关键字参数不允许默认值。")

                name = arg_decl[2:].strip()
                if not name:
                    raise ValueError("可变关键字参数名不能为空。")

                kind = IRArgumentKind.VAR_KEYWORD
                state = _ArgsParseState.FINISHED
            elif arg_decl.startswith("*"):
                if state not in (_ArgsParseState.POSITIONAL, _ArgsParseState.POSITIONAL_OR_KEYWORD):
                    raise ValueError("可变位置参数必须出现在普通参数之后。")
                if default_str is not None:
                    raise ValueError("可变位置参数不允许默认值。")

                name = arg_decl[1:].strip()
                if not name:
                    raise ValueError("可变位置参数名不能为空。")

                kind = IRArgumentKind.VAR_POSITIONAL
                state = _ArgsParseState.KEYWORD_ONLY
            else:
                name = arg_decl.strip()
                if not name:
                    raise ValueError("参数名不能为空。")

                if state is _ArgsParseState.KEYWORD_ONLY:
                    kind = IRArgumentKind.KEYWORD_ONLY
                else:
                    kind = IRArgumentKind.POSITIONAL_OR_KEYWORD

            result.append(
                IRArgument(
                    name=name,
                    default_value=default_str,
                    has_default=default_str is not None,
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
    ) -> list[tuple[str, str | None, str | None]]:
        """按顶层逗号拆参数，再分阶段解析注解和默认值。"""
        if not args_str.strip():
            return []

        arg_blocks = self._split_top_level(args_str, ",")

        result: list[tuple[str, str | None, str | None]] = []
        for arg_block in arg_blocks:
            if not arg_block.strip():
                raise ValueError("参数列表中存在空参数块。")

            # 先拆=号，再拆:
            nametype_default_parts = self._split_top_level(arg_block, "=")
            if len(nametype_default_parts) > 2:
                raise ValueError("参数默认值声明中包含多个 '='。")

            nametype = nametype_default_parts[0]
            default = (
                nametype_default_parts[1].strip()
                if len(nametype_default_parts) == 2
                else None
            )

            name_type_parts = self._split_top_level(nametype, ":")
            if len(name_type_parts) > 2:
                raise ValueError("参数注解声明中包含多个 ':'。")

            name = name_type_parts[0].strip()
            type_ = name_type_parts[1].strip() if len(name_type_parts) == 2 else None
            result.append((name, type_, default))

        return result

    def _split_top_level(self, text: str, delim: str) -> list[str]:
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
                idx = str_end + 1
                continue

            if ch in left_to_right:
                stack.append(left_to_right[ch])  # 进右侧符号
            elif ch in rights:
                # 栈为空或ch不为栈顶
                if not stack or ch != stack[-1]:
                    raise ValueError("括号不匹配。")
                stack.pop()
            elif not stack and ch == delim:
                parts.append(text[start:idx])
                start = idx + 1
            idx += 1

        if stack:
            raise ValueError("存在未闭合的括号。")

        parts.append(text[start:])
        return parts

    def _find_str_end(self, s: str, start: int) -> int:
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
        raise ValueError("字符串字面量未闭合。")

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
