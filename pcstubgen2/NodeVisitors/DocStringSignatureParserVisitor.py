from __future__ import annotations

import ast
import re

from ..ErrorCollector import ErrorCollector
from ..IR import (
    IRArgument,
    IRArgumentKind,
    IRFunction,
    InvalidExpression,
    ResolvedType,
    IRValue,
    QualifiedName,
    IRClass,
    IRMethod,
    IRModule,
)
from .NodeVisitor import NodeVisitor
from ..Errors import InvalidExpressionError

_generic_args = [
    IRArgument(name="args", kind=IRArgumentKind.VAR_POSITIONAL),
    IRArgument(name="kwargs", kind=IRArgumentKind.VAR_KEYWORD),
]


class DocStringSignatureParserVisitor(NodeVisitor):
    '''
    解析文档字符串中的函数和方法的签名
    '''
    _arg_star_name_regex = re.compile(
        r"^\s*(?P<stars>\*{1,2})?" r"\s*(?P<name>\w+)\s*$"
    )
    _pybind11_enum_pattern = re.compile(r"<(?P<enum>\w+(\.\w+)+): (?P<value>-?\d+)>")

    def __init__(
        self,
        error_collector: ErrorCollector,
        enum_class_locations: dict[re.Pattern, str] | None = None,
    ):
        self.error_collector = error_collector
        self.enum_class_locations = enum_class_locations or {}

    def visit_module(self, node: IRModule) -> None:

        new_funcs = []
        for func in node.functions:
            parsed = self._parse_function(func)
            new_funcs.extend(parsed)
        node.functions.clear()
        node.functions.extend(new_funcs)

        # 递归
        super().visit_module(node)

    def visit_class(self, node: IRClass) -> None:
        new_methods = []
        for method in node.methods:
            funcs = self._parse_function(method.function)
            if len(funcs) == 1 and funcs[0] is method.function:
                new_methods.append(method)
            else:
                # 它已扩展或更改
                for f in funcs:
                    new_methods.append(IRMethod(function=f, decorator=method.decorator))
        node.methods = new_methods

        # 递归
        super().visit_class(node)

    def _parse_function(self, func: IRFunction) -> list[IRFunction]:
        # 仅当我们具有泛型 (*args, **kwargs) 签名时才从文档字符串解析
        is_generic = func.is_generic_signature()
        
        if not is_generic:
            return [func]
        
        if not func.doc:
            return [func]

        doc_lines = func.doc.splitlines()
        parsed_funcs = self.parse_function_docstring(func.name, doc_lines)
        
        if len(parsed_funcs) > 0:
            return parsed_funcs
        
        return [func]

    def parse_function_docstring(
        self, func_name: str, doc_lines: list[str]
    ) -> list[IRFunction]:
        '''
        解析函数文档字符串中的签名。

        Example（单签名）:
            add(a: int, b: int = 0) -> int
            Return a + b.

        Example（pybind11 重载）:
            add(*args, **kwargs)
            Overloaded function.
            1. add(a: int, b: int) -> int
            2. add(a: float, b: float) -> float
        '''
        if len(doc_lines) == 0:
            return []

        # 正则表达式
        top_signature_regex = re.compile(
            rf"^{re.escape(func_name)}\((?P<args>.*)\)\s*(->\s*(?P<returns>.+))?$"
        )

        match = top_signature_regex.match(doc_lines[0])
        if match is None:
            return []

        # 在 pybind11 中，重载格式固定为 "Overloaded function." 这一行
        if len(doc_lines) < 2 or doc_lines[1].strip() != "Overloaded function.":
            returns_str = match.group("returns")
            if returns_str is not None:
                returns = self.parse_annotation_str(returns_str)
            else:
                returns = None

            return [
                IRFunction(
                    name=func_name,
                    args=self.parse_args_str(match.group("args")),
                    doc=self._strip_empty_lines(doc_lines[1:]),
                    return_annotation=returns,
                )
            ]

        overload_signature_regex = re.compile(
            rf"^(\s*(?P<overload_number>\d+).\s*)"
            rf"{re.escape(func_name)}\((?P<args>.*)\)\s*->\s*(?P<returns>.+)$"
        )

        doc_start = 0
        _dummy = IRFunction("")
        overloads = [_dummy]

        for i in range(2, len(doc_lines)):
            match = overload_signature_regex.match(doc_lines[i])
            if match:
                if match.group("overload_number") != f"{len(overloads)}":
                    continue
                overloads[-1].doc = self._strip_empty_lines(doc_lines[doc_start:i])
                doc_start = i + 1
                
                # 检查 "typing.overload"
                decorators = ["typing.overload"]

                overloads.append(
                    IRFunction(
                        name=func_name,
                        args=self.parse_args_str(match.group("args")),
                        return_annotation=self.parse_annotation_str(match.group("returns")),
                        doc=None,
                        decorators=decorators,
                    )
                )

        overloads[-1].doc = self._strip_empty_lines(doc_lines[doc_start:])

        return overloads[1:]

    def parse_args_str(self, args_str: str) -> list[IRArgument]:
        split_args = self._split_args_str(args_str)
        if split_args is None:
            return _generic_args

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
                return _generic_args
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

            if annotation_str is not None:
                annotation = self.parse_annotation_str(annotation_str)
            else:
                annotation = None

            if default_str is not None:
                default = self.parse_value_str(default_str)
            else:
                default = None

            result.append(
                IRArgument(
                    name=name,
                    default=default,
                    annotation=annotation,
                    kind=kind,
                )
            )
        return result

    def parse_annotation_str(
        self, annotation_str: str
    ) -> ResolvedType | InvalidExpression | IRValue:
        variants = self._split_type_union_str(annotation_str)
        if variants is None or len(variants) == 0:
            self.error_collector.report_error(InvalidExpressionError(annotation_str))
            return InvalidExpression(annotation_str)
        if len(variants) == 1:
            return self.parse_type_str(variants[0])
        
        # 这里我们没有直接访问权限来检查 typing.Union 是否为有效导入，
        # 但我们可以生成 ResolvedType。
        return ResolvedType(
            name=QualifiedName.from_str("typing.Union"),
            parameters=[self.parse_type_str(variant) for variant in variants],
        )

    def parse_type_str(
        self, annotation_str: str
    ) -> ResolvedType | InvalidExpression | IRValue:
        qname_regex = re.compile(
            r"^\s*(?P<qual_name>([_A-Za-z]\w*)?(\s*\.\s*[_A-Za-z]\w*)*)"
        )
        annotation_str = annotation_str.strip()
        match = qname_regex.match(annotation_str)
        if match is None:
            return self.parse_value_str(annotation_str)
        qual_name = QualifiedName(
            part for part in match.group("qual_name").replace(" ", "").split(".")
        )
        parameters_str = annotation_str[match.end("qual_name") :].strip()

        if len(parameters_str) == 0:
            parameters = None
        else:
            if parameters_str[0] != "[" or parameters_str[-1] != "]":
                return self.parse_value_str(annotation_str)

            split_parameters = self._split_parameters_str(parameters_str[1:-1])
            if split_parameters is None:
                return self.parse_value_str(annotation_str)

            parameters = [
                self.parse_annotation_str(param_str) for param_str in split_parameters
            ]
        return ResolvedType(name=qual_name, parameters=parameters)

    def parse_value_str(self, value: str) -> IRValue | InvalidExpression:
        strip_expr = value.strip()

        match = self._pybind11_enum_pattern.match(strip_expr)
        if match is not None:
            enum_qual_name = match.group("enum")
            class_path, entry = enum_qual_name.rsplit(".", maxsplit=1)
            for pattern, prefix in self.enum_class_locations.items():
                if pattern.match(class_path):
                    return IRValue(
                        repr=f"{prefix}.{class_path}.{entry}",
                        is_print_safe=True,
                    )

        try:
            ast.parse(strip_expr)
            print_safe = False
            try:
                ast.literal_eval(strip_expr)
                print_safe = True
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                pass
            return IRValue(strip_expr, is_print_safe=print_safe)
        except SyntaxError:
            self.error_collector.report_error(InvalidExpressionError(strip_expr))
            return InvalidExpression(strip_expr)

    # --- 字符串分割辅助函数 ---

    def _split_args_str(
        self, args_str: str
    ) -> list[tuple[str, str | None, str | None]] | None:
        result = []
        closing = {"(": ")", "{": "}", "[": "]"}
        stack = []
        i = 0
        arg_begin = 0
        semicolon_pos: int | None = None
        eq_sign_pos: int | None = None

        def add_arg():
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

    def _split_type_union_str(self, type_str: str) -> list[str] | None:
        return self._split_str(type_str, delim="|")

    def _split_parameters_str(self, param_str: str) -> list[str] | None:
        return self._split_str(param_str, delim=",")

    def _split_str(self, param_str: str, delim: str) -> list[str] | None:
        result = []
        closing = {"(": ")", "{": "}", "[": "]"}
        stack = []
        i = 0
        arg_begin = 0

        def add_arg():
            arg_end = i
            param = param_str[arg_begin:arg_end]
            result.append(param)

        while i < len(param_str):
            c = param_str[i]
            if c in "\"'":
                str_end = self._find_str_end(param_str, i)
                if str_end is None:
                    return None
                i = str_end
            elif c in closing:
                stack.append(closing[c])
            elif len(stack) == 0:
                if c == delim:
                    add_arg()
                    arg_begin = i + 1
            elif stack[-1] == c:
                stack.pop()
            i += 1
        if len(stack) != 0:
            return None
        if len(param_str[arg_begin:i].strip()) != 0:
            add_arg()
        return result

    def _find_str_end(self, s: str, start: int) -> int | None:
        for i in range(start + 1, len(s)):
            c = s[i]
            if c == "\\":  # 跳过转义字符
                continue
            if c == s[start]:
                return i
        return None

    def _strip_empty_lines(self, doc_lines: list[str]) -> str | None:
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
