from __future__ import annotations

import sys

from ..ir_modules import (
    IRArgument,
    IRArgumentKind,
    IRClass,
    IRFunction,
    IRMethod,
    IRMethodDecorator,
    IRModule,
    IRSignature,
)


class StubRenderer:
    """将 IR 渲染为 stub 文本。"""

    def __init__(
        self,
        include_docstrings: bool = False,
        include_module_type_comment: bool = False,
        include_c_inferred_source_comment: bool = False,
    ):
        self.include_docstrings = include_docstrings
        self.include_module_type_comment = include_module_type_comment
        self.include_c_inferred_source_comment = include_c_inferred_source_comment

    @staticmethod
    def indent_lines(lines: list[str], by: int = 4) -> list[str]:
        """按指定空格数缩进多行文本。"""
        return [" " * by + line for line in lines]

    def print_module(self, node: IRModule) -> list[str]:
        """渲染整个模块。"""
        result: list[str] = []

        if self.include_module_type_comment:
            result.append(f"# module type: {node.module_type.value}")

        if self.include_docstrings and node.doc is not None:
            result.extend(self.print_docstring(node.doc))

        for import_name in self._collect_module_imports(node):
            result.append(f"import {import_name}")

        for sub_module in node.sub_modules:
            result.extend(self.print_submodule_import(sub_module.full_name.name))

        for class_ in sorted(node.classes, key=lambda c: c.name):
            result.extend(self.print_class(class_))

        for func in sorted(node.functions, key=lambda f: f.name):
            result.extend(self.print_function(func))

        return result

    def print_class(self, irclass: IRClass) -> list[str]:
        """渲染类定义。"""
        signature = f"class {irclass.name}"
        if irclass.bases:
            signature += f"({', '.join(str(base) for base in irclass.bases)})"
        signature += ":"

        body = self._print_class_body(irclass)
        return [signature, *self.indent_lines(body)]

    def _print_class_body(self, irclass: IRClass) -> list[str]:
        """渲染类体。"""
        result: list[str] = []
        if self.include_docstrings and irclass.doc is not None:
            result.extend(self.print_docstring(irclass.doc))

        for sub_class in sorted(irclass.classes, key=lambda c: c.name):
            result.extend(self.print_class(sub_class))

        decorator_order: dict[IRMethodDecorator, int] = {
            "staticmethod": 0,
            "classmethod": 1,
            None: 2,
        }
        for method in sorted(
            irclass.methods, key=lambda m: (decorator_order.get(m.decorator, 2), m.function.name)
        ):
            result.extend(self.print_method(method))

        if not result:
            result = ["pass"]

        return result

    def print_method(self, node: IRMethod) -> list[str]:
        """渲染类方法。"""
        result: list[str] = []
        overload = len(node.function.signatures) > 1
        for signature in self._get_printable_signatures(node.function):
            result.extend(
                self._print_function_block(
                    func_name=node.function.name,
                    signature=signature,
                    func_doc=node.function.doc,
                    overload=overload,
                    method_decorator=node.decorator,
                )
            )
        return result

    def print_argument(self, arg: IRArgument) -> str:
        """渲染单个参数。"""
        parts = []
        if arg.kind is IRArgumentKind.VAR_POSITIONAL:
            parts.append("*")
        if arg.kind is IRArgumentKind.VAR_KEYWORD:
            parts.append("**")
        parts.append(f"{arg.name}")
        if arg.type is not None:
            parts.append(f": {arg.type.render()}")
        if arg.default_value is not None:
            parts.append(f" = {arg.default_value}")
        elif arg.has_default:
            parts.append(" = ...")

        return "".join(parts)

    def print_docstring(self, doc: str) -> list[str]:
        """渲染文档字符串。"""
        return [
            '"""',
            *(line.replace("\\", r"\\").replace('"""', r"\"\"\"") for line in doc.splitlines()),
            '"""',
        ]

    def print_function(self, func: IRFunction) -> list[str]:
        """渲染模块级函数。"""
        result: list[str] = []
        overload = len(func.signatures) > 1
        for signature in self._get_printable_signatures(func):
            result.extend(
                self._print_function_block(
                    func_name=func.name,
                    signature=signature,
                    func_doc=func.doc,
                    overload=overload,
                    method_decorator=None,
                )
            )

        if self.include_c_inferred_source_comment and func.c_inferred_source_comment is not None:
            result.extend(
                self.print_c_inferred_source_comment(
                    func_name=func.name,
                    source_text=func.c_inferred_source_comment,
                )
            )
        return result

    def print_c_inferred_source_comment(self, *, func_name: str, source_text: str) -> list[str]:
        """渲染由 C AST 推断签名来源的源码注释。"""
        result = [f"#   C inferred source for {func_name}:"]
        for line in source_text.splitlines():
            if line:
                result.append(f"#   {line}")
            else:
                result.append("#")
        return result

    def _print_function_block(
        self,
        *,
        func_name: str,
        signature: IRSignature,
        func_doc: str | None,
        overload: bool,
        method_decorator: IRMethodDecorator,
    ) -> list[str]:
        """渲染一条可打印的函数签名块。"""
        result: list[str] = []
        if method_decorator is not None:
            result.append(f"@{method_decorator}")
        if overload:
            result.append("@typing.overload")
        result.extend(self._build_function_signature(func_name=func_name, signature=signature))

        if self.include_docstrings and func_doc is not None:
            body = self.print_docstring(func_doc)
        else:
            body = ["..."]

        result.extend(self.indent_lines(body))
        return result

    def _build_function_signature(self, *, func_name: str, signature: IRSignature) -> list[str]:
        """构建单条 def 头。"""
        args = self._format_arguments(signature.args)
        if len(signature.args) <= 1:
            return [self._build_single_line_function_signature(func_name=func_name, args=args, signature=signature)]

        return self._build_multiline_function_signature(func_name=func_name, args=args, signature=signature)

    def _build_single_line_function_signature(
        self,
        *,
        func_name: str,
        args: list[str],
        signature: IRSignature,
    ) -> str:
        """构建单行 def 头。"""
        rendered = [f"def {func_name}(", ", ".join(args), ")"]
        if signature.return_type is not None:
            rendered.append(f" -> {signature.return_type.render()}")
        rendered.append(":")
        return "".join(rendered)

    def _build_multiline_function_signature(
        self,
        *,
        func_name: str,
        args: list[str],
        signature: IRSignature,
    ) -> list[str]:
        """构建多行 def 头，每个参数独占一行。"""
        rendered = [f"def {func_name}("]
        rendered.extend(self.indent_lines([f"{arg}," for arg in args]))

        closing_line = ")"
        if signature.return_type is not None:
            closing_line += f" -> {signature.return_type.render()}"
        closing_line += ":"
        rendered.append(closing_line)
        return rendered

    def _format_arguments(self, args: list[IRArgument]) -> list[str]:
        """渲染函数参数列表。"""
        rendered_args: list[str] = []
        has_pos_only = any(arg.kind is IRArgumentKind.POSITIONAL_ONLY for arg in args)
        pos_only_boundary: int | None = None
        if has_pos_only:
            pos_only_boundary = next(
                (
                    index
                    for index, arg in enumerate(args)
                    if arg.kind is not IRArgumentKind.POSITIONAL_ONLY
                ),
                len(args),
            )

        kw_only_marker_inserted = False
        has_var_positional = False
        for index, arg in enumerate(args):
            if (
                pos_only_boundary is not None
                and pos_only_boundary > 0
                and index == pos_only_boundary
                and sys.version_info >= (3, 8)
            ):
                rendered_args.append("/")

            if (
                arg.kind is IRArgumentKind.KEYWORD_ONLY
                and not kw_only_marker_inserted
                and not has_var_positional
            ):
                rendered_args.append("*")
                kw_only_marker_inserted = True

            if arg.kind is IRArgumentKind.VAR_POSITIONAL:
                has_var_positional = True
                kw_only_marker_inserted = True

            rendered_args.append(self.print_argument(arg))

        if (
            pos_only_boundary is not None
            and pos_only_boundary == len(args)
            and pos_only_boundary > 0
            and sys.version_info >= (3, 8)
        ):
            rendered_args.append("/")

        return rendered_args

    def _get_printable_signatures(self, func: IRFunction) -> list[IRSignature]:
        """返回可打印签名，缺失时合成占位签名。"""
        if func.signatures:
            return func.signatures
        return [self._build_placeholder_signature(func)]

    def _build_placeholder_signature(self, func: IRFunction) -> IRSignature:
        """为未知签名函数合成仅用于输出的占位签名。"""
        return IRSignature(
            args=[
                IRArgument(name="args", kind=IRArgumentKind.VAR_POSITIONAL),
                IRArgument(name="kwargs", kind=IRArgumentKind.VAR_KEYWORD),
            ],
        )

    def _collect_module_imports(self, node: IRModule) -> list[str]:
        """收集模块内函数/方法签名依赖的 import。"""
        imports: set[str] = set()
        for class_ in node.classes:
            imports.update(self._collect_class_imports(class_))
        for func in node.functions:
            imports.update(self._collect_function_imports(func))
        return sorted(imports)

    def _collect_class_imports(self, irclass: IRClass) -> set[str]:
        """递归收集类内函数/方法签名依赖的 import。"""
        imports: set[str] = set()
        for sub_class in irclass.classes:
            imports.update(self._collect_class_imports(sub_class))
        for method in irclass.methods:
            imports.update(self._collect_function_imports(method.function))
        return imports

    def _collect_function_imports(self, func: IRFunction) -> set[str]:
        """收集函数签名依赖的 import。"""
        imports: set[str] = set()
        if len(func.signatures) > 1:
            imports.add("typing")
        for signature in func.signatures:
            imports.update(self._collect_signature_imports(signature))
        return imports

    @staticmethod
    def _collect_signature_imports(signature: IRSignature) -> set[str]:
        """收集单条签名依赖的 import。"""
        imports: set[str] = set()
        if signature.return_type is not None:
            imports.update(signature.return_type.collect_imports())
        for arg in signature.args:
            if arg.type is not None:
                imports.update(arg.type.collect_imports())
        return imports

    def print_submodule_import(self, name: str) -> list[str]:
        """渲染子模块导入。"""
        return [f"from . import {name}"]
