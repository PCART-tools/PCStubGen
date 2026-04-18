from __future__ import annotations

import sys

from ..models import (
    Argument,
    ArgumentKind,
    Class,
    Decorator,
    Function,
    Module,
    Signature,
)


class StubRenderer:
    """将模型渲染为 stub 文本。"""

    def __init__(self, include_docstrings: bool = False):
        self.include_docstrings = include_docstrings

    @staticmethod
    def indent_lines(lines: list[str], by: int = 4) -> list[str]:
        """按指定空格数缩进多行文本。"""
        return [" " * by + line for line in lines]

    def render_module(self, node: Module) -> list[str]:
        """渲染整个模块。"""
        result: list[str] = []

        if self.include_docstrings and node.doc is not None:
            result.extend(self.render_docstring(node.doc))

        for import_name in self._collect_module_imports(node):
            result.append(f"import {import_name}")

        for sub_module in node.sub_modules:
            result.extend(self.render_submodule_import(sub_module.full_name.name))

        for class_ in sorted(node.classes, key=lambda c: c.name):
            result.extend(self.render_class(class_))

        for func in sorted(node.functions, key=lambda f: f.name):
            result.extend(self.render_function(func))

        return result

    def render_class(self, class_node: Class) -> list[str]:
        """渲染类定义。"""
        signature = f"class {class_node.name}"
        if class_node.bases:
            signature += f"({', '.join(str(base) for base in class_node.bases)})"
        signature += ":"

        body = self._render_class_body(class_node)
        return [signature, *self.indent_lines(body)]

    def _render_class_body(self, class_node: Class) -> list[str]:
        """渲染类体。"""
        result: list[str] = []
        if self.include_docstrings and class_node.doc is not None:
            result.extend(self.render_docstring(class_node.doc))

        for sub_class in sorted(class_node.classes, key=lambda c: c.name):
            result.extend(self.render_class(sub_class))

        decorator_order: dict[Decorator, int] = {
            "staticmethod": 0,
            "classmethod": 1,
            None: 2,
        }
        for method in sorted(
            class_node.methods,
            key=lambda method: (
                decorator_order.get(method.decorator, 2),
                method.name,
            ),
        ):
            result.extend(self.render_method(method))

        if not result:
            result = ["pass"]

        return result

    def render_method(self, func: Function) -> list[str]:
        """渲染类方法。"""
        result: list[str] = []
        overload = len(func.signatures) > 1
        for signature in self._get_renderable_signatures(func):
            result.extend(
                self._render_function_block(
                    func_name=func.name,
                    signature=signature,
                    func_doc=func.doc,
                    overload=overload,
                    decorator=func.decorator,
                )
            )
        return result

    def render_argument(self, arg: Argument) -> str:
        """渲染单个参数。"""
        parts = []
        if arg.kind is ArgumentKind.VAR_POSITIONAL:
            parts.append("*")
        if arg.kind is ArgumentKind.VAR_KEYWORD:
            parts.append("**")
        parts.append(f"{arg.name}")
        if arg.type is not None:
            parts.append(f": {arg.type.render()}")
        if arg.default_value is not None:
            parts.append(f" = {arg.default_value}")

        return "".join(parts)

    def render_docstring(self, doc: str) -> list[str]:
        """渲染文档字符串。"""
        return [
            '"""',
            *(line.replace("\\", r"\\").replace('"""', r"\"\"\"") for line in doc.splitlines()),
            '"""',
        ]

    def render_function(self, func: Function) -> list[str]:
        """渲染模块级函数。"""
        result: list[str] = []
        overload = len(func.signatures) > 1
        for signature in self._get_renderable_signatures(func):
            result.extend(
                self._render_function_block(
                    func_name=func.name,
                    signature=signature,
                    func_doc=func.doc,
                    overload=overload,
                    decorator=None,
                )
            )

        if func.comment is not None:
            result.extend(
                self.render_comment(
                    comment_text=func.comment,
                )
            )
        return result

    def render_comment(self, *, comment_text: str) -> list[str]:
        """渲染由 C AST 推断签名来源的源码注释。"""
        result: list[str] = []
        for line in comment_text.splitlines():
            if line:
                result.append(f"#   {line}")
            else:
                result.append("#")
        return result

    def render_function_signature(self, *, func_name: str, signature: Signature) -> str:
        """将单条函数签名渲染为单行字符串。"""
        args = ", ".join(self._format_arguments(signature.args))
        rendered = [f"def {func_name}(", args, ")"]
        if signature.return_type is not None:
            rendered.append(f" -> {signature.return_type.render()}")
        rendered.append(":")
        return "".join(rendered)

    def render_function_signature_lines(self, *, func_name: str, signature: Signature) -> list[str]:
        """将单条函数签名渲染为 def 头行列表。"""
        return self._build_function_signature(func_name=func_name, signature=signature)

    def _render_function_block(
        self,
        *,
        func_name: str,
        signature: Signature,
        func_doc: str | None,
        overload: bool,
        decorator: Decorator,
    ) -> list[str]:
        """渲染一条可渲染的函数签名块。"""
        result: list[str] = []
        if decorator is not None:
            result.append(f"@{decorator}")
        if overload:
            result.append("@typing.overload")
        result.extend(self._build_function_signature(func_name=func_name, signature=signature))

        if self.include_docstrings and func_doc is not None:
            body = self.render_docstring(func_doc)
        else:
            body = ["..."]

        result.extend(self.indent_lines(body))
        return result

    def _build_function_signature(self, *, func_name: str, signature: Signature) -> list[str]:
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
        signature: Signature,
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
        signature: Signature,
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

    def _format_arguments(self, args: list[Argument]) -> list[str]:
        """渲染函数参数列表。"""
        rendered_args: list[str] = []
        has_pos_only = any(arg.kind is ArgumentKind.POSITIONAL_ONLY for arg in args)
        pos_only_boundary: int | None = None
        if has_pos_only:
            pos_only_boundary = next(
                (
                    index
                    for index, arg in enumerate(args)
                    if arg.kind is not ArgumentKind.POSITIONAL_ONLY
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
                arg.kind is ArgumentKind.KEYWORD_ONLY
                and not kw_only_marker_inserted
                and not has_var_positional
            ):
                rendered_args.append("*")
                kw_only_marker_inserted = True

            if arg.kind is ArgumentKind.VAR_POSITIONAL:
                has_var_positional = True
                kw_only_marker_inserted = True

            rendered_args.append(self.render_argument(arg))

        if (
            pos_only_boundary is not None
            and pos_only_boundary == len(args)
            and pos_only_boundary > 0
            and sys.version_info >= (3, 8)
        ):
            rendered_args.append("/")

        return rendered_args

    def _get_renderable_signatures(self, func: Function) -> list[Signature]:
        """返回可渲染签名，缺失时合成占位签名。"""
        if func.signatures:
            return func.signatures
        return [self._build_placeholder_signature(func)]

    def _build_placeholder_signature(self, func: Function) -> Signature:
        """为未知签名函数合成仅用于输出的占位签名。"""
        return Signature(
            args=[
                Argument(name="args", kind=ArgumentKind.VAR_POSITIONAL),
                Argument(name="kwargs", kind=ArgumentKind.VAR_KEYWORD),
            ],
        )

    def _collect_module_imports(self, node: Module) -> list[str]:
        """收集模块内函数/方法签名依赖的 import。"""
        imports: set[str] = set()
        for class_ in node.classes:
            imports.update(self._collect_class_imports(class_))
        for func in node.functions:
            imports.update(self._collect_function_imports(func))
        return sorted(imports)

    def _collect_class_imports(self, class_node: Class) -> set[str]:
        """递归收集类内函数/方法签名依赖的 import。"""
        imports: set[str] = set()
        for sub_class in class_node.classes:
            imports.update(self._collect_class_imports(sub_class))
        for method in class_node.methods:
            imports.update(self._collect_function_imports(method))
        return imports

    def _collect_function_imports(self, func: Function) -> set[str]:
        """收集函数签名依赖的 import。"""
        imports: set[str] = set()
        if len(func.signatures) > 1:
            imports.add("typing")
        for signature in func.signatures:
            imports.update(self._collect_signature_imports(signature))
        return imports

    @staticmethod
    def _collect_signature_imports(signature: Signature) -> set[str]:
        """收集单条签名依赖的 import。"""
        imports: set[str] = set()
        if signature.return_type is not None:
            imports.update(signature.return_type.collect_imports())
        for arg in signature.args:
            if arg.type is not None:
                imports.update(arg.type.collect_imports())
        return imports

    def render_submodule_import(self, name: str) -> list[str]:
        """渲染子模块导入。"""
        return [f"from . import {name}"]
