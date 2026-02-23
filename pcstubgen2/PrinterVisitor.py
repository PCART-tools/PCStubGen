from __future__ import annotations

import sys

from .IR import (
    IRAnnotation,
    IRArgument,
    IRArgumentKind,
    IRClass,
    IRFunction,
    InvalidExpression,
    IRMethod,
    IRModifier,
    IRModule,
    ResolvedType,
    IRValue,
)


class PrinterVisitor:
    def __init__(self, invalid_expr_as_ellipses: bool = True):
        self.invalid_expr_as_ellipses = invalid_expr_as_ellipses

    @staticmethod
    def indent_lines(lines: list[str], by: int = 4) -> list[str]:
        return [" " * by + line for line in lines]

    def visit_module(self, node: IRModule) -> list[str]:
        result: list[str] = []

        if node.doc is not None:
            result.extend(self.print_docstring(node.doc))

        for sub_module in node.sub_modules:
            result.extend(self.print_submodule_import(sub_module.Name))

        for class_ in sorted(node.classes, key=lambda c: c.name):
            result.extend(self.print_class(class_))

        for func in sorted(node.functions, key=lambda f: f.name):
            result.extend(self.print_function(func))

        return result

    def print_class(self, irclass: IRClass) -> list[str]:
        signature = f"class {irclass.name}"
        if irclass.bases:
            signature += f"({', '.join(str(base) for base in irclass.bases)})"
        signature += ":"

        body = self._print_class_body(irclass)
        return [signature, *self.indent_lines(body)]

    def _print_class_body(self, irclass: IRClass) -> list[str]:
        result: list[str] = []
        if irclass.doc is not None:
            result.extend(self.print_docstring(irclass.doc))

        for sub_class in sorted(irclass.classes, key=lambda c: c.name):
            result.extend(self.print_class(sub_class))

        modifier_order: dict[IRModifier, int] = {
            "static": 0,
            "class": 1,
            None: 2,
        }
        for method in sorted(
            irclass.methods, key=lambda m: (modifier_order.get(m.modifier, 2), m.function.name)
        ):
            result.extend(self.print_method(method))

        if not result:
            result = ["pass"]

        return result

    def print_method(self, node: IRMethod) -> list[str]:
        result: list[str] = []
        if node.modifier == "static":
            result.append("@staticmethod")
        elif node.modifier == "class":
            result.append("@classmethod")

        result.extend(self.print_function(node.function))
        return result

    def print_argument(self, arg: IRArgument) -> str:
        parts = []
        if arg.kind is IRArgumentKind.VAR_POSITIONAL:
            parts.append("*")
        if arg.kind is IRArgumentKind.VAR_KEYWORD:
            parts.append("**")
        parts.append(f"{arg.name}")
        if arg.annotation is not None:
            parts.append(f": {self.print_annotation(arg.annotation)}")
        if isinstance(arg.default, IRValue):
            if arg.default.is_print_safe:
                parts.append(f" = {self.print_value(arg.default)}")
            else:
                parts.append(" = ...")
        elif isinstance(arg.default, InvalidExpression):
            parts.append(f" = {self.print_invalid_exp(arg.default)}")

        return "".join(parts)

    def print_docstring(self, doc: str) -> list[str]:
        return [
            '"""',
            *(line.replace("\\", r"\\").replace('"""', r"\"\"\"") for line in doc.splitlines()),
            '"""',
        ]

    def print_function(self, func: IRFunction) -> list[str]:
        args = []
        has_pos_only = any(arg.kind is IRArgumentKind.POSITIONAL_ONLY for arg in func.args)
        pos_only_boundary: int | None = None
        if has_pos_only:
            pos_only_boundary = next(
                (
                    index
                    for index, arg in enumerate(func.args)
                    if arg.kind is not IRArgumentKind.POSITIONAL_ONLY
                ),
                len(func.args),
            )

        kw_only_marker_inserted = False
        has_var_positional = False
        for index, arg in enumerate(func.args):
            if (
                pos_only_boundary is not None
                and pos_only_boundary > 0
                and index == pos_only_boundary
                and sys.version_info >= (3, 8)
            ):
                args.append("/")

            if (
                arg.kind is IRArgumentKind.KEYWORD_ONLY
                and not kw_only_marker_inserted
                and not has_var_positional
            ):
                args.append("*")
                kw_only_marker_inserted = True

            if arg.kind is IRArgumentKind.VAR_POSITIONAL:
                has_var_positional = True
                kw_only_marker_inserted = True

            args.append(self.print_argument(arg))

        if (
            pos_only_boundary is not None
            and pos_only_boundary == len(func.args)
            and pos_only_boundary > 0
            and sys.version_info >= (3, 8)
        ):
            args.append("/")

        signature = [f"def {func.name}(", ", ".join(args), ")"]
        if func.return_annotation is not None:
            signature.append(f" -> {self.print_annotation(func.return_annotation)}")
        signature.append(":")

        result: list[str] = [
            *(f"@{decorator}" for decorator in func.decorators),
            "".join(signature),
        ]

        if func.doc is not None:
            body = self.print_docstring(func.doc)
        else:
            body = ["..."]

        result.extend(self.indent_lines(body))
        return result

    def print_submodule_import(self, name: str) -> list[str]:
        return [f"from . import {name}"]

    def print_value(self, value: IRValue) -> str:
        split = value.repr.split("\n", 1)
        if len(split) == 1:
            return split[0]
        return split[0] + "..."

    def print_type(self, type_: ResolvedType) -> str:
        if (
            str(type_.name) == "typing.Optional"
            and type_.parameters is not None
            and len(type_.parameters) == 1
        ):
            return f"{self.print_annotation(type_.parameters[0])} | None"
        if str(type_.name) == "typing.Union" and type_.parameters is not None:
            return " | ".join(self.print_annotation(p) for p in type_.parameters)
        if type_.parameters:
            param_str = "[" + ", ".join(self.print_annotation(p) for p in type_.parameters) + "]"
        else:
            param_str = ""
        return f"{type_.name}{param_str}"

    def print_annotation(self, annotation: IRAnnotation) -> str:
        if isinstance(annotation, ResolvedType):
            return self.print_type(annotation)
        if isinstance(annotation, IRValue):
            return self.print_value(annotation)
        if isinstance(annotation, InvalidExpression):
            return self.print_invalid_exp(annotation)
        return str(annotation)

    def print_invalid_exp(self, invalid_expr: InvalidExpression) -> str:
        if self.invalid_expr_as_ellipses:
            return "..."
        return invalid_expr.text
