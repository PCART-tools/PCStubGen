from __future__ import annotations

import dataclasses
import sys
from typing import Any

from .IR import (
    IRAlias,
    IRAnnotation,
    IRArgument,
    IRArgumentKind,
    IRVariable,
    IRClass,
    IRField,
    IRFunction,
    IRImport,
    InvalidExpression,
    IRMethod,
    IRModifier,
    IRModule,
    IRProperty,
    ResolvedType,
    IRTypeVar,
    IRValue,
)


class PrinterVisitor:
    def __init__(self, invalid_expr_as_ellipses: bool = True):
        self.invalid_expr_as_ellipses = invalid_expr_as_ellipses
        self._lines: list[str] = []

    @staticmethod
    def indent_lines(lines: list[str], by: int = 4) -> list[str]:
        return [" " * by + line for line in lines]

    def visit_module(self, node: IRModule) -> list[str]:
        # 重置行以进行新的模块打印，或者只返回此模块的行
        # 如果这对子模块进行递归调用，我们需要处理它。
        # 但通常我们将一个模块打印到一个文件。
        # 对于此访问者，我们假设它返回正在访问的节点的行。
        
        result = []

        if node.doc is not None:
            result.extend(self.print_docstring(node.doc))

        for import_ in sorted(node.imports, key=lambda i: i.origin):
            result.extend(self.print_import(import_))

        for sub_module in node.sub_modules:
            result.extend(self.print_submodule_import(sub_module.Name))

        # 将 __all__ 放在所有内容之上
        if node.all is not None:
            result.extend(self.print_variable(node.all))

        for type_var in sorted(node.type_vars, key=lambda t: t.name):
            result.extend(self.print_type_var(type_var))

        for class_ in sorted(node.classes, key=lambda c: c.name):
            result.extend(self.print_class(class_))

        for func in sorted(node.functions, key=lambda f: f.name):
            result.extend(self.print_function(func))

        for variable in sorted(node.variables, key=lambda v: v.name):
            result.extend(self.print_variable(variable))

        for alias in node.aliases:
            result.extend(self.print_alias(alias))

        return result

    def _uses_typing(self, node: IRModule) -> bool:
        def annotation_uses_typing(annotation: IRAnnotation | None) -> bool:
            if isinstance(annotation, ResolvedType):
                if len(annotation.name) > 0 and annotation.name[0] == "typing":
                    return True
                if annotation.parameters:
                    return any(annotation_uses_typing(p) for p in annotation.parameters)
            return False

        def function_uses_typing(func: IRFunction) -> bool:
            if annotation_uses_typing(func.returns):
                return True
            if any(annotation_uses_typing(arg.annotation) for arg in func.args):
                return True
            if any("typing." in decorator for decorator in func.decorators):
                return True
            return False

        def class_uses_typing(cls: IRClass) -> bool:
            if any(str(base).startswith("typing.") for base in cls.bases):
                return True
            if any(class_uses_typing(c) for c in cls.classes):
                return True
            if any(function_uses_typing(m.function) for m in cls.methods):
                return True
            if any(function_uses_typing(p.getter) for p in cls.properties if p.getter):
                return True
            if any(function_uses_typing(p.setter) for p in cls.properties if p.setter):
                return True
            if any(annotation_uses_typing(f.variable.annotation) for f in cls.fields):
                return True
            return False

        if any(class_uses_typing(c) for c in node.classes):
            return True
        if any(function_uses_typing(f) for f in node.functions):
            return True
        if node.all is not None and annotation_uses_typing(node.all.annotation):
            return True
        if any(annotation_uses_typing(v.annotation) for v in node.variables):
            return True
        if any(annotation_uses_typing(t.bound) for t in node.type_vars):
            return True
        return False

    def print_class(self, irclass: IRClass) -> list[str]:
        s = f"class {irclass.name}"
        if irclass.bases:
            s += f"({', '.join(str(base) for base in irclass.bases)})"
        s += ":"

        body = self._print_class_body(irclass)
        
        return [s, *self.indent_lines(body)]

    def _print_class_body(self, irclass: IRClass) -> list[str]:
        result = []
        if irclass.doc is not None:
            result.extend(self.print_docstring(irclass.doc))

        for sub_class in sorted(irclass.classes, key=lambda c: c.name):
            result.extend(self.print_class(sub_class))

        modifier_order: dict[IRModifier, int] = {
            "static": 0,
            "class": 1,
            None: 2,
        }
        for field in sorted(
            irclass.fields, key=lambda f: (modifier_order.get(f.modifier, 2), f.variable.name)
        ):
            result.extend(self.print_field(field))

        for alias in sorted(irclass.aliases, key=lambda a: a.name):
            result.extend(self.print_alias(alias))

        for method in sorted(
            irclass.methods, key=lambda m: (modifier_order.get(m.modifier, 2), m.function.name)
        ):
            result.extend(self.print_method(method))

        for prop in sorted(irclass.properties, key=lambda p: p.name):
            result.extend(self.visit_property(prop))

        if not result:
            result = ["pass"]

        return result

    def print_method(self, node: IRMethod) -> list[str]:
        result = []
        if node.modifier == "static":
            result.append("@staticmethod")
        elif node.modifier == "class":
            result.append("@classmethod")
        elif node.modifier is None:
            pass
        else:
            # 如果类型正确，则不应发生
            pass
            
        result.extend(self.print_function(node.function))
        return result

    def print_field(self, node: IRField) -> list[str]:
        # 待完善：如有需要，处理修饰符（例如 C++ 中的静态字段通常是隐含的）
        return self.print_variable(node.variable)

    def visit_property(self, node: IRProperty) -> list[str]:
        if not node.getter:
            # 待完善：支持仅 setter 的属性
            return []

        result = []

        result.extend(
            [
                "@property",
                *self.print_function(
                    dataclasses.replace(
                        node.getter,
                        name=node.name,
                        # 如果 prop.doc 存在，则替换 getter 文档字符串
                        doc=node.doc if node.doc is not None else node.getter.doc,
                    )
                ),
            ]
        )
        if node.setter:
            result.extend(
                [
                    f"@{node.name}.setter",
                    *self.print_function(
                        dataclasses.replace(
                            node.setter,
                            name=node.name,
                            # 如果 prop.doc 存在，则移除 setter 文档字符串
                            doc=None if node.doc is not None else node.setter.doc,
                        )
                    ),
                ]
            )

        return result

    # --- 辅助方法（私有类） ---

    def print_alias(self, alias: IRAlias) -> list[str]:
        return [f"{alias.name} = {alias.origin}"]

    def print_variable(self, variable: IRVariable) -> list[str]:
        parts = [
            f"{variable.name}",
        ]
        if variable.annotation is not None:
            parts.append(f": {self.print_annotation(variable.annotation)}")

        if variable.value is not None and variable.value.is_print_safe:
            parts.append(f" = {self.print_value(variable.value)}")
        else:
            if variable.annotation is None:
                parts.append(" = ...")
            if variable.value is not None:
                parts.append(f"  # value = {self.print_value(variable.value)}")

        return ["".join(parts)]

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
    
    def print_type_var(self, type_var: IRTypeVar) -> list[str]:
        return [str(type_var)]

    def print_docstring(self, doc: str) -> list[str]:
        return [
            '"""',
            *(
                line.replace("\\", r"\\").replace('"""', r"\"\"\"")
                for line in doc.splitlines()
            ),
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
        signature = [
            f"def {func.name}(",
            ", ".join(args),
            ")",
        ]

        if func.returns is not None:
            signature.append(f" -> {self.print_annotation(func.returns)}")
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

    def print_import(self, import_: IRImport) -> list[str]:
        parent = str(import_.origin.parent)
        if import_.name is None:
            return [f"import {import_.origin}"]

        if len(parent) == 0:
            return [f"import {import_.origin} as {import_.name}"]

        result = f"from {parent} import {import_.origin.name}"
        if import_.name != import_.origin.name:
            result += f" as {import_.name}"
        return [result]

    def print_value(self, value: IRValue) -> str:
        split = value.repr.split("\n", 1)
        if len(split) == 1:
            return split[0]
        else:
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
            param_str = (
                "["
                + ", ".join(self.print_annotation(p) for p in type_.parameters)
                + "]"
            )
        else:
            param_str = ""
        return f"{type_.name}{param_str}"

    def print_annotation(self, annotation: IRAnnotation) -> str:
        if isinstance(annotation, ResolvedType):
            return self.print_type(annotation)
        elif isinstance(annotation, IRValue):
            return self.print_value(annotation)
        elif isinstance(annotation, InvalidExpression):
            return self.print_invalid_exp(annotation)
        else:
            return str(annotation)

    def print_invalid_exp(self, invalid_expr: InvalidExpression) -> str:
        if self.invalid_expr_as_ellipses:
            return "..."
        return invalid_expr.text
