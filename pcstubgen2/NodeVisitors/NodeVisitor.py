from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..IR import (
        IRModule,
        IRClass,
        IRFunction,
        IRMethod,
        IRProperty,
        IRField,
        IRVariable,
    )


class NodeVisitor(abc.ABC):
    """所有访问者的抽象基类。"""

    def finalize(self) -> None:
        """访问完成后的收尾钩子。子类可按需覆盖。"""
        return None

    def visit_module(self, node: IRModule) -> None:
        """访问模块节点。"""
        for sub_module in node.sub_modules:
            self.visit_module(sub_module)
        for cls in node.classes:
            self.visit_class(cls)
        for func in node.functions:
            self.visit_function(func)
        if node.all is not None:
            self.visit_variable(node.all)
        for variable in node.variables:
            self.visit_variable(variable)
        return None

    def visit_class(self, node: IRClass) -> None:
        """访问类节点。"""
        for nested_cls in node.classes:
            self.visit_class(nested_cls)
        for method in node.methods:
            self.visit_method(method)
        for prop in node.properties:
            self.visit_property(prop)
        for field in node.fields:
            self.visit_field(field)
        return None

    def visit_function(self, node: IRFunction) -> None:
        """访问函数节点。"""
        return None

    def visit_method(self, node: IRMethod) -> None:
        """访问方法节点。"""
        self.visit_function(node.function)
        return None

    def visit_property(self, node: IRProperty) -> None:
        """访问属性节点。"""
        if node.getter:
            self.visit_function(node.getter)
        if node.setter:
            self.visit_function(node.setter)
        return None

    def visit_field(self, node: IRField) -> None:
        """访问字段节点。"""
        self.visit_variable(node.variable)
        return None

    def visit_variable(self, node: IRVariable) -> None:
        """访问变量节点。"""
        return None
