from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..IR import (
        IRModule,
        IRClass,
        IRFunction,
        IRMethod,
    )


class NodeVisitor(abc.ABC):
    """所有访问者的抽象基类。"""

    def visit_module(self, node: IRModule) -> None:
        """访问模块节点。"""
        for sub_module in node.sub_modules:
            self.visit_module(sub_module)
        new_classes: list[IRClass] = []
        for cls in node.classes:
            visited_cls = self.visit_class(cls)
            if visited_cls is not None:
                new_classes.append(visited_cls)
        node.classes = new_classes

        new_functions: list[IRFunction] = []
        for func in node.functions:
            visited_func = self.visit_function(func)
            if visited_func is not None:
                new_functions.append(visited_func)
        node.functions = new_functions
        return None

    def visit_class(self, node: IRClass) -> IRClass | None:
        """访问类节点。"""
        new_classes: list[IRClass] = []
        for nested_cls in node.classes:
            visited_cls = self.visit_class(nested_cls)
            if visited_cls is not None:
                new_classes.append(visited_cls)
        node.classes = new_classes

        new_methods: list[IRMethod] = []
        for method in node.methods:
            visited_method = self.visit_method(method)
            if visited_method is not None:
                new_methods.append(visited_method)
        node.methods = new_methods

        return node

    def visit_function(self, node: IRFunction) -> IRFunction | None:
        """访问函数节点。"""
        return node

    def visit_method(self, node: IRMethod) -> IRMethod | None:
        """访问方法节点。"""
        visited_function = self.visit_function(node.function)
        if visited_function is None:
            return None
        node.function = visited_function
        return node
