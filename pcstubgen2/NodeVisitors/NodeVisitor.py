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
        for cls in node.classes:
            self.visit_class(cls)

        for func in node.functions:
            self.visit_function(func)

    def visit_class(self, node: IRClass) -> None:
        """访问类节点。"""
        for nested_cls in node.classes:
            self.visit_class(nested_cls)

        for method in node.methods:
            self.visit_method(method)

    def visit_function(self, node: IRFunction) -> None:
        """访问函数节点。"""

    def visit_method(self, node: IRMethod) -> None:
        """访问方法节点。"""
        self.visit_function(node.function)
