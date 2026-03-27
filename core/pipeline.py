from __future__ import annotations

from typing import Sequence

from .ir import IRClass, IRMethod, IRModule
from .node_visitors.node_visitor import NodeVisitor


class Pipeline:
    def __init__(self, visitors: Sequence[NodeVisitor]):
        self.visitors = visitors

    def run(self, module: IRModule) -> IRModule:
        """按 visitor 顺序遍历整棵 IR 树。"""
        for visitor in self.visitors:
            self._visit_module(visitor, module)
        return module

    @staticmethod
    def _visit_module(visitor: NodeVisitor, module: IRModule) -> None:
        """访问模块，再递归访问其子模块、类和函数。"""
        visitor.visit_module(module)

        for sub_module in module.sub_modules:
            Pipeline._visit_module(visitor, sub_module)

        for cls in module.classes:
            Pipeline._visit_class(visitor, cls, module)

        for func in module.functions:
            visitor.visit_function(func, module)

    @staticmethod
    def _visit_class(
        visitor: NodeVisitor,
        node: IRClass,
        module: IRModule,
    ) -> None:
        """访问类，再递归访问其嵌套类和方法。"""
        visitor.visit_class(node, module)

        for nested_cls in node.classes:
            Pipeline._visit_class(visitor, nested_cls, module)

        for method in node.methods:
            Pipeline._visit_method(visitor, method, module)

    @staticmethod
    def _visit_method(
            visitor: NodeVisitor,
        method: IRMethod,
        module: IRModule,
    ) -> None:
        """访问方法节点，并继续触发其函数级钩子。"""
        visitor.visit_method(method, module)
        visitor.visit_function(method.function, module)
