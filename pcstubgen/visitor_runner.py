from __future__ import annotations

from collections.abc import Sequence

from .ir import IRClass, IRMethod, IRModule
from .visitors.node_visitor import NodeVisitor


def run_visitors(module: IRModule, visitors: Sequence[NodeVisitor]) -> IRModule:
    """按 visitor 顺序遍历整棵 IR 树。"""
    for visitor in visitors:
        _visit_module(visitor, module)
    return module


def _visit_module(visitor: NodeVisitor, module: IRModule) -> None:
    """访问模块，再递归访问其子模块、类和函数。"""
    visitor.visit_module(module)

    for sub_module in module.sub_modules:
        _visit_module(visitor, sub_module)

    for cls in module.classes:
        _visit_class(visitor, cls, module)

    for func in module.functions:
        visitor.visit_function(func, module)


def _visit_class(
    visitor: NodeVisitor,
    node: IRClass,
    module: IRModule,
) -> None:
    """访问类，再递归访问其嵌套类和方法。"""
    visitor.visit_class(node, module)

    for nested_cls in node.classes:
        _visit_class(visitor, nested_cls, module)

    for method in node.methods:
        _visit_method(visitor, method, module)


def _visit_method(
    visitor: NodeVisitor,
    method: IRMethod,
    module: IRModule,
) -> None:
    """访问方法节点，并继续触发其函数级钩子。"""
    visitor.visit_method(method, module)
    visitor.visit_function(method.function, module)
