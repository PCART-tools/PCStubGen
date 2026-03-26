from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ir import (
        IRModule,
        IRClass,
        IRFunction,
        IRMethod,
    )


class NodeVisitor:
    """所有访问者的抽象基类。"""

    def visit_module(self, node: IRModule) -> None:
        """访问模块节点。"""

    def visit_class(self, node: IRClass, module: IRModule) -> None:
        """访问类节点。"""

    def visit_function(self, node: IRFunction, module: IRModule) -> None:
        """访问函数节点。"""

    def visit_method(self, node: IRMethod, module: IRModule) -> None:
        """访问方法节点。"""
