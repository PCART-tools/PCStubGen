from __future__ import annotations

from ...IR import IRMethod
from ..NodeVisitor import NodeVisitor


class InferMethodModifierVisitor(NodeVisitor):
    """根据方法首参统一推导方法修饰器。"""

    def visit_method(self, node: IRMethod) -> IRMethod | None:
        node.decorator = self._infer_decorator(node)
        return super().visit_method(node)

    def _infer_decorator(self, node: IRMethod) -> str | None:
        args = node.function.args
        if args:
            first = args[0].name
            if first == "self":
                return None
            if first == "cls":
                return "classmethod"
        return "staticmethod"
