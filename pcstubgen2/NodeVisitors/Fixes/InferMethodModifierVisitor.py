from __future__ import annotations

from ...IR import IRMethod
from ..NodeVisitor import NodeVisitor


class InferMethodModifierVisitor(NodeVisitor):
    """根据方法首参统一推导 method modifier。"""

    def visit_method(self, node: IRMethod) -> None:
        node.modifier = self._infer_modifier(node)
        super().visit_method(node)

    def _infer_modifier(self, node: IRMethod) -> str | None:
        args = node.function.args
        if len(args) == 0:
            return "static"
        first = args[0].name
        if first == "self":
            return None
        if first == "cls":
            return "class"
        return "static"
