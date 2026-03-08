from __future__ import annotations

from ...IR import IRClass, IRModule
from ..NodeVisitor import NodeVisitor

class FixRedundantMethodsFromBuiltinObjectVisitor(NodeVisitor):
    """过滤掉与 object.__init__ 具有相同文档字符串的 __init__ 方法。"""
    
    def visit_class(self, node: IRClass, module: IRModule) -> None:
        node.methods = [
            m for m in node.methods
            if not (
                m.function.name == "__init__"
                and m.function.doc == object.__init__.__doc__
            )
        ]
        super().visit_class(node, module)
