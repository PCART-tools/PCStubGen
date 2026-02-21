from __future__ import annotations

from ...IR import IRVariable
from ..NodeVisitor import NodeVisitor

class FixRedundantBuiltinsAnnotationVisitor(NodeVisitor):
    """移除 None 和模块类型的冗余注释。"""
    
    def visit_variable(self, node: IRVariable) -> None:
        if node.value is not None:
            # __hash__ 在存根里通常需要保留显式注解
            if node.name == "__hash__":
                super().visit_variable(node)
                return
            # 检查值是否为 None
            if node.value.repr == "None":
                node.annotation = None
        super().visit_variable(node)
