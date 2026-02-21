from __future__ import annotations

from ...IR import IRModule, IRImport, QualifiedName
from ..NodeVisitor import NodeVisitor

class FixMissingFutureAnnotationsVisitor(NodeVisitor):
    """向模块添加 `from __future__ import annotations`。"""
    
    def visit_module(self, node: IRModule) -> None:
        node.imports.add(
            IRImport(
                name="annotations",
                origin=QualifiedName.from_str("__future__.annotations"),
            )
        )
        super().visit_module(node)
