from __future__ import annotations

from ...IR import (
    IRClass, ResolvedType, QualifiedName
)
from ..NodeVisitor import NodeVisitor

class FixMissingNoneHashFieldAnnotationVisitor(NodeVisitor):
    """为 __hash__ = None 字段添加正确的注释。"""
    
    def visit_class(self, node: IRClass) -> None:
        for field in node.fields:
            if (
                field.variable.name == "__hash__"
                and field.variable.value is not None
                and field.variable.value.repr == "None"
            ):
                field.variable.annotation = ResolvedType(
                    name=QualifiedName.from_str("typing.ClassVar"),
                    parameters=[ResolvedType(name=QualifiedName.from_str("None"))],
                )
        super().visit_class(node)
