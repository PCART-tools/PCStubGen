from __future__ import annotations

from typing import Any

from ...IR import (
    IRFunction, IRVariable, ResolvedType
)
from ..NodeVisitor import NodeVisitor

class FixScipyTypeArgumentsVisitor(NodeVisitor):
    """从 scipy.sparse 数组/矩阵中移除类型参数（它们不是泛型的）。"""
    
    def visit_function(self, node: IRFunction) -> None:
        if node.return_annotation:
            node.return_annotation = self._fix_type(node.return_annotation)
        for arg in node.args:
            if arg.annotation:
                arg.annotation = self._fix_type(arg.annotation)
        super().visit_function(node)
    
    def visit_variable(self, node: IRVariable) -> None:
        if node.annotation:
            node.annotation = self._fix_type(node.annotation)
        super().visit_variable(node)
    
    def _fix_type(self, annotation: Any) -> Any:
        if not isinstance(annotation, ResolvedType):
            return annotation
        
        # `scipy.sparse` 数组/矩阵目前不是泛型
        if len(annotation.name) >= 2 and annotation.name[0] == "scipy" and annotation.name[1] == "sparse":
            annotation.parameters = None
        
        # 递归修复参数
        if annotation.parameters:
            annotation.parameters = [
                self._fix_type(p) for p in annotation.parameters
            ]
        
        return annotation
