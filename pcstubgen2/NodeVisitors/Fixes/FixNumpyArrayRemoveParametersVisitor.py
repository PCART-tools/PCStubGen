from __future__ import annotations

from typing import Any

from ...IR import (
    IRFunction, IRVariable, ResolvedType, QualifiedName
)
from ..NodeVisitor import NodeVisitor

class FixNumpyArrayRemoveParametersVisitor(NodeVisitor):
    """简化 numpy.ndarray[...] 为 numpy.ndarray（移除类型参数）。"""
    
    __ndarray_name = QualifiedName.from_str("numpy.ndarray")
    
    def visit_function(self, node: IRFunction) -> None:
        if node.returns:
            node.returns = self._fix_type(node.returns)
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
        
        # 从 numpy.ndarray 移除参数
        if annotation.name == self.__ndarray_name:
            annotation.parameters = None
        
        # 递归修复参数（针对嵌套类型）
        if annotation.parameters:
            annotation.parameters = [
                self._fix_type(p) for p in annotation.parameters
            ]
        
        return annotation
