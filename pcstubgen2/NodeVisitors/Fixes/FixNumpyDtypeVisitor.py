from __future__ import annotations

from typing import Any

from ...IR import (
    IRFunction, IRVariable, ResolvedType, QualifiedName
)
from ..NodeVisitor import NodeVisitor

class FixNumpyDtypeVisitor(NodeVisitor):
    """确保 numpy.dtype 具有类型参数。"""
    
    __numpy_dtype = QualifiedName.from_str("numpy.dtype")
    
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
        
        # `numpy.dtype` 是泛型，应该带一个类型参数
        if annotation.parameters is None or len(annotation.parameters) == 0:
            name = annotation.name
            if (
                (len(name) == 1 and name[0] == "dtype")
                or (len(name) == 2 and name[0] == "numpy" and name[1] == "dtype")
            ):
                annotation.name = self.__numpy_dtype
                annotation.parameters = [
                    ResolvedType(name=QualifiedName.from_str("typing.Any"))
                ]
        
        # 递归修复参数
        if annotation.parameters:
            annotation.parameters = [
                self._fix_type(p) for p in annotation.parameters
            ]
        
        return annotation
