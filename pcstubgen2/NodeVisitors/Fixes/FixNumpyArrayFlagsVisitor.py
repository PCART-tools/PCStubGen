from __future__ import annotations

from typing import Any

from ...IR import (
    IRFunction, IRVariable, ResolvedType, QualifiedName
)
from ..NodeVisitor import NodeVisitor

class FixNumpyArrayFlagsVisitor(NodeVisitor):
    """修复 numpy 数组标志注释（例如，flags.writeable -> numpy.ndarray.flags.writeable）。"""
    
    __ndarray_name = QualifiedName.from_str("numpy.ndarray")
    __flags = {
        QualifiedName.from_str("flags.writeable"),
        QualifiedName.from_str("flags.c_contiguous"),
        QualifiedName.from_str("flags.f_contiguous"),
    }
    
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
        
        if annotation.name == self.__ndarray_name and annotation.parameters:
            for param in annotation.parameters:
                if isinstance(param, ResolvedType) and param.name in self.__flags:
                    param.name = QualifiedName.from_str(f"numpy.ndarray.{param.name}")
        
        # 递归修复参数
        if annotation.parameters:
            annotation.parameters = [
                self._fix_type(p) for p in annotation.parameters
            ]
        
        return annotation
