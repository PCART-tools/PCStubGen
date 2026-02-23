from __future__ import annotations

from typing import Any

from ...IR import (
    IRClass, IRFunction, IRVariable, ResolvedType, QualifiedName
)
from ..NodeVisitor import NodeVisitor

class FixBuiltinTypesVisitor(NodeVisitor):
    """修复内置类型名称（例如，builtins.NoneType -> None）。"""
    
    def visit_class(self, node: IRClass) -> None:
        # 修复基类
        new_bases = []
        for base in node.bases:
            if (len(base) >= 1 and base[0] == "PyCapsule") or (
                len(base) >= 2 and base[0] == "builtins" and base[1] == "PyCapsule"
            ):
                new_bases.append(QualifiedName.from_str("typing.Any"))
                continue
            if len(base) >= 2 and base[0] == "builtins":
                if base[1] == "PyCapsule":
                    new_bases.append(QualifiedName.from_str("typing.Any"))
                    continue
                # 也为基类剥离 'builtins.' 前缀
                new_bases.append(QualifiedName(base[1:]))
            else:
                new_bases.append(base)
        node.bases = new_bases
        super().visit_class(node)
    
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
        
        name = annotation.name
        
        # 将 `builtins.X` 规范化为 `X`
        if len(name) == 1 and name[0] == "PyCapsule":
            return ResolvedType(name=QualifiedName.from_str("typing.Any"))
        if len(name) >= 2 and name[0] == "builtins":
            if name[1] == "PyCapsule":
                return ResolvedType(name=QualifiedName.from_str("typing.Any"))
            if name[1] == "NoneType":
                return ResolvedType(name=QualifiedName.from_str("None"))
            if name[1] in ("function", "builtin_function_or_method"):
                return ResolvedType(name=QualifiedName.from_str("typing.Callable"))
            # 剥离 'builtins.' 前缀
            annotation.name = QualifiedName(name[1:])
        
        # 递归修复参数
        if annotation.parameters:
            annotation.parameters = [
                self._fix_type(p) for p in annotation.parameters
            ]
        
        return annotation
