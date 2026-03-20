from __future__ import annotations

from typing import Any

from ...ir import (
    IRClass, IRFunction, IRModule, ResolvedType, QualifiedName
)
from ..NodeVisitor import NodeVisitor

class FixTypingTypeNamesVisitor(NodeVisitor):
    """修复 typing 模块类型名称（例如，sequence -> Sequence）。"""
    
    __typing_names = {
        "Annotated", "Any", "Buffer", "Callable", "Dict", "ItemsView",
        "Iterable", "Iterator", "KeysView", "List", "Literal", "Optional",
        "Sequence", "Set", "Tuple", "Union", "ValuesView",
        # 旧的 pybind11 注释未大写
        "buffer", "iterable", "iterator", "sequence",
    }
    
    def visit_class(self, node: IRClass, module: IRModule) -> None:
        # 修复基类
        new_bases = []
        for base in node.bases:
            if len(base) == 1:
                word = base[0]
                if word in self.__typing_names:
                    capitalized = word[0].upper() + word[1:]
                    new_bases.append(QualifiedName.from_str(f"typing.{capitalized}"))
                elif word == "function":
                    new_bases.append(QualifiedName.from_str("typing.Callable"))
                elif word in ("object", "handle"):
                    new_bases.append(QualifiedName.from_str("typing.Any"))
                else:
                    new_bases.append(base)
            else:
                new_bases.append(base)
        node.bases = new_bases
        super().visit_class(node, module)

    def visit_function(self, node: IRFunction) -> None:
        if node.return_annotation:
            node.return_annotation = self._fix_type(node.return_annotation)
        for arg in node.args:
            if arg.annotation:
                arg.annotation = self._fix_type(arg.annotation)
        super().visit_function(node)
    
    def _fix_type(self, annotation: Any) -> Any:
        if not isinstance(annotation, ResolvedType):
            return annotation
        
        # 修复单字 typing 名称
        if len(annotation.name) == 1:
            word = annotation.name[0]
            if word in self.__typing_names:
                # 首字母大写
                capitalized = word[0].upper() + word[1:]
                annotation.name = QualifiedName.from_str(f"typing.{capitalized}")
            if word == "function" and annotation.parameters is None:
                annotation.name = QualifiedName.from_str("typing.Callable")
            if word in ("object", "handle") and annotation.parameters is None:
                annotation.name = QualifiedName.from_str("typing.Any")
        
        # 递归修复参数
        if annotation.parameters:
            annotation.parameters = [
                self._fix_type(p) for p in annotation.parameters
            ]
        
        return annotation
