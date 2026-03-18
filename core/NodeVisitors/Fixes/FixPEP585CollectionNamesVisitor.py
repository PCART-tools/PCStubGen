from __future__ import annotations

from typing import Any

from ...IR import (
    IRClass, IRFunction, IRModule, ResolvedType, QualifiedName
)
from ..NodeVisitor import NodeVisitor

class FixPEP585CollectionNamesVisitor(NodeVisitor):
    """转换 typing.List -> list, typing.Dict -> dict 等 (PEP 585)。"""
    
    __typing_collection_names = {
        "Dict": "dict",
        "List": "list",
        "Set": "set",
        "Tuple": "tuple",
        "FrozenSet": "frozenset",
        "Type": "type",
    }
    
    def visit_class(self, node: IRClass, module: IRModule) -> None:
        # 修复基类
        new_bases = []
        for base in node.bases:
            if len(base) == 2 and base[0] == "typing":
                word = base[1]
                if word in self.__typing_collection_names:
                    new_bases.append(QualifiedName.from_str(
                        self.__typing_collection_names[word]
                    ))
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
        
        # 检查是否是应该为 x 的 typing.X
        if len(annotation.name) == 2 and annotation.name[0] == "typing":
            word = annotation.name[1]
            if word in self.__typing_collection_names:
                annotation.name = QualifiedName.from_str(
                    self.__typing_collection_names[word]
                )
        
        # 递归修复参数
        if annotation.parameters:
            annotation.parameters = [
                self._fix_type(p) for p in annotation.parameters
            ]
        
        return annotation
